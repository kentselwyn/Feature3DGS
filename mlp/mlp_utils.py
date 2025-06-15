import torch
from .tensor import batch_to_device
from .tools import AverageMetric
import torch.nn.functional as F


@torch.no_grad() 
def do_evaluation2(model, loader, device):
    model.eval()
    results = {}
    loss_fn = model.loss
    for index, data in enumerate(loader):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses, metrics = loss_fn(pred, data)
            del pred, data
        numbers = {**metrics, **{"loss/" + key: value for key, value in losses.items()}}
        for key, value in numbers.items():
            if key not in results:
                results[key] = AverageMetric()
            results[key].update(value)
        del numbers
    results = {key: results[key].compute() for key in results}

    return results




def train_one_epoch(model, trainloader, epoch, device, optimizer):
    model.train(True)
    for i, data in enumerate(trainloader):
        data = batch_to_device(data, device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(data)
        losses, _ = model.loss(pred, data)
        loss = torch.mean(losses["total"])
        loss.backward()
        optimizer.step()
        if i % 100 == 99:
            for key in sorted(losses.keys()):
                losses[key] = torch.mean(losses[key], -1)
                losses[key] = losses[key].item()
            str_losses = [f"{key} {value:.3E}" for key, value in losses.items()]
            print( "[E {} | iter {}] loss {{{}}}".format(epoch, i, ", ".join(str_losses)))
    # torch.cuda.empty_cache()
    losses[key] = torch.mean(losses[key], -1)
    losses[key] = losses[key].item()
    return losses



@torch.no_grad() 
def do_evaluation3(model, loader, device):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    total_loss = 0
    for index, data in enumerate(loader):
        # data = batch_to_device(data, device, non_blocking=True)
        # with torch.no_grad():
        #     pred = model(data)
        #     loss = loss_fn(pred, data)
        #     del pred, data
        #     total_loss+=loss
        with torch.no_grad():
            desc0 = data[0].to(device)
            # desc1 = data[1].to(device)
            pred0 = model(desc0)
            # pred1 = model(desc1)
            loss0 = loss_fn(pred0, desc0)
            # loss1 = loss_fn(pred1, desc1)
            loss = loss0
            total_loss+=loss
    average_loss = total_loss/len(loader)
    return average_loss


def contrastive_loss(z1, z2, label, margin=1.0):
    d = F.pairwise_distance(z1, z2, p=2)  # L2 距離
    loss = label * d.pow(2) + (1 - label) * F.relu(margin - d).pow(2)
    return loss.mean()

def l2_loss(z1, z2, label):
    # 只對正樣本計算 L2，相當於加強收斂性
    pos_mask = label == 1
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=z1.device)
    return F.mse_loss(z1[pos_mask], z2[pos_mask])


@torch.no_grad() 
def do_evaluation_pair(args, model, loader, device):
    model.eval()
    total_loss = 0
    total_l2 = 0
    total_contrastive = 0
    for index, (f1, f2, label) in enumerate(loader):
        with torch.no_grad():
            f1, f2, label = f1.to(device), f2.to(device), label.to(device)
            z1, z2 = model(f1), model(f2)
            loss_contrast = contrastive_loss(z1, z2, label, margin=args.margin)
            loss_l2 = l2_loss(z1, z2, label)
            loss = loss_contrast + args.l2_weight * loss_l2
            total_loss+=loss
            total_l2+=loss_l2
            total_contrastive+=loss_contrast
    return total_loss/len(loader), total_l2/len(loader), total_contrastive/len(loader)





def triplet_loss(f_pos0, f_pos1, f_neg0, f_neg1, margin=0.2):

    N = min(f_pos0.shape[0], f_neg1.shape[0], f_neg0.shape[0], f_pos1.shape[0])
    f_pos0 = f_pos0[:N]
    f_pos1 = f_pos1[:N]
    f_neg0 = f_neg0[:N]
    f_neg1 = f_neg1[:N]

    loss1 = F.relu(F.pairwise_distance(f_pos0, f_pos1) -
                   F.pairwise_distance(f_pos0, f_neg1) + margin)
    loss2 = F.relu(F.pairwise_distance(f_pos1, f_pos0) -
                   F.pairwise_distance(f_pos1, f_neg0) + margin)

    return (loss1.mean() + loss2.mean()) / 2


@torch.no_grad() 
def do_evaluation_pair2(args, model, loader, device):
    model.eval()
    total_loss = 0
    total_l2 = 0
    total_triplet = 0
    mse_weight = 1.0
    for index, batch in enumerate(loader):
        with torch.no_grad():
            batch = batch_to_device(batch, device)

            c_desc0, rec_desc0 = model(batch["desc0"])  # [B, L, D]
            c_desc1, rec_desc1 = model(batch["desc1"])

            # 只對有效位置計算重建 MSE
            mse0 = F.mse_loss(rec_desc0[batch["mask0"]], batch["desc0"][batch["mask0"]])
            mse1 = F.mse_loss(rec_desc1[batch["mask1"]], batch["desc1"][batch["mask1"]])
            mse_loss = (mse0 + mse1) / 2

            # 計算 triplet loss（使用 encoder 輸出的 compressed descriptor）
            c_pos0, _ = model(batch["pos0"])
            c_pos1, _ = model(batch["pos1"])
            c_neg0, _ = model(batch["neg0"])
            c_neg1, _ = model(batch["neg1"])
            loss_triplet = triplet_loss(c_pos0, c_pos1, c_neg0, c_neg1)
            loss = loss_triplet + mse_weight * mse_loss

            total_loss+=loss
            total_l2 += mse_loss
            total_triplet+=loss_triplet

    return total_loss/len(loader), total_l2/len(loader), total_triplet/len(loader)
