import torch
from .tensor import batch_to_device
from .tools import AverageMetric


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
            desc1 = data[1].to(device)
            pred0 = model(desc0)
            pred1 = model(desc1)
            loss0 = loss_fn(pred0, desc0)
            loss1 = loss_fn(pred1, desc1)
            loss = loss0 + loss1
            total_loss+=loss
    average_loss = total_loss/len(loader)

    return average_loss
