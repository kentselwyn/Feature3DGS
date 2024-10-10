import torch
import torch.nn as nn
from omegaconf import OmegaConf


def weight_loss(log_assignment, weights, gamma=0.0):
    batch, row_num, col_num = log_assignment.shape
    row_num -= 1
    col_num -= 1

    loss_score = log_assignment * weights

    # num_neg computation
    num_neg0 = weights[:, :row_num, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :col_num].sum(-1).clamp(min=1.0)
    
    # num_pos computation
    num_pos = weights[:, :row_num, :col_num].sum((-1, -2)).clamp(min=1.0)

    # nll_pos computation
    nll_pos = -loss_score[:, :row_num, :col_num].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    # nll_neg computation
    nll_neg0 = -loss_score[:, :row_num, -1].sum(-1)
    nll_neg1 = -loss_score[:, -1, :col_num].sum(-1)
    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)
    # breakpoint()

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.compute_weights = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment_compress"]

        # always produce gt weights
        if weights is None:
            weights = self.compute_weights(log_assignment, data)
        
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(log_assignment, weights, gamma=self.conf.gamma_f)
        
        nll = (self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg)

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data):
        row_num, col_num = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        positive = data["gt_assignment"].float()
        
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :row_num, :col_num] = positive

        weights[:, :row_num, -1] = neg0
        weights[:, -1, :col_num] = neg1
        
        
        return weights





class NLLLoss_lg(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.compute_weights = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]

        # always produce gt weights
        if weights is None:
            weights = self.compute_weights(log_assignment, data)
        
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(log_assignment, weights, gamma=self.conf.gamma_f)
        
        nll = (self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg)

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data):
        row_num, col_num = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        positive = data["gt_assignment"].float()
        
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :row_num, :col_num] = positive

        weights[:, :row_num, -1] = neg0
        weights[:, -1, :col_num] = neg1
        
        
        return weights











# python -m core.models.utils.losses
if __name__=="__main__":
    log_assignment = torch.ones(5, 10, 2)
    weights = torch.ones(5, 10, 2)
    nll_pos, nll_neg, num_pos, avg_neg = weight_loss(log_assignment, weights)

    loss_fn = NLLLoss({})



    
    print(nll_pos)
    print(nll_neg)

    print(num_pos)
    print(avg_neg)








