import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import Callable, List, Optional



FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True


# normaliza key points position
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)

    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts




# position embedding
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        

        return emb.repeat_interleave(2, dim=-1)





# for width dropping, or point pruning
def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    matching0, matching1 = max0.indices, max1.indices
    
    indices0 = torch.arange(matching0.shape[1], device=matching0.device)[None]
    indices1 = torch.arange(matching1.shape[1], device=matching1.device)[None]



    mutual0 = indices0 == matching1.gather(1, matching0)
    mutual1 = indices1 == matching0.gather(1, matching1)

    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, matching1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, matching1)
    
    matching0 = torch.where(valid0, matching0, -1)
    matching1 = torch.where(valid1, matching1, -1)
    
    return matching0, matching1, mscores0, mscores1

# for depth pruning
class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), 
                                   nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices)
        correct1 = (la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices)
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0





# all for attention mechanism
class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)

class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))

class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None
    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)
    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            matching0 = self.flash(qk0, qk1, v1, mask)
            matching1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            matching0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            matching1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                matching0, matching1 = matching0.nan_to_num(), matching1.nan_to_num()
        matching0, matching1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), matching0, matching1)
        matching0, matching1 = self.map_(self.to_out, matching0, matching1)
        x0 = x0 + self.ffn(torch.cat([x0, matching0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, matching1], -1))
        return x0, x1

class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)






def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    batch, kpt0_num, kpt1_num = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    
    scores = sim.new_full((batch, kpt0_num + 1, kpt1_num + 1), 0)
    
    scores[:, :kpt0_num, :kpt1_num] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores

# find match assignment by descriptors
class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, desc_dim = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / desc_dim**0.25, mdesc1 / desc_dim**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)






class LightGlue(nn.Module):
    
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "weights": "superpoint",  # either a path or the name of pretrained weights (disk, ...), superpoint
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]
    # required_data_keys = ["keypoints0", "keypoints1", "descriptors_back0", "descriptors_back1"]

    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(2+2*conf.add_scale_ori, head_dim, head_dim)

        num_heads, n_layers, descriptor_dim = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList([TransformerLayer(descriptor_dim, num_heads, conf.flash) for _ in range(n_layers)])
        self.log_assignment = nn.ModuleList([MatchAssignment(descriptor_dim) for _ in range(n_layers)])
        self.token_confidence = nn.ModuleList([TokenConfidence(descriptor_dim) for _ in range(n_layers - 1)])

        state_dict = None

        if conf.weights is not None:
            # weights can be either a path or an existing file from official LG
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, 
                                        map_location="cpu")
            else:
                fname = (f"{conf.weights}_{conf.weights_from_version}".replace(".", "-") + ".pth")
                state_dict = torch.hub.load_state_dict_from_url(
                                                        self.url.format(conf.weights_from_version, conf.weights), 
                                                        file_name=fname,)
        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn("Point pruning is partially disabled for compiled forward.", stacklevel=2,)

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(self.transformers[i], mode=mode, fullgraph=True)




    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        batch, kpt0_num, _ = kpts0.shape
        batch, kpt1_num, _ = kpts1.shape
        device = kpts0.device
        
        size0 = data.get("image_size")
        size1 = data.get("image_size")


        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat([kpts0, 
                               sc0 if sc0.dim() == 3 else sc0[..., None], 
                                o0 if o0.dim() == 3 else o0[..., None],],  -1,)
            kpts1 = torch.cat([kpts1, 
                               sc1 if sc1.dim() == 3 else sc1[..., None],
                                o1 if o1.dim() == 3 else o1[..., None],],  -1,)

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()
        # desc0 = data["descriptors_back0"].contiguous()
        # desc1 = data["descriptors_back1"].contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()


        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)


        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training

        all_desc0, all_desc1 = [], []
        all_assignments = []

        if do_point_pruning:
            ind0 = torch.arange(0, kpt0_num, device=device)[None]
            ind1 = torch.arange(0, kpt1_num, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = checkpoint(self.transformers[i], desc0, desc1, encoding0, encoding1)
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)
            
            if i==2 or i==5 or i==8:
                score, _ = self.log_assignment[i](desc0, desc1)
                # print(score.shape)
                all_assignments.append(score)
            
            all_desc0.append(desc0)
            all_desc1.append(desc1)
            if self.training or i == self.conf.n_layers - 1:
                continue
            
            # only for eval
            if do_early_stop:
                assert batch == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :kpt0_num, :], token1[..., :kpt1_num, :], i, kpt0_num + kpt1_num):
                    break
            if do_point_pruning:
                assert batch == 1
                
                # ####################
                # for desc0
                # ####################
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                # ####################
                # for desc1
                # ####################
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1
            
            

        desc0, desc1 = desc0[..., :kpt0_num, :], desc1[..., :kpt1_num, :]

        scores, _ = self.log_assignment[i](desc0, desc1)
        matching0, matching1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        if do_point_pruning:
            m0_ = torch.full((batch, kpt0_num), -1, device=matching0.device, dtype=matching0.dtype)
            m1_ = torch.full((batch, kpt1_num), -1, device=matching1.device, dtype=matching1.dtype)
            
            m0_[:, ind0] = torch.where(matching0 == -1, -1, ind1.gather(1, matching0.clamp(min=0)))
            m1_[:, ind1] = torch.where(matching1 == -1, -1, ind0.gather(1, matching1.clamp(min=0)))
            
            mscores0_ = torch.zeros((batch, kpt0_num), device=mscores0.device)
            mscores1_ = torch.zeros((batch, kpt1_num), device=mscores1.device)
            
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            
            matching0, matching1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "m0": matching0,
            "m1": matching1,
            
            "mscores0": mscores0,
            "mscores1": mscores1,
            
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            
            "log_assignment": scores,
            "all_log_assignments": all_assignments,
            
            "prune0": prune0,
            "prune1": prune1,
        }

        return pred



    def confidence_thresholds(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep





    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]








__main_model__ = LightGlue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def test_model():
    model = __main_model__({}).train().to(device)
    # data = get_homography_data(batch=4)

    init_cp = torch.load("/home/koki/gluetrain/outputs/train/records/superglue_lightglue/240410_sp_lg/checkpoint_best.tar", map_location="cpu")

    model_dict = init_cp['model']
    state_dict = {k.replace('matcher.','',1): v for k, v in model_dict.items() if k.split('.')[0] == "matcher"}


    model.load_state_dict(state_dict)
    # pred = model(data)

    # loss, metric = model.loss(pred, data)
    # gt_neg0 = (data["gt_matches0"] == -1).float()
    # gt_neg1 = (data["gt_matches1"] == -1).float()


    # print(pred.keys())
    # from pprint import pprint
    # pprint(loss)
    # pprint(metric)



# python -m core.models.matcher.lightglue
if __name__=="__main__":
    test_model()




