"""
Base class for trainable models.
"""

from abc import abstractmethod
from copy import copy
from omegaconf import OmegaConf
from torch import nn




class BaseModel(nn.Module):
    base_default_conf = {
        "name": None,
        "trainable": True,
        "freeze_batch_normalization": True,  # use test-time statistics
        "timeit": False,
    }
    default_conf = {}
    required_data_keys = []
    strict_conf = False
    are_weights_initialized = False

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()

        default_conf = OmegaConf.merge(self.base_default_conf, OmegaConf.create(self.default_conf))

        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self.required_data_keys = copy(self.required_data_keys)

        self._init(conf)

        if not conf.trainable:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        if self.conf.freeze_batch_normalization:
            self.apply(freeze_bn)

        return self

    def load_state_dict(self, *args, **kwargs):
        """Load the state dict of the model, and set the model to initialized."""
        ret = super().load_state_dict(*args, **kwargs)
        self.set_initialized()
        return ret

    def is_initialized(self):
        """Recursively check if the model is initialized, i.e. weights are loaded"""
        is_initialized = True  # initialize to true and perform recursive and
        for _, w in self.named_children():
            if isinstance(w, BaseModel):
                # if children is BaseModel, we perform recursive check
                is_initialized = is_initialized and w.is_initialized()
            else:
                # else, we check if self is initialized or the children has no params
                n_params = len(list(w.parameters()))
                is_initialized = is_initialized and (
                    n_params == 0 or self.are_weights_initialized
                )
        return is_initialized

    def set_initialized(self, to: bool = True):
        """Recursively set the initialization state."""
        self.are_weights_initialized = to
        for _, w in self.named_parameters():
            if isinstance(w, BaseModel):
                w.set_initialized(to)

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""

        def recursive_key_check(requirekeys, data):
            for key in requirekeys:
                assert key in data, f"Missing key {key} in data"
                if isinstance(requirekeys, dict):
                    recursive_key_check(requirekeys[key], data[key])

        recursive_key_check(self.required_data_keys, data)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError



 


if __name__=="__main__":
    basemodel = BaseModel({"jjj":"hhh"})


