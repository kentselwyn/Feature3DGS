import torch
import collections.abc as collections


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


string_classes = (str, bytes)
def map_tensor(input_, func):
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif input_ is None:
        return None
    else:
        return func(input_)


def batch_to_device(batch, device, non_blocking=True):
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking)
    return map_tensor(batch, _func)



def get_homography_data(batch = 4):
    if batch==1:
        homography_data = torch.load('/home/koki/gluetrain/data/testdata/batch1_two_view_homography.pth')
    else:
        homography_data = torch.load('/home/koki/gluetrain/data/testdata/batch4_two_view_homography.pth')
    return batch_to_device(homography_data, device=device)


def count_trainable_params(model):
    count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return count
