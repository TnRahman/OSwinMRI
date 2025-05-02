import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


"""
# --------------------------------------------
# Custom layers for weighting parameters
# --------------------------------------------
"""

class DCWeights(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DCWeights,self).__init__(*args, **kwargs)
        self.dcw = nn.Parameter(torch.tensor(1.0, dtype=torch.float, requires_grad=True))
        # Other necessary setup
        

    def forward(self, x):
        # Necessary forward computations
        return self.dcw*x

    def save_dcw(self, save_dir, dcw, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        state_dict = dcw.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_dcw(self, load_path, dcw, strict=True, param_key='params'):
        network = self.get_bare_model(dcw)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old), (key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

class DCWeightsCC(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DCWeightsCC,self).__init__(*args, **kwargs)
        self.dcw = nn.Parameter(torch.tensor(1.0, dtype=torch.float, requires_grad=True))
        # Other necessary setup
        

    def forward(self, x, y):
        # Necessary forward computations
        return self.dcw*x + (1-self.dcw)*y

    def save_dcw(self, save_dir, dcw, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        state_dict = dcw.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_dcw(self, load_path, dcw, strict=True, param_key='params'):
        network = self.get_bare_model(dcw)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old), (key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network



