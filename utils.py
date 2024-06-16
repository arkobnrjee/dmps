import torch


def polyak_update(net_params, targ_params, tau):
    with torch.no_grad():
        for param, target_param in zip(net_params, targ_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
