from nested_dict import nested_dict
from functools import partial
import torch
from torch.nn.init import kaiming_normal_
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F


def conv_params(ni, no, k=1):
    """ initialize convolutional network's parameters normal distribution
    :param ni: input dimension
    :param no: output dimension
    :param k: conv size (default: 1)

    kaiming_normal_: N(0, std^2), std = gain/sqrt(fan_mode)
    """
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    """ initialize the linear weight and bias
    :param ni: input dimension
    :param no: output dimension
    """
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bn_params(n):
    """ initialize the batch normalization
    :param n: BN input size
    :return: BN parameters
    """
    return {
        'weight': torch.rand(n),  # random
        'bias': torch.zeros(n),   # all zeros
        'running_mean': torch.zeros(n),
        'running_var': torch.ones(n)
    }


def set_requires_grad_except_bn_(params):
    """ set `requires_grad = True`
    :param params: all parameters except bn
    """
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True


def flatten(params):
    """ flatten dictionary
    :return: data type e.g. {'a.b.c': value1, 'a.b.d': value2, ...}
    """
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def cast(params, dtype='float'):
    """ recursively access the dictionary data and store into CUDA
    :return: data type e.g. {'a.b.c': tensor(value1, float), 'a.b.d': tensor(value2, float), ...}
    """
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)



def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    device_ids = list(device_ids)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)







def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


