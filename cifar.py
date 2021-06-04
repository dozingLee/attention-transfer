"""
    PyTorch training code for
    "Paying More Attention to Attention: Improving the Performance of
                Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    
    This file includes:
     * CIFAR ResNet and Wide ResNet training code which exactly reproduces
       https://github.com/szagoruyko/wide-residual-networks
     * Activation-based attention transfer
     * Knowledge distillation implementation

    2017 Sergey Zagoruyko
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import utils

'''
    The built-in auto tuner of cudnn automatically finds the most suitable 
    algorithm for the current convolution network structure, which is suitable 
    for the situation that the network structure and network input do not change.
'''
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--depth', default=16, type=int,
                    metavar="D", help="residual network depth (default: 16, depth must be 6n+4)")
parser.add_argument('--width', default=1, type=int,
                    metavar="W", help="(default: 1)")
parser.add_argument('--dataset', default='CIFAR10', type=str,
                    metavar="DATASET", help="training dataset (default: CIFAR10)")
parser.add_argument('--dataroot', default='.', type=str,
                    metavar="PATH", help="dataset root (default:  ., current directory)")
parser.add_argument('--dtype', default='float', type=str,
                    metavar="DTYPE", help="data type (default: float)")
parser.add_argument('--nthread', default=4, type=int,
                    metavar="N", help="number of dataloader working thread (default: 4)")
parser.add_argument('--teacher_id', default='', type=str,
                    metavar="ID", help="teacher id (default: none)")

# Training options
parser.add_argument('--batch_size', default=128, type=int,
                    metavar="N", help="input batch size for training (default: 128)")
parser.add_argument('--lr', default=0.1, type=float,
                    metavar="LR", help="learning rate (default: 0.1)")
parser.add_argument('--epochs', default=200, type=int,
                    metavar='N', help='number of total epochs to run (default: 200)')
parser.add_argument('--weight_decay', '-wd', default=0.0005, type=float,
                    metavar="W", help="weight decay (default: 0.0005)")
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    metavar="N", help='json list with epochs to drop lr on (default: [60,120,160])')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float,
                    metavar="R", help="learning rate decay ratio (default: 0.2)")
parser.add_argument('--resume', default='', type=str,
                    metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument('--randomcrop_pad', default=4, type=int,
                    metavar="R", help="random crop padding (default: 4)")
parser.add_argument('--temperature', default=4, type=float,
                    metavar="T")
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)

# Device options
parser.add_argument('--cuda', action='store_true', help="uses CUDA training")
parser.add_argument('--save', default='', type=str,
                    metavar="PATH", help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    metavar="N", help='number of GPUs to use for training (default: 1)')
parser.add_argument('--gpu_id', default='0', type=str,
                    metavar="ID", help='id(s) for CUDA_VISIBLE_DEVICES (default: 0)')


def create_dataset(opt, train):
    """
    :param opt: argument parser
    :param train: True or False
    :return: load dataset
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0)
    ])
    if train:
        transform = T.Compose([
            T.Pad(opt.randomcrop_pad, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])

    '''
    getattr(object, name[, default])
        :param object: Object with several attributes
        :param name: Data type is string (so the reflection is used here)
        :return: object['name']
    '''
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6  # divide, round down
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        """ Block
        :param ni: input size
        :param no: output size
        :return:
            conv0 weight: ni×no×3×3
            conv1 weight: no×no×3×3
            BN parameters: weight, bias, running_mean, running_var (4×n)
            convdim weight: ni×no×1×1, mapping ni to no
        """
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),  #
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        """ Group
        :param ni: input size
        :param no: output size
        :param count: group size
        :return:
            
        """
        return {
            'block%d' % i: gen_block_params(ni if i == 0 else no, no) for i in range(count)
        }

    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride):
        for i in range(n):
            o = block(o, params, f'{base}.block{i}', mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode, base=''):
        x = F.conv2d(input, params[f'{base}conv0'], padding=1)
        g0 = group(x, params, f'{base}group0', mode, 1)
        g1 = group(g0, params, f'{base}group1', mode, 2)
        g2 = group(g1, params, f'{base}group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, f'{base}bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params[f'{base}fc.weight'], params[f'{base}fc.bias'])
        return o, (g0, g1, g2)

    return f, flat_params


def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    # deal with student first
    f_s, params_s = resnet(opt.depth, opt.width, num_classes)

    # deal with teacher
    if opt.teacher_id:
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:])
        f_t = resnet(info['depth'], info['width'], num_classes)[0]
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7'))
        params_t = model_data['params']

        # merge teacher and student params
        params = {'student.' + k: v for k, v in params_s.items()}
        for k, v in params_t.items():
            params['teacher.' + k] = v.detach().requires_grad_(False)

        def f(inputs, params, mode):
            y_s, g_s = f_s(inputs, params, mode, 'student.')
            with torch.no_grad():
                y_t, g_t = f_t(inputs, params, False, 'teacher.')
            return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]
    else:
        f, params = f_s, params_s

    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD((v for v in params.values() if v.requires_grad), lr,
                   momentum=0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    utils.print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in list(params_s.values()))
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = utils.cast(sample[0], opt.dtype).detach()
        targets = utils.cast(sample[1], 'long')
        if opt.teacher_id != '':
            y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
            loss_groups = [v.sum() for v in loss_groups]
            [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
            return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) \
                   + opt.beta * sum(loss_groups), y_s
        else:
            y = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))[0]
            return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in params.items()},
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy();
        z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(state['loss'].item())

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iterator'] = tqdm(train_loader)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        train_loss = meter_loss.mean
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, test_loader)

        test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss,
            "train_acc": train_acc[0],
            "test_loss": meter_loss.mean,
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
            "at_losses": [m.value() for m in meters_at],
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
              (opt.save, state['epoch'], opt.epochs, test_acc))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
