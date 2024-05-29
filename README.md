# deep-learning-2024
# 导入相关模块
'''
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
'''

#设置设备，如果有cuda，就用cuda/GPU，否则用CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#定义一个类class，名为LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)   #卷积层，2维的
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)   #全连接层，线性函数
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):     #定义前馈网络
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #最大池化，2维，relu函数
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)  #model是LeNet的实例

module = model.conv1    #标记model的卷积层1为module
print(list(module.named_parameters()))    #打印module的所有参数

prune.random_unstructured(module, name="weight", amount=0.3)
#使用非结构化的随机剪枝，对module的权重进行操作，剪枝去掉30%的权重


print(list(module.named_parameters()))   #打印参数
print(list(module.named_buffers()))      #打印缓冲区
print(module.weight)
print(module._forward_pre_hooks)         #打印前向预钩子
prune.l1_unstructured(module, name="bias", amount=3)  #使用L1范数的非结构化剪枝
print(list(module.named_parameters()))
print(list(module.named_buffers()))
print(module.bias)
print(module._forward_pre_hooks)

# iteration pruning 迭代剪枝
prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)   #结构化剪枝
print(module.weight)

for hook in module._forward_pre_hooks.values():  #循环语句，挨个找hook的值
    if hook._tensor_name == "weight":  # select out the correct hook
        break       #如果hook是权重，则跳出/退出循环

print(list(hook))  # pruning history in the container

#serializing a pruned model 序列化剪枝模型
print(model.state_dict().keys())   #打印model的参数和键
# remove pruning and re-parametrization 移除剪枝，然后重新参数化
print(list(module.named_parameters()))
print(list(module.named_buffers()))
print(module.weight)
prune.remove(module, 'weight')
print(list(module.named_parameters()))
print(list(module.named_buffers()))

# Pruning multiple parameters in a model 批量剪枝
#定义一个新模型
new_model = LeNet()
for name, module in new_model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):     #如果是卷积层，就剪枝20%
        prune.l1_unstructured(module, name='weight', amount=0.2)
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):    #如果是线性层，就剪枝40%
        prune.l1_unstructured(module, name='weight', amount=0.4)

print(dict(new_model.named_buffers()).keys())  # to verify that all masks exist

# Global pruning 全局剪枝
model = LeNet()
parameters_to_prune = (      #将各个层附到一个参数上，然后对这个参数进行全局剪枝
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,   #剪枝方法
    amount=0.2,
)

# 打印各层的稀疏度和全局的稀疏度
print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    )
)
print(
    "Sparsity in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv2.weight == 0))
        / float(model.conv2.weight.nelement())
    )
)
print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc1.weight == 0))
        / float(model.fc1.weight.nelement())
    )
)
print(
    "Sparsity in fc2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc2.weight == 0))
        / float(model.fc2.weight.nelement())
    )
)
print(
    "Sparsity in fc3.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc3.weight == 0))
        / float(model.fc3.weight.nelement())
    )
)
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
            + torch.sum(model.fc3.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
            + model.fc3.weight.nelement()
        )
    )
)



# Extending with custom pruning functions 扩展自定义的剪枝函数
class FooBarPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask

    def foobar_unstructured(module, name):
        FooBarPruningMethod.apply(module, name)
        return module
model = LeNet()
FooBarPruningMethod.foobar_unstructured(model.fc3, name='bias')

print(model.fc3.bias_mask)


#dot.py
from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable
import warnings

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"

def get_fn_name(fn, show_attrs, max_attr_chars):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX):]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params


def make_dot(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50):
    """ Produces Graphviz representation of PyTorch autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
        show_attrs: whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """
    if LooseVersion(torch.__version__) < LooseVersion("1.9") and \
        (show_attrs or show_saved):
        warnings.warn(
            "make_dot: showing grad_fn attributes and saved variables"
            " requires PyTorch version >= 1.9. (This does NOT apply to"
            " saved tensors saved by custom autograd functions.)")

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return '%s\n %s' % (name, size_to_str(var.size()))

    def add_nodes(fn):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        if show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX):]
                if torch.is_tensor(val):
                    dot.edge(str(id(fn)), str(id(val)), dir="none")
                    dot.node(str(id(val)), get_var_name(val, attr), fillcolor='orange')
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + '[%s]' % str(i)
                            dot.edge(str(id(fn)), str(id(t)), dir="none")
                            dot.node(str(id(t)), get_var_name(t, name), fillcolor='orange')

        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            seen.add(var)
            dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
            dot.edge(str(id(var)), str(id(fn)))

        # add the node for this grad_fn
        dot.node(str(id(fn)), get_fn_name(fn, show_attrs, max_attr_chars))

        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                dot.edge(str(id(t)), str(id(fn)))
                dot.node(str(id(t)), get_var_name(t), fillcolor='orange')


    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")


    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    return dot


def make_dot_from_trace(trace):
    """ This functionality is not available in pytorch core at
    https://pytorch.org/docs/stable/tensorboard.html
    """
    # from tensorboardX
    raise NotImplementedError("This function has been moved to pytorch core and "
                              "can be found here: https://pytorch.org/docs/stable/tensorboard.html")


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


# MNIST.py
#导入python的标准库，作用分别是执行文件操作、时间处理、命令行参数解析、系统交互以及日期时间处理
import os
import time
import argparse
import sys
import datetime
#导入PyTorch及相关的库
import torch      #PyTorch
import torch.nn as nn  #从PyTorch导入神经网络模块，并将其别名设置为nn。这个模块包含了构建神经网络所需的各种层和功能。
import torch.nn.functional as F  #导入functional子模块，并将其别名设置为F。此模块包含激活函数和损失函数
import torch.utils.data as data  #这个模块提供了数据加载和处理的工具，如Dataset和DataLoader
from torch.cuda import amp     #amp是PyTorch中的一个高级API，用于简化混合精度训练的实现
from torch.utils.tensorboard import SummaryWriter  #可视化工具
import torchvision    #torchvision包含处理图像和视频的常用工具和预训练模型
import numpy as np    #NumPy提供了大量的数学函数和对多维数组的支持，常用于数据的预处理和后处理

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
#spikingjelly是一个库的名称，是专门用于处理尖峰神经网络（Spiking Neural Networks, SNNs）的库。

class SNN(nn.Module):
    def __init__(self, tau):   #_init__是类的构造函数，当创建类的新实例时会自动调用
        super().__init__()     #super用来初始化
        #nn.Sequential是一个容器，按顺序包含多个模块（层）。它允许将多个层组合成一个序列，数据将按顺序流经这些层
        self.layer = nn.Sequential(
            layer.Flatten(),        #flatten层，将输入张量展平为一维张量
            layer.Linear(28 * 28, 10, bias=False),   #全连接层，将输入张量（展平后的784个特征）转换为10个输出值。该层不使用偏置项。
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),   #neuron.LIFNode：一个神经元模型，代表具有Leaky Integrate-and-Fire（LIF）特性的神经元
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)
#forward接受一个输入张量 x，将该张量通过网络层 self.layer 进行前向传播，并返回传播的结果。
#这是神经网络模型定义中的核心部分，它指定了数据如何通过网络流动

def main():
    '''
    :return: None

    * :ref:`API in English <lif_fc_mnist.main-en>`

    .. _lif_fc_mnist.main-cn:

    使用全连接-LIF的网络结构，进行MNIST识别。\n
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <lif_fc_mnist.main-cn>`

    .. _lif_fc_mnist.main-en:

    The network with FC-LIF structure for classifying MNIST.\n
    This function initials the network, starts trainingand shows accuracy on test dataset.
    '''
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam',
                        help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

    args = parser.parse_args()
    print(args)

    net = SNN(tau=args.tau)

    print(net)

    net.to(args.device)

    # 初始化数据加载器
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')

    if args.amp:
        out_dir += '_amp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / args.T
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(
            f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 保存绘图用数据
    net.eval()
    # 注册钩子
    output_layer = net.layer[-1]  # 输出层
    output_layer.v_seq = []
    output_layer.s_seq = []

    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)

    with torch.no_grad():
        img, label = test_dataset[0]
        img = img.to(args.device)
        out_fr = 0.
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy", v_t_array)
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy", s_t_array)


if __name__ == '__main__':
    main()

# simple_model.py
# 导入绘图模块
import matplotlib.pyplot as plt
import numpy as np

# 导入相关模块
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# for graphs
import igraph as ig

#设置设备，如果有cuda，就用cuda/GPU，否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input definition
W_in = 5
H_in = 5
in_ch = 1

input = torch.randn(W_in,H_in, in_ch)

# Network parameters,a filter K, stride S,padding P
K = 3
P = 1
S = 1

ch_conv1 = 1

#定义一个类class，名为LeNet
class conv_layer(nn.Module):
    def __init__(self):
        super(conv_layer, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=ch_conv1, kernel_size=K, stride=S, padding=P)   #卷积层，2维的

    def forward(self, x):     #定义前馈网络
        x = F.relu(self.conv1(x))
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #最大池化，2维，relu函数
        return x

model = conv_layer().to(device=device)  #model是LeNet的实例

plt.imshow(input)
plt.show()

size_in = W_in*H_in
id_nodes_input = ['in_' + str(i) for i in range(size_in)]

# define node ids for conv1
size_conv1 = int(((W_in -K)/S+1)*((H_in-K)/S+1)) * ch_conv1
print(size_conv1)
id_nodes_per_ch = ['_' + str(i) for i in range(size_conv1)]
id_nodes_conv1 = list()
for ch in range(ch_conv1):
    id_nodes_conv1.extend(['conv1_ch_' + str(ch) + '_unit' + i for i in id_nodes_per_ch])


#Create graph
g = ig.Graph()
g.add_vertices(size_in+size_conv1)

all_labels = id_nodes_input + id_nodes_conv1
g.vs["label"] = all_labels


#this prints out the weights of conv1 (there are only 9, because each unit in conv1
# uses the same weights to define their individual activation)
for name, param in model.named_parameters():
    print(name)
    print(param)

# This loops through all conv1 units (i) and defines incoming edges from all
#  input units (j). However, we know that each conv1 unit (i) receives non-zero
#  weights from a subset of input units (j). We need to define these weights based
#  on our knowledge of the network architecture.
n_weights_filter = size_conv1
total_n_weights = n_weights_filter**2
for i in id_nodes_conv1:
    for j in id_nodes_input:
        g.add_edges([(g.vs["label"].index(j), g.vs["label"].index(i))])
        g[g.vs["label"].index(j),g.vs["label"].index(i)] =
g.es["weight"] = 0

for i in id_nodes_conv1:
    for j in id_nodes_input:
        g.add_edges([(g.vs["label"].index(j), g.vs["label"].index(i))])

layout = g.layout_reingold_tilford()

fig, ax = plt.subplots()
ig.plot(g, layout=layout, target=ax)

plt.show()

g.add_edges([(g.vs["label"].index(0), g.vs["label"].index(1))])


