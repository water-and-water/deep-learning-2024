# deep-learning-2024
# 导入相关模块

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

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
