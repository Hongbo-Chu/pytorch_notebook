# torch.nn.Module
## 1. 源代码分析：
```py
class Module:
    def __init__(self):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        torch._C._log_api_usage_once("python.nn_module")

        self.training = True
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._non_persistent_buffers_set = set()
        self._backward_hooks = OrderedDict()
        self._is_full_backward_hook = None
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
```
一个Net，也就是继承自nn.Module的类，当实例化后，本质上就是维护了以下8个字典(OrderedDict)：
这8个字典用于网络的前向、反向、序列化、反序列化中。

因此，当实例化你定义的Net(nn.Module的子类)时，要确保父类的构造函数首先被调用，这样才能确保上述8个OrderedDict被create出来，否则，后续任何的初始化操作将抛出类似这样的异常：cannot assign module before Module.__init__() call。
#### 举例：
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
class CivilNet(nn.Module):
    def __init__(self):
        super(CivilNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.gemfield = "gemfield.org"
        self.syszux = torch.zeros([1,1])
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
对于前述的CivilNet而言，当CivilNet被实例化后，CivilNet本身维护了这8个OrderedDict，更重要的是，CivilNet中的conv1和conv2(类型为nn.modules.conv.Conv2d）、pool（类型为nn.modules.pooling.MaxPool2d）、fc1、fc2、fc3（类型为torch.nn.modules.linear.Linear）<font color = yellow>均维护了8个OrderedDict</font>，因为它们的父类都是nn.Module，而gemfield（类型为str）、syszux（类型为torch.Tensor)则没有这8个OrderedDict。

也因此，在你定义的网络投入运行前，必然要确保和上面一样——构造出那8个OrderedDict，这个构造，就在nn.Module的构造函数中。如此以来，你定义的Net就必须继承自nn.Module；如果你的Net定义了__init__()方法，则必须在你的__init__方法中调用nn.Module的构造函数，比如super(your_class).__init__() ，注意，如果你的子类没有定义__init__()方法，则在实例化的时候会默认用nn.Module的，这种情况也对。

nn.Module通过使用__setattr__机制，使得定义在类中（不一定要定义在构造函数里）的成员（比如各种layer），被有序归属到_parameters、_modules、_buffers或者普通的attribute里；那具体怎么归属呢？很简单，当类成员的type 派生于Parameter类时（比如conv的weight，在CivilNet类中，就是self.conv1中的weight属性），该属性就会被划归为_parameters；当类成员的type派生于Module时（比如CivilNet中的self.conv1，其实除了gemfield和syszux外都是），该成员就会划归为_modules

如果知道了这个机制，就会自然而然的知道，如果上面的CivilNet里的成员封装到一个list里，像下面这样：
```py
class CivilNet(nn.Module):
    def __init__(self):
        super(CivilNet, self).__init__()
        conv1 = nn.Conv2d(3, 6, 5)
        pool = nn.MaxPool2d(2, 2)
        conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = [conv1, pool, conv2]
        ...
```
那么在运行的时候，可能optimizer就会提示parameters为empty。这就是因为成员layer1的type派生自list，而非Module；而像CivilNet这样的Net，在取所有的parameters的时候，都是通过_modules桥梁去取得的...

### 1，_parameters
<b><font color = RED>用来存放注册的 Parameter 对象</font></b>
前述说到了parameters就是Net的权重参数（比如conv的weight、conv的bias、fc的weight、fc的bias)，类型为tensor，用于前向和反向；比如，你针对Net使用cpu()、cuda()等调用的时候，实际上调用的就是parameter这个tensor的cpu()、cuda()等方法；再比如，你保存模型或者重新加载pth文件的时候，针对的都是parameter的操作或者赋值。

如果你针对的是CivilNet直接取_parameters属性的值的话，很遗憾是空的，因为CivilNet的成员并没有直接派生自Parameter类；但是当针对CivilNet取parameters()函数的返回值（是个iter）时，则会递归拿到所有的，比如conv的weight、bias等；
### 2. _modules
<b><font color =RED> 用来保存注册的 Module 对象。</font></b>

### 3. training
<b><font color = RED>用来表示是不是在 training 状态下</font></b> 

### 4. _buffers 
<b><font color = RED>用来存放注册的 Buffer 对象。（pytorch 中 buffer 的概念就是 不需要反向传导更新的值）</font></b>

### model.state_dict()

例子：
```py
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
"""
out>
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])

Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
"""
```
`model.state_dict`存储了`model`的全部`parameter`的参数正如上面例子所示。
## 相关函数：
### 1. `torch.save()`
    `torch.save(model, PATH)`
`model`是一个字典的形式，`path`是存储的路径。`torch.save`会调用`pickle module`将，dict二值序列化存储。

## 模型的存储：
### 1. Save/Load state_dict (Recommended)
save:
```py
torch.save(model.state_dict(), PATH)
```
load:
```py
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```
    note:model.state_dict() returns a reference to the state and not its copy! 
    返回值是浅拷贝!!!

### 2. Save/Load Entire Model
save:
```py
torch.save(model, PATH)
```
load:
```py
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()
```

### 3.Saving & Loading a General Checkpoint for Inference and/or Resuming Training

save:
```py
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
```
load:
```py
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

