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

前述说到了parameters就是Net的权重参数（比如conv的weight、conv的bias、fc的weight、fc的bias)，类型为tensor，用于前向和反向；比如，你针对Net使用cpu()、cuda()等调用的时候，实际上调用的就是parameter这个tensor的cpu()、cuda()等方法；再比如，你保存模型或者重新加载pth文件的时候，针对的都是parameter的操作或者赋值。

如果你针对的是CivilNet直接取_parameters属性的值的话，很遗憾是空的，因为CivilNet的成员并没有直接派生自Parameter类；但是当针对CivilNet取parameters()函数的返回值（是个iter）时，则会递归拿到所有的，比如conv的weight、bias等；