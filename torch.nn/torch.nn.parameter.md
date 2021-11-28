# torch.nn.parameter
>CLASStorch.nn.parameter.Parameter(data=None, requires_grad=True)[SOURCE]
>>A kind of Tensor that is to be considered a module parameter. Parameters are Tensor subclasses, that have a very special property when used with Module s - when they’re assigned as Module attributes they are automatically added to the list of its parameters, and will appear e.g. in parameters() iterator. Assigning a Tensor doesn’t have such effect. This is because one might want to cache some temporary state, like last hidden state of the RNN, in the model. If there was no such class as Parameter, these temporaries would get registered too.
```python
class Parameter(torch.Tensor):
```
## parameter
![avatar](..\torch.nn\imgs\parameter.par.jpg)

## 功能
<b>首先可以把这个函数理解为类型转换函数</b>:将一个不可训练的类型`Tensor`转换成可以训练的类型`parameter`并将这个`parameter`绑定到这个`module`里面`(net.parameter()`中就有这个绑定的`parameter`，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。

## 目的
<font color=RED>使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。</font>

## <b><font color = #FF0>与Variable区别！！！</font><b>
>"`Variable` – If tensor is just an array, how can we know how this was created, how can we store its gradient? So, to solve this problem a wrapper around tensor was created called Variable. Variable have certain properties – .data (the tensor under the variable), .grad (the gradient computed for this variable, must be of the same shape and type of .data), .requires_grad (boolean indicating whether to calculate gradient for the Variable during backpropagation), .grad_fn (the function that created this Variable, used when backproping the gradients). There is one more attribute, .volatile, whose function will be explained later on. Variable is available under torch.autograd as torch.autograd.Variable"

>"`Parameter` – Whenever we create a model, we have some layers as the components of our model that transforms our input. So, when we print summary of our model, we can see the configuration of that model. But, sometimes we want some tensor as the parameter of some module. <font color=#FF0>For example, learnable initial state for RNNs, input image tensor while doing neural style transfer, even the weights and biases of a layer etc. This cannot be achieved with either Tensor (as they cannot have gradient) nor Variable (as they are not module parameters). </font>So a wrapper around Variable is created called Parameter. This is available under torch.nn as torch.nn.Parameter"

<b>不同其实显而易见：就是字面意思上的区别：`parameter`是参数，是模型中的需要被学习更改的值，而`variable`是变量，是输入进来要计算梯度的值。</b>
`parameter`会添加到模型中去,是模型的参数，`nn.parameter`都是在Model的constructor中使用的。
而`variable`是作用在输入变量身上的，他用来计算梯度。
e.g1:
```python
import torch

class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.variable = torch.autograd.Variable(torch.Tensor([5]))
        self.parameter = torch.nn.Parameter(torch.Tensor([10]))

net = MyModule()
for param in net.parameters():
    print(param)

"""
output:
Parameter containing:
tensor([10.], requires_grad=True)
"""
```




e.g2:
```python
import torch
import torch.nn as nn
from torch.optim import Adam

class NN_Network(nn.Module):
    def __init__(self,in_dim,hid,out_dim):
        super(NN_Network, self).__init__()
        self.linear1 = nn.Linear(in_dim,hid)
        self.linear2 = nn.Linear(hid,out_dim)
        self.linear1.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
        self.linear1.bias = torch.nn.Parameter(torch.ones(hid))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
        self.linear2.bias = torch.nn.Parameter(torch.ones(hid))

    def forward(self, input_array):
        h = self.linear1(input_array)
        y_pred = self.linear2(h)
        return y_pred

in_d = 5
hidn = 2
out_d = 3
net = NN_Network(in_d, hidn, out_d)


for param in net.parameters():
    print(type(param.data), param.size())

""" Output
<class 'torch.FloatTensor'> torch.Size([5, 2])
<class 'torch.FloatTensor'> torch.Size([2])
<class 'torch.FloatTensor'> torch.Size([5, 2])
<class 'torch.FloatTensor'> torch.Size([2])
"""
""" this can easily be fed to your optimizer -"""
opt = Adam(net.parameters(), learning_rate=0.001)

```
通过使用parameterm,可以轻松的将参数传递给优化器。

e.g3:
```python
x_tensor = torch.randn(2,5)
y_tensor = torch.randn(2,5)
#将tensor转换成Variable
x = Variable(x_tensor) #Varibale 默认时不要求梯度的，如果要求梯度，需要说明
y = Variable(y_tensor,requires_grad=True)
z = torch.sum(x + y)
```
通常variable都加载输入变量身上。

## parameter相关方法：
### 1. requir_grad:
虽然已经作为参数添加到模型里面去了，但是还是可以选择是否训练这一层参数。
e.g1:

```python
for param in vgg.features.parameters():
 
    param.requires_grad=False
"""
>output:
<class 'generator'>
0.weight torch.Size([3, 4])
0.bias torch.Size([3])
2.weight torch.Size([1, 3])
2.bias torch.Size([1])
"""
```
e.g2:
```python
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))
"""
>output
weight torch.Size([3, 4]) <class 'torch.nn.parameter.Parameter'>
bias torch.Size([3]) <class 'torch.nn.parameter.Parameter'>

"""
```
e.g3:
```python
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass
    
n = MyModel()
for name, param in n.named_parameters():
    print(name)
"""
>output
weight1
"""
```
e.g4:
```python
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad) # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)

"""
>output
tensor([[ 0.2719, -0.0898, -0.2462,  0.0655],
        [-0.4669, -0.2703,  0.3230,  0.2067],
        [-0.2708,  0.1171, -0.0995,  0.3913]])
None
tensor([[-0.2281, -0.0653, -0.1646, -0.2569],
        [-0.1916, -0.0549, -0.1382, -0.2158],
        [ 0.0000,  0.0000,  0.0000,  0.0000]])
"""
```
### 2. named_parameter:
返回参数和层的名字

e.g1:
```python
    net = model()
    for name, param in net.named_parameters():
        print(name, param.size())
```
e.g2:
初始化每层的weight为特定值。
```python
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
```



## ..
`nn.Linear.weigth`和`nn.Linear.bias`都是parameter类的变量，是可以训练的。