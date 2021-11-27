# torch.nn.parameter
## 功能
<b>首先可以把这个函数理解为类型转换函数</b>:将一个不可训练的类型`Tensor`转换成可以训练的类型`parameter`并将这个`parameter`绑定到这个`module`里面`(net.parameter()`中就有这个绑定的`parameter`，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。

## 目的
<font color=RED>使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。</font>