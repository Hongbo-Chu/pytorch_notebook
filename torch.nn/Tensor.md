# Tensor
在pytorch里面，所有的操作都是基于tensor,可以理解为是torch的基础数据结构，是一种包含单一数据类型元素的多维矩阵。
## 1. tensor和Tensor
从命名可以看出`tensor`是函数，而`Tensor`是类名。
torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor
tensor函数可以将`list`, `tuple`, `numpy`, `array`, `scalar`变成`Tensor`

## 2. Tensor Attribute
官方文档：
>"Each torch.Tensor has a torch.dtype, torch.device, and torch.layout."

为什么是`torch.`呢，因为这些类型是torch自己定义的，是torch自己定义的类型（相当于int,float之类的），而这里这些类型作为Tensor的属性。
Tensor{
    torch.dtype;
    torch.device
    torch.layout
};

<b>pytorch原码中截取：</b>

tensor的定义：
```python
def tensor(data: Any, dtype: Optional[_dtype]=None, device: Union[_device, str, None]=None, requires_grad: _bool=False) -> Tensor: ...
```
```python
_dtype = torch.dtype
_device = torch.device
_qscheme = torch.qscheme
_size = Union[torch.Size, List[_int], Tuple[_int, ...]]
_layout = torch.layout
```


使用e.g.
```python
a = torch.tensor([1], dtype=torch.float, device=torch.device('cuda'), requires_grad=False)
```

### 1.  <b>torch.dtype</b>
>"A torch.dtype is an object that represents the data type of a torch.Tensor. PyTorch has twelve different data types:"

torch.dtype 是展示 torch.Tensor 数据类型的类，pytorch 有八个不同的数据类型,下表是完整的 dtype 列表.
![avatar](D:/pytorch_notebook/torch.nn/imgs/tensor.dtype.jpg)

### 2.  <b>torch.device</b>
>"A torch.device is an object representing the device on which a torch.Tensor is or will be allocated."

没啥好说的，表示这个tensor存在哪里，cpu or cuda

### 3.  torch.layoyut
>"A torch.layout is an object that represents the memory layout of a torch.Tensor. Currently, we support torch.strided (dense Tensors) and have beta support for torch.sparse_coo (sparse COO Tensors)."

