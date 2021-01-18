# 实验记录

## 1.3
- 解决了validate时显存占用过多爆炸的问题
- ProtoNet基线测试：cat_fuse + 投影矩阵 + batchNorm/dropout 均会导致距离爆炸，最终结果nan
- 只使用seq特征，sgd优化太慢，改用adam，loss降到1时，acc甚至不到0.2，说明计算可能有问题

## 1.9
- 修复了因为metric的label_cache没有在query标签归一化之后再更新，导致的metric计算有误的bug。修复后metric值恢复正常
- 训练速度较慢，存在问题，相同条件下原项目只用序列只用20s，此处需要两倍时间40s
    - 问题已修复：img部分的episode采样时间消耗过长（可能是collect时的tolist导致）。隔离数据源解决
- 存在训练时loss异常过高的问题，该问题导致即使是adam优化，训练过程也会在训练正确率在70%时停止拟合，经过debug已发现similarity
    部分的的距离尺度较大，该问题导致在softmax时分布直接变为长尾分布（经同数据debug验证，问题可能在forward过程）
  
## 1.17
- 修复了forward过程优化停滞的问题，该问题由没有调用zero_grad()清空梯度导致
- 初始化dataset时读入内存，但是没有读入显存，导致每次在取数据的时候都需要tolist+cuda，使得数据总是在内存和显存之间
运输，大大增加了运行时间。因此考虑读取dataset时就将其读入显存，然后在取数据时预留出一个0位置的维度，直接使用torch.cat
  进行数据的拼接，大大减少了image的读取时间