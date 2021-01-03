# 实验记录

## 1.3
- 解决了validate时显存占用过多爆炸的问题
- ProtoNet基线测试：cat_fuse + 投影矩阵 + batchNorm/dropout 均会导致距离爆炸，最终结果nan
- 只使用seq特征，sgd优化太慢，改用adam，loss降到1时，acc甚至不到0.2，说明计算可能有问题