<!-- visual-lab:generated:start -->
# BatchNorm与LayerNorm沿哪个轴统计

<!-- bilingual-en:start -->
*Batch normalization vs layer normalization*
<!-- bilingual-en:end -->

把同一4×3矩阵沿不同轴归一化，再施加仿射参数，查看每一个元素的实际变化。

> [!question] 先预测，再操作
> 归一化输出一定均值0、方差1吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/normalization.mp4]]

约 39 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 39 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先回答：哪些数一起算均值？
- **0:14** 同一组，减同一个均值、除同一个尺度
- **0:22** 仿射参数在标准化之后作用
- **0:29** Layer Norm 改的是分组方向

<!-- bilingual-en:start -->
- **0:00** Choose the statistical group before normalizing.
- **0:14** Center and scale using that group only.
- **0:22** Apply the learned scale and shift afterward.
- **0:29** Layer normalization uses features within each observation.
<!-- bilingual-en:end -->

## 亲手实验

<!-- bilingual-en:start -->
*Try it yourself*
<!-- bilingual-en:end -->

在下方展开“影片参数与延伸对照”，选择“载入本片的模型与数据”，先核对影片中的一个对象或运算，再改变一个条件。其他对照用于继续探索。

<!-- bilingual-en:start -->
Expand the film inputs below and load the model and data used in this film. Check one object or calculation, then change one condition. The other comparisons provide further exploration.
<!-- bilingual-en:end -->

<!-- visual-lab:film:end -->
```visual-lab
id: normalization
height: 980
```

## 观察路线

1. 先数清行是样本、列是特征，再选择统计模式。
2. 只改变第3特征的偏移，比较BN其他列与LN同一行的变化。
3. 选择含恒定列矩阵，检查方差为0时epsilon和beta各起什么作用。
4. 把gamma设为0，观察所有输出是否等于beta。

## 公式与适用条件

$$
z=\frac{x-\mu_G}{\sqrt{\sigma_G^2+\varepsilon}},\quad y=\gamma z+\beta,\quad \sigma_G^2=|G|^{-1}\sum_{j\in G}(x_j-\mu_G)^2
$$

- 矩阵固定为4个样本、3个特征，使用分母组大小的方差和ε=10⁻⁵。
- BN仅展示训练态批次统计；实际推理通常使用训练期间估计的固定统计。
- 本页γ、β在各特征设为同一个数以减少控件，实际一般可逐特征学习。

## 回到知识点

- [[00_Knowledge/LayerNorm|LayerNorm]]：同一个样本沿特征维计算统计；epsilon、逐特征仿射与批次归一化轴的区别。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/initialization|初始化如何改变信号与梯度尺度]] · [[07_Programme/可视化实验室/实验/residual|残差捷径怎样改变梯度乘积]] · [[07_Programme/可视化实验室/实验/attention|注意力从分数到读取向量]]

## 来源与核验

- [Ioffe & Szegedy · Batch Normalization (ICML 2015)](https://proceedings.mlr.press/v37/ioffe15.pdf) — §3; Algorithm 1; Algorithm 2。支持：当前批次均值/方差、epsilon、gamma/beta；训练态与使用固定统计的推理态不同。
- [Ba, Kiros & Hinton · Layer Normalization (2016)](https://arxiv.org/pdf/1607.06450) — §3 Layer normalization; Eq. (3)。支持：同一样本内跨特征的归一化统计，区别于跨样本的BatchNorm。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
