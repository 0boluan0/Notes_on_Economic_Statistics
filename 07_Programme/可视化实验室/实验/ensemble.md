<!-- visual-lab:generated:start -->
# 平均集成为何受相关性限制

<!-- bilingual-en:start -->
*Ensemble averaging & correlated errors*
<!-- bilingual-en:end -->

实际生成等相关成员误差，观察增加成员数能消除的波动与无法消除的共享误差。

> [!question] 先预测，再操作
> 把同一个预测复制一百次再平均，会更准确吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/ensemble.mp4]]

约 40 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 40 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 平均多个预测，会抵消多少误差？
- **0:15** 独有误差被平均，共同误差留在原地
- **0:30** 模型数量，不能替代误差多样性

<!-- bilingual-en:start -->
- **0:00** Averaging can cancel errors that disagree.
- **0:15** Independent noise shrinks; common noise remains.
- **0:30** More members do not guarantee independent errors.
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
id: ensemble
height: 980
```

## 观察路线

1. 先设 ρ=0，预测 m 从 1 到 100 的理论方差。
2. 提高 ρ，比较理论平台与实际重复模拟曲线。
3. 设 ρ=1，再增加 m，解释为何同一重复中的预测不改变。
4. 增加 bias，分清方差、偏差与对目标 0 的 MSE。

## 公式与适用条件

$$
e_j=b+\sigma(\sqrt\rho Z_0+\sqrt{1-\rho}Z_j),\quad\operatorname{Var}(\bar e_m)=\sigma^2\left(\rho+\frac{1-\rho}{m}\right)
$$

- 所有成员同方差 σ²、同偏差 b，且两两相关都为 ρ∈[0,1]。
- 目标固定为 0；成员数 m 和重复实验次数 B 是不同轴。
- 这是可解析的误差生成模型，不是对任意集成方法的性能保证。

## 回到知识点

- [[00_Knowledge/组合误差协方差|组合误差协方差]]：等权组合成员误差，实际改变共享误差相关性，直接演示固定单位和权重下组合方差由误差协方差决定。
- [[00_Knowledge/和的方差协方差项|和的方差协方差项]]：σ²[ρ+(1−ρ)/m] 是有限成员和的方差展开除以 m²；ρ=0 和 1 两个预设核验交叉项。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/bootstrap|Bootstrap 能告诉我们什么]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]] · [[07_Programme/可视化实验室/实验/boosting|提升算法怎样逐轮修正残差]] · [[07_Programme/可视化实验室/实验/sampling|抽样均值、大数与中心极限定理]]

## 来源与核验

- [scikit-learn · Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) — §1.11.3 Bagging meta-estimator; §1.11.2 Random forests and other randomized tree ensembles。支持：平均多个学习器减少方差的机制与随机化；本实验等相关方差公式自行推导，不把模拟误差冒充森林训练。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
