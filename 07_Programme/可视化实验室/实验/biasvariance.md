<!-- visual-lab:generated:start -->
# 真正的偏差—方差分解

<!-- bilingual-en:start -->
*Repeated-fit bias–variance decomposition*
<!-- bilingual-en:end -->

重新抽取训练噪声并实际拟合多次，在同一个 x₀ 比较预测分布、偏差平方、方差和新响应噪声。

> [!question] 先预测，再操作
> 对真实函数值的估计误差，为什么比预测一次新响应少一项 σ²？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/biasvariance.mp4]]

约 38 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 38 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 数据换一次，拟合曲线也换一次
- **0:14** 固定一个 x，看预测值如何散开
- **0:29** 平均值的偏移，和围绕平均值的波动，要分开

<!-- bilingual-en:start -->
- **0:00** A fitted function varies across datasets.
- **0:14** Fix one input and inspect the distribution of predictions.
- **0:29** Bias is displacement; variance is spread.
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
id: biasvariance
height: 980
```

## 观察路线

1. 固定 x₀，先看每次预测的分布，再区分中心偏离与围绕中心的散布。
2. 将 noise 设为 0，保留较低次数，检验随机方差消失后模型偏差是否仍存在。
3. 比较 degree 和 λ；每个配置都重新实际拟合，图形不预设 U 形。
4. 查看新响应风险多出的 σ²，并说明一次实际新响应抽样不必精确等于其期望。

## 公式与适用条件

$$
\frac1R\sum_r(\hat f_r-f)^2=\underbrace{\frac1R\sum_r(\hat f_r-\bar f)^2}_{\widehat{\mathrm{Var}}_R}+\underbrace{(\bar f-f)^2}_{\widehat{\mathrm{Bias}}^2},\quad E_\epsilon[(f+\epsilon-\hat f)^2]=(f-\hat f)^2+\sigma^2
$$

- 真实 f(x)=sin(πx)，固定等距训练设计；只对训练响应噪声重复，未对训练 x 的随机性取平均。
- Legendre 多项式拟合，训练平方损失求和，截距不受 λ 惩罚。
- 方差分母为 R；指标是有限 R 的经验分解，总体偏差/方差仍有 Monte Carlo 误差。

## 回到知识点

- [[00_Knowledge/MSE偏差方差分解|MSE偏差方差分解]]：实际重新拟合 R 次，把固定目标 f(x₀) 的经验 MSE 精确分成分母 R 的方差与经验偏差平方；新响应噪声另列，避免与固定目标恒等式混淆。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/overfit|模型复杂度与泛化]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]] · [[07_Programme/可视化实验室/实验/ensemble|平均集成为何受相关性限制]] · [[07_Programme/可视化实验室/实验/sampling|抽样均值、大数与中心极限定理]]

## 来源与核验

- [Stanford CS229 · Lecture Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — §8.1.1, printed pp.110–111; PDF pp.111–112。支持：支持固定测试点处训练随机性、偏差平方、方差与不可约噪声的分解；有限 R 版本由平方和恒等式独立推导。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
