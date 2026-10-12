<!-- visual-lab:generated:start -->
# Bootstrap 能告诉我们什么

<!-- bilingual-en:start -->
*Bootstrap uncertainty & its limits*
<!-- bilingual-en:end -->

从同一份观测样本反复有放回抽取，估计均值的不确定性并检验百分位区间的边界。

> [!question] 先预测，再操作
> 把重采样次数增加十倍，是否等价于收集十倍的新数据？

## 动画：一份样本怎样变成一幅分布

<!-- bilingual-en:start -->
*Animation: from one sample to a distribution*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/bootstrap.mp4]]

81 秒，中英字幕。可暂停、点击进度条定位或全屏观看。合成示例使用指数总体，n=8、B=500、seed=42；数值显示三位小数。这个小样本用于说明机制。

<!-- bilingual-en:start -->
81 seconds, with Chinese and English captions. Pause, click the timeline to seek or watch full screen. The synthetic example uses an exponential population with n=8, B=500 and seed=42; values are displayed to three decimal places. This small sample illustrates the mechanism.
<!-- bilingual-en:end -->

## 亲手实验

<!-- bilingual-en:start -->
*Try it yourself*
<!-- bilingual-en:end -->

先打开“有放回地重新抽取”，确认 n=8、B=500、指数总体、种子42，可复现短片；再只改变 B 或 n，比较各自改变了什么。

<!-- bilingual-en:start -->
Open the with-replacement step and set n=8, B=500, an exponential population and seed 42 to reproduce the film. Then change only B or n and compare what changes.
<!-- bilingual-en:end -->

```visual-lab
id: bootstrap
height: 980
```

## 观察路线

1. 先看原始样本均值与理论总体均值；Bootstrap 围绕当前样本，而非直接围绕总体真值。
2. 只增加 B，确认原始样本与样本均值保持不变；观察直方图和分位数更稳定。
3. 改变 n 或种子相当于更换原始样本，区间位置与宽度都可能改变。
4. 比较 Bootstrap SE 与 s/√n；用下面的精确条件方差解释有限 n 差异。

## 公式与适用条件

$$
X_1^*,\ldots,X_n^*\overset{iid}{\sim}\hat F_n,\quad \bar X^*=\frac1n\sum_iX_i^*,\quad \widehat{\mathrm{SE}}_* =\mathrm{sd}(\bar X_1^*,\ldots,\bar X_B^*),\quad I_*=[q^*_{.025},q^*_{.975}]
$$

- 原始观测 IID；普通逐点 Bootstrap 不适合直接处理时间依赖、聚类或复杂抽样。
- 均值的常规 Bootstrap 近似需要适当的有限矩和代表性；本页提供正态与指数两种有限方差总体。
- s 使用 n−1 分母；Bootstrap SE 是条件于原始样本的模拟标准差，分位数也随有限 B 有 Monte Carlo 误差。

## 回到知识点

- [[00_Knowledge/抽样分布|抽样分布]]：对比总体重复抽样分布与给定样本后的 Bootstrap 条件分布。
- [[00_Knowledge/样本量与模拟重复次数|样本量与模拟重复次数]]：增加 B 不增加原始观测信息。
- [[00_Knowledge/历史模拟Bootstrap|历史模拟Bootstrap]]：迁移有放回重抽和无法创造未观测尾部的机制；本实验计算均值，不能替代 VaR/ES 的专门有效性检验。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/sampling|抽样均值、大数与中心极限定理]] · [[07_Programme/可视化实验室/实验/confidence|置信区间的覆盖率]] · [[07_Programme/可视化实验室/实验/overfit|模型复杂度与泛化]]

## 来源与核验

- [UC Berkeley · Stat 20 Bootstrapping](https://stat20.berkeley.edu/fall-2025/3-generalization/09-bootstrapping/notes.html) — Bootstrap; Bootstrap Confidence Interval (percentile method); Summary cautions。支持：从经验分布有放回重采样，百分位区间及代表性限制；区间覆盖并非有限样本精确保证。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
