<!-- visual-lab:generated:start -->
# 检验、p 值与统计功效

<!-- bilingual-en:start -->
*Two-sided z tests & power*
<!-- bilingual-en:end -->

分开查看零假设分布、指定备择分布与一次观测，理解误报、漏报和样本量。

> [!question] 先预测，再操作
> 同一个观测均值，为什么样本量不同会改变 p 值？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/testing.mp4]]

约 33 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 33 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先假设零假设成立，再放入观测
- **0:08** 两侧更极端的区域，加起来是 p 值
- **0:16** 显著性阈值，先于这次数据规定
- **0:23** p 值不是零假设为真的概率

<!-- bilingual-en:start -->
- **0:00** Locate the observation under the null model.
- **0:08** A two-sided p-value counts both extreme tails.
- **0:16** The significance level specifies a rejection rule.
- **0:23** A p-value is not the probability that the null is true.
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
id: testing
height: 980
```

## 观察路线

1. 先固定 n 与 α，移动 observed；它改变本次证据，不改变检验设计的功效。
2. 再移动 effect；它改变功效和备择曲线，不改变同一个 observed 对 H₀ 的 p 值。
3. 让 effect=0，检查拒绝概率是否等于 α。
4. 比较 n=10 与 100，解释拒绝阈值与实际效应大小各扮演什么角色。

## 公式与适用条件

$$
Z=\sqrt n\,\bar X,\quad p=2\{1-\Phi(|Z|)\},\quad c=z_{1-\alpha/2},\quad \mathrm{Power}(\delta)=\Phi(-c-\delta\sqrt n)+1-\Phi(c-\delta\sqrt n)
$$

- 每个样本来自 IID N(μ,1)，双侧检验规则事先指定。
- effect 是用来计算功效的假定真实均值；observed 是一次样本均值，两者可以独立设置。
- 单次检验的 α 控制这里的误报率；反复筛选、提前停试或多重比较需要另行处理。

## 回到知识点

- [[00_Knowledge/p值条件解释|p值条件解释]]：固定 H₀ 的双侧尾概率，与观测参数独立变化。
- [[00_Knowledge/显著性解释边界|显著性解释边界]]：功效依赖指定备择值、噪声和 n，显著性不等于效应大小。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/sampling|抽样均值、大数与中心极限定理]] · [[07_Programme/可视化实验室/实验/confidence|置信区间的覆盖率]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]]

## 来源与核验

- [NIST/SEMATECH · Test of a Process Mean](https://www.itl.nist.gov/div898/handbook/prc/section2/prc22.htm) — §7.2.2; test statistic and rejection region。支持：已知标准差时的 z 统计量与双侧拒绝域。
- [NIST/SEMATECH · Critical Values and p Values](https://www.itl.nist.gov/div898/handbook/prc/section1/prc131.htm) — §7.1.3.1 Critical values and p values。支持：p 值是在零假设与参考分布下的尾概率，和预先指定显著性水平比较。
- [NIST/SEMATECH · Sample Sizes Required](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm) — §7.2.2.2; Type II error and power。支持：功效取决于指定备择效应与样本量；本实验双侧精确功效由正态尾概率自行计算。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
