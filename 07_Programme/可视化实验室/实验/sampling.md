<!-- visual-lab:generated:start -->
# 抽样均值、大数与中心极限定理

<!-- bilingual-en:start -->
*Sampling means, LLN & CLT*
<!-- bilingual-en:end -->

把一次累积均值路径和多次独立样本均值分布分开观察，并用柯西分布检验定理边界。

> [!question] 先预测，再操作
> 样本量很大时，任何分布的平均值都会越来越稳定吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/sampling.mp4]]

约 51 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 51 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 八个观测，只产生一个样本均值
- **0:09** 换一整份样本，才得到下一个均值
- **0:31** 很多个均值，才形成抽样分布
- **0:42** n 控制每个均值，B 控制模拟精细度

<!-- bilingual-en:start -->
- **0:00** Eight observations produce one sample mean.
- **0:09** A new sample gives the next sample mean.
- **0:31** The distribution is built from repeated sample means.
- **0:42** Sample size and repetition count play different roles.
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
id: sampling
height: 980
```

## 观察路线

1. 保持同一个种子与 B，比较指数总体下 n=3、30、100 的均值直方图。
2. 区分横轴上的“一个样本均值”和样本内部的单次观测。
3. 只增加 B：预测图更平滑还是均值分布更窄，再观察。
4. 换成柯西分布，查看理论均值和标准误为何显示未定义；检查分位数裁切说明与省略计数，并从导出查看极端值。

## 公式与适用条件

$$
\bar X_n=\frac1n\sum_{i=1}^nX_i,\quad E\bar X_n=\mu,\quad \mathrm{SE}(\bar X_n)=\frac\sigma{\sqrt n},\quad \frac{\sqrt n(\bar X_n-\mu)}\sigma\Rightarrow N(0,1)
$$

- 正态和指数样本 IID；经典 CLT 使用有限均值和 0<σ²<∞。
- 正态总体的均值在每个 n 下都精确正态；指数总体的均值在有限 n 下仍有偏斜。
- 一条累积路径只能展示一种实现，不能证明或否定一个渐近定理；有限浮点模拟也不能穷尽重尾。

## 回到知识点

- [[00_Knowledge/抽样分布|抽样分布]]：均值直方图对应统计量在固定 n 下的分布。
- [[00_Knowledge/经典中心极限定理|经典中心极限定理]]：有限正方差、中心化与标准化边界；柯西为不适用例。
- [[00_Knowledge/弱大数定律|弱大数定律]]：累积均值路径提供直觉，不以单条路径证明依概率收敛。
- [[00_Knowledge/样本量与模拟重复次数|样本量与模拟重复次数]]：n 与 B 的不同作用通过独立控件比较。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/confidence|置信区间的覆盖率]] · [[07_Programme/可视化实验室/实验/testing|检验、p 值与统计功效]] · [[07_Programme/可视化实验室/实验/bootstrap|Bootstrap 能告诉我们什么]]

## 来源与核验

- [OpenStax · Introductory Statistics 2e](https://openstax.org/books/introductory-statistics-2e/pages/7-1-the-central-limit-theorem-for-sample-means-averages) — §7.1 The Central Limit Theorem for Sample Means。支持：样本均值的抽样分布与标准误；实验补足 IID 与有限方差条件，正态总体的结果为精确分布。
- [Rick Durrett · Probability: Theory and Examples, 5th ed.](https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf) — Theorem 2.2.14, printed p.64 (PDF p.72); Theorem 3.4.1, printed p.144 (PDF p.152)。支持：IID 且一阶绝对矩有限的弱大数定律；IID 且正有限方差的标准化 CLT。
- [NIST/SEMATECH · Cauchy Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm) — §1.3.6.6.3; Common Statistics。支持：标准柯西的均值、标准差未定义；对称中心不是期望。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
