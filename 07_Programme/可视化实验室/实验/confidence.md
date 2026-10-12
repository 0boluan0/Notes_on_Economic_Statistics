<!-- visual-lab:generated:start -->
# 置信区间的覆盖率

<!-- bilingual-en:start -->
*Confidence intervals & coverage*
<!-- bilingual-en:end -->

反复从同一正态总体抽样，观察区间宽度、漏盖次数和模拟覆盖率的波动。

> [!question] 先预测，再操作
> 95% 置信区间意味着已经算出的这个区间有 95% 的概率包含 μ 吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/confidence.mp4]]

约 44 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 44 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先抽样，才有一个会变的区间
- **0:10** 重复整个抽样过程，再看覆盖频率
- **0:29** 95% 描述的是这种造区间的方法
- **0:35** 固定参数，随机区间

<!-- bilingual-en:start -->
- **0:00** A random sample produces a random interval.
- **0:10** Repeat sampling to assess the interval procedure.
- **0:29** Coverage is a property of the procedure.
- **0:35** A fixed parameter and a random interval.
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
id: confidence
height: 980
```

## 观察路线

1. 先预测 100 个 95% 区间中大约有几个漏掉 μ=0，再看模拟结果。
2. 保持其余参数，比较 n=25 与 n=100 的宽度。
3. 把置信水平改为 99%，说明覆盖与精度的交换。
4. 当 B>100 时区间图仅展示前 100 个；使用指标中所有 B 个区间的覆盖率与分母。

## 公式与适用条件

$$
I(\mathbf X)=\left[\bar X-z_{1-\alpha/2}\frac\sigma{\sqrt n},\ \bar X+z_{1-\alpha/2}\frac\sigma{\sqrt n}\right],\quad P_\mu\{\mu\in I(\mathbf X)\}=1-\alpha
$$

- 观测独立且同分布为 N(0,σ²)，σ 作为已知总体参数输入。
- 公式没有用样本标准差替换 σ；若 σ 未知且仍假设正态，通常需要 t 区间。
- B 个区间的覆盖指示相互独立；模拟标准误描述覆盖率估计的精度，不是单个均值的标准误。

## 回到知识点

- [[00_Knowledge/置信区间覆盖率|置信区间覆盖率]]：已知 σ 正态情形的精确长期覆盖，以及经验覆盖的 Monte Carlo 波动。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/sampling|抽样均值、大数与中心极限定理]] · [[07_Programme/可视化实验室/实验/testing|检验、p 值与统计功效]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]] · [[07_Programme/可视化实验室/实验/bootstrap|Bootstrap 能告诉我们什么]]

## 来源与核验

- [OpenStax · Introductory Statistics 2e](https://openstax.org/books/introductory-statistics-2e/pages/8-1-a-single-population-mean-using-the-normal-distribution) — §8.1; known population standard deviation and error bound。支持：已知总体标准差的 z 区间与宽度；本实验额外固定正态总体以保证有限样本精确覆盖。
- [NIST/SEMATECH · Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm) — §1.3.5.2; Purpose and confidence interpretation。支持：置信水平属于重复构造区间的方法；该页主要 t 区间公式不作为本实验 z 公式来源。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
