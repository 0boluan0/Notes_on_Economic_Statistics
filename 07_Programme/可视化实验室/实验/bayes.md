<!-- visual-lab:generated:start -->
# 先验与数据怎样形成后验

<!-- bilingual-en:start -->
*Beta–Binomial updating*
<!-- bilingual-en:end -->

在成功率推断中并列观察先验、缩放似然和后验，解释收缩与可信区间。

> [!question] 先预测，再操作
> 相同的七次成功、三次失败，为什么不同先验会给出不同后验？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/bayes.mp4]]

约 49 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 49 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先验是更新之前的分布
- **0:04** 每个新观测，贡献一个似然因子
- **0:21** 相乘后，再归一化成新的分布
- **0:39** 后验，仍然表达不确定性

<!-- bilingual-en:start -->
- **0:00** A prior describes uncertainty before these observations.
- **0:04** Each observation contributes a likelihood factor.
- **0:21** Multiply prior and likelihood, then normalize.
- **0:39** Updating changes uncertainty; it does not eliminate it.
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
id: bayes
height: 980
```

## 观察路线

1. 保持 s=7、f=3，比较 Beta(2,2) 与 Beta(15,15)；先预测后验均值更接近哪里。
2. 设置 s=f=0，检查后验是否与先验一致。
3. 使用全部成功预设，比较 MLE=1 与后验均值。
4. 观察图例：似然只为比较形状而缩放，后验区间来自积分概率。

## 公式与适用条件

$$
p\sim\mathrm{Beta}(a,b),\quad L(p)\propto p^s(1-p)^f,\quad p\mid D\sim\mathrm{Beta}(a+s,b+f),\quad E[p\mid D]=\frac{a+s}{a+b+s+f}
$$

- 给定固定成功概率 p，每次试验条件独立且同分布为 Bernoulli(p)。
- 这里 a,b≥1；所有后验都是适当分布，不涉及不适当先验。
- 95% 区间是后验 2.5% 与 97.5% 分位数构成的等尾可信区间，不一定是最短区间。

## 回到知识点

- [[00_Knowledge/先验分布|先验分布]]：改变 Beta 先验参数，比较相同数据下的更新。
- [[00_Knowledge/后验分布|后验分布]]：条件分布、归一化和顺序更新；明确后验概率依赖先验和模型。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/confidence|置信区间的覆盖率]] · [[07_Programme/可视化实验室/实验/testing|检验、p 值与统计功效]] · [[07_Programme/可视化实验室/实验/sampling|抽样均值、大数与中心极限定理]]

## 来源与核验

- [Holmes & Huber · Modern Statistics for Modern Biology](https://web.stanford.edu/class/bios221/book/02-chap.html) — §2.9.3–2.9.6; especially §2.9.5 The posterior distribution is also a Beta。支持：Beta 先验与二项数据更新；先验参数加成功/失败次数，允许顺序更新。
- [NIST/SEMATECH · Beta Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm) — §1.3.6.6.17; density, CDF, percent point function, mean。支持：Beta 密度和均值；分位数需数值求逆，实验区间为中央等尾区间。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
