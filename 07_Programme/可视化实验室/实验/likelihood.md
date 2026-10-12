<!-- visual-lab:generated:start -->
# 似然曲线与最大似然估计

<!-- bilingual-en:start -->
*Likelihood & boundary MLEs*
<!-- bilingual-en:end -->

并列计算 Bernoulli 和已知方差 Gaussian 的相对似然，观察样本量、候选参数与边界最优解。

> [!question] 先预测，再操作
> 似然曲线被缩放到峰值 1 后，能当作参数的概率密度吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/likelihood.mp4]]

约 40 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 40 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 数据已经固定，移动的是参数假设
- **0:04** 一个候选概率，给整份数据一个相对支持度
- **0:24** 最大似然选择最支持这份数据的参数
- **0:30** 把概率看作数据的函数，和看作参数的函数，要分清

<!-- bilingual-en:start -->
- **0:00** The data stay fixed while the candidate parameter varies.
- **0:04** A candidate probability assigns support to the whole dataset.
- **0:24** Maximum likelihood selects the best-supported candidate.
- **0:30** Likelihood is a function of the parameter for fixed observations.
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
id: likelihood
height: 980
```

## 观察路线

1. 先看实际整数成功计数，手算 s/n，确认它对应曲线最高处。
2. 固定观测数据，只移动候选 p 或 μ，区分参数评估与更换数据。
3. 把成功比例设为 0 或 1，解释为什么最优点移到参数边界。
4. 增加 n 或改变 Gaussian σ，比较曲线宽度；两幅图是独立示例，不是共享同一参数。

## 公式与适用条件

$$
\ell_B(p)=s\log p+(n-s)\log(1-p),\quad\hat p=s/n;\qquad\ell_N(\mu)=\mathrm{const}-\frac1{2\sigma^2}\sum_i(x_i-\mu)^2,\quad\hat\mu=\bar x
$$

- Bernoulli 成功计数为 round(n×所选观测比例)，控件表示构造出的观测情况。
- Gaussian 样本独立来自 N(0.8,σ²)，σ 是已知值且不进行估计。
- 若使用二项计数模型，组合系数与 p 无关，MLE 不变；当前 log likelihood 使用固定序列约定。

## 回到知识点

- [[00_Knowledge/似然函数|似然函数]]：两幅图固定观测改变候选参数；峰值归一化不等于参数概率密度。
- [[00_Knowledge/极大似然估计|极大似然估计]]：Bernoulli s/n 与已知 σ Gaussian 样本均值最大化各自似然，边界例子不要求内点导数为零。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]] · [[07_Programme/可视化实验室/实验/logistic|概率、阈值与交叉熵]] · [[07_Programme/可视化实验室/实验/gmm|EM 怎样做软分配]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]]

## 来源与核验

- [NIST/SEMATECH · Maximum Likelihood](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3652.htm) — §1.3.6.5.2 Maximum Likelihood。支持：给定观测后的参数似然、独立密度乘积及优化含义；两个具体 MLE 由本项目独立推导。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
