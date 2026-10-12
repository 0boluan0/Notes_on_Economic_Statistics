<!-- visual-lab:generated:start -->
# 模型复杂度与泛化

<!-- bilingual-en:start -->
*Polynomial fitting & generalization*
<!-- bilingual-en:end -->

用同一组带噪训练数据拟合不同阶多项式，比较训练、独立验证与已知真实曲线。

> [!question] 先预测，再操作
> 训练误差更低，为什么新数据预测可能更差？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/overfit.mp4]]

约 38 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 38 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 模型首先看见的是这批训练点
- **0:10** 把没参与拟合的数据放进来
- **0:16** 复杂度变化，用两条误差曲线一起判断
- **0:29** 拟合能力，是可以使用也可以误用的自由度

<!-- bilingual-en:start -->
- **0:00** The fit is chosen using these training observations.
- **0:10** Now reveal observations the fit did not use.
- **0:16** Track training and validation error across model complexity.
- **0:29** Flexibility must be assessed using independent evidence.
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
id: overfit
height: 980
```

## 观察路线

1. 固定种子、n 和噪声，只修改 degree，比较同一训练与验证数据。
2. 从 d=1 开始预测偏差形状，再观察更高阶模型是否追随噪声。
3. 固定高阶模型，逐渐提高 λ，比较训练误差、验证误差和曲线边界行为。
4. 至少换三个种子，检查你观察到的规律是否只是某次数据的偶然；持续调参使用的验证集不能再当作最终测试集。

## 公式与适用条件

$$
y=\sin(\pi x)+\varepsilon,\quad \hat f(x)=\sum_{j=0}^{d}\beta_jP_j(x),\quad \min_\beta\sum_i\{y_i-\hat f(x_i)\}^2+\lambda\sum_{j=1}^{d}\beta_j^2
$$

- 训练与验证样本从同一 x∼Uniform[−1,1]、y=sin(πx)+ε 机制独立生成。
- ε 为独立零均值正态噪声；控件 noise 是标准差。噪声为零时没有不可约噪声，但仍可能拟合不足或数值不稳。
- Pⱼ 为 Legendre 基；常数项不惩罚。λ=0 且列满秩时为普通最小二乘；参数多于信息时要报告秩或非唯一性。

## 回到知识点

- [[00_Knowledge/训练验证测试划分|训练验证测试划分]]：拟合与选择的角色区分；交互中反复查看的是验证数据。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]] · [[07_Programme/可视化实验室/实验/sampling|抽样均值、大数与中心极限定理]] · [[07_Programme/可视化实验室/实验/gradient|梯度下降何时收敛]] · [[07_Programme/可视化实验室/实验/bootstrap|Bootstrap 能告诉我们什么]]

## 来源与核验

- [Cosma Shalizi · CMU Optimism and Over-Fitting](https://www.stat.cmu.edu/~cshalizi/sml/21/lectures/04/lecture-04.html) — Lecture 4; Estimating the true risk; The moral: approximation-estimation trade-off。支持：训练误差的乐观性、嵌套模型类训练误差不增，以及逼近误差与估计误差的交换。
- [Ryan Tibshirani · CMU Modern Regression 1](https://stat.cmu.edu/~ryantibs/datamining/lectures/16-modr1.pdf) — slides 3–8; Important details (slide 7)。支持：岭回归的平方和目标、闭式解、标准化和不惩罚截距；方差讨论属于重复样本层面。
- [Cawley & Talbot · JMLR 11 (2010)](https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf) — Abstract; §5.1 An Unbiased Performance Evaluation Methodology; §5.3 Another Example of Biased Evaluation Methodology。支持：对有限样本的模型选择准则也会过拟合；评估需隔离选择过程。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
