<!-- visual-lab:generated:start -->
# 共线性与岭回归

<!-- bilingual-en:start -->
*Multicollinearity & ridge regression*
<!-- bilingual-en:end -->

用两个相关解释变量比较 OLS 与岭回归，区分系数不稳定、预测拟合和正则化。

> [!question] 先预测，再操作
> 两个变量几乎一样时，为什么预测看起来不错，系数却可能变化很大？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/regression.mp4]]

约 39 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 39 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 线性预测，是几项贡献相加
- **0:09** 残差连接预测与真实观测
- **0:22** 拟合目标，把所有残差平方汇总
- **0:30** 预测稳定，不代表每个系数都稳定

<!-- bilingual-en:start -->
- **0:00** A linear prediction adds feature contributions.
- **0:09** A residual connects a prediction to the observed response.
- **0:22** The fitting objective aggregates squared residuals.
- **0:30** Correlated predictors can destabilize individual coefficients.
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
id: regression
height: 980
```

## 观察路线

1. 保持种子和样本参数，只改变 λ，以确保比较同一份数据。
2. 把 ρ 从 0 调到 0.98，检查样本相关、VIF 和条件数；它们由当前设计矩阵计算。
3. 在高相关下更换几个种子，观察系数变化，再判断是否有重复样本层面的稳定现象。
4. 观察残差与预测；本页训练拟合不能证明新数据上的泛化改进。

## 公式与适用条件

$$
\hat\beta_\lambda=\arg\min_\beta\{\|y_c-X_c\beta\|_2^2+\lambda\|\beta\|_2^2\}=(X_c^TX_c+\lambda I)^{-1}X_c^Ty_c,\quad \mathrm{VIF}=\frac1{1-r^2}
$$

- y=Xβ+ε，真实 β=(1,1)，误差与设计变量独立、均值为 0，噪声控件给出其标准差。
- ρ 是生成机制参数，有限样本的 r 不必等于 ρ；VIF=1/(1−r²) 只针对此处两个解释变量的设计。
- 目标使用残差平方和而非平均平方误差；若换为平均损失，同等惩罚强度需 λ_mean=λ_sum/n。

## 回到知识点

- [[00_Knowledge/多重共线性|多重共线性]]：相关设计列导致弱识别；有限样本相关和生成参数区分。
- [[00_Knowledge/近似共线性精度|近似共线性精度]]：系数可能不稳定而拟合值较稳定；换种子观察不能替代重复样本证明。
- [[00_Knowledge/岭正则化|岭正则化]]：平方和损失加系数二范数惩罚，λ 正值改变估计目标。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/projection|投影、残差与最短距离]] · [[07_Programme/可视化实验室/实验/gradient|梯度下降何时收敛]] · [[07_Programme/可视化实验室/实验/overfit|模型复杂度与泛化]] · [[07_Programme/可视化实验室/实验/pca|PCA、尺度与重构]]

## 来源与核验

- [Georgia Tech · Interactive Linear Algebra](https://textbooks.math.gatech.edu/ila/least-squares.html) — §6.5.1 Least-Squares Solutions。支持：最小二乘与正交投影、正规方程、系数解唯一性的满列秩条件。
- [Ryan Tibshirani · CMU Modern Regression 1](https://stat.cmu.edu/~ryantibs/datamining/lectures/16-modr1.pdf) — slides 3–8; Important details (slide 7)。支持：岭回归的平方和目标、闭式解、标准化和不惩罚截距；方差讨论属于重复样本层面。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
