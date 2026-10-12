<!-- visual-lab:generated:start -->
# 参数不确定性与新观测噪声怎样分开

<!-- bilingual-en:start -->
*Bayesian linear posterior & predictive uncertainty*
<!-- bilingual-en:end -->

解析计算二维线性参数后验，同时显示潜在均值可信区间、新观测预测区间与噪声方差。

> [!question] 先预测，再操作
> 数据越来越多以后，为什么新观测仍然不能被精确预测？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/uncertainty.mp4]]

约 31 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 31 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 同一个位置，可以问两种不确定性
- **0:07** 先只保留参数不确定性
- **0:14** 预测新观测，还要加入观测噪声
- **0:21** 增加数据，不能消除新观测的固有噪声

<!-- bilingual-en:start -->
- **0:00** At one location, distinguish mean uncertainty from response noise.
- **0:07** Parameter uncertainty creates a latent-mean interval.
- **0:14** A new response adds another source of variation.
- **0:21** More data do not remove irreducible response noise.
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
id: uncertainty
height: 980
```

## 观察路线

1. 在同一x位置比较均值区间与新观测区间，确认它们中心相同。
2. 阅读方差分解图，直接核对两条方差曲线之差等于σ²。
3. 固定种子、σ和输入范围，增加n观察参数后验如何收缩。
4. 把查询点移出设计范围，再看斜率不确定性如何放大预测均值的不确定性。

## 公式与适用条件

$$
S=(\tau_0^{-2}I+\sigma^{-2}X^TX)^{-1},\quad m=\sigma^{-2}SX^Ty,\quad \mathrm{Var}(f_*\mid D)=\phi_*^TS\phi_*,\quad \mathrm{Var}(y_*\mid D)=\phi_*^TS\phi_*+\sigma^2
$$

- 真实合成均值为0.4+1.2x，参数先验为N(0,τ₀²I)，噪声σ已知。
- 固定种子时n增加使用同一个30点样本池；改σ会同时改变生成数据与似然假设。
- 95%函数区间和预测区间都是逐点区间；参数图使用二维联合95%椭圆。

## 回到知识点

- [[00_Knowledge/后验分布|后验分布]]：后验是给定模型、先验和数据的条件分布；本页解析计算Gaussian参数后验及其预测推送。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]] · [[07_Programme/可视化实验室/实验/confidence|置信区间的覆盖率]] · [[07_Programme/可视化实验室/实验/bootstrap|Bootstrap 能告诉我们什么]]

## 来源与核验

- [Rasmussen & Williams · Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/RW2.pdf) — §2.1，式(2.8)–(2.11)；§2.2 noisy predictions。支持：Gaussian线性模型权重后验、潜在函数预测及加上观测噪声的预测分布。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
