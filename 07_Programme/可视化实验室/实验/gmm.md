<!-- visual-lab:generated:start -->
# EM 怎样做软分配

<!-- bilingual-en:start -->
*Gaussian mixtures & EM*
<!-- bilingual-en:end -->

对一维双高斯数据执行实际 E/M 步，观察责任度、均值、混合权重和对数似然。

> [!question] 先预测，再操作
> 两个分量从完全相同的位置出发，EM 会自己打破对称性吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/gmm.mp4]]

约 42 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 42 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一个点，不必硬归到某一组
- **0:06** E 步：相对密度，变成分担比例
- **0:15** M 步：用责任概率作权重，重算均值
- **0:32** 软分组，依然依赖模型与初始值

<!-- bilingual-en:start -->
- **0:00** A point can share responsibility across components.
- **0:06** The E-step normalizes component support.
- **0:15** The M-step uses responsibilities as averaging weights.
- **0:32** Soft assignments still depend on assumptions and initialization.
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
id: gmm
height: 980
```

## 观察路线

1. 查看 0 轮的初始混合密度，再预测靠左和靠右点的责任度。
2. 逐轮增加 steps，检查观测数据对数似然是否不下降。
3. 把 initial 设为 0，解释为什么两个分量仍重合。
4. 增加 σ 或减小生成分离度，比较软分配的不确定性。

## 公式与适用条件

$$
\gamma_{i1}=\frac{\pi\phi_\sigma(x_i-\mu_1)}{\pi\phi_\sigma(x_i-\mu_1)+(1-\pi)\phi_\sigma(x_i-\mu_2)},\quad\pi^+=\frac1n\sum_i\gamma_{i1},\quad\mu_1^+=\frac{\sum_i\gamma_{i1}x_i}{\sum_i\gamma_{i1}}
$$

- 双分量一维 Gaussian 混合；生成权重为 0.65/0.35，模型初始权重均为 0.5。
- 共同 σ 已知固定，不估计协方差，因而不展示自由方差坍缩为零的病态。
- EM 使用 log-sum-exp/稳定 logistic 计算，分量数值质量极小时保留参数并报告。

## 回到知识点

- [[00_Knowledge/Bayes法则|Bayes法则]]：责任度将 Gaussian 密度乘混合先验再对两个分量归一化；这是连续观测密度版本，非把点密度当点概率。
- [[00_Knowledge/似然函数|似然函数]]：观测数据固定后计算每轮混合模型 log likelihood；不把其当参数概率密度。
- [[00_Knowledge/聚类解释边界|聚类解释边界]]：分量交换不改变混合密度；责任度只是当前模型内软归属，不证明真实自然类别。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/kmeans|K-means 的分配与中心更新]] · [[07_Programme/可视化实验室/实验/likelihood|似然曲线与最大似然估计]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]] · [[07_Programme/可视化实验室/实验/kde|带宽怎样改变密度估计]]

## 来源与核验

- [scikit-learn · Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html) — §2.1.1; Estimation algorithm expectation-maximization; Singularities。支持：软分配与 EM 交替、初始化与协方差退化风险；本页固定共同方差，只更新权重和均值。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
