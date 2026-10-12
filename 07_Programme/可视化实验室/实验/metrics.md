<!-- visual-lab:generated:start -->
# 阈值、混淆矩阵与 ROC/PR

<!-- bilingual-en:start -->
*Confusion matrix, ROC & precision–recall*
<!-- bilingual-en:end -->

从同一批实际生成的分数计算所有阈值，观察类别比例、误报、召回率和不同面积指标。

> [!question] 先预测，再操作
> ROC 看起来不错，为什么正类很少时 precision 仍可能很低？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/metrics.mp4]]

约 36 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 36 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先划阈值，才产生四种结果
- **0:04** 按真实与预测两个维度，整理同一批样本
- **0:12** 分母不同，问题就不同
- **0:26** 阈值是取舍，指标要对应使用问题

<!-- bilingual-en:start -->
- **0:00** A threshold turns scores into four types of outcomes.
- **0:04** Cross-tabulate true and predicted classes.
- **0:12** Changing the denominator changes the question.
- **0:26** Threshold choice expresses a tradeoff.
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
id: metrics
height: 980
```

## 观察路线

1. 固定数据，逐步升高阈值，先预测 TP 与 FP 的变化，再查看混淆矩阵。
2. 观察当前点沿 ROC 和 PR 移动；完整 AUC 与 AP 应不随单一选定阈值改变。
3. 改变正类比例，比较相同分离度下 precision 的变化。
4. 将阈值设为 1，确认 precision 显示未定义；区分绘图起点约定与实际指标。

## 公式与适用条件

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\quad\mathrm{Recall}=\frac{TP}{TP+FN},\quad\mathrm{FPR}=\frac{FP}{FP+TN},\quad\mathrm{AP}=\sum_k(R_k-R_{k-1})P_k
$$

- 正例数 round(n×prevalence)，所有参数范围内正负两类都存在。
- 相同分数作为一组同时进入阈值；ROC 用梯形面积，AP 用召回率增量加权 precision。
- 混淆矩阵列是真实类别、行是预测类别；分数等于阈值时预测为正。

## 回到知识点

- [[00_Knowledge/分类评估口径|分类评估口径]]：真实 TP/FP/FN/TN 计数对应 precision、recall、FPR 三种条件分母；逐阈值 ROC/PR 分开排序与行动。
- [[00_Knowledge/灵敏度与后验概率|灵敏度与后验概率]]：稀有正类挑战将 TPR/FPR 和基率转成 precision，说明条件方向与基率不可交换。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/logistic|概率、阈值与交叉熵]] · [[07_Programme/可视化实验室/实验/testing|检验、p 值与统计功效]] · [[07_Programme/可视化实验室/实验/modelselection|验证集怎样被选择过程过拟合]] · [[07_Programme/可视化实验室/实验/fairness|相同错误率为何仍有不同 PPV]]

## 来源与核验

- [scikit-learn · Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) — Classification metrics: precision, recall, ROC and average precision。支持：混淆计数、ROC、PR、Average Precision 的定义与零分母问题。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
