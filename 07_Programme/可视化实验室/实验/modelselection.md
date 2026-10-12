<!-- visual-lab:generated:start -->
# 验证集怎样被选择过程过拟合

<!-- bilingual-en:start -->
*Validation selection optimism*
<!-- bilingual-en:end -->

让所有候选拥有相同真实风险，再在有限验证数据上择优，直接分离“挑中好运气”的效应。

> [!question] 先预测，再操作
> 候选越多，最低验证误差越低，是否一定是模型能力提升了？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/modelselection.mp4]]

约 34 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 34 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 验证分数也会带着噪声
- **0:06** 挑最小值，也挑中了有利的噪声
- **0:17** 用独立测试，重新衡量这个赢家
- **0:25** 反复看测试集，也会把它变成选择集

<!-- bilingual-en:start -->
- **0:00** Validation scores contain sampling noise.
- **0:06** Selecting the minimum selects favorable noise.
- **0:17** Evaluate the selected model on independent data.
- **0:25** Repeated test-driven selection destroys that independence.
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
id: modelselection
height: 980
```

## 观察路线

1. 先确认所有候选的理论风险相同，预测最小验证误差相对真实风险的位置。
2. 保持种子与 nval，只增加候选数，观察第一次重复的候选前缀保持相同。
3. 看多次重复中的所选验证与全新测试误差，区分偶然波动与平均乐观性。
4. 增加验证样本量，讨论它为什么缓解但不能自动消除所有选择偏差。

## 公式与适用条件

$$
f_j(X)=aX_j,\quad Y,X_j\overset{iid}{\sim}N(0,1),\quad R_j=E(Y-aX_j)^2=1+a^2,\quad j^*=\arg\min_j\hat R_j^{val}
$$

- 每个候选固定为 aXⱼ；没有使用验证标签训练系数。所有真实风险为 1+a²。
- 同一重复内各候选共享验证目标 Y，但输入坐标独立；候选得分不被错误假设为完全独立。
- 每次选择后用 400 个全新观测估计测试风险；重复次数控制模拟精度。

## 回到知识点

- [[00_Knowledge/训练验证测试划分|训练验证测试划分]]：固定候选在验证集上选择，然后以全新数据估计赢家风险，直接隔离两种信息角色。
- [[00_Knowledge/测试集自适应复用|测试集自适应复用]]：全等真实风险候选的择优产生乐观性，挑战明确测试反馈一旦调参便转为验证。
- [[00_Knowledge/验证管线隔离|验证管线隔离]]：本页候选已冻结，验证数据只择优；该选择分数不能同时作为独立最终表现。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/overfit|模型复杂度与泛化]] · [[07_Programme/可视化实验室/实验/tree|决策树怎样划分空间]] · [[07_Programme/可视化实验室/实验/boosting|提升算法怎样逐轮修正残差]] · [[07_Programme/可视化实验室/实验/metrics|阈值、混淆矩阵与 ROC/PR]]

## 来源与核验

- [Cawley & Talbot · JMLR (2010)](https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf) — Abstract; §5.1 and §5.3 Performance evaluation protocols。支持：有限模型选择准则也会过拟合；选择必须作为完整拟合流程的一部分被独立评价。
- [scikit-learn · Nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) — Opening explanation and experiment comparison。支持：在用于选择的同一证据上报告性能会偏乐观；嵌套评价隔离选择。当前实验不是交叉验证实现。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
