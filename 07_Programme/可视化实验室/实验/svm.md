<!-- visual-lab:generated:start -->
# 软间隔与 hinge 损失

<!-- bilingual-en:start -->
*Linear margins & hinge loss*
<!-- bilingual-en:end -->

手动旋转、缩放和平移线性判别函数，核对每个样本的函数间隔与软间隔目标。

> [!question] 先预测，再操作
> 分类正确的点，为什么仍可能有正的 hinge 损失？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/svm.mp4]]

约 29 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 29 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 分类边界之外，还要看离边界多远
- **0:06** 把真实类别乘进去，正确一侧才是正
- **0:19** 大间隔与违例代价，需要一起权衡

<!-- bilingual-en:start -->
- **0:00** A separator also defines a margin.
- **0:06** A signed margin includes the true class label.
- **0:19** The objective trades margin width against violations.
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
id: svm
height: 980
```

## 观察路线

1. 先读函数间隔 m，判断 m<0、0≤m<1、m≥1 各意味着什么。
2. 固定 b=0，增加 ‖w‖，比较同一决策线与两条间隔线。
3. 只改变 C，观察目标值变化；手动参数在本页不会自动重训练。
4. 设置 ‖w‖=0，检查哪些几何量必须显示未定义。

## 公式与适用条件

$$
m_i=y_i(w^Tx_i+b),\quad\xi_i=\max(0,1-m_i),\quad J=\frac12\|w\|^2+C\sum_i\xi_i
$$

- 二类标签使用 −1 和 +1，线性 f(x)=wᵀx+b。
- 采用 ½‖w‖²+C∑hinge 的和损失约定；截距不惩罚。
- 噪声加在观测坐标上；只展示有限二维合成样本，没有核映射或自动优化。

## 回到知识点

- [[00_Knowledge/梯度与水平集|梯度与水平集]]：f(x)=wᵀx+b 的 f=0、±1 三个水平集共享法向量 w；零法向控件显式展示正则条件边界。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/logistic|概率、阈值与交叉熵]] · [[07_Programme/可视化实验室/实验/regularization|Dropout保留均值，却怎样改变损失]] · [[07_Programme/可视化实验室/实验/metrics|阈值、混淆矩阵与 ROC/PR]]

## 来源与核验

- [scikit-learn · Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) — §1.4.7.1 Mathematical formulation; LinearSVC primal form。支持：½‖w‖²+C∑max(0,1−y f) 的线性软间隔目标；最优性与支持向量需另行求解。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
