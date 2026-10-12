<!-- visual-lab:generated:start -->
# 相同错误率为何仍有不同 PPV

<!-- bilingual-en:start -->
*Base rates, equalized odds & predictive values*
<!-- bilingual-en:end -->

构造具有相同 TPR/FPR 的两个抽象组，直接算出混淆概率、选择率和阳性预测值。

> [!question] 先预测，再操作
> 两组真实正类比例不同，equalized odds 能否同时保证相同 PPV？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/fairness.mp4]]

约 28 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 28 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 同样的错误率，会给出同样的预测可靠性吗？
- **0:05** 现在只看“预测为正”的那一列
- **0:19** 先说清分母，再讨论公平约束

<!-- bilingual-en:start -->
- **0:00** Equal error rates need not imply equal predictive value.
- **0:05** Condition on positive predictions in each group.
- **0:19** Specify the conditioning event before comparing fairness metrics.
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
id: fairness
height: 980
```

## 观察路线

1. 先用热图的四格联合概率手算两个组的 PPV。
2. 只改变基准率，核对共同 TPR/FPR 保持不变但选择率和 PPV 可以改变。
3. 把两个基准率设成相同，检查差异消失。
4. 依次看 FPR=0 和 TPR=FPR=0 的边界，区分 PPV=1 与 PPV 未定义。

## 公式与适用条件

$$
\mathrm{PPV}(\pi)=\frac{\pi\,\mathrm{TPR}}{\pi\,\mathrm{TPR}+(1-\pi)\mathrm{FPR}},\quad P(\hat Y=1)=\pi\,\mathrm{TPR}+(1-\pi)\mathrm{FPR}
$$

- 组 A/B 完全合成，基准率均严格位于 0 与 1；共同 TPR/FPR 是构造条件。
- 热图为组内联合概率，每组四格合计 1；每千人数据只是期望计数，可为非整数。
- 当 TPR>0、FPR>0 且基准率不同时，PPV 严格不同；零率边界单独处理。

## 回到知识点

- [[00_Knowledge/Bayes法则|Bayes法则]]：从真实标签条件下的 TPR/FPR 反转为预测正类条件下的 PPV，必须使用组内基准率。
- [[00_Knowledge/条件概率|条件概率]]：混淆矩阵三个条件分母；没有正预测时 PPV 的条件事件概率为零，不能填作 0。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/metrics|阈值、混淆矩阵与 ROC/PR]] · [[07_Programme/可视化实验室/实验/testing|检验、p 值与统计功效]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]]

## 来源与核验

- [Hardt, Price & Srebro · Equality of Opportunity in Supervised Learning](https://arxiv.org/pdf/1610.02413) — §2 Definition 2.1; binary equalized odds; §2.1。支持：相同真实标签下的跨组条件预测率；本页 PPV 差异由 Bayes 公式自行推导，未把其称为所有公平定义的一般不可能定理。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
