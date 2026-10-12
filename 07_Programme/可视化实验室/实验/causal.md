<!-- visual-lab:generated:start -->
# 混杂怎样让观察关联与干预效应反向

<!-- bilingual-en:start -->
*Known structural model, Simpson reversal & intervention*
<!-- bilingual-en:end -->

精确枚举一个明确已知的二元结构模型，比较组内响应、观察混合比例、标准化和do干预。

> [!question] 先预测，再操作
> 每个Z组中T都提高结果均值，总体观察均值差为什么仍可为负？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/causal.mp4]]

约 36 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 36 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 比较两组之前，先看组里都有谁
- **0:06** 观察差异，使用两套不同的人群权重
- **0:13** 干预比较，保留同一套总体组成
- **0:27** 调整有效，需要因果假设支持

<!-- bilingual-en:start -->
- **0:00** Group composition can change an aggregate comparison.
- **0:06** Observed groups can have different mixtures of Z.
- **0:13** An intervention compares treatments in the same population.
- **0:27** Adjustment requires a justified causal model.
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
id: causal
height: 980
```

## 观察路线

1. 先分别比较两个Z组内T从0到1的变化。
2. 再读总体观察曲线，并用第二图解释两个T组的组成差异。
3. 将选择强度设0，使T独立于Z，检查观察差与干预差重合。
4. 保持τ=0而改变选择强度，区分“存在关联”与“存在干预效应”。

## 公式与适用条件

$$
Z\sim\mathrm{Bernoulli}(q),\ e_0=.5+.4s,\ e_1=.5-.4s,\quad P(Y=1\mid T=t,Z=z)=.2+.5z+\tau t,\quad E[Y\mid do(t)]=\sum_z E[Y\mid t,z]P(z)
$$

- 三个相互独立的Uniform外生变量生成Z、T、Y；T由e_Z阈值产生，Y由0.2+0.5Z+τT阈值产生。
- e_Z在0.1至0.9，结果概率在0.05至0.85；所有条件格都有正概率。
- 模型已知且不存在给定Z后剩余的共同原因；展示精确总体量，不从观察数据发现因果图。

## 回到知识点

- [[00_Knowledge/概率依赖与因果|概率依赖与因果]]：观察条件关系不等于替换结构方程的干预关系，需要因果结构与识别假设。
- [[00_Knowledge/均值差的因果分解|均值差的因果分解]]：已知同质组内均值效应的合成模型中，观察组均值差含选择组成偏差。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/testing|检验、p 值与统计功效]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]] · [[07_Programme/可视化实验室/实验/fairness|相同错误率为何仍有不同 PPV]] · [[07_Programme/可视化实验室/实验/rl-policy|策略梯度如何改变行动概率]]

## 来源与核验

- [Pearl · Causal inference in statistics: An overview (2009)](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf) — §2.2；§3.2.1；§3.3.1，式(25)。支持：观察和干预的区别、替换结构方程、满足后门条件时的协变量调整。具体二元SCM及所有概率为本实验设定。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
