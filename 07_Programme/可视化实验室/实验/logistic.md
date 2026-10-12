<!-- visual-lab:generated:start -->
# 概率、阈值与交叉熵

<!-- bilingual-en:start -->
*Logistic probabilities & decisions*
<!-- bilingual-en:end -->

手动改变 logistic 模型参数，分开观察概率曲线、类别决策和真实数据上的交叉熵。

> [!question] 先预测，再操作
> 把分类阈值从 0.5 改成 0.2，会让模型概率本身变化吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/logistic.mp4]]

约 41 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 41 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先产生概率，再决定类别
- **0:11** 阈值只是把概率切成两边
- **0:25** 训练看概率是否支持真实标签
- **0:32** 分数、概率、决策，是三层不同对象

<!-- bilingual-en:start -->
- **0:00** A probability model and a decision rule are separate.
- **0:11** The threshold splits probabilities into decisions.
- **0:25** Log loss evaluates probability assigned to the observed label.
- **0:32** Score, probability and decision are distinct objects.
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
id: logistic
height: 980
```

## 观察路线

1. 先设置 w=b=0，预测每个点概率和平均交叉熵。
2. 改变 w 的符号和大小，观察方向与自信程度；再改变 b 看曲线平移。
3. 固定 w,b，只调整阈值，检查交叉熵不变而准确率可能改变。
4. 在第二张图中找到较低损失区域；它固定当前 b，只扫描 w，并不显示二维联合最优解。

## 公式与适用条件

$$
p(x)=\sigma(wx+b),\quad\ell(z,y)=\log(1+e^z)-yz,\quad\hat y=\mathbb1\{p(x)\ge t\}
$$

- 数据由固定 σ(1.8x−0.3) 的 Bernoulli 条件模型生成，x 均匀分布在 [−3,3]。
- 显示无正则的平均负对数似然；w,b 是手动参数。
- 输入维度为一；实际多维 logistic 模型的 log-odds 对特征线性。

## 回到知识点

- [[00_Knowledge/Logit模型|Logit模型]]：手动 w、b 经 sigmoid 映射为概率；模型曲线直接演示线性 log-odds 与概率效应的区别。
- [[00_Knowledge/二元响应MLE|二元响应MLE]]：平均交叉熵是 Bernoulli 对数似然的负值；本页显示该目标的切片，不声称已求 MLE。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/likelihood|似然曲线与最大似然估计]] · [[07_Programme/可视化实验室/实验/metrics|阈值、混淆矩阵与 ROC/PR]] · [[07_Programme/可视化实验室/实验/gradient|梯度下降何时收敛]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]]

## 来源与核验

- [scikit-learn · Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) — §1.1.11.1 Binary Case。支持：sigmoid 条件概率与二元负对数似然；本页探索手动参数，不使用文档软件求解器。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
