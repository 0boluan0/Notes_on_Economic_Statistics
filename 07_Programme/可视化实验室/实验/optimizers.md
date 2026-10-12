<!-- visual-lab:generated:start -->
# GD、动量与Adam实际走哪条路

<!-- bilingual-en:start -->
*Gradient descent, momentum & Adam*
<!-- bilingual-en:end -->

在同一个可解析的旋转二次函数上逐步执行三种更新，比较轨迹与完整损失。

> [!question] 先预测，再操作
> 同一个学习率下最快的方法，就一定是最好的优化器吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/optimizers.mp4]]

约 43 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 43 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 同一个起点，三种更新记忆
- **0:19** 路径不同，因为状态不同
- **0:33** 这张轨迹图不构成方法排名

<!-- bilingual-en:start -->
- **0:00** Three update rules start at the same point.
- **0:19** Different internal states produce different updates.
- **0:33** A single parameter setting is not a benchmark.
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
id: optimizers
height: 980
```

## 观察路线

1. 先查看同一起点(3,2)，计算GD稳定界2/最大特征值。
2. 增加共同学习率，观察轨迹离开窗口后完整损失是否继续增长。
3. 把动量系数设为0，检查Momentum轨迹与GD重合。
4. 保持特征值不变旋转谷底，理解Adam逐坐标缩放的坐标依赖。

## 公式与适用条件

$$
f(\theta)=\tfrac12\theta^TH\theta,\quad m_t=\beta m_{t-1}+(1-\beta)g_t,\quad \theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
$$

- 精确全梯度、二维正定二次目标，无小批噪声、权重衰减或学习率日程。
- Adam固定β₂=0.999、ε=10⁻⁸；ε在平方根之外。
- 参数轨迹只显示|θⱼ|≤4的部分；完整原始路径与损失在数据中保留。

## 回到知识点

- [[00_Knowledge/梯度最陡方向|梯度最陡方向]]：欧氏最陡下降只解释GD当前方向；有限步长、动量记忆和Adam缩放需要另行分析。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/gradient|梯度下降何时收敛]] · [[07_Programme/可视化实验室/实验/initialization|初始化如何改变信号与梯度尺度]] · [[07_Programme/可视化实验室/实验/regularization|Dropout保留均值，却怎样改变损失]]

## 来源与核验

- [Kingma & Ba · Adam (ICLR 2015)](https://arxiv.org/pdf/1412.6980) — Algorithm 1, p.2; §2–3。支持：一阶/二阶矩指数平均、偏差修正、epsilon在平方根之外的更新式。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
