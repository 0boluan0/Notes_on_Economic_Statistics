<!-- visual-lab:generated:start -->
# 梯度下降何时收敛

<!-- bilingual-en:start -->
*Gradient descent & conditioning*
<!-- bilingual-en:end -->

在可精确分析的椭圆二次函数上，观察步长、曲率和起点怎样决定收敛、振荡与发散。

> [!question] 先预测，再操作
> 每步都沿负梯度走，为什么函数值还可能越来越大？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/gradient.mp4]]

约 42 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 42 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 梯度给方向，步长给距离
- **0:11** 再算梯度，再迈下一步
- **0:26** 陡峭方向，对步长更敏感
- **0:33** 步子更大，不一定更快

<!-- bilingual-en:start -->
- **0:00** The gradient gives a direction; the step size gives distance.
- **0:11** Recompute the gradient at every new point.
- **0:26** The steep direction sets the stability limit.
- **0:33** A larger step can oscillate or diverge.
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
id: gradient
height: 980
```

## 观察路线

1. 保留 κ=4，预测 η=0.2、0.5、0.6 的轨迹，再逐个使用预设。
2. 对照参数轨迹与损失曲线：振荡可以同时伴随损失下降。
3. 把 κ 提高到 20，保留原学习率，先算 2/κ 再判断稳定性。
4. 在越界步长下设 y₀=0，解释为什么这条特殊轨迹仍可能收敛。

## 公式与适用条件

$$
f(x,y)=\frac{x^2+\kappa y^2}{2},\quad x_{t+1}=(1-\eta)x_t,\quad y_{t+1}=(1-\eta\kappa)y_t,\quad 0<\eta<\frac2\kappa
$$

- κ≥1，Hessian=diag(1,κ) 正定，唯一最小点为原点。
- 严格界 0<η<2/κ 保证所有起点收敛；若起点在不稳定特征方向的分量恰为零，单条轨迹可例外。
- 这是固定步长、精确梯度、二维二次损失；不能把其精确界直接当作神经网络训练保证。

## 回到知识点

- [[00_Knowledge/梯度最陡方向|梯度最陡方向]]：局部最陡下降的欧氏含义；本实验另行展示有限步长的稳定条件，不把方向定理当作收敛定理。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/tangent|割线如何变成切线]] · [[07_Programme/可视化实验室/实验/linear|矩阵怎样改变空间]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]]

## 来源与核验

- [MIT · Foundations of Computer Vision](https://visionbook.mit.edu/gradient_descent.html) — §10.3 Basic Gradient Descent; §10.4 Learning Rate Schedules。支持：梯度迭代和学习率作用；本实验严格稳定界由二次函数递推自行推导，不外推到任意损失。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
