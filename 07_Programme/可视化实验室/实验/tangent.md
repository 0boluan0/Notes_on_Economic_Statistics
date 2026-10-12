<!-- visual-lab:generated:start -->
# 割线如何变成切线

<!-- bilingual-en:start -->
*Derivatives & finite differences*
<!-- bilingual-en:end -->

移动观察点与差分步长，把局部变化率、解析导数和数值近似放到同一张图上。

> [!question] 先预测，再操作
> 把步长缩小十倍，差商误差一定也缩小十倍吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/tangent.mp4]]

约 36 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 36 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 斜率从两个点开始
- **0:06** 横向间隔变小，平均变化率怎样变？
- **0:27** 导数是局部线性近似的系数

<!-- bilingual-en:start -->
- **0:00** A secant measures change between two points.
- **0:06** Shrink the interval while keeping the first point fixed.
- **0:27** The derivative is the coefficient of a local linear approximation.
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
id: tangent
height: 980
```

## 观察路线

1. 先只看原函数：预测当前点的斜率正负，再读解析导数。
2. 固定 x=1、三次函数，分别设置 h=0.5、0.1、0.01，比较前向和中心差分的绝对误差。
3. 把 x 改为 0，解释为什么两种误差相同；用下方挑战的展开式核对。
4. 换成 sin x 或 exp x，区分函数值的变化与导数值的变化。

## 公式与适用条件

$$
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}h,\quad D_h^+f=\frac{f(x+h)-f(x)}h,\quad D_h^0f=\frac{f(x+h)-f(x-h)}{2h}
$$

- 当前三个函数处处光滑；中心差分需要能在 x−h 和 x+h 处评价函数。
- 在通常的光滑性条件下，前向截断误差为 O(h)，中心为 O(h²)；个别位置主项可能消失。
- 数值计算使用有限精度；本页 h≥0.001，不能用此范围证明无限缩小 h 永远改善精度。

## 回到知识点

- [[00_Knowledge/导数|导数]]：差商极限、切线与局部线性近似；实验把有限 h 和极限值并列。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/gradient|梯度下降何时收敛]] · [[07_Programme/可视化实验室/实验/overfit|模型复杂度与泛化]]

## 来源与核验

- [OpenStax · Calculus Volume 1](https://openstax.org/books/calculus-volume-1/pages/3-1-defining-the-derivative) — §3.1 Tangent Lines; The Derivative of a Function at a Point。支持：差商、导数极限和切线；本实验的三次函数误差式为直接代数推导。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
