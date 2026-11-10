---
aliases:
  - Pappus theorem for volume
  - Pappus centroid theorem for volume
student_os: knowledge-atom
atom_id: CALC-DEFINT-030
atom_type: theorem
status: source-checked
requires:
  - "[[质心]]"
  - "[[圆柱壳法]]"
related:
  - "[[旋转体重复覆盖]]"
part_of:
  - "[[定积分与应用.canvas]]"
---

<!-- bilingual-en:start -->
# Pappus's volume theorem
<!-- bilingual-en:end -->

*Pappus 体积定理*

<!-- bilingual-en:start -->
A plane region can sweep out a solid by making one full turn around a line in its plane. **Pappus's volume theorem** finds that volume from two simpler quantities: the region's area and the distance travelled by its area centroid. The centroid is the average position when equal areas receive equal weight; it agrees with the [[质心|centre of mass]] of a uniform thin plate.
<!-- bilingual-en:end -->

平面区域绕同一平面内的一条直线转一整圈，可以扫出一个立体。**Pappus 体积定理**用两个较简单的量求体积：区域的面积，以及面积形心转一圈经过的路程。形心是给相同面积相同权重所得到的平均位置，等于均匀薄片的[[质心]]。

<!-- bilingual-en:start -->
For example, rotate the rectangle $2\le x\le3$, $0\le y\le2$ about the $y$-axis, with coordinate units fixed. Its area is $2$ and its centroid is $(5/2,1)$. The centroid travels around a circle of radius $5/2$, hence a distance $5\pi$. Multiplication gives volume $2\cdot5\pi=10\pi$. The same solid is a hollow cylinder of height $2$, outer radius $3$ and inner radius $2$: its volume is $2\pi(3^2-2^2)=10\pi$.
<!-- bilingual-en:end -->

例如固定坐标单位，把矩形 $2\le x\le3$、$0\le y\le2$ 绕 $y$ 轴旋转。它的面积为 $2$，形心为 $(5/2,1)$。形心沿半径 $5/2$ 的圆转一圈，路程为 $5\pi$，因此体积为 $2\cdot5\pi=10\pi$。同一个立体也是高 $2$、外半径 $3$、内半径 $2$ 的空心圆柱，直接计算得 $2\pi(3^2-2^2)=10\pi$。

> [!important] Area times centroid travel · 面积乘形心路程
> <!-- bilingual-en:start -->
> Let $D$ be a bounded plane region with piecewise-smooth boundary and area $A>0$. Let the rotation axis be a line in the same plane. Assume **the whole region lies in one closed half-plane bounded by the axis**: it is on one side, with boundary contact allowed. Thus the axis does not cross its interior. If $d$ is the perpendicular distance from the area centroid to the axis, the geometric volume swept out in one full revolution is
> <!-- bilingual-en:end -->
>
> 设 $D$ 是边界分段光滑的有界平面区域，面积 $A>0$。旋转轴是一条与区域共面的直线。假设**整个区域位于该轴划分出的同一个闭半平面内**，也就是全在轴的一侧，但允许边界接触轴，因此轴不穿过区域内部。若面积形心到轴的垂直距离为 $d$，转一整圈扫出的几何体积为
>
> $$V=A\,(2\pi d).$$

<!-- bilingual-en:start -->
Here $d$ is a distance to the axis; $2\pi d$ is the travel distance. Area times length has volume units. Use the **area centroid**, even if the physical material has nonuniform density: its actual mass centre can move when mass is redistributed, while the solid swept out by the same region is unchanged. Positive area is needed to define the centroid's area average; a zero-area figure is outside this theorem's stated scope.
<!-- bilingual-en:end -->

这里 $d$ 是到轴的距离，$2\pi d$ 才是走过的路程；面积乘长度具有体积的单位。必须用**面积形心**：即使实际材料密度不均匀，重新分配质量也只会改变质量质心，同一区域扫出的立体并不会改变。形心的面积平均需要正面积；面积为零的图形不属于这里所述定理的范围。

<!-- bilingual-en:start -->
## Why the centroid gives exactly the right average
<!-- bilingual-en:end -->

*为什么形心恰好给出所需的平均*

<!-- bilingual-en:start -->
Choose coordinates so that the axis is the $y$-axis and $D$ lies in $x\ge0$. At horizontal position $x$, let $h(x)$ be the total length of the vertical parts of $D$. If there are several separated pieces, add their lengths. An empty slice has length zero. Take $R$ large enough to contain all these slices in $0\le x\le R$.
<!-- bilingual-en:end -->

选择坐标，使旋转轴为 $y$ 轴，并让 $D$ 位于 $x\ge0$。在横坐标 $x$ 处，记 $h(x)$ 为区域沿竖直方向占据的总长度；如果有几个分离部分，就把它们的长度相加，空切片取零。选一个足够大的 $R$，使所有切片都落在 $0\le x\le R$ 内。

<!-- bilingual-en:start -->
A narrow strip has area approximately $h(x)\,\Delta x$. The [[质心积分计算|area-centroid calculation]] therefore averages $x$ with this area weight, while the [[圆柱壳法|shell method]] multiplies the same area by the circumference $2\pi x$:
<!-- bilingual-en:end -->

一条窄带的面积近似为 $h(x)\,\Delta x$。[[质心积分计算|形心计算]]以这份面积为权重平均 $x$；[[圆柱壳法]]则把同一份面积乘以周长 $2\pi x$：

$$
A=\int_0^R h(x)\,dx,\qquad
\bar x=\frac{1}{A}\int_0^R xh(x)\,dx,\qquad
V=2\pi\int_0^R xh(x)\,dx.
$$

<!-- bilingual-en:start -->
The bar in $\bar x$ denotes the centroid's horizontal coordinate. Since the region is on the nonnegative side, $d=\bar x$. Substituting the middle formula into the last gives $V=2\pi A\bar x=2\pi Ad$. In area-element notation this is $V=2\pi\int_D x\,dA$, where $dA$ represents a small piece of area. The one-sided assumption makes each positive radius correspond to just one position $x$, avoiding [[旋转体重复覆盖|duplicate rotational coverage]].
<!-- bilingual-en:end -->

$\bar x$ 上的横线表示形心的横坐标。区域在非负一侧，因此 $d=\bar x$。把中间公式代入最后一个公式，就得到 $V=2\pi A\bar x=2\pi Ad$。用面积微元记号写，就是 $V=2\pi\int_D x\,dA$，其中 $dA$ 表示一小块面积。同侧条件让每个正半径只对应一个位置 $x$，避免[[旋转体重复覆盖]]。

<!-- bilingual-en:start -->
## Touching the axis is different from crossing it
<!-- bilingual-en:end -->

*接触轴与穿过轴不同*

<!-- bilingual-en:start -->
Take a right triangle with vertices $(0,0)$, $(r,0)$ and $(0,h)$, where $r,h>0$. Its area is $rh/2$. To find its centroid distance from the $y$-axis, use the vertical height $h(1-x/r)$:
<!-- bilingual-en:end -->

取顶点为 $(0,0)$、$(r,0)$、$(0,h)$ 的直角三角形，其中 $r,h>0$。面积为 $rh/2$。为了求形心到 $y$ 轴的距离，用竖直切片高度 $h(1-x/r)$ 计算：

$$
\bar x
=\frac{\int_0^r xh(1-x/r)\,dx}{rh/2}
=\frac{hr^2/6}{rh/2}
=\frac r3.
$$

<!-- bilingual-en:start -->
The corresponding horizontal calculation gives $\bar y=h/3$. Rotation about the $y$-axis gives a cone, and Pappus's formula gives $V=(rh/2)(2\pi r/3)=\pi r^2h/3$. The axis lies on one boundary side of the triangle. Points on that axis stay on the axis, adding no three-dimensional volume, and the shell factor $x$ is zero there. Boundary contact therefore does not spoil the proof.
<!-- bilingual-en:end -->

相应的水平切片计算给出 $\bar y=h/3$。绕 $y$ 轴旋转得到圆锥，Pappus 公式给出 $V=(rh/2)(2\pi r/3)=\pi r^2h/3$。轴沿着三角形的一条边；轴上的点旋转后仍在轴上，不贡献三维体积，壳层公式中的因子 $x$ 在那里也为零，所以触边不破坏证明。

> [!warning] A crossing axis can invalidate the formula · 穿过内部时不能直接套用
> <!-- bilingual-en:start -->
> A disk centred at the origin, rotated about the $y$-axis, generates a ball of positive volume. Its centroid is on the axis, so blindly using $A(2\pi d)$ would give zero. The disk's two halves sweep out the same ball. Taking the absolute value of the final zero cannot repair this; first choose a generating half-region or calculate the actual occupied sections.
> <!-- bilingual-en:end -->
>
> 以原点为圆心的圆盘绕 $y$ 轴旋转，得到正体积的球。形心却在轴上，直接套 $A(2\pi d)$ 会得到零。圆盘的两个半边扫出的是同一个球；对最后的零取绝对值无法修复错误，应先选择能生成立体的半区域，或计算实际占据的截面。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [OpenStax, Calculus Volume 2, §2.6, Theorem 2.14 and its proof; Theorems 2.12–2.13](https://openstax.org/books/calculus-volume-2/pages/2-6-moments-and-centers-of-mass): the external-axis formula, shell/centroid identity and uniform-density interpretation. The boundary-contact version is checked directly by the same nonnegative-radius calculation above; exercises 282–283 also use boundary axes.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Problems.pdf#page=4|MIT 18.01SC Exam 3, problem 4, p.4]] and [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Solutions.pdf#page=6|solutions, pp.5–6]]: the uniform-triangle centroid and the cone application. Coordinates are specified explicitly here and the integrals are independently evaluated.
<!-- bilingual-en:end -->

- [OpenStax《Calculus Volume 2》第2.6节，定理2.14及证明、定理2.12–2.13](https://openstax.org/books/calculus-volume-2/pages/2-6-moments-and-centers-of-mass)：外轴公式、壳层与形心的积分关系、均匀密度解释。触边版本由正文同一个非负半径计算直接核对；习题282–283也采用边界轴。
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Problems.pdf#page=4|MIT 18.01SC Exam 3，第4题，第4页]]及[[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Solutions.pdf#page=6|答案第5–6页]]：均匀三角形形心与圆锥应用；正文明确坐标，并独立完成积分计算。
