---
aliases:
  - Simpson's trapezoidal-midpoint identity
student_os: knowledge-atom
atom_id: CALC-DEFINT-035
atom_type: theorem
status: source-checked
requires:
  - "[[梯形法]]"
  - "[[中点法]]"
  - "[[Simpson法]]"
part_of:
  - "[[定积分与应用.canvas]]"
---

<!-- bilingual-en:start -->
# Simpson's trapezoidal–midpoint identity
<!-- bilingual-en:end -->
*Simpson组合恒等式*

<!-- bilingual-en:start -->
If you have estimated the same integral using the [[梯形法|trapezoidal rule]] and the [[中点法|midpoint rule]] on the **same equal partition**, you already have every sampled value needed for a finer [[Simpson法|Simpson rule]]. Giving the midpoint estimate twice the weight of the trapezoidal estimate produces that Simpson value exactly. The equality concerns the computed approximations; it does not say that the integral itself is known exactly.
<!-- bilingual-en:end -->

如果你已经在**同一个等宽分割**上用 [[梯形法]] 和 [[中点法]] 估计了一个积分，那么计算更细网格的 [[Simpson法]] 所需要的取样值其实已经齐了。把中点法的权重设为梯形法的两倍，得到的数恰好就是该 Simpson 近似值。这是几种近似值之间的精确相等，不代表积分真值已经精确算出。

> [!tip] Keep the grid counts visible · 先分清粗细网格
> <!-- bilingual-en:start -->
> With $n\ge1$ coarse intervals, $T_n$ uses their $n+1$ endpoints and $M_n$ uses their $n$ midpoints. Together these are the $2n+1$ endpoints of a fine grid with **$2n$ intervals**. Hence
> <!-- bilingual-en:end -->
>
> 粗网格有 $n\ge1$ 段时，$T_n$ 使用 $n+1$ 个端点，$M_n$ 使用 $n$ 个中点。合起来正好是细网格的 $2n+1$ 个端点，而细网格有 **$2n$ 段**。因此
> $$\boxed{S_{2n}=\frac{T_n+2M_n}{3}.}$$

<!-- bilingual-en:start -->
## Where the coefficients come from
<!-- bilingual-en:end -->
*系数从哪里来*

<!-- bilingual-en:start -->
Take finite real $a<b$ and a positive integer $n$. Let the fine width be $h=(b-a)/(2n)$, with nodes $x_i=a+ih$ and sampled values $f_i=f(x_i)$, for $i=0,\ldots,2n$. Assume these values are finite real numbers. The coarse width is $2h$: its endpoints are the even-indexed nodes, and its midpoints are the odd-indexed nodes.
<!-- bilingual-en:end -->

取有限实数 $a<b$ 和正整数 $n$。细步长为 $h=(b-a)/(2n)$，节点是 $x_i=a+ih$，取样值记作 $f_i=f(x_i)$，其中 $i=0,\ldots,2n$；要求这些值是有限实数。粗步长为 $2h$：粗端点对应偶数下标，粗中点对应奇数下标。

$$
T_n=h\left[f_0+2\sum_{j=1}^{n-1}f_{2j}+f_{2n}\right],
\qquad
M_n=2h\sum_{j=1}^{n}f_{2j-1}.
$$

<!-- bilingual-en:start -->
A coarse trapezoid contributes its width $2h$ times the average of its endpoint values. This explains the leading $h$ in $T_n$ and why internal coarse endpoints count twice. Each midpoint rectangle contributes $2h$ times its middle value. Multiplying $M_n$ by two, adding $T_n$, and dividing by three gives
<!-- bilingual-en:end -->

一个粗梯形的贡献是宽度 $2h$ 乘两端高度的平均，因此 $T_n$ 前面是 $h$，内部粗端点因被相邻两段共用而出现两次。每个中点矩形则贡献 $2h$ 乘中间高度。把 $M_n$ 乘二，加上 $T_n$，再除以三：

$$
\frac{T_n+2M_n}{3}
=\frac h3\left[f_0+4\sum_{j=1}^{n}f_{2j-1}
+2\sum_{j=1}^{n-1}f_{2j}+f_{2n}\right]
=S_{2n}.
$$

<!-- bilingual-en:start -->
The resulting weights are $1,4,2,4,\ldots,2,4,1$, precisely the composite Simpson weights. When $n=1$, the internal-even-node sum is empty and contributes zero. This derivation is algebra on sampled values: continuity, derivatives and even integrability are unnecessary for the identity itself. Conditions that justify approximation or control error belong to the [[Simpson误差界|Simpson error theorem]].
<!-- bilingual-en:end -->

结果中的权重正是复合 Simpson 公式的 $1,4,2,4,\ldots,2,4,1$。当 $n=1$ 时，内部偶数节点的求和为空，取零即可。整个推导只对采样值做代数运算；恒等式本身不要求连续、可导，甚至不要求函数可积。保证近似质量或控制误差的条件，另见 [[Simpson误差界]]。

<!-- bilingual-en:start -->
## An identity need not be the true integral
<!-- bilingual-en:end -->
*近似之间相等，不等于积分精确*

> [!example] One coarse interval · 一段粗网格
> <!-- bilingual-en:start -->
> For $f(x)=x^2$ on $[0,1]$, $T_1=1/2$ and $M_1=1/4$, so $S_2=(1/2+2\cdot1/4)/3=1/3$. This happens to equal the integral because Simpson's rule integrates quadratics exactly.
> <!-- bilingual-en:end -->
>
> 对 $[0,1]$ 上的 $f(x)=x^2$，$T_1=1/2$，$M_1=1/4$，所以 $S_2=(1/2+2\cdot1/4)/3=1/3$。它在这里恰好等于积分，因为 Simpson 法对二次多项式精确。
>
> <!-- bilingual-en:start -->
> For $f(x)=x^4$ on the same interval, $T_1=1/2$ and $M_1=1/16$. The identity still holds, but now $S_2=5/24$, while the exact integral is $1/5$. The error $S_2-I=1/120$ remains.
> <!-- bilingual-en:end -->
>
> 换成同一区间的 $f(x)=x^4$，$T_1=1/2$，$M_1=1/16$。恒等式仍成立，但 $S_2=5/24$，积分真值却是 $1/5$，仍有 $S_2-I=1/120$ 的误差。

<!-- bilingual-en:start -->
The matching partition is essential. Combining a trapezoidal estimate on four intervals with a midpoint estimate on a different partition is not covered by the formula. Also, $T_n$ and $M_n$ do not combine into $S_n$: the extra midpoint samples halve each coarse interval. In floating-point arithmetic, independently evaluating equivalent sums can introduce small rounding differences; the stated equality is an exact-arithmetic identity.
<!-- bilingual-en:end -->

两种方法的分割必须对应。在四段上算出的梯形值，不能直接和另一种分割的中点值套用此式。$T_n$ 与 $M_n$ 组合后也不是 $S_n$：新增的中点把每个粗区间分成了两段。浮点运算中，分别计算等价的求和式可能出现微小舍入差异；这里说的是精确算术下的恒等式。

> [!question]- Count the intervals · 数一数区间
> <!-- bilingual-en:start -->
> If you have $T_4$ and $M_4$ on the same equal partition, do you obtain $S_4$ or $S_8$? **Answer:** $S_8$, using five coarse endpoints and four midpoints: nine distinct nodes.
> <!-- bilingual-en:end -->
>
> 已经在同一等宽分割上得到 $T_4$ 与 $M_4$，组合得到的是 $S_4$ 还是 $S_8$？**答案：** $S_8$；五个粗端点加四个中点，共九个不同节点。

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Problems.pdf#page=5|MIT 18.01SC Exam 3, PDF p. 5, Problem 5]] and [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Solutions.pdf#page=6|solutions, PDF pp. 6–7]]: matching coarse/fine grids and the exact weighted identity. The coefficient proof was checked term by term; the quadratic and fourth-power examples were calculated directly.
<!-- bilingual-en:end -->

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Problems.pdf#page=5|MIT 18.01SC Exam 3，PDF 第5页第5题]]与 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Solutions.pdf#page=6|解答，PDF 第6–7页]]：对应的粗细网格及精确加权恒等式。系数证明逐项核对，二次与四次函数的例子直接重算。
