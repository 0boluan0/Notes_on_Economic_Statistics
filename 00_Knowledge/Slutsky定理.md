---
aliases:
  - "当一列随机变量依分布收敛且另一列依概率收敛到常数时，和、积与非零分母之比保留相应分布极限"
  - "Slutsky theorem"
  - "Slutsky 定理"
student_os: knowledge-atom
atom_id: PROB-LIM-005
atom_set: convergence-and-limit-theorems
atom_type: theorem
status: source-checked
mastery_state: unassessed
part_of:
  - "[[收敛与极限定理.canvas|收敛与极限定理]]"
requires:
  - "[[依概率收敛]]"
  - "[[依分布收敛]]"
  - "[[连续映射定理]]"
related:
  - "[[经典中心极限定理]]"
  - "[[标准误含义]]"
  - "[[Monte Carlo均值标准误]]"
---

# 当一列随机变量依分布收敛且另一列依概率收敛到常数时，和、积与非零分母之比保留相应分布极限
<!-- bilingual-en:start -->
*When one sequence converges in distribution and another converges in probability to a constant, their sum, product, and ratio with a nonzero limiting denominator retain the corresponding distributional limits*
<!-- bilingual-en:end -->

> [!summary] 把一致估计量代入渐近分布
> 对每个 $n$，设 $X_n$ 与 $Y_n$ 能在同一概率空间上联合定义。若 $X_n\Rightarrow X$ 且 $Y_n\xrightarrow p c$，则 $(X_n,Y_n)\Rightarrow(X,c)$。连续映射定理于是给出和与积的极限；只有 $c\ne0$ 且比值在有限样本中定义良好时，才能对除法使用同一论证。$X_n$ 与 $Y_n$ 不需要独立。
> <!-- bilingual-en:start -->
> Suppose $X_n$ and $Y_n$ are jointly defined for each $n$. If $X_n\Rightarrow X$ and $Y_n\xrightarrow p c$, then $(X_n,Y_n)\Rightarrow(X,c)$. The continuous mapping theorem gives the limits of sums and products. Division uses the same argument only when $c\ne0$ and the finite-sample ratio is well defined. Independence is not required.
> <!-- bilingual-en:end -->

## 三个结论与分母条件

对每个 $n$，$X_n$ 与 $Y_n$ 必须能在同一概率空间上联合定义，否则 $X_n+Y_n$、$X_nY_n$ 与 $X_n/Y_n$ 本身没有共同样本结果可供计算。在此前提与上述收敛条件下，

$$
X_n+Y_n\Rightarrow X+c,
\qquad
X_nY_n\Rightarrow cX.
$$

若进一步 $c\ne0$，并且 $X_n/Y_n$ 已在 $Y_n=0$ 的事件上作出明确、不会改变渐近结论的定义（常见充分做法是每个 $n$ 都有 $P(Y_n=0)=0$），则

$$
\frac{X_n}{Y_n}\Rightarrow\frac{X}{c}.
$$

因为 $Y_n\xrightarrow p c\ne0$，对任意 $0<\delta<|c|$，

$$
P(|Y_n|\le |c|-\delta)
\le P(|Y_n-c|\ge\delta)\longrightarrow0,
$$

所以分母以趋近 1 的概率远离 0；但这不替代有限样本中“比值必须有定义”的实现约定。
<!-- bilingual-en:start -->
The sum converges to $X+c$ and the product to $cX$. For ratios, the limiting constant must be nonzero and the ratio must be defined at every finite sample size, for example because $P(Y_n=0)=0$ or because a harmless convention is specified on zero-denominator events. Convergence of $Y_n$ to a nonzero constant makes it bounded away from zero with probability tending to one, but does not itself define an otherwise undefined finite-sample ratio.
<!-- bilingual-en:end -->

## 为什么不需要独立

关键是常数极限：$Y_n\xrightarrow p c$ 使联合向量 $(X_n,Y_n)$ 的第二坐标渐近退化，从而即使两列来自同一数据、彼此高度相关，也有联合收敛到 $(X,c)$。随后把联合向量代入 [[连续映射定理]] 中的三个函数：

$$
g_+(x,y)=x+y,
\quad g_\times(x,y)=xy,
\quad g_/(x,y)=x/y
$$

前两个函数处处连续；最后一个函数只在 $y\ne0$ 的区域连续。
<!-- bilingual-en:start -->
Independence is unnecessary because convergence of $Y_n$ to a constant makes the second coordinate asymptotically degenerate. Joint convergence to $(X,c)$ follows even when both sequences use the same data. Addition and multiplication are continuous everywhere, while division is continuous only away from zero.
<!-- bilingual-en:end -->

## 学生化是典型用途

若经典 CLT 给出

$$
\frac{\sqrt n(\bar X_n-\mu)}{\sigma}\Rightarrow N(0,1),
$$

而样本标准差满足 $s_n\xrightarrow p\sigma>0$，则

$$
\frac{\sqrt n(\bar X_n-\mu)}{s_n}
=
\frac{\sqrt n(\bar X_n-\mu)}{\sigma}
\frac{\sigma}{s_n}
\Rightarrow N(0,1).
$$

这里用的是 $\sigma/s_n\xrightarrow p1$ 与乘积结论。Slutsky 只负责把一致尺度估计代入；它不负责证明 $s_n$ 一致，也不修复原始 CLT 的条件失败。
<!-- bilingual-en:start -->
Studentisation is the standard application. A CLT supplies the normal limit with the true scale, and consistency of $s_n$ supplies $\sigma/s_n\to1$ in probability. Slutsky combines them. It does not prove consistency of the scale estimator or repair a failed CLT.
<!-- bilingual-en:end -->

## 零分母不是小技术细节

若 $c=0$，一般没有 $X_n/Y_n\Rightarrow X/c$ 这样的结论。取 $X_n=Y_n=1/n$，两者都依概率收敛到 0，但

$$
\frac{X_n}{Y_n}=1
$$

恒成立，而“$0/0$”没有定义。即使 $Y_n$ 很少等于 0，任何实际计算仍应明确这些事件如何处理，而不能让软件静默产生无穷或缺失值后继续套定理。
<!-- bilingual-en:start -->
When $c=0$, there is no general ratio conclusion. Taking $X_n=Y_n=1/n$ makes both sequences converge to zero while their ratio is identically one, and the putative limit $0/0$ is undefined. Implementations must also specify how finite-sample zero denominators are handled.
<!-- bilingual-en:end -->

> [!question]- 可核验自检
> 已知 $T_n\Rightarrow N(0,4)$、$A_n\xrightarrow p3$、$B_n\xrightarrow p2$。分别写出 $T_n+A_n$、$B_nT_n$ 与 $T_n/B_n$ 的分布极限。需要假设三列独立吗？
>
> **答案：** 依次为 $N(3,4)$、$N(0,16)$ 与 $N(0,1)$；最后一项还要求比值在有限样本中定义良好。无需独立，因为另两列收敛到常数。
> <!-- bilingual-en:start -->
> The limits are respectively $N(3,4)$, $N(0,16)$, and $N(0,1)$, with a well-defined finite-sample ratio required for the last result. Independence is unnecessary because the other sequences converge to constants.
> <!-- bilingual-en:end -->

## 来源与原文定位

- A. W. van der Vaart, [*Asymptotic Statistics* (1998), Lemma 2.8, pp. 11–12](https://doi.org/10.1017/CBO9780511802256)：精确定位 Slutsky 的加法、乘法、倒数与商，以及常数极限下无需独立的条件。
- UC Berkeley Statistics 210A, [Lecture 19, “Continuous Mapping / Slutsky's Theorem,” p. 7](https://www.stat.berkeley.edu/~wfithian/courses/stat210a/lectures/lecture19.pdf)：定位联合收敛、连续映射和非零常数分母条件。
