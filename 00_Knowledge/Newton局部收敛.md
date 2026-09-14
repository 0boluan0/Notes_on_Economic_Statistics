---
aliases:
  - "当初值足够接近光滑函数的简单根且导数不退化时，Newton 误差受前一轮误差平方控制；这是局部结论而非全局成功保证"
  - When the initial value is sufficiently close to a simple root of a smooth function with a nondegenerate derivative, Newton error is bounded by a constant times the previous error squared, but only locally
student_os: knowledge-atom
atom_id: CS-NR-008
atom_set: numerical-root-finding
atom_type: convergence-theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Newton迭代]]"
related:
  - "[[残差控制根误差]]"
  - "[[算法成本模型]]"
leads_to:
  - "[[Newton失效边界]]"
  - "[[求根策略]]"
part_of:
  - "[[数值求根.canvas|数值求根]]"
  - "[[导数的应用.canvas]]"
---

# 当初值足够接近光滑函数的简单根且导数不退化时，Newton 误差受前一轮误差平方控制；这是局部结论而非全局成功保证
<!-- bilingual-en:start -->
*When the initial value is sufficiently close to a simple root of a smooth function with a nondegenerate derivative, Newton error is bounded by a constant times the previous error squared; this is local, not a global guarantee of success*
<!-- bilingual-en:end -->

> [!summary] 原子收敛定理
> Newton 法的“快”来自一个有条件的误差递推：迭代点先要落在简单根附近，函数要有受控的二阶曲率，导数还要远离 0。满足这些条件后，下一轮误差由当前误差的平方控制；这些条件没有说明任意初值都能进入该邻域。
>
> <!-- bilingual-en:start -->
> Newton's speed comes from a conditional error recurrence. The iterates must already lie near a simple root, the second derivative must be controlled, and the first derivative must stay away from zero. Then the next error is bounded by the square of the current error. These conditions do not say that an arbitrary starting point will enter that neighbourhood.
> <!-- bilingual-en:end -->

## Taylor 余项给出误差平方界

设 $f(\alpha)=0$，$f'(\alpha)\ne0$，并记 $e_k=x_k-\alpha$。取一个包含 $\alpha$ 的开区间 $I$，假设 $f$ 在 $I$ 上二阶连续可微。对任意满足 $f'(x_k)\ne0$ 的 $x_k\in I$，Taylor 公式给出某个介于 $x_k$ 与 $\alpha$ 之间的 $\xi_k\in I$，使

$$
e_{k+1}
=\frac{f''(\xi_k)}{2f'(x_k)}e_k^2.
$$

进一步，若在整个 $I$ 上

$$
|f'(x)|\ge m>0,
\qquad
|f''(x)|\le M,
$$

则

$$
|e_{k+1}|
\le \frac{M}{2m}|e_k|^2.
$$

令 $C=M/(2m)$。选半径 $\rho>0$，使闭区间 $[\alpha-\rho,\alpha+\rho]\subset I$ 且 $q=C\rho<1$。若 $e_0=0$，初值已经是根；否则，只要 $|e_0|\le\rho$，就有

$$
|e_1|\le C|e_0|^2\le q|e_0|<|e_0|\le\rho.
$$

同一论证可归纳到每一轮；零误差是平凡固定情形，非零误差则至多按因子 $q<1$ 收缩。因此迭代点不会离开这个闭区间，误差趋于 0，并始终满足平方界。这里的“初值足够接近”有一个明确的充分条件，而不是先假定后续点会留在邻域。

只要后续误差非零，带符号的渐近系数为

$$
\lim_{k\to\infty}\frac{e_{k+1}}{e_k^2}
=\frac{f''(\alpha)}{2f'(\alpha)}
$$

若 $f''(\alpha)\ne0$ 且迭代没有有限步命中根，则绝对值比率趋向正的有限常数，Newton 法在该根附近的收敛阶恰为 2。若 $f''(\alpha)=0$，上述比率趋于 0；平方上界仍成立，但实际可能高于二阶，不能无条件说“阶恰为 2”。若有限步已经命中根，序列直接结束，也无需再给后续误差比率定义。这些都是局部误差结论，不是单步函数计算成本，也不是任意初值下的全局复杂度。

## 简单根条件不能删掉

对 $f(x)=x^3$，根 $\alpha=0$ 的重数为 3。当 $x_k\ne0$ 时，普通 Newton 更新为

$$
x_{k+1}=x_k-\frac{x_k^3}{3x_k^2}
=\frac23x_k,
$$

因此 $|e_{k+1}|=(2/3)|e_k|$，只有线性收敛。迭代仍可能趋近根，但简单根附近的平方误差界不能移植到这个重根。若初值不在局部收敛邻域，序列能否进来还要由 [[Newton失效边界]] 和实际问题结构判断。

> [!question]- 自检
> 一次 Newton 运行前几轮误差下降很慢，能否仅凭这一现象否定简单根附近的平方误差界？
>
> **答案：** 不能。前几轮可能还没有进入定理要求的局部邻域；也要核对根是否简单、导数是否远离 0，以及误差是否用同一个真实根定义。平方误差界只描述条件满足后的局部阶段；是否恰为二阶还要看渐近系数是否非零。

## 来源与核验

- MIT 18.330, [*Introduction to Numerical Analysis, Chapter 4: Nonlinear equations*](https://ocw.mit.edu/courses/18-330-introduction-to-numerical-analysis-spring-2012/5b325bfa56a599794c7196de926844b0_MIT18_330S12_Chapter4.pdf)：核对简单根附近的 Newton 误差展开、平方误差界，以及渐近系数非零时恰为二阶的结论。重根反例由本卡把 $f(x)=x^3$ 直接代入更新式验证。
- MIT 6.006, [*Lecture 12: Square Roots, Newton's Method*](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-12-square-roots-newtons-method/)：交叉核验平方根 Newton 迭代的误差分析与精度增长。
- [[Newton迭代]]：复用切线更新式；本卡只拥有局部平方误差界及其收敛阶边界。

> [!success] 独立内容审核通过
> 误差恒等式、局部不变区域、平方误差界、收敛阶边界、重根反例与来源均已通过第二位模型复审；`status: source-checked`。学习证据尚未评估，`mastery_state: unassessed`。
