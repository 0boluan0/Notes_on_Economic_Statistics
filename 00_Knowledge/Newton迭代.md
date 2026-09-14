---
aliases:
  - "Newton 迭代用当前点处切线的零点生成下一次求根近似"
  - Newton iteration uses the zero of the tangent at the current point as the next root approximation
student_os: knowledge-atom
atom_id: CS-NR-004
atom_set: numerical-root-finding
atom_type: algorithm-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[导数]]"
related:
  - "[[线性近似]]"
leads_to:
  - "[[Newton局部收敛]]"
  - "[[Newton失效边界]]"
  - "[[混合求根]]"
part_of:
  - "[[数值求根.canvas|数值求根]]"
  - "[[导数的应用.canvas]]"
---

# Newton 迭代用当前点处切线的零点生成下一次求根近似
<!-- bilingual-en:start -->
*Newton iteration uses the zero of the tangent at the current point as the next root approximation*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> Newton 迭代在当前点 $x_k$ 用原函数的切线作局部线性模型，并把这条切线的零点作为下一次求根近似：$x_{k+1}=x_k-f(x_k)/f'(x_k)$。若 $f(x_k)=0$，应先直接返回当前根；否则只有在 $f'(x_k)\ne0$ 时才能计算这一步。
>
> <!-- bilingual-en:start -->
> Newton iteration uses the tangent at the current point $x_k$ as a local linear model and takes the tangent's zero as the next root approximation: $x_{k+1}=x_k-f(x_k)/f'(x_k)$. If $f(x_k)=0$, return the current root before dividing; otherwise the update can be computed only when $f'(x_k)\ne0$.
> <!-- bilingual-en:end -->

## 更新式来自一阶局部模型

在 $x_k$ 附近写一阶 Taylor 近似

$$
f(x_k+s)\approx f(x_k)+f'(x_k)s.
$$

令右侧等于 0，得到

$$
s_k=-\frac{f(x_k)}{f'(x_k)},
\qquad
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)},
$$

前提是 $f'(x_k)\ne0$。这一步求的是切线的零点，不是原函数零点；只有当局部线性模型足够好时，它才是有用的修正。

实际顺序应先检查 $f(x_k)=0$。若当前点已经是根，就不需要、也不应该再计算 $f(x_k)/f'(x_k)$；这对导数同样为 0 的重根尤其重要。只有在当前点尚不是根时，才要求分母非零并生成下一候选。

例如求 $x^2-2=0$ 时，

$$
x_{k+1}=\frac12\left(x_k+\frac{2}{x_k}\right).
$$

从 $x_0=1.5$ 出发，$x_1\approx1.4166667$，$x_2\approx1.4142157$。这个例子展示了局部速度，却不能替所有初值提供收敛证明。

## 收敛与失效由独立原子判断

本卡只定义切线更新。它何时具有局部二次速度，由 [[Newton局部收敛]] 给出条件；零导数、巨大步长、循环、其他吸引域和越界等机制，由 [[Newton失效边界]] 单独诊断。实现层面的残差、最大迭代和非有限值出口统一见 [[数值迭代停止条件]]；若还能先找到异号区间，[[混合求根]] 可以在不丢掉括区间的前提下尝试 Newton 候选。

<!-- bilingual-en:start -->
This card owns only the tangent update. [[Newton局部收敛|Newton local convergence]] states the conditions under which the update becomes locally quadratic, while [[Newton失效边界|Newton failure boundaries]] separately diagnose zero derivatives, large steps, cycles, competing basins, and domain escape. [[数值迭代停止条件|Numerical stopping criteria]] governs residual, iteration-limit, and non-finite exits; when a sign-changing bracket is also available, [[混合求根|hybrid root finding]] can try Newton candidates without discarding that bracket.
<!-- bilingual-en:end -->

> [!question]- 自检
> Newton 更新式本身回答了什么，又没有回答什么？
>
> **答案：** 它回答怎样从 $x_k$ 计算切线零点 $x_{k+1}$；它不独自回答序列是否收敛、收敛到哪个根或误差多大。这些问题分别需要局部定理、失败诊断或括区间证书。

## 来源与核验

- MIT 18.330, [*Introduction to Numerical Analysis, Chapter 4: Nonlinear equations*](https://ocw.mit.edu/courses/18-330-introduction-to-numerical-analysis-spring-2012/5b325bfa56a599794c7196de926844b0_MIT18_330S12_Chapter4.pdf)：核对切线更新推导，以及“更新式有定义”与“序列收敛”必须分开的边界。
- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec06.pdf|MIT 6.100L Lecture 6 slides]]，slides 30–35：核对从导数产生下一猜测以及平方根迭代的课程语境。
- SciPy, [`newton`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html)：核对初值与导数接口、步长停止不保证已找到根，以及有括区间时更安全方法的官方边界。
- [[01_Math/01_calculus/02_Applications_of_Differentiation.md#33a：迭代公式的推导|微积分课程页]]：复用切线近似的几何推导；本卡只拥有更新式，局部收敛条件与失败机制分别由 [[Newton局部收敛]]、[[Newton失效边界]] 拥有。

> [!success] 独立内容审核通过
> 切线推导、精确根先返回、公式定义域、关系与来源均已通过第二位模型复审；`status: source-checked`。局部收敛与失效机制已保持为独立原子；学习证据尚未评估，`mastery_state: unassessed`。
