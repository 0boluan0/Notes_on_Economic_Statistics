---
aliases:
  - LICQ
  - Linear independence constraint qualification
  - 线性无关约束资格
student_os: knowledge-atom
atom_id: OPT-MV-007
atom_set: multivariable-optimization
atom_type: definition
status: source-checked
requires:
  - "[[全微分]]"
  - "[[线性无关]]"
related:
  - "[[Lagrange必要条件]]"
  - "[[互补松弛]]"
  - "[[Slater条件]]"
  - "[[隐函数比较静态]]"
leads_to:
  - "[[KKT必要性]]"
  - "[[LICQ保证乘子唯一]]"
  - "[[LICQ不是KKT必要条件]]"
  - "[[Slater不等于LICQ]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

<!-- bilingual-en:start -->
# LICQ
<!-- bilingual-en:end -->

*线性无关约束资格*

<!-- bilingual-en:start -->
**LICQ** checks whether the constraints that bind at a feasible point have independent first-order effects. Each constraint has a [[全微分#实值函数、梯度与全微分|gradient]], a column of partial derivatives describing its first-order change. LICQ asks whether the equality gradients and the binding inequality gradients form a [[线性无关|linearly independent list]]: the only way to combine them into zero must be to give every gradient coefficient zero.
<!-- bilingual-en:end -->

**LICQ** 检查一个可行点处，取等号的约束是否具有彼此独立的一阶作用。每条约束都有一个[[全微分#实值函数、梯度与全微分|梯度]]，即描述一阶变化的偏导数列。LICQ 要求等式梯度与绑定不等式梯度组成[[线性无关|线性无关的向量组]]：只有给每个梯度的系数都取零，才能把它们组合成零。

<!-- bilingual-en:start -->
Work in an open set $U\subseteq\mathbb R^n$, with positive integer $n$. The constraints are real-valued functions $h_j(x)=0$ for $j=1,\ldots,p$ and $g_i(x)\le0$ for $i=1,\ldots,m$, where $p,m$ may be zero. A **feasible point** $x^*\in U$ satisfies every stated equality and inequality. Assume the constraint functions are differentiable at that point. An inequality is **active** or **binding** when its value is zero; a negative value means it is inactive, or strictly slack. Every equality constraint is included.
<!-- bilingual-en:end -->

在开集 $U\subseteq\mathbb R^n$ 上考虑问题，$n$ 为正整数。约束为实值函数组成的 $h_j(x)=0$（$j=1,\ldots,p$）和 $g_i(x)\le0$（$i=1,\ldots,m$），其中 $p,m$ 可以为零。**可行点** $x^*\in U$ 满足所有给定等式与不等式。假设约束函数在该点可微。不等式的值为零时称为**活动**或**绑定**；值为负则为非活动或严格松弛。所有等式约束都纳入检查。

> [!definition] The gradient test · 梯度判据
> <!-- bilingual-en:start -->
> At a feasible point $x^*$, first define the active set $\mathcal A(x^*)=\{i:g_i(x^*)=0\}$. The linear independence constraint qualification (LICQ) requires the gradients of every equality constraint together with every active inequality constraint to be linearly independent at $x^*$.
> <!-- bilingual-en:end -->
>
> 对可行点 $x^*$，先由
> $$
> \mathcal A(x^*)=\{i:g_i(x^*)=0\}
> $$
> 找出所有活动不等式。线性无关约束资格（linear independence constraint qualification, LICQ）要求
> $$
> \{\nabla h_j(x^*)\}_{j=1}^p
> \cup
> \{\nabla g_i(x^*)\}_{i\in\mathcal A(x^*)}
> $$
> 线性无关。

<!-- bilingual-en:start -->
The braces above describe an **indexed family**, with one gradient for each constraint. Equal vectors coming from different constraints remain separate members. For example, gradients $(1,0)^T$ and $(1,0)^T$ are dependent because coefficients $1,-1$ cancel them. Removing one as a “duplicate set element” would change the LICQ test. This condition concerns the given constraint representation at the given feasible point; it is not an intrinsic property of the feasible set alone.
<!-- bilingual-en:end -->

上面的花括号表示**按约束索引的向量族**，每条约束贡献一个梯度。不同约束即使给出相同向量，也必须分别保留。例如两个 $(1,0)^T$ 相关，因为系数 $1,-1$ 能使它们抵消；不能把其中一个当作“重复集合元素”删掉。该条件针对指定可行点处的指定约束表示，并非仅由可行集合本身决定。

<!-- bilingual-en:start -->
## Checking the gradients and the matrix
<!-- bilingual-en:end -->

## 怎样实际检查

<!-- bilingual-en:start -->
First identify the active set from constraint values, then list every equality gradient and active-inequality gradient, and finally test this family for linear independence. A quick preliminary check is dimensional: more such gradients than variables makes LICQ impossible.
<!-- bilingual-en:end -->

第一步只看候选点和约束值，列出活动集 $\mathcal A(x^*)$；第二步把所有等式梯度与活动不等式梯度作为向量列出；第三步检查这些向量是否线性无关。最直接的预检是计数：若等式数加活动不等式数超过变量维数 $n$，LICQ 必然失败。

<!-- bilingual-en:start -->
More explicitly, test whether the following implication holds for real coefficients $\alpha_j,\beta_i$: the zero on the left is the zero vector in $\mathbb R^n$.
<!-- bilingual-en:end -->

更具体地，检查下面对实系数 $\alpha_j,\beta_i$ 的蕴含是否成立；左边的零是 $\mathbb R^n$ 中的零向量。

$$
\sum_{j=1}^{p}\alpha_j\nabla h_j(x^*)+
\sum_{i\in\mathcal A(x^*)}\beta_i\nabla g_i(x^*)=0
\quad\Longrightarrow\quad
\alpha_j=0\ \text{for all }j,\quad\beta_i=0\ \text{for all }i\in\mathcal A(x^*).
$$

<!-- bilingual-en:start -->
Let $q=|\mathcal A(x^*)|$, the number of active inequalities. Put the equality gradients first, then the active gradients, into the columns of $G\in\mathbb R^{n\times(p+q)}$. The displayed test is $Gc=0\Rightarrow c=0$, where $c$ lists all coefficients. Equivalently, $G$ has full column [[矩阵秩|rank]] $p+q$. The [[Jacobian矩阵|Jacobian]] of the corresponding constraint map instead puts each gradient transpose in a **row**: $J=G^T\in\mathbb R^{(p+q)\times n}$. LICQ says $J$ has full row rank, not necessarily that $J$ is square or has a nonzero determinant.
<!-- bilingual-en:end -->

令 $q=|\mathcal A(x^*)|$，即活动不等式的条数。把等式梯度排在前面，再把活动不等式梯度排入 $G\in\mathbb R^{n\times(p+q)}$ 的各列。上述检查就是 $Gc=0\Rightarrow c=0$，其中 $c$ 列出全部系数；等价地，$G$ 的列[[矩阵秩|秩]]为 $p+q$，即满列秩。相应约束映射的 [[Jacobian矩阵|Jacobian]] 则把每个梯度的转置放在一**行**：$J=G^T\in\mathbb R^{(p+q)\times n}$。LICQ 要求它满行秩，并不要求它是方阵或具有非零行列式。

> [!example] One equality, three inequalities · 一个等式、三个不等式
> <!-- bilingual-en:start -->
> Take variables $(x,y)\in\mathbb R^2$ and constraints $h=x+y-1=0$, $g_1=-x\le0$, $g_2=-y\le0$, $g_3=x-2\le0$. At $x^*=(0,1)$ their values are $h=0$, $(g_1,g_2,g_3)=(0,-1,-2)$, so only $g_1$ is active. The relevant Jacobian and its coefficient test are
> <!-- bilingual-en:end -->
>
> 取变量 $(x,y)\in\mathbb R^2$，约束为 $h=x+y-1=0$、$g_1=-x\le0$、$g_2=-y\le0$、$g_3=x-2\le0$。在 $x^*=(0,1)$，约束值为 $h=0$、$(g_1,g_2,g_3)=(0,-1,-2)$，因此只有 $g_1$ 活动。对应 Jacobian 与系数检查为
>
> $$J=\begin{pmatrix}1&1\\-1&0\end{pmatrix},\qquad
> \alpha\binom11+\beta\binom{-1}0=\binom{\alpha-\beta}{\alpha}=0.
> $$
>
> <!-- bilingual-en:start -->
> The second coordinate gives $\alpha=0$, then the first gives $\beta=0$: LICQ holds. At $(1/2,1/2)$ all three inequalities are slack; the one nonzero equality gradient still gives LICQ, with a rectangular $1\times2$ Jacobian. If we append $g_4=-2x\le0$, the feasible set stays the same, but at $(0,1)$ the two active inequality gradients satisfy $2\nabla g_1-\nabla g_4=0$. LICQ now fails.
> <!-- bilingual-en:end -->
>
> 第二坐标给出 $\alpha=0$，第一坐标再给出 $\beta=0$，因此 LICQ 成立。在 $(1/2,1/2)$，三条不等式都松弛；唯一的非零等式梯度仍保证 LICQ，对应的是 $1\times2$ 矩形 Jacobian。若添加 $g_4=-2x\le0$，可行集不变，但在 $(0,1)$ 两个活动不等式梯度满足 $2\nabla g_1-\nabla g_4=0$，此时 LICQ 失败。

<!-- bilingual-en:start -->
If there are no equalities and no active inequalities, the family is empty and is independent by convention. Any included zero gradient makes a nonempty family dependent: give that member coefficient $1$ and all others coefficient $0$. These rules cover interior points and degenerate constraint functions without inventing an extra rank exception.
<!-- bilingual-en:end -->

若没有等式也没有活动不等式，梯度族为空，按约定线性无关。只要纳入一个零梯度，非空向量族就相关：给它系数 $1$，其他系数全取 $0$ 即可。这些规则自然覆盖内部点和退化约束函数，不需另造秩的例外。

<!-- bilingual-en:start -->
## Activity is determined by constraint values
<!-- bilingual-en:end -->

## 活动集不能由乘子反推

<!-- bilingual-en:start -->
The active set is defined by $g_i(x^*)=0$ and must include binding constraints with zero multipliers. Selecting constraints by $\lambda_i>0$ would omit normals and falsely weaken LICQ; see [[绑定不推正乘子|the binding-zero-multiplier boundary]].
<!-- bilingual-en:end -->

活动集由 $g_i(x^*)=0$ 定义，必须包括绑定但零乘子的约束。若改用 $\lambda_i>0$ 选约束，便会漏项并错误地放宽 LICQ；具体反例见 [[绑定不推正乘子]]。

<!-- bilingual-en:start -->
A multiplier is a coefficient in [[KKT条件|KKT stationarity]]. [[互补松弛|Complementary slackness]] forces an inactive inequality's multiplier to zero, but permits an active inequality to have zero multiplier too. For example, maximizing $-x^2$ subject to $x\le0$ gives $x^*=0$ and stationarity $-2x^*-\lambda=0$, hence $\lambda=0$. Its gradient $1$ must nevertheless be included in LICQ.
<!-- bilingual-en:end -->

乘子是 [[KKT条件|KKT 驻点方程]]中的系数。[[互补松弛]]迫使非活动不等式的乘子为零，但也允许活动不等式的乘子为零。例如最大化 $-x^2$ 且要求 $x\le0$，最优点 $x^*=0$ 的驻点方程为 $-2x^*-\lambda=0$，所以 $\lambda=0$；梯度 $1$ 仍必须纳入 LICQ。

> [!warning] Active does not always mean a geometric boundary · 活动未必是真正的几何边界
> <!-- bilingual-en:start -->
> The constraint $g(x)=-x^2\le0$ allows every real $x$. At $x=0$ it is active, yet $0$ is an interior point of the feasible set $\mathbb R$, and $g'(0)=0$. The zero gradient is still included, so LICQ fails. An active value alone therefore supplies neither a nonzero normal nor a genuine boundary of the feasible set. The test uses the constraint functions as written.
> <!-- bilingual-en:end -->
>
> 约束 $g(x)=-x^2\le0$ 允许每个实数 $x$。它在 $x=0$ 活动，但 $0$ 是可行集 $\mathbb R$ 的内点，而且 $g'(0)=0$。这个零梯度仍需纳入，因此 LICQ 失败。仅有活动约束值，既不保证非零法向量，也不保证该点是真正的可行域边界；检查对象是写出的约束函数。

<!-- bilingual-en:start -->
## What follows from the definition
<!-- bilingual-en:end -->

## 定义之外的结论分别处理

<!-- bilingual-en:start -->
LICQ itself is only a linear-independence test on active-constraint gradients. The linked results explain KKT necessity, multiplier uniqueness, the failed converse, and the distinction from Slater.
<!-- bilingual-en:end -->

LICQ 本身只是一项活动约束梯度的线性无关检查。它如何支持 [[KKT必要性]]、为何会 [[LICQ保证乘子唯一|保证乘子唯一]]、以及它为何 [[LICQ不是KKT必要条件|不是 KKT 的必要条件]]，见各自的结论。它与 Slater 检查的对象不同，见 [[Slater不等于LICQ]]。

<!-- bilingual-en:start -->
Checking gradients does not replace checking feasibility, and the objective function does not enter LICQ. To infer KKT **existence at a local optimum**, the usual theorem additionally assumes the objective and constraints are $C^1$ on a neighborhood, meaning continuously differentiable there. For [[LICQ保证乘子唯一|uniqueness of multipliers that already exist at one point]], linear independence at that point is enough. Neither statement says every feasible LICQ point is optimal. LICQ failure also does not rule out KKT multipliers.
<!-- bilingual-en:end -->

梯度检查不能代替可行性检查，目标函数也不参与 LICQ 定义。若要推出**局部最优点存在 KKT 乘子**，常用定理还要求目标和约束在邻域内为 $C^1$，即连续可微。若只讨论[[LICQ保证乘子唯一|同一点已经存在的乘子是否唯一]]，该点的线性无关就足够。这两件事都不表示每个可行 LICQ 点都最优；LICQ 失败也不排除 KKT 乘子存在。

> [!question]- Keep every active gradient · 保留每个活动梯度
> <!-- bilingual-en:start -->
> Why must LICQ include a constraint with $g_i(x^*)=0$ but $\lambda_i=0$?
> <!-- bilingual-en:end -->
>
> 为什么定义 LICQ 时必须纳入 $g_i(x^*)=0$ 但 $\lambda_i=0$ 的约束？
>
> <!-- bilingual-en:start -->
> **Answer:** Activity is defined by the constraint value, not its multiplier. Every indexed active gradient enters the test, including a zero gradient, which immediately makes the family dependent.
> <!-- bilingual-en:end -->
>
> **答案：** 活动性由约束值定义，不由乘子决定。每个按约束索引的活动梯度都要纳入，包括零梯度；零梯度一旦被纳入，就直接使向量族相关。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- Stanford CME307/MS&E311 Lecture Note 7 was checked for regular points and linear independence of active-constraint gradients.
- MIT 6.7220 Lecture 7 was used to cross-check the standard LICQ definition and active-set convention.
<!-- bilingual-en:end -->

- [Stanford CME307/MS&E311, Lecture Note 7](https://web.stanford.edu/class/msande311/lecture07.pdf)：直接核对正则点与活动约束梯度的线性无关定义。
- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：交叉核对 LICQ 的标准定义与活动集口径。

<!-- bilingual-en:start -->
- Precise locations: Stanford PDF pp.2–3, 5, 7–9 distinguish the $C^1$ problem, Jacobian rows, active-set regularity and KKT consequences. MIT PDF pp.2–4, definition L7.1 and theorem L7.3, provide the active set and gradient criterion; footnote 3 explicitly retains repeated gradients. PDF p.5 distinguishes pointwise differentiability from the neighborhood assumptions for a necessity theorem.
- The two-coordinate example, repeated-gradient relation and $g=-x^2$ boundary are verified directly. Row/column dimensions use the linked Jacobian definition; no objective-optimality conclusion is inferred merely from LICQ.
<!-- bilingual-en:end -->

- 精确定位：Stanford PDF 第2–3、5、7–9页区分 $C^1$ 问题、Jacobian 的行、活动集正则性与 KKT 后果。MIT PDF 第2–4页定义 L7.1、定理 L7.3 给出活动集与梯度判据；脚注3明确保留重复梯度。第5页区分逐点可微与必要性定理所需的邻域假设。
- 二维例子、重复梯度关系和 $g=-x^2$ 边界直接核算。行列尺寸与所链 Jacobian 定义一致；没有单凭 LICQ 推出目标最优性。
