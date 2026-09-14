---
aliases:
  - 'LICQ 要求等式约束与所有活动不等式约束的梯度在候选点线性无关'
  - LICQ requires the gradients of all equality constraints and all active inequality constraints to be linearly independent at the candidate point
student_os: knowledge-atom
atom_id: OPT-MV-007
atom_set: multivariable-optimization
atom_type: definition
status: source-checked
mastery_state: unassessed
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

# LICQ 要求等式约束与所有活动不等式约束的梯度在候选点线性无关
<!-- bilingual-en:start -->
*LICQ requires the gradients of all equality constraints and all active inequality constraints to be linearly independent at the candidate point*
<!-- bilingual-en:end -->

> [!summary] 它是什么
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
>
> <!-- bilingual-en:start -->
> At a feasible point $x^*$, first define the active set $\mathcal A(x^*)=\{i:g_i(x^*)=0\}$. The linear independence constraint qualification (LICQ) requires the gradients of every equality constraint together with every active inequality constraint to be linearly independent at $x^*$.
> <!-- bilingual-en:end -->

## 怎样实际检查
<!-- bilingual-en:start -->
*How to check it in practice*
<!-- bilingual-en:end -->

第一步只看候选点和约束值，列出活动集 $\mathcal A(x^*)$；第二步把所有等式梯度与活动不等式梯度作为向量列出；第三步检查这些向量是否线性无关。最直接的预检是计数：若等式数加活动不等式数超过变量维数 $n$，LICQ 必然失败。

<!-- bilingual-en:start -->
First identify the active set from constraint values, then list every equality gradient and active-inequality gradient, and finally test this family for linear independence. A quick preliminary check is dimensional: more such gradients than variables makes LICQ impossible.
<!-- bilingual-en:end -->

## 活动集不能由乘子反推
<!-- bilingual-en:start -->
*The active set cannot be inferred from multipliers*
<!-- bilingual-en:end -->

活动集由 $g_i(x^*)=0$ 定义，必须包括绑定但零乘子的约束。若改用 $\lambda_i>0$ 选约束，便会漏项并错误地放宽 LICQ；具体反例见 [[绑定不推正乘子]]。

<!-- bilingual-en:start -->
The active set is defined by $g_i(x^*)=0$ and must include binding constraints with zero multipliers. Selecting constraints by $\lambda_i>0$ would omit normals and falsely weaken LICQ; see [[绑定不推正乘子|the binding-zero-multiplier boundary]].
<!-- bilingual-en:end -->

## 定义之外的结论分别处理
<!-- bilingual-en:start -->
*Keep the consequences separate from the definition*
<!-- bilingual-en:end -->

LICQ 本身只是一项活动约束梯度的线性无关检查。它如何支持 [[KKT必要性]]、为何会 [[LICQ保证乘子唯一|保证乘子唯一]]、以及它为何 [[LICQ不是KKT必要条件|不是 KKT 的必要条件]]，见各自的结论。它与 Slater 检查的对象不同，见 [[Slater不等于LICQ]]。

<!-- bilingual-en:start -->
LICQ itself is only a linear-independence test on active-constraint gradients. The linked results explain KKT necessity, multiplier uniqueness, the failed converse, and the distinction from Slater.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么定义 LICQ 时必须纳入 $g_i(x^*)=0$ 但 $\lambda_i=0$ 的约束？
>
> <!-- bilingual-en:start -->
> Why must LICQ include a constraint with $g_i(x^*)=0$ but $\lambda_i=0$?
> <!-- bilingual-en:end -->
>
> **答案：** 它仍是可行域边界的一条活动约束，并贡献一个法向量；乘子为零不把这条几何边界从可行域中删除。
>
> <!-- bilingual-en:start -->
> **Answer:** It remains an active boundary of the feasible set and contributes a normal vector. A zero multiplier does not remove that geometric boundary.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Stanford CME307/MS&E311, Lecture Note 7](https://web.stanford.edu/class/msande311/lecture07.pdf)：直接核对正则点与活动约束梯度的线性无关定义。
- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：交叉核对 LICQ 的标准定义与活动集口径。

<!-- bilingual-en:start -->
- Stanford CME307/MS&E311 Lecture Note 7 was checked for regular points and linear independence of active-constraint gradients.
- MIT 6.7220 Lecture 7 was used to cross-check the standard LICQ definition and active-set convention.
<!-- bilingual-en:end -->
