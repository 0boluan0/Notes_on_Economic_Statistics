---
aliases:
- LICQ failure and KKT multipliers
- LICQ 失效与 KKT 乘子
student_os: knowledge-atom
atom_id: OPT-MV-021
atom_set: multivariable-optimization
atom_type: implication-boundary
status: source-checked
requires:
- '[[LICQ条件]]'
- '[[KKT条件]]'
related:
- '[[KKT必要性]]'
- '[[绑定不推正乘子]]'
part_of:
- '[[多元优化.canvas|多元优化]]'
---

<!-- bilingual-en:start -->
# LICQ failure and KKT multipliers
<!-- bilingual-en:end -->

*LICQ 失效与 KKT 乘子*

<!-- bilingual-en:start -->
A constraint qualification tells us when an optimum must admit a particular gradient balance. It is tempting to reverse that statement: if the qualification fails, perhaps the balance is impossible. That reversal is invalid for [[LICQ条件|LICQ]]. Dependent constraint gradients can still represent the objective gradient; we have to check the actual equations.
<!-- bilingual-en:end -->

约束资格说明什么时候最优点必定满足某种梯度平衡。人们容易倒过来想：资格失败，平衡是不是就不可能？对[[LICQ条件|LICQ]]，这个反推不成立。相关的约束梯度仍可能表示目标梯度，必须检查具体方程。

<!-- bilingual-en:start -->
Use a maximization problem with differentiable objective $f$ and constraints $g_i(x)\le0$, $h_j(x)=0$, where $x\in\mathbb R^n$ is the choice. A feasible point satisfies all these restrictions. An inequality is **active**, or binding, when $g_i(x)=0$. [[KKT条件|The KKT conditions]] require feasibility, nonnegative inequality multipliers $\lambda_i$, unrestricted equality multipliers $\nu_j$, stationarity, and the products $\lambda_i g_i(x)=0$. A multiplier is a coefficient in the gradient balance, not another choice variable in the original problem.
<!-- bilingual-en:end -->

采用最大化问题，目标为可微函数 $f$，约束为 $g_i(x)\le0$、$h_j(x)=0$，选择向量是 $x\in\mathbb R^n$。可行点满足这些限制；$g_i(x)=0$ 的不等式称为**活动**或绑定。[[KKT条件]]要求可行性、非负不等式乘子 $\lambda_i$、不限制符号的等式乘子 $\nu_j$、驻点方程及乘积 $\lambda_i g_i(x)=0$。乘子是梯度平衡中的系数，不是原问题中额外的选择变量。

$$
\nabla f(x^*)=\sum_i\lambda_i\nabla g_i(x^*)+\sum_j\nu_j\nabla h_j(x^*).
$$

<!-- bilingual-en:start -->
[[LICQ条件|LICQ]] asks whether the family consisting of every equality gradient and every active inequality gradient is [[线性无关|linearly independent]], keeping each constraint index. For the usual [[KKT必要性|necessity theorem]], assume the objective and constraints are continuously differentiable near $x^*$ and that $x^*$ is a local maximizer. LICQ then supplies a sufficient qualification. The following conclusion concerns that complete set of hypotheses.
<!-- bilingual-en:end -->

[[LICQ条件|LICQ]]检查全部等式梯度和全部活动不等式梯度按约束索引组成的族是否[[线性无关]]。调用通常的[[KKT必要性]]定理时，还要假设目标与约束在 $x^*$ 附近连续可微，且 $x^*$ 是局部最大点；LICQ 在这些前提下提供一项充分资格。下面的结论要在这整组条件下理解。

<!-- bilingual-en:start -->
LICQ is one sufficient route from local optimality to the existence of KKT multipliers. Its failure removes that guarantee but does not logically imply that KKT multipliers fail to exist in the particular problem.
<!-- bilingual-en:end -->

[[KKT必要性]] 可以用 LICQ 作为一条充分路线：局部最优 + LICQ $\Rightarrow$ 存在 KKT 乘子。但 LICQ 失败只表示这条定理不能负责保证；某个具体问题仍可能直接解出满足 [[KKT条件]] 的乘子。

<!-- bilingual-en:start -->
## A dependent gradient family with valid multipliers
<!-- bilingual-en:end -->

## 反例：重复活动约束使 LICQ 失败，KKT 仍成立

<!-- bilingual-en:start -->
Maximising $-x^2$ subject to the duplicate constraints $x\le0$ and $2x\le0$ has the unique optimum zero. Both active gradients are dependent, so LICQ fails. Nevertheless, setting both multipliers to zero satisfies feasibility, stationarity, signs, and complementary slackness.
<!-- bilingual-en:end -->

考虑
$$
\max_x -x^2\quad\text{s.t.}\quad x\le0,\qquad 2x\le0.
$$
唯一最优点是 $x^*=0$。两条约束都活动，梯度 $1$ 与 $2$ 在线性空间 $\mathbb R$ 中相关，所以 LICQ 失败。取
$$
\mathcal L=-x^2-\lambda_1x-\lambda_2(2x).
$$
在 $x^*=0$ 令 $\lambda_1^*=\lambda_2^*=0$，原问题可行、对偶可行、驻点和互补松弛全部成立。因此 KKT 乘子确实存在。

<!-- bilingual-en:start -->
Here every feasible $x$ satisfies $x\le0$, and $-x^2\le0$ with equality only at $x=0$. This proves the unique global optimum directly. At that point the two active gradients are $g_1^{\prime}(0)=1$ and $g_2^{\prime}(0)=2$. They are dependent because $2\cdot1+(-1)\cdot2=0$ uses coefficients that are not both zero. Although the second restriction duplicates the first feasible restriction after positive rescaling, both indices remain in the LICQ test.
<!-- bilingual-en:end -->

每个可行点都满足 $x\le0$，而 $-x^2\le0$ 仅在 $x=0$ 时取等号，直接证明了唯一全局最优点。该点的两个活动梯度是 $g_1^{\prime}(0)=1$、$g_2^{\prime}(0)=2$；由于 $2\cdot1+(-1)\cdot2=0$ 使用了不全为零的系数，它们相关。第二条限制是第一条的正比例重复，但 LICQ 检验仍须保留两个约束索引。

<!-- bilingual-en:start -->
Differentiating the displayed Lagrangian gives $-2x-\lambda_1-2\lambda_2=0$. At zero this becomes $\lambda_1+2\lambda_2=0$. With $\lambda_1,\lambda_2\ge0$, both coefficients must be zero. Thus the pair is actually unique in this example despite LICQ failure. Primal feasibility holds because both constraint values are zero; multiplier signs hold; stationarity holds; and both complementary products are $0\cdot0=0$. This checks the entire KKT system.
<!-- bilingual-en:end -->

对上面的 Lagrangian 求导得到 $-2x-\lambda_1-2\lambda_2=0$，在零点化为 $\lambda_1+2\lambda_2=0$。由于 $\lambda_1,\lambda_2\ge0$，两者只能都为零。因此，这个例子虽不满足 LICQ，乘子对实际上仍唯一。两条约束值为零保证原问题可行；乘子符号正确；驻点式成立；两个互补乘积都是 $0\cdot0=0$，完整 KKT 系统全部通过。

<!-- bilingual-en:start -->
A different failed qualification can give a different outcome. Maximize $f(x)=x$ subject to $g(x)=x^2\le0$. The only feasible point, hence the unique optimum, is zero. But $g^{\prime}(0)=0$, so stationarity would demand $1=\lambda\cdot0$, which no multiplier can satisfy. These two examples establish precisely what failure of LICQ does and does not decide: it leaves multiplier existence to further analysis.
<!-- bilingual-en:end -->

另一种资格失效可以给出不同结果。最大化 $f(x)=x$，约束为 $g(x)=x^2\le0$。唯一可行点、因而唯一最优点是零；但 $g^{\prime}(0)=0$，驻点式要求 $1=\lambda\cdot0$，任何乘子都做不到。两个例子共同说明 LICQ 失败的准确含义：乘子是否存在，仍需进一步判断。

<!-- bilingual-en:start -->
## Use the theorem in its stated direction
<!-- bilingual-en:end -->

## 正确读法

<!-- bilingual-en:start -->
Keep the neighborhood smoothness and local-optimum hypotheses when invoking necessity. [[LICQ保证乘子唯一|Multiplier uniqueness under LICQ]] is a separate conditional statement at the same feasible point for the same constraint functions, assuming KKT multipliers already exist. Neither statement makes LICQ a necessary condition for existence or uniqueness.
<!-- bilingual-en:end -->

调用必要性时，要保留邻域光滑性和局部最优前提。[[LICQ保证乘子唯一]]是另一个条件命题：固定同一可行点和同一组约束函数，并假设 KKT 乘子已经存在。两条结论都不把 LICQ 变成乘子存在或唯一的必要条件。

<!-- bilingual-en:start -->
When LICQ holds, its KKT-necessity and multiplier-uniqueness theorems are available. When it fails, one should seek another qualification or inspect the KKT system directly rather than concluding that no optimum or multipliers can exist.
<!-- bilingual-en:end -->

若 LICQ 成立，可以安全调用相应的必要性与 [[LICQ保证乘子唯一|乘子唯一]] 定理。若 LICQ 失败，应寻找其他约束资格或直接检查 KKT 系统，而不是宣布“最优点不存在”或“KKT 必然失败”。

> [!question]- Check the implication · 自检
> <!-- bilingual-en:start -->
> What is wrong with the inference, “LICQ fails, so this point cannot satisfy KKT”?
>
> **Answer:** LICQ is only a sufficient condition for KKT necessity. Failure of a sufficient condition does not imply failure of the conclusion.
> <!-- bilingual-en:end -->
>
> “LICQ 失败，所以这个点不可能满足 KKT”错在什么逻辑方向？
>
>
>
> **答案：** LICQ 只是 KKT 必要性的一项充分条件。充分条件失败不能推出目标结论失败。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.7220, Lecture7, Definition L7.1 and Theorem L7.3, PDF pp.3–5](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/mit6_7220_s25_lec07.pdf#page=3): the four KKT conditions, C¹ qualification, and repeated active-gradient family. The notes minimize; here replacing the objective by its negative gives the stated maximization signs. Both one-dimensional examples above are verified directly, including every KKT condition.
<!-- bilingual-en:end -->

- MIT 6.7220 讲义7，定义L7.1与定理L7.3，PDF3–5页：核对四类 KKT 条件、连续可微资格及保留重复索引的活动梯度族。原讲义使用最小化；目标取负得到本文最大化的符号。本文两个一维例子均直接验证，逐项检查 KKT。

<!-- bilingual-en:start -->
- MIT 6.7220 Lecture 7 was checked for LICQ as a sufficient constraint qualification.
- The duplicated-constraint example directly verifies failure of LICQ together with valid KKT multipliers.
<!-- bilingual-en:end -->

- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：核对 LICQ 作为约束资格的充分地位。
- 正文重复约束例子直接核验 LICQ 失败而 KKT 成立。
