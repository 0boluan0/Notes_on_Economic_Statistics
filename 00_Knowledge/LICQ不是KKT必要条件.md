---
aliases:
  - 'LICQ 是保证局部最优满足 KKT 的充分约束资格，但不是 KKT 乘子存在的必要条件'
  - LICQ is a sufficient constraint qualification for KKT necessity but is not necessary for KKT multipliers to exist
student_os: knowledge-atom
atom_id: OPT-MV-021
atom_set: multivariable-optimization
atom_type: implication-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[LICQ条件]]"
  - "[[KKT条件]]"
related:
  - "[[KKT必要性]]"
  - "[[绑定不推正乘子]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# LICQ 是保证局部最优满足 KKT 的充分约束资格，但不是 KKT 乘子存在的必要条件
<!-- bilingual-en:start -->
*LICQ is a sufficient constraint qualification for KKT necessity but is not necessary for KKT multipliers to exist*
<!-- bilingual-en:end -->

> [!summary] 保证失效不等于结论为假
> [[KKT必要性]] 可以用 LICQ 作为一条充分路线：局部最优 + LICQ $\Rightarrow$ 存在 KKT 乘子。但 LICQ 失败只表示这条定理不能负责保证；某个具体问题仍可能直接解出满足 [[KKT条件]] 的乘子。
>
> <!-- bilingual-en:start -->
> LICQ is one sufficient route from local optimality to the existence of KKT multipliers. Its failure removes that guarantee but does not logically imply that KKT multipliers fail to exist in the particular problem.
> <!-- bilingual-en:end -->

## 反例：重复活动约束使 LICQ 失败，KKT 仍成立
<!-- bilingual-en:start -->
*Counterexample: duplicated active constraints break LICQ while KKT still holds*
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
Maximising $-x^2$ subject to the duplicate constraints $x\le0$ and $2x\le0$ has the unique optimum zero. Both active gradients are dependent, so LICQ fails. Nevertheless, setting both multipliers to zero satisfies feasibility, stationarity, signs, and complementary slackness.
<!-- bilingual-en:end -->

## 正确读法
<!-- bilingual-en:start -->
*The correct interpretation*
<!-- bilingual-en:end -->

若 LICQ 成立，可以安全调用相应的必要性与 [[LICQ保证乘子唯一|乘子唯一]] 定理。若 LICQ 失败，应寻找其他约束资格或直接检查 KKT 系统，而不是宣布“最优点不存在”或“KKT 必然失败”。

<!-- bilingual-en:start -->
When LICQ holds, its KKT-necessity and multiplier-uniqueness theorems are available. When it fails, one should seek another qualification or inspect the KKT system directly rather than concluding that no optimum or multipliers can exist.
<!-- bilingual-en:end -->

> [!question]- 自检
> “LICQ 失败，所以这个点不可能满足 KKT”错在什么逻辑方向？
>
> <!-- bilingual-en:start -->
> What is wrong with the inference, “LICQ fails, so this point cannot satisfy KKT”?
> <!-- bilingual-en:end -->
>
> **答案：** LICQ 只是 KKT 必要性的一项充分条件。充分条件失败不能推出目标结论失败。
>
> <!-- bilingual-en:start -->
> **Answer:** LICQ is only a sufficient condition for KKT necessity. Failure of a sufficient condition does not imply failure of the conclusion.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：核对 LICQ 作为约束资格的充分地位。
- 正文重复约束例子直接核验 LICQ 失败而 KKT 成立。

<!-- bilingual-en:start -->
- MIT 6.7220 Lecture 7 was checked for LICQ as a sufficient constraint qualification.
- The duplicated-constraint example directly verifies failure of LICQ together with valid KKT multipliers.
<!-- bilingual-en:end -->
