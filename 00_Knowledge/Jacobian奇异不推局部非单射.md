---
aliases:
  - "Jacobian 奇异只使逆函数定理失效而不推出映射局部非单射"
  - A singular Jacobian does not imply local non-injectivity
  - Failure of the inverse function theorem converse
student_os: knowledge-atom
atom_id: CALC-MV-019
atom_set: multivariable-differentiation
atom_type: implication-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[逆函数定理]]"
related:
  - "[[Jacobian行列式]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

# Jacobian 奇异只使逆函数定理失效而不推出映射局部非单射
<!-- bilingual-en:start -->
*A singular Jacobian makes the inverse function theorem inapplicable but does not imply local non-injectivity*
<!-- bilingual-en:end -->

> [!summary] 推理边界
> 若 $\det J_F(a)=0$，只能断定 $DF(a)$ 不可逆，因而不能用 [[逆函数定理]] 在 $a$ 处保证 $C^1$ 局部逆。它不推出 $F$ 在 $a$ 附近一定不是一一对应。
>
> <!-- bilingual-en:start -->
> If $\det J_F(a)=0$, the derivative is singular and the inverse function theorem cannot guarantee a $C^1$ local inverse at $a$. This does not imply that $F$ must fail to be locally injective.
> <!-- bilingual-en:end -->

反例是
$$
F(t)=t^3.
$$
它满足 $F'(0)=0$，却在 $\mathbb R$ 上严格递增，因此全局一一对应。逆映射 $F^{-1}(y)=\sqrt[3]{y}$ 连续，但在 $0$ 处不可微，所以这里没有逆函数定理所保证的 $C^1$ 局部逆。

<!-- bilingual-en:start -->
Although $F'(0)=0$, the function is strictly increasing on $\mathbb R$ and therefore globally injective. Its inverse $F^{-1}(y)=\sqrt[3]{y}$ is continuous but not differentiable at $0$, so what fails is the $C^1$ local-inverse conclusion of the inverse function theorem.
<!-- bilingual-en:end -->

逻辑形式是
$$
DF(a)\text{ 可逆}
\Longrightarrow
\text{存在 }C^1\text{ 局部逆},
$$
但不能把前件否定改写成后件否定。

<!-- bilingual-en:start -->
The valid implication is that an invertible derivative yields a $C^1$ local inverse. Negating the hypothesis does not negate the conclusion.
<!-- bilingual-en:end -->

> [!question]- 自检
> 对 $F(t)=t^3$，$F'(0)=0$ 为什么不证明它在 $0$ 附近非单射？
>
> <!-- bilingual-en:start -->
> For $F(t)=t^3$, why does $F'(0)=0$ not prove that the map is non-injective near $0$?
> <!-- bilingual-en:end -->
>
> **答案：** 零导数只让一阶近似在该点退化；原函数仍严格递增。失败的是 $C^1$ 局部逆保证，不是单射性本身。
>
> <!-- bilingual-en:start -->
> **Answer:** A zero derivative makes the first-order approximation degenerate at that point, but the function itself remains strictly increasing. What fails is the guarantee of a $C^1$ local inverse, not injectivity.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Harvard Math 25b course page](https://people.math.harvard.edu/~elkies/M25b.13/index.html)：核对 $t^3$ 严格递增、在 $0$ 处导数为零，以及逆函数在 $0$ 处不可微的标准边界例子。
- [Stanford Math 174A notes, Theorem 1](https://math.stanford.edu/~andras/174A-2.pdf)：核对逆函数定理是从导数可逆推出局部微分同胚的充分条件，而不是其逆命题。

<!-- bilingual-en:start -->
- The Harvard Math 25b course page was checked for the standard boundary example: $t^3$ is strictly increasing, has derivative zero at $0$, and has an inverse that is not differentiable there.
- Stanford Math 174A notes, Theorem 1, was checked for the sufficient implication from an invertible derivative to a local diffeomorphism, rather than its converse.
<!-- bilingual-en:end -->
