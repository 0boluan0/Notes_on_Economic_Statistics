---
aliases:
  - "奇异半正定矩阵可以有 $LDL^T$ 分解，但零主元可能使标准递推中断并令分解不唯一"
  - "奇异半正定矩阵可以有 $LDL^T$ 分解，但零主元会破坏标准递推与唯一性"
  - Zero-pivot boundary of LDLT factorization
student_os: knowledge-atom
atom_id: LA-SPD-027
atom_set: symmetric-positive-definite
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[LDL正定判别]]"
  - "[[半正定矩阵]]"
related:
  - "[[Cholesky 正定判据]]"
  - "[[半正定零方向]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

# 奇异半正定矩阵可以有 $LDL^T$ 分解，但零主元可能使标准递推中断并令分解不唯一
<!-- bilingual-en:start -->
*A singular positive-semidefinite matrix may have an $LDL^T$ factorization, but a zero pivot can break the standard recurrence and uniqueness*
<!-- bilingual-en:end -->

> [!summary] 零主元的影响
> $A\succeq0$ 并不保证标准 $LDL^T$ 递推（不做行交换）像正定情形那样顺利、唯一。零主元可能使递推中的除法失效；即使某个分解存在，单位下三角因子也可能不唯一。
> <!-- bilingual-en:start -->
> Positive semidefiniteness does not guarantee that the standard unpivoted $LDL^T$ recurrence remains well defined or unique. A zero pivot can make the recurrence divide by zero, and an existing factorization may still have a nonunique unit lower-triangular factor.
> <!-- bilingual-en:end -->

取
$$
A=D=\operatorname{diag}(0,1),
\qquad
L_t=\begin{bmatrix}1&0\\t&1\end{bmatrix}.
$$
对每个实数 $t$，都有
$$
L_tDL_t^T=A.
$$
因此同一个奇异半正定矩阵对应无穷多个这样的 $L$。正定情形不会发生这一现象，因为全部主元严格为正，标准递推中的除数不会为零。
<!-- bilingual-en:start -->
For $A=D=\operatorname{diag}(0,1)$, every $L_t=\begin{bmatrix}1&0\\t&1\end{bmatrix}$ satisfies $L_tDL_t^T=A$. Thus the same singular PSD matrix has infinitely many such factors. Positive definiteness excludes this example by making every pivot strictly positive.
<!-- bilingual-en:end -->

置换或分块主元可以继续处理更一般的矩阵，但那是另一种分解流程；此时必须按块判断惯性，不能逐个套用标量主元判据。
<!-- bilingual-en:start -->
Pivoted or block factorizations can continue on more general matrices, but they use a different procedure: inertia must be read blockwise rather than from scalar diagonal entries one at a time.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么正定矩阵不做行交换的 $LDL^T$ 分解不会碰到这个零主元障碍？
>
> **答案：** 正定等价于全部对称消元主元严格为正；因此递推中的除数都非零。
> <!-- bilingual-en:start -->
> Why does an unpivoted $LDL^T$ factorisation of a positive-definite matrix avoid the zero-pivot obstruction?
>
> **Answer:** Positive definiteness is equivalent to every symmetric-elimination pivot being strictly positive, so no divisor in the recurrence is zero.
> <!-- bilingual-en:end -->

## 来源与核验

- [[LDL正定判别]]：核对正定情形中正主元、存在性与唯一性。
- [UCLA, *Symmetric factorizations*](https://math.ucla.edu/~njhu/notes/nla/lin-direct/symfact.pdf)：核对零主元导致的递推失败、非唯一性及分块置换边界。
<!-- bilingual-en:start -->
- [[LDL正定判别|The positive-definite LDL criterion]] was checked for existence and uniqueness in the positive-definite case.
- UCLA notes on symmetric factorizations were checked for zero-pivot recurrence failure, nonuniqueness, and the pivoted block boundary.
<!-- bilingual-en:end -->
