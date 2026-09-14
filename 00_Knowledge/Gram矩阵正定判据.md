---
aliases:
  - "$A^TA$ 正定当且仅当 $A$ 满列秩"
  - Gram matrix positive-definiteness criterion
  - Full column rank and positive-definite Gram matrix
student_os: knowledge-atom
atom_id: LA-PROJ-019
atom_set: orthogonal-projection-least-squares
atom_type: equivalence
status: source-checked
mastery_state: unassessed
requires:
  - "[[正定矩阵]]"
  - "[[矩阵秩]]"
leads_to:
  - "[[满列秩投影公式]]"
related:
  - "[[正定与半正定可逆性]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---

# $A^TA$ 正定当且仅当 $A$ 满列秩
<!-- bilingual-en:start -->
*$A^TA$ is positive definite if and only if $A$ has full column rank*
<!-- bilingual-en:end -->

> [!summary] 等价关系
> 对任意实矩阵 $A\in\mathbb R^{m\times n}$，$A^TA$ 总是半正定，而且
> $$
> A^TA\succ0
> \quad\Longleftrightarrow\quad
> N(A)=\{0\}
> \quad\Longleftrightarrow\quad
> \operatorname{rank}(A)=n.
> $$
> 因此 $A^TA$ 正定、$A^TA$ 可逆与 $A$ 满列秩是同一个条件的三种表达。
> <!-- bilingual-en:start -->
> For every real matrix $A\in\mathbb R^{m\times n}$, the Gram matrix $A^TA$ is positive semidefinite, and $A^TA\succ0$ exactly when $N(A)=\{0\}$, equivalently when $A$ has full column rank. Positive definiteness of $A^TA$, invertibility of $A^TA$, and full column rank of $A$ are therefore three forms of the same condition.
> <!-- bilingual-en:end -->

## 一行证明中的两个方向
<!-- bilingual-en:start -->
*Both directions in one identity*
<!-- bilingual-en:end -->

对任意 $x\in\mathbb R^n$，
$$
x^TA^TAx=\|Ax\|_2^2\ge0.
$$
所以 $A^TA$ 总是半正定。若 $A$ 满列秩，则 $x\ne0$ 推出 $Ax\ne0$，上式严格为正，因而 $A^TA\succ0$。反之，若 $A$ 不满列秩，存在 $0\ne x\in N(A)$，此时 $x^TA^TAx=0$，故 $A^TA$ 不可能正定。
<!-- bilingual-en:start -->
For every $x$, the identity $x^TA^TAx=\|Ax\|_2^2$ proves positive semidefiniteness. If $A$ has full column rank, then every nonzero $x$ gives $Ax\ne0$, so the quadratic form is strictly positive. Conversely, if $A$ is rank deficient, some nonzero $x$ satisfies $Ax=0$, and the quadratic form vanishes on that direction.
<!-- bilingual-en:end -->

## 最小反例与复数情形
<!-- bilingual-en:start -->
*Minimal counterexample and the complex case*
<!-- bilingual-en:end -->

若 $A=[a\ 2a]$ 且 $a\ne0$，则 $A$ 的两列相关。令 $z=(2,-1)^T$，则 $z\in N(A)$，因此
$$
z^TA^TAz=0.
$$
$A^TA$ 虽是方阵且半正定，仍然奇异。对复矩阵应把转置换成共轭转置：$A^*A\succ0$ 当且仅当 $A$ 满列秩。
<!-- bilingual-en:start -->
If $A=[a\ 2a]$ with $a\ne0$, then $z=(2,-1)^T$ lies in $N(A)$ and the Gram quadratic form vanishes in that nonzero direction. Thus $A^TA$ can be square and positive semidefinite yet singular. For a complex matrix, replace transpose by conjugate transpose: $A^*A\succ0$ exactly when $A$ has full column rank.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 为什么“$A^TA$ 是方阵”不能推出它可逆？
> <!-- bilingual-en:start -->
> Why does the fact that $A^TA$ is square not imply that it is invertible?
> <!-- bilingual-en:end -->
>
> **答案：** 可逆性要求 $N(A^TA)=N(A)=\{0\}$，也就是 $A$ 满列秩；方阵只说明尺寸匹配。
> <!-- bilingual-en:start -->
> **Answer:** Invertibility requires $N(A^TA)=N(A)=\{0\}$, equivalently full column rank of $A$; being square only fixes the dimensions.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf|MIT 18.06SC Session 2.1 summary]]：核验 $N(A^TA)=N(A)$、$\operatorname{rank}(A^TA)=\operatorname{rank}(A)$ 以及满列秩时 $A^TA$ 可逆。
- [[正定与半正定可逆性]]：核验正定矩阵可逆而半正定矩阵可以奇异的边界。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 2.1 summary supports $N(A^TA)=N(A)$, equality of the two ranks, and invertibility of $A^TA$ under full column rank.
- [[正定与半正定可逆性|The PD-versus-PSD invertibility result]] supports the boundary that positive-definite matrices are invertible whereas positive-semidefinite matrices may be singular.
<!-- bilingual-en:end -->
