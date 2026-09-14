---
aliases:
  - "实对称矩阵正定当且仅当它有主元全正的无行交换 $LDL^T$ 分解"
  - "实对称矩阵正定当且仅当无行交换 $LDL^T$ 分解的主元全正"
  - "LDL 转置分解把正定性变成全部主元为正"
  - Positive pivots criterion
  - Completing the square via LDLT
student_os: knowledge-atom
atom_id: LA-SPD-013
atom_set: symmetric-positive-definite
atom_type: equivalence
status: source-checked
mastery_state: unassessed
requires:
  - "[[正定矩阵]]"
related:
  - "[[Sylvester 正定判据]]"
leads_to:
  - "[[Cholesky 正定判据]]"
contrasts_with:
  - "[[LDL零主元边界]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

# 实对称矩阵正定当且仅当它有主元全正的无行交换 $LDL^T$ 分解
<!-- bilingual-en:start -->
*An $LDL^T$ factorization turns positive definiteness into positivity of every pivot*
<!-- bilingual-en:end -->

> [!summary] $LDL^T$ 正定判据
> 对实对称 $A$，若保持对称且不做行交换的消元给出
> $$A=LDL^T,$$
> 其中 $L$ 为单位下三角矩阵、$D=\operatorname{diag}(d_1,\ldots,d_n)$，则
> $$A\succ0\iff d_i>0\quad\text{对所有 }i.$$
> 正定矩阵保证这种不做行交换的分解存在。
> <!-- bilingual-en:start -->
> If symmetry-preserving elimination without row exchanges gives $A=LDL^T$, with $L$ unit lower triangular and $D=\operatorname{diag}(d_i)$, then $A$ is positive definite exactly when every pivot $d_i$ is positive. Positive definiteness guarantees that this factorization exists without exchanges.
> <!-- bilingual-en:end -->

令 $y=L^Tx$。因为 $L$ 可逆，$x\ne0\iff y\ne0$，并且
$$
x^TAx=y^TDy=\sum_i d_i y_i^2.
$$
这就是高维配方：$L^Tx$ 产生新的线性组合，$d_i$ 是各平方项系数。
<!-- bilingual-en:start -->
With $y=L^Tx$, invertibility of $L$ preserves nonzero vectors and the quadratic form becomes $\sum_i d_i y_i^2$. This is completing the square in higher dimensions: the transformed coordinates are linear combinations and the pivots are their squared-term coefficients.
<!-- bilingual-en:end -->

若 $A$ 的全部顺序主子矩阵都非奇异，则这种单位下三角 $L$ 和对角 $D$ 的无行交换分解存在且唯一；正定保证这个前提。奇异半正定矩阵出现零主元时，递推可能中断，分解也可能不唯一，见 [[LDL零主元边界]]。
<!-- bilingual-en:start -->
If every leading principal submatrix is nonsingular, the unpivoted factorization with unit lower-triangular $L$ and diagonal $D$ exists and is unique; positive definiteness guarantees this premise. [[LDL零主元边界|The zero-pivot boundary for an $LDL^T$ factorization]] explains how a zero pivot can break the recurrence and uniqueness for a singular semidefinite matrix.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若 $D=\operatorname{diag}(2,-3)$，怎样立刻构造一个负方向？
>
> **答案：** 取 $y=e_2$，再令 $x=L^{-T}y$；此时 $x^TAx=-3$。
>
> <!-- bilingual-en:start -->
> **Question:** If $D=\operatorname{diag}(2,-3)$, how can you immediately construct a direction in which the quadratic form is negative?
>
> **Answer:** Take $y=e_2$ and then set $x=L^{-T}y$. This gives $x^TAx=-3$.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S03_Lecture_Lecture_27_Positive_Definite_Matrices_and_Minima.pdf|MIT Lecture 27 transcript]]：核对配方、主元与二次型的对应。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.3.2 配方就是 $LDL^T$|课程 3.3.2]]：核对 $LDL^T$ 推导及不做行交换时的边界。
- [UCLA, *Symmetric factorizations*](https://math.ucla.edu/~njhu/notes/nla/lin-direct/symfact.pdf)：核对非零顺序主子式下的存在唯一性、零主元失败与分块置换边界。
<!-- bilingual-en:start -->
- The MIT Lecture 27 transcript was checked for the link among completing the square, pivots, and quadratic forms.
- Course Section 3.3.2 was checked for the $LDL^T$ derivation and the no-row-exchange boundary.
- UCLA notes on symmetric factorizations were checked for existence and uniqueness under nonzero leading principal minors, zero-pivot failure, and pivoted block boundaries.
<!-- bilingual-en:end -->
