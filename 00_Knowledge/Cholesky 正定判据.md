---
aliases:
  - "实对称矩阵正定当且仅当存在正对角 Cholesky 分解"
  - Cholesky factorization criterion
  - SPD Cholesky factorization
student_os: knowledge-atom
atom_id: LA-SPD-016
atom_set: symmetric-positive-definite
atom_type: equivalence
status: source-checked
mastery_state: unassessed
requires:
  - "[[LDL正定判别]]"
  - "[[正定矩阵]]"
related:
  - "[[半正定主平方根]]"
  - "[[LDL零主元边界]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

# 实对称矩阵正定当且仅当存在正对角 Cholesky 分解
<!-- bilingual-en:start -->
*A real symmetric matrix is positive definite exactly when it has a Cholesky factor with positive diagonal*
<!-- bilingual-en:end -->

> [!summary] Cholesky 判据
> 对实对称 $A\in\mathbb R^{n\times n}$，
> $$A\succ0\iff A=LL^T$$
> 对某个对角元全正的可逆下三角矩阵 $L$ 成立。该 $L$ 唯一。等价地，可写 $A=R^TR$，其中 $R=L^T$ 为正对角上三角矩阵。
> <!-- bilingual-en:start -->
> A real symmetric matrix is positive definite exactly when $A=LL^T$ for a lower-triangular $L$ with positive diagonal. This factor is unique. Equivalently, $A=R^TR$ with $R=L^T$ upper triangular and positive diagonal.
> <!-- bilingual-en:end -->

若 $A=\widetilde L D\widetilde L^T$ 且所有主元 $d_i>0$，令 $D^{1/2}=\operatorname{diag}(\sqrt{d_i})$，则 $L=\widetilde L D^{1/2}$ 给出 Cholesky。反向若 $A=LL^T$ 且 $L$ 可逆，则对 $x\ne0$，
$$
x^TAx=\|L^Tx\|_2^2>0.
$$
<!-- bilingual-en:start -->
Positive pivots in $A=\widetilde L D\widetilde L^T$ can be split as $D^{1/2}D^{1/2}$ to form a Cholesky factor. Conversely, an invertible triangular $L$ gives $x^TAx=\|L^Tx\|^2>0$ for every nonzero $x$.
<!-- bilingual-en:end -->

每个实对称半正定矩阵都能写成 $A=BB^T$；例如可取 $B=A^{1/2}$。但只有正定时，才保证存在对角元全正、可逆且唯一的三角 Cholesky 因子。精确算术中，在已确认对称后，标准无主元 Cholesky 出现非正平方根说明 $A$ 不正定；浮点计算还需考虑近奇异与容差，不能把微小负数无条件解释成数学反例。
<!-- bilingual-en:start -->
Every real symmetric positive-semidefinite matrix admits a factorization $A=BB^T$; one choice is $B=A^{1/2}$. What fails at the semidefinite boundary is the guarantee of a unique, invertible triangular factor with positive diagonal. In exact arithmetic, failure of an unpivoted Cholesky test on a symmetric matrix rules out positive definiteness; floating-point near-singularity requires a stated tolerance.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 $A=R^TR$ 还必须要求 $R$ 可逆，才能推出正定？
>
> **答案：** 若 $R$ 奇异，会存在非零 $x$ 使 $Rx=0$，于是 $x^TAx=0$，只能推出半正定。
>
> <!-- bilingual-en:start -->
> **Question:** Why must $R$ be invertible before $A=R^TR$ implies positive definiteness?
>
> **Answer:** If $R$ is singular, some nonzero $x$ satisfies $Rx=0$, giving $x^TAx=0$. The factorization then implies only positive semidefiniteness.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT 18.06 Spring 2010 Problem Set 9 solutions](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/58e62cc93e3e8b6cb52e72fa5830fc4e_MIT18_06S10_pset9_s10_soln.pdf)：核对由 $LDL^T$ 的正主元构造 Cholesky 因子。
- [Stanford EE263, *Cholesky Factorization*](https://ee263.stanford.edu/lectures/cholesky.pdf)：直接核对 SPD 与正对角 Cholesky 分解的存在性、反向性与唯一性。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.3.2 配方就是 $LDL^T$|课程 3.3.2]]：核对本地 $LDL^T$ 结构与主元前提。
<!-- bilingual-en:start -->
- MIT 18.06 Spring 2010 Problem Set 9 solutions were checked for constructing Cholesky from positive $LDL^T$ pivots.
- Stanford EE263 notes directly support existence, converse, and uniqueness of the positive-diagonal Cholesky factor for SPD matrices.
- Course Section 3.3.2 was checked for the local $LDL^T$ structure and pivot assumptions.
<!-- bilingual-en:end -->
