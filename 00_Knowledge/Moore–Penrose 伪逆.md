---
aliases:
  - "Moore–Penrose 伪逆由四个 Penrose 方程定义"
  - The Moore-Penrose pseudoinverse is defined by the four Penrose equations
  - Moore-Penrose pseudoinverse
  - Moore–Penrose 四条件
  - Pseudoinverse
  - 伪逆
student_os: knowledge-atom
atom_id: LA-PINV-004
atom_set: pseudoinverse-one-sided-inverses
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[广义逆]]"
  - "[[Hermitian 矩阵]]"
part_of:
  - "[[广义逆与最小范数解.canvas]]"
leads_to:
  - "[[伪逆SVD公式]]"
  - "[[AA+列空间投影]]"
  - "[[A+A行空间投影]]"
related:
  - "[[伪逆存在唯一性]]"
---

# Moore–Penrose 伪逆由四个 Penrose 方程定义
<!-- bilingual-en:start -->
*The Moore–Penrose pseudoinverse is defined by the four Penrose equations*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 给定 $A\in\mathbb F^{m\times n}$，其中 $\mathbb F\in\{\mathbb R,\mathbb C\}$。若 $G\in\mathbb F^{n\times m}$ 满足
> $$AGA=A,\qquad GAG=G,$$
> $$(AG)^*=AG,\qquad(GA)^*=GA,$$
> 则称 $G$ 为 $A$ 的 Moore–Penrose 伪逆，记作 $A^+$。这里 $^*$ 表示共轭转置；对实矩阵，它就是普通转置。
> <!-- bilingual-en:start -->
> Given $A\in\mathbb F^{m\times n}$ with $\mathbb F\in\{\mathbb R,\mathbb C\}$, a matrix $G\in\mathbb F^{n\times m}$ is called the Moore–Penrose pseudoinverse of $A$ if it satisfies the four displayed equations. It is denoted by $A^+$. Here $^*$ denotes the conjugate transpose, which reduces to the ordinary transpose over the reals.
> <!-- bilingual-en:end -->

前两个方程使 $A$ 与 $G$ 相互反身，并保证 $AG$ 和 $GA$ 都是幂等矩阵；后两个方程进一步要求这两个幂等矩阵为 Hermitian。因此，$AA^+$ 是到列空间 $C(A)$ 的正交投影，$A^+A$ 是到行空间 $C(A^*)$ 的正交投影。只满足 $AGA=A$ 的 [[广义逆]] 是更宽泛的概念，并不自动具有这种正交几何。
<!-- bilingual-en:start -->
The first two equations make $A$ and $G$ reflexive with respect to one another and ensure that both $AG$ and $GA$ are idempotent. The last two equations further require those idempotents to be Hermitian. Consequently, $AA^+$ is the orthogonal projector onto the column space $C(A)$, while $A^+A$ is the orthogonal projector onto the row space $C(A^*)$. A [[广义逆|generalised inverse]] satisfying only $AGA=A$ is a broader notion and does not automatically have this orthogonal geometry.
<!-- bilingual-en:end -->

## 定义与定理的边界
<!-- bilingual-en:start -->
*Boundary between the definition and the theorem*
<!-- bilingual-en:end -->

上面的四个方程给出定义；“每个实或复矩阵恰有一个这样的矩阵”则是另一个需要证明的结论，见 [[伪逆存在唯一性]]。[[伪逆SVD公式]] 给出具体构造。方阵可逆时，这一定义退化为 $A^+=A^{-1}$；矩形或秩亏时，$AA^+$ 与 $A^+A$ 通常是投影而不是单位矩阵。
<!-- bilingual-en:start -->
The four equations above give the definition. The statement that every real or complex matrix has exactly one such matrix is a separate theorem; see [[伪逆存在唯一性|existence and uniqueness of the pseudoinverse]]. The [[伪逆SVD公式|SVD formula for the pseudoinverse]] supplies a concrete construction. When $A$ is square and invertible, the definition reduces to $A^+=A^{-1}$. For a rectangular or rank-deficient matrix, $AA^+$ and $A^+A$ are generally projectors rather than identity matrices.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么只验证 $AGA=A$ 还不足以确认 $G$ 是 Moore–Penrose 伪逆？
> <!-- bilingual-en:start -->
> Why is verifying only $AGA=A$ insufficient to establish that $G$ is the Moore–Penrose pseudoinverse?
> <!-- bilingual-en:end -->
>
> **答案：** 这只说明 $G$ 是一种广义逆。还必须验证反身条件 $GAG=G$，以及 $AG$、$GA$ 的两个 Hermitian 条件。
> <!-- bilingual-en:start -->
> **Answer:** That equation establishes only that $G$ is a generalised inverse. One must also verify the reflexive condition $GAG=G$ and the two Hermitian conditions on $AG$ and $GA$.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- R. Penrose, [“A generalized inverse for matrices”](https://doi.org/10.1017/S0305004100030401), *Proceedings of the Cambridge Philosophical Society* 51 (1955), 406–413：核对复矩阵情形的四个定义方程，以及与存在唯一性定理的区分。
  <!-- bilingual-en:start -->
  *English:* Checked the four defining equations over complex matrices and their distinction from the existence-and-uniqueness theorem.
  <!-- bilingual-en:end -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|MIT 18.06SC Session 3.8 summary]]：核对 SVD 构造以及 $AA^+$、$A^+A$ 的正交投影解释。
  <!-- bilingual-en:start -->
  *English:* Checked the SVD construction and the interpretation of $AA^+$ and $A^+A$ as orthogonal projectors.
  <!-- bilingual-en:end -->
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.8.4 Moore–Penrose 四条件|课程 3.8.4]]：核对课程使用的四条件、记号与实矩阵形式。
  <!-- bilingual-en:start -->
  *English:* Checked the course's four conditions, notation, and real-matrix formulation in [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.8.4 Moore–Penrose 四条件|Section 3.8.4]].
  <!-- bilingual-en:end -->
