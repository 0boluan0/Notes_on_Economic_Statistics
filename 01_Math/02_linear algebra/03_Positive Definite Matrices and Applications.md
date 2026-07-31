---
aliases:
  - MIT 18.06SC Unit III
  - 正定矩阵及其应用
tags:
  - 线性代数
  - mit-ocw
  - course-note
---

# Positive Definite Matrices and Applications

> [!info] 课程来源与范围
> 本笔记依照 MIT OCW 18.06SC *Linear Algebra*（Fall 2011）Unit III 的官方顺序整理：Session 3.1–3.8 → Session 3.9 Exam 3 Review → Unit 3 Exam。Lecture 33 的 left/right inverse 与 pseudoinverse 已按课程知识主线放入 Session 3.8；Lecture 32 是 Exam 3 review，而不是伪逆课。
> <!-- bilingual-en:start -->
> This notebook follows the official sequence of Unit III in MIT OCW 18.06SC *Linear Algebra* (Fall 2011): Sessions 3.1–3.8 → Session 3.9 Exam 3 Review → Unit 3 Exam. Lecture 33 on left and right inverses and the pseudoinverse is placed in Session 3.8 to match the course's conceptual progression; Lecture 32 is the Exam 3 review, not the pseudoinverse lecture.
> <!-- bilingual-en:end -->

## 0. 本单元要解决什么
<!-- bilingual-en:start -->
*0. What is addressed in this module*
<!-- bilingual-en:end -->

前两单元已经回答了“怎样解 $Ax=b$”和“怎样利用正交、行列式与特征值理解方阵”。Unit III 进一步追问：
<!-- bilingual-en:start -->
The first two units answered “How do we solve $Ax=b$?” and “How do orthogonality, determinants, and eigenvalues help us understand square matrices?” Unit III asks five further questions:
<!-- bilingual-en:end -->

1. 哪些矩阵拥有最稳定、最清晰的谱结构？答案是对称矩阵（symmetric matrix）与正定矩阵（positive definite matrix）。
2. 一个方阵不能对角化时，怎样准确描述缺失的特征向量？答案是相似变换与 Jordan 形。
3. 一个矩阵非方阵、秩亏、甚至没有特征分解时，怎样找到最合适的输入—输出坐标？答案是奇异值分解（singular value decomposition, SVD）。
4. 为什么同一个线性映射会有许多不同矩阵？答案是基、坐标与换基。
5. 当真正的逆不存在时，怎样保留“能逆的那一部分”？答案是左右逆与 Moore–Penrose 伪逆。
<!-- bilingual-en:start -->
1. Which matrices have the most stable and transparent spectral structure? Symmetric and positive-definite matrices.
2. When a square matrix is not diagonalizable, how do we describe the missing eigenvectors precisely? Similarity transformations and Jordan form.
3. For a rectangular or rank-deficient matrix, or one without an eigendecomposition, how do we find the most useful input and output coordinates? The singular value decomposition (SVD).
4. Why can the same linear map be represented by many different matrices? Bases, coordinates, and change of basis.
5. When an ordinary inverse does not exist, how do we retain the invertible part of the map? Left and right inverses and the Moore–Penrose pseudoinverse.
<!-- bilingual-en:end -->

本单元的统一图景是
<!-- bilingual-en:start -->
The unified picture for this module is
<!-- bilingual-en:end -->

$$
\text{对称/正定}\longrightarrow\text{正交谱分解},\qquad
\text{一般方阵}\longrightarrow\text{相似/Jordan},
$$

$$
\text{任意 }m\times n\text{ 矩阵}\longrightarrow\text{SVD}\longrightarrow\text{换基、压缩、最小二乘与伪逆}.
$$

## 1. 全局约定与阅读导航
<!-- bilingual-en:start -->
*1. Global Engagement and Reading Navigation*
<!-- bilingual-en:end -->

### 1.1 域、尺寸和共轭
<!-- bilingual-en:start -->
*1.1 Fields, dimensions, and conjugate transposes*
<!-- bilingual-en:end -->

- 未特别说明时，矩阵在实数域 $\mathbb R$ 上；涉及复特征值、Fourier 矩阵和 Hermitian 矩阵时，底层域改为 $\mathbb C$。
- $A\in\mathbb F^{m\times n}$ 表示 $A$ 有 $m$ 行、$n$ 列，并定义线性映射 $x\in\mathbb F^n\mapsto Ax\in\mathbb F^m$。
- 实矩阵的转置是 $A^T$；复矩阵的共轭转置是
  $$
  A^*=\overline{A}^{\,T}.
  $$
- 复向量内积约定为 $\langle x,y\rangle=x^*y$，因此 $\|x\|^2=x^*x>0$（只要 $x\ne0$）。不能把它误写成 $x^Tx$。
- “正定”在本课程中默认指实对称正定，或复 Hermitian 正定。若矩阵不对称，二次型只看它的对称部分：
  $$
  x^TAx=x^T\frac{A+A^T}{2}x.
  $$
<!-- bilingual-en:start -->
- Unless stated otherwise, matrices are over the real field $\mathbb R$; when complex eigenvalues, Fourier matrices, or Hermitian matrices appear, the underlying field is $\mathbb C$.
- $A\in\mathbb F^{m\times n}$ means that $A$ has $m$ rows, $n$ columns, and defines the linear mapping $x\in\mathbb F^n\mapsto Ax\in\mathbb F^m$.
- The transpose of a real matrix is $A^T$; the conjugate transpose of a complex matrix is
  $$
  A^*=\overline{A}^{\,T}.
  $$
- For complex vectors, the inner product is $\langle x,y\rangle=x^*y$, so $\|x\|^2=x^*x>0$ whenever $x\ne0$; it must not be replaced by $x^Tx$.
- In this course, “positive definite” means real symmetric positive definite or complex Hermitian positive definite. For a nonsymmetric real matrix, the quadratic form depends only on its symmetric part:
  $$
  x^TAx=x^T\frac{A+A^T}{2}x.
  $$
<!-- bilingual-en:end -->

### 1.2 Session 导航

- [[#Session 3.1 Symmetric matrices and positive definiteness|3.1 对称矩阵与正定性]]
- [[#Session 3.2 Complex matrices and fast Fourier transform|3.2 复矩阵与 FFT]]
- [[#Session 3.3 Positive definite matrices and minima|3.3 二次型、正定矩阵与极小值]]
- [[#Session 3.4 Similar matrices and Jordan form|3.4 相似矩阵与 Jordan 形]]
- [[#Session 3.5 Singular value decomposition|3.5 奇异值分解]]
- [[#Session 3.6 Linear transformations and their matrices|3.6 线性变换及其矩阵]]
- [[#Session 3.7 Change of basis and image compression|3.7 换基与图像压缩]]
- [[#Session 3.8 Left and right inverses and pseudoinverse|3.8 左右逆与伪逆]]
- [[#Session 3.9 Exam 3 review|3.9 Exam 3 Review]]
- [[#Unit 3 Exam 完整题解|Unit 3 Exam 完整题解]]

---

## Session 3.1 Symmetric matrices and positive definiteness

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节要回答：为什么实对称矩阵是特征值理论中最“友好”的矩阵？怎样从特征值、主元和顺序主子式判断正定？
<!-- bilingual-en:start -->
This section asks why real symmetric matrices are the most tractable matrices in eigenvalue theory, and how positive definiteness can be diagnosed from eigenvalues, pivots, and leading principal minors.
<!-- bilingual-en:end -->

前置知识：特征值与特征向量、对角化、正交矩阵、行列式和无行交换的消元。设 $A\in\mathbb R^{n\times n}$；证明实特征值时允许特征向量 $x\in\mathbb C^n$，因此必须使用共轭转置。
<!-- bilingual-en:start -->
Prerequisites: eigenvalues and eigenvectors, diagonalization, orthogonal matrices, determinants, and elimination without row exchanges. Let $A\in\mathbb R^{n\times n}$. When proving that its eigenvalues are real, an eigenvector may lie in $\mathbb C^n$, so the proof must use the conjugate transpose.
<!-- bilingual-en:end -->

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.1sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S01_Lecture_Lecture_25_Symmetric_Matrices_and_Positive_Definiteness.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S01_Recitation_Problem_Solving_Symmetric_Matrices_and_Positive_Definiteness.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.1prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.1sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:start -->
Data portal: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.1sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S01_Lecture_Lecture_25_Symmetric_Matrices_and_Positive_Definiteness.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S01_Recitation_Problem_Solving_Symmetric_Matrices_and_Positive_Definiteness.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.1prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.1sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:end -->

关联卡片：[[对称矩阵与正定二次型#对称矩阵与谱定理|对称矩阵]]、[[对称矩阵与正定二次型#二次型与正定性|正定矩阵]]、[[对称矩阵与正定二次型#对称矩阵与谱定理|谱分解]]、[[对称矩阵与正定二次型#二次型与正定性|正定性判别]]。
<!-- bilingual-en:start -->
Associated cards: [[对称矩阵与正定二次型#对称矩阵与谱定理|symmetric matrices]], [[对称矩阵与正定二次型#二次型与正定性|positive-definite matrices]], [[对称矩阵与正定二次型#对称矩阵与谱定理|spectral decomposition]], [[对称矩阵与正定二次型#二次型与正定性|tests for positive definiteness]].
<!-- bilingual-en:end -->

### 3.1.1 实对称矩阵的谱定理
<!-- bilingual-en:start -->
*3.1.1 Spectral Theorems of Real Symmetric Matrices*
<!-- bilingual-en:end -->

若 $A=A^T$，称 $A$ 为实[[对称矩阵与正定二次型#对称矩阵与谱定理|对称矩阵]]。课程采用的实谱定理（spectral theorem）是：
<!-- bilingual-en:start -->
If $A=A^T$, call $A$ [[对称矩阵与正定二次型#对称矩阵与谱定理|symmetric matrix]].  The real spectrum theorem adopted in the course is:
<!-- bilingual-en:end -->

> [!theorem] [[对称矩阵与正定二次型#对称矩阵与谱定理|实谱定理]]
> 对每个 $A\in\mathbb R^{n\times n}$，若 $A=A^T$，则：
> 1. $A$ 的所有特征值都是实数；
> 2. 不同特征值的特征向量彼此正交；
> 3. $\mathbb R^n$ 存在一组由 $A$ 的特征向量组成的标准正交基。
>
> 因而存在正交矩阵 $Q=[q_1\ \cdots\ q_n]$ 与实对角矩阵 $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$，使
> $$
> A=Q\Lambda Q^T,\qquad Q^TQ=QQ^T=I.
> $$
> <!-- bilingual-en:start -->
> For each $A\in\mathbb R^{n\times n}$, if $A=A^T$:
> 1. All the eigenvalues of $A$ are real numbers;
> 2. Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal;
> 3. $\mathbb R^n$ has an orthonormal basis consisting of eigenvectors of $A$.
> Thus there exist an orthogonal matrix $Q=[q_1\ \cdots\ q_n]$ and a real diagonal matrix $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$ such that
> <!-- bilingual-en:end -->

本课程完整证明前两项，并把“重特征值的每个特征子空间内可选标准正交基”与对称算子的归纳分解作为第三项的课程级论证。严格的完整存在性证明需要先保证特征多项式在 $\mathbb C$ 分裂；这是代数学基本定理提供的背景。
<!-- bilingual-en:start -->
This course proves the first two statements in full. For the third, it gives a course-level argument: within each eigenspace for a repeated eigenvalue, choose an orthonormal basis, and decompose the symmetric operator inductively. A fully rigorous existence proof also uses the fact that the characteristic polynomial splits over $\mathbb C$, supplied by the fundamental theorem of algebra.
<!-- bilingual-en:end -->

#### 证明一：特征值为何一定是实数
<!-- bilingual-en:start -->
*Proof 1: Why Eigenvalues Must Be Real Numbers*
<!-- bilingual-en:end -->

**目标。** 已知 $Ax=\lambda x$、$x\ne0$，证明 $\lambda=\overline\lambda$。
<!-- bilingual-en:start -->
**Goal.** Given $Ax=\lambda x$ with $x\ne0$, prove that $\lambda=\overline\lambda$.
<!-- bilingual-en:end -->

**构造。** 虽然 $A$ 是实矩阵，$x$ 和 $\lambda$ 事先可能是复数。计算同一标量 $x^*Ax$ 的两种表达。
<!-- bilingual-en:start -->
**Construction.** Although $A$ is real, the eigenvector $x$ and eigenvalue $\lambda$ may initially be complex. Compute the same scalar $x^*Ax$ in two ways.
<!-- bilingual-en:end -->

由 $Ax=\lambda x$，左乘 $x^*$：
<!-- bilingual-en:start -->
Starting from $Ax=\lambda x$, multiply on the left by $x^*$:
<!-- bilingual-en:end -->

$$
x^*Ax=x^*(\lambda x)=\lambda x^*x. \tag{1}
$$

另一方面，对 $Ax=\lambda x$ 取共轭转置：
<!-- bilingual-en:start -->
On the other hand, conjugate transpose $Ax=\lambda x$:
<!-- bilingual-en:end -->

$$
x^*A^*=\overline\lambda x^*.
$$

因为 $A$ 实对称，所以 $A^*=A^T=A$。右乘 $x$ 得
<!-- bilingual-en:start -->
$A^*=A^T=A$ because $A$ is symmetric.  Right by $x$
<!-- bilingual-en:end -->

$$
x^*Ax=\overline\lambda x^*x. \tag{2}
$$

比较 (1)、(2)：
<!-- bilingual-en:start -->
Compare (1), (2):
<!-- bilingual-en:end -->

$$
(\lambda-\overline\lambda)x^*x=0.
$$

而
<!-- bilingual-en:start -->
but
<!-- bilingual-en:end -->

$$
x^*x=\sum_{j=1}^n|x_j|^2>0
$$

是严格正实数，所以只能有 $\lambda=\overline\lambda$，即 $\lambda\in\mathbb R$。
<!-- bilingual-en:start -->
is strictly positive real, so you can only have $\lambda=\overline\lambda$, or $\lambda\in\mathbb R$.
<!-- bilingual-en:end -->

> [!warning] 本单元必须修正的旧稿错误
> 对复特征向量不能写“$x^Tx>0$”。例如 $x=(i,1)^T\ne0$，但 $x^Tx=i^2+1=0$。正确的正量是 $x^*x=|i|^2+|1|^2=2$。
> <!-- bilingual-en:start -->
> "$x^Tx>0$" cannot be written for complex eigenvectors.  For example, $x=(i,1)^T\ne0$, but $x^Tx=i^2+1=0$.  The correct positive is $x^*x=|i|^2+|1|^2=2$.
> <!-- bilingual-en:end -->

#### 证明二：不同特征值的特征向量正交
<!-- bilingual-en:start -->
*Proof 2: Orthogonality of eigenvectors corresponding to distinct eigenvalues*
<!-- bilingual-en:end -->

设 $Ax=\lambda x$、$Ay=\mu y$，其中 $\lambda\ne\mu$。因为 $A=A^*$，
<!-- bilingual-en:start -->
Set $Ax=\lambda x$, $Ay=\mu y$, where $\lambda\ne\mu$.  Because $A=A^*$,
<!-- bilingual-en:end -->

$$
\langle Ax,y\rangle=(Ax)^*y=x^*A^*y=x^*Ay=\langle x,Ay\rangle.
$$

代入特征方程：
<!-- bilingual-en:start -->
Substitute characteristic equation:
<!-- bilingual-en:end -->

$$
\overline\lambda\langle x,y\rangle=\mu\langle x,y\rangle.
$$

上一证明已知 $\lambda$ 为实数，所以
<!-- bilingual-en:start -->
The previous proof knew that $\lambda$ was real, so
<!-- bilingual-en:end -->

$$
(\lambda-\mu)\langle x,y\rangle=0.
$$

因 $\lambda\ne\mu$，得到 $\langle x,y\rangle=0$。
<!-- bilingual-en:start -->
Because of $\lambda\ne\mu$, we got $\langle x,y\rangle=0$.
<!-- bilingual-en:end -->

#### 证明三：为什么能得到一整组标准正交特征向量
<!-- bilingual-en:start -->
*Proof 3: Why can we get a whole set of orthonormal eigenvectors?*
<!-- bilingual-en:end -->

前两步已经说明：在 $\mathbb C$ 中出现的特征值实际都为实数。取一个实特征值 $\lambda_1$，因为 $A-\lambda_1I$ 是实奇异矩阵，它有非零实零空间；从中选单位特征向量 $q_1$。
<!-- bilingual-en:start -->
The first two steps have shown that the eigenvalues that appear in $\mathbb C$ are actually real numbers.  We choose a real eigenvalue $\lambda_1$, because $A-\lambda_1I$ is a real singular matrix and it has a non-zero real nullspace, and select the unit eigenvector $q_1$ from it.
<!-- bilingual-en:end -->

关键是证明正交补 $q_1^\perp$ 在 $A$ 下保持不变。若 $y\in q_1^\perp$，则
<!-- bilingual-en:start -->
The key point is to prove that the orthogonal complement $q_1^\perp$ is invariant under $A$.  If $y\in q_1^\perp$,
<!-- bilingual-en:end -->

$$
q_1^TAy=(A^Tq_1)^Ty=(Aq_1)^Ty=(\lambda_1q_1)^Ty=0.
$$

所以 $Ay\in q_1^\perp$。把 $A$ 限制到这个 $(n-1)$ 维子空间，它仍然是对称算子；在其中重复同一过程。对维数作归纳，得到 $q_1,\ldots,q_n$ 的标准正交特征基。把这些向量作列即得 $Q$，并由 $AQ=Q\Lambda$ 推出
<!-- bilingual-en:start -->
Thus $Ay\in q_1^\perp$. Restrict $A$ to this $(n-1)$-dimensional invariant subspace; the restriction is still symmetric. Repeating the argument and inducting on the dimension produces an orthonormal eigenbasis $q_1,\ldots,q_n$. Placing these vectors in the columns of $Q$ gives $AQ=Q\Lambda$, and hence
<!-- bilingual-en:end -->

$$
A=Q\Lambda Q^{-1}=Q\Lambda Q^T.
$$

这一步使用的背景事实只有：特征多项式在 $\mathbb C$ 上至少有一个根；这是代数学基本定理在本课程中的使用位置。
<!-- bilingual-en:start -->
The only background facts used in this step are that the characteristic polynomial has at least one root on $\mathbb C$; this is where the basic theorem of algebra is used in this course.
<!-- bilingual-en:end -->

#### 从谱分解到投影分解
<!-- bilingual-en:start -->
*From Spectral Decomposition to Projection Decomposition*
<!-- bilingual-en:end -->

把 $Q\Lambda Q^T$ 按列展开：
<!-- bilingual-en:start -->
Expand $Q\Lambda Q^T$ by column:
<!-- bilingual-en:end -->

$$
A=\sum_{j=1}^n\lambda_jq_jq_j^T.
$$

$q_jq_j^T$ 是投影到一维子空间 $\operatorname{span}(q_j)$ 的正交投影。对任意 $x$，
<!-- bilingual-en:start -->
$q_jq_j^T$ is an orthogonal projection onto a one-dimensional subspace $\operatorname{span}(q_j)$.  For any $x$,
<!-- bilingual-en:end -->

$$
Ax=\sum_{j=1}^n\lambda_jq_j(q_j^Tx).
$$

因此 $A$ 的作用可以逐方向理解：先把 $x$ 分解到互相垂直的特征方向，再把第 $j$ 个分量乘以 $\lambda_j$。
<!-- bilingual-en:start -->
Thus the action of $A$ can be understood one direction at a time: decompose $x$ into orthogonal eigendirections, then multiply its $j$th component by $\lambda_j$.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-spectral-positive-definite.png|820]]

### 3.1.2 正定的定义与等价判据
<!-- bilingual-en:start -->
*3.1.2 Positive definiteness and equivalent criteria*
<!-- bilingual-en:end -->

> [!definition] 正定矩阵
> 实矩阵 $A\in\mathbb R^{n\times n}$ 若满足 $A=A^T$，且对每个 $x\ne0$ 都有
> $$
> x^TAx>0,
> $$
> 则称 $A$ 为正定矩阵。若改为 $x^TAx\ge0$，则称[[对称矩阵与正定二次型#二次型与正定性|正半定（positive semidefinite）]]。复数情形把条件改成 $A=A^*$ 与 $x^*Ax>0$。
> <!-- bilingual-en:start -->
> A real matrix $A\in\mathbb R^{n\times n}$ is **positive definite** if $A=A^T$ and
> $$
> x^TAx>0
> $$
> for every nonzero $x$. Replacing $>$ by $\geq$ gives [[对称矩阵与正定二次型#二次型与正定性|positive semidefiniteness]]. Over $\mathbb C$, require $A=A^*$ and $x^*Ax>0$ for every nonzero $x$.
> <!-- bilingual-en:end -->

对实对称 $A$，下列条件等价：
<!-- bilingual-en:start -->
For real symmetric $A$, the following conditions are equivalent:
<!-- bilingual-en:end -->

1. $x^TAx>0$ 对所有 $x\ne0$ 成立；
2. 所有特征值 $\lambda_i>0$；
3. 在不需要换行的对称消元中，所有主元 $d_i>0$；
4. 所有顺序主子式
   $$
   \Delta_k=\det A_{1:k,1:k}>0,\qquad k=1,\ldots,n
   $$
   都为正（Sylvester criterion）。
<!-- bilingual-en:start -->
1. $x^TAx>0$ for every $x\ne0$;
2. All eigenvalues $\lambda_i>0$;
3. Every pivot $d_i$ in symmetry-preserving elimination without row exchanges is positive;
4. Every leading principal minor
   $$
   \Delta_k=\det A_{1:k,1:k}>0,\qquad k=1,\ldots,n,
   $$
   is positive (Sylvester's criterion).
<!-- bilingual-en:end -->

这里“主元为正”必须解释为保持对称结构的 $LDL^T$ 分解中的对角元，或无行交换消元得到的主元；不能对任意行交换后的消元结果直接套用。
<!-- bilingual-en:start -->
Here, “positive pivots” means the diagonal entries of a symmetry-preserving $LDL^T$ factorization, equivalently the pivots obtained by elimination without row exchanges. The criterion cannot be applied blindly after arbitrary row exchanges.
<!-- bilingual-en:end -->

#### 二次型判据与特征值判据等价
<!-- bilingual-en:start -->
*Quadratic criterion and eigenvalue criterion are equivalent*
<!-- bilingual-en:end -->

由谱分解 $A=Q\Lambda Q^T$，令 $y=Q^Tx$。因为 $Q$ 可逆，$x\ne0\iff y\ne0$。于是
<!-- bilingual-en:start -->
Using the spectral decomposition $A=Q\Lambda Q^T$, set $y=Q^Tx$. Because $Q$ is invertible, $x\ne0\iff y\ne0$. Hence
<!-- bilingual-en:end -->

$$
x^TAx=x^TQ\Lambda Q^Tx=y^T\Lambda y=\sum_{i=1}^n\lambda_i y_i^2.
$$

- 若全部 $\lambda_i>0$，则至少一个 $y_i\ne0$，所以和严格为正。
- 反之，若某个 $\lambda_k\le0$，取 $x=q_k$，便有 $x^TAx=\lambda_k\le0$，违反正定。
<!-- bilingual-en:start -->
- If every $\lambda_i>0$, then at least one $y_i\ne0$, so the sum is strictly positive.
- Conversely, if some $\lambda_k\le0$, choose $x=q_k$. Then $x^TAx=\lambda_k\le0$, contradicting positive definiteness.
<!-- bilingual-en:end -->

因此二者等价。
<!-- bilingual-en:start -->
So the two are equivalent.
<!-- bilingual-en:end -->

#### 主元判据为何成立
<!-- bilingual-en:start -->
*Why the pivot criterion works*
<!-- bilingual-en:end -->

若对称消元无换行得到
<!-- bilingual-en:start -->
If symmetry-preserving elimination requires no row exchanges, then
<!-- bilingual-en:end -->

$$
A=LDL^T,
$$

其中 $L$ 为单位下三角矩阵，$D=\operatorname{diag}(d_1,\ldots,d_n)$ 为主元对角阵。令 $y=L^Tx$，则 $L^T$ 可逆，所以 $x\ne0\iff y\ne0$，并且
<!-- bilingual-en:start -->
where $L$ is unit lower triangular and $D=\operatorname{diag}(d_1,\ldots,d_n)$ is the diagonal matrix of pivots. Let $y=L^Tx$. Since $L^T$ is invertible, $x\ne0\iff y\ne0$, and
<!-- bilingual-en:end -->

$$
x^TAx=x^TLDL^Tx=y^TDy=\sum_{i=1}^n d_i y_i^2.
$$

故 $A$ 正定当且仅当所有 $d_i>0$。此外，前 $k$ 个顺序主子式满足
<!-- bilingual-en:start -->
Thus $A$ is positive definite if and only if every $d_i>0$. Moreover, the leading principal minors satisfy
<!-- bilingual-en:end -->

$$
\Delta_k=d_1d_2\cdots d_k,
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
d_k=\frac{\Delta_k}{\Delta_{k-1}},\qquad \Delta_0=1.
$$

这又说明“所有主元正”与“所有顺序主子式正”等价。
<!-- bilingual-en:start -->
Therefore, “all pivots are positive” is equivalent to “all leading principal minors are positive.”
<!-- bilingual-en:end -->

> [!example] 课件例题：三条路线判断正定
> 设
> $$
> A=\begin{bmatrix}5&2\\2&3\end{bmatrix}.
> $$
> 由两个非对角元都等于 $2$，可见 $A=A^T$。
>
> **主元法：** 第一主元 $d_1=5$；消去第二行第一项时，
> $$
> d_2=3-\frac25\cdot2=\frac{11}{5}>0.
> $$
>
> **顺序主子式法：** $\Delta_1=5>0$，$\Delta_2=15-4=11>0$。
>
> **特征值法：**
> $$
> \det(A-\lambda I)=(5-\lambda)(3-\lambda)-4
> =\lambda^2-8\lambda+11,
> $$
> 所以 $\lambda=4\pm\sqrt5>0$。
>
> 三种方法都证明 $A$ 正定。
> <!-- bilingual-en:start -->
> Since the two off-diagonal entries are both $2$, the matrix is symmetric.
> **Pivot method:** The first pivot is $d_1=5$. After eliminating the first entry in the second row,
> **Leading-principal-minor method:** $\Delta_1=5>0$ and $\Delta_2=15-4=11>0$.
> **Eigenvalue method:**
> Thus $\lambda=4\pm\sqrt5>0$.
> All three methods show that $A$ is positive definite.
> <!-- bilingual-en:end -->

### 3.1.3 Recitation：四个性质
<!-- bilingual-en:start -->
*3.1.3 Recitation: Four Properties*
<!-- bilingual-en:end -->

1. **正定矩阵必可逆。** 所有特征值正，因此 $\det A=\prod_i\lambda_i>0$。
2. **唯一正定投影矩阵是 $I$。** 投影矩阵满足 $P^2=P$，故特征值只可能是 $0,1$；正定排除 $0$，谱分解于是给出 $P=QIQ^T=I$。
3. **正对角矩阵正定。** 若 $D=\operatorname{diag}(d_i)$ 且 $d_i>0$，则 $x^TDx=\sum d_ix_i^2>0$。
4. **$\det S>0$ 不足以推出正定。** 例如
   $$
   S=\begin{bmatrix}-3&1\\1&-2\end{bmatrix},\qquad \det S=5>0,
   $$
   但取 $x=e_1$，有 $x^TSx=-3<0$。
<!-- bilingual-en:start -->
1.**The positive definite matrix must be invertible.**All eigenvalues are positive, so $\det A=\prod_i\lambda_i>0$.
2.**The only positive and definite projection matrix is $I$.**The projection matrix satisfies $P^2=P$, so the eigenvalue can only be $0,1$; positive definite exclusion $0$, spectral decomposition is given $P=QIQ^T=I$.
3.**The positive diagonal matrix is positive definite.**$x^TDx=\sum d_ix_i^2>0$ if $D=\operatorname{diag}(d_i)$ and $d_i>0$.
4.**$\det S>0$ is not enough to launch Positive Determination.**For example
   But take $x=e_1$, there is $x^TSx=-3<0$.
<!-- bilingual-en:end -->

### 3.1.4 Homework：完整题解
<!-- bilingual-en:start -->
*3.1.4 Homework: Complete solutions*
<!-- bilingual-en:end -->

> [!question]- Problem 25.1：找出“所有实矩阵的特征值都为实数”这一伪证明的漏洞
> 题目声称：由 $Ax=\lambda x$ 得 $x^TAx=\lambda x^Tx$，所以 $\lambda=(x^TAx)/(x^Tx)$ 为实数。用
> $$
> A=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\quad \lambda=i,\quad x=\begin{bmatrix}i\\1\end{bmatrix}
> $$
> 检查。
> <!-- bilingual-en:start -->
> The purported proof argues that $Ax=\lambda x$ implies $x^TAx=\lambda x^Tx$, and therefore that $\lambda=(x^TAx)/(x^Tx)$ is real. Test the argument using the displayed matrix $A$, eigenvalue $\lambda=i$, and eigenvector $x$.
> <!-- bilingual-en:end -->

> [!success]- 解答
> **已知。** $A$ 是逆时针 $90^\circ$ 旋转矩阵。
>
> **验证特征方程：**
> $$
> Ax=\begin{bmatrix}-1\\i\end{bmatrix}
> =i\begin{bmatrix}i\\1\end{bmatrix}=ix.
> $$
>
> **检查伪证明前一步：**
> $$
> x^TAx=\begin{bmatrix}i&1\end{bmatrix}\begin{bmatrix}-1\\i\end{bmatrix}=0,
> $$
> 且
> $$
> \lambda x^Tx=i(i^2+1)=0.
> $$
> 所以前一步没有错；真正的问题是
> $$
> x^Tx=i^2+1=0,
> $$
> 最后一步除以 $x^Tx$ 是除以零。复向量应使用 $x^*x=2$。此外，即便改用 $x^*x$，一般实矩阵的 $x^*Ax$ 也未必为实；只有 $A=A^*$ 时才能保证。
> <!-- bilingual-en:start -->
> **Given.** $A$ is the matrix for a counterclockwise rotation through $90^\circ$.
>
> **Verify the eigenvalue equation:**
> $$
> Ax=\begin{bmatrix}-1\\i\end{bmatrix}
> =i\begin{bmatrix}i\\1\end{bmatrix}=ix.
> $$
>
> **Check the preceding step in the false proof:**
> $$
> x^TAx=0,\qquad \lambda x^Tx=i(i^2+1)=0.
> $$
> That step is valid. The failure occurs in the final division, because
> $$
> x^Tx=i^2+1=0.
> $$
> For complex vectors the positive norm is $x^*x=2$. Even after that correction, $x^*Ax$ need not be real for a general real matrix; it is guaranteed to be real when $A=A^*$.
> <!-- bilingual-en:end -->

> [!question]- Problem 25.2：哪些矩阵集合构成群？
> 以矩阵乘法为运算，判断：(a) 对称正定矩阵；(b) 正交矩阵；(c) 固定矩阵 $A$ 的所有 $e^{tA}$；(d) 行列式为 $1$ 的矩阵。
> <!-- bilingual-en:start -->
> Determine which of the following sets are closed under matrix multiplication: (a) symmetric positive-definite matrices; (b) orthogonal matrices; (c) the family $\{e^{tA}:t\in\mathbb R\}$ for a fixed matrix $A$; and (d) matrices with determinant $1$.
> <!-- bilingual-en:end -->

> [!success]- 解答
> **(a) 不构成群。** 逆仍然正定，但乘积未必对称。取
> $$
> A=\begin{bmatrix}2&1\\1&1\end{bmatrix},\qquad
> B=\begin{bmatrix}1&1/2\\1/2&1\end{bmatrix}.
> $$
> 两者顺序主子式均正，故正定；然而
> $$
> AB=\begin{bmatrix}5/2&2\\3/2&3/2\end{bmatrix}
> $$
> 不对称，因此不在集合中。
>
> **(b) 构成群。** 若 $Q^TQ=I$，则 $Q^{-1}=Q^T$ 仍正交；若 $A,B$ 正交，
> $$
> (AB)^T(AB)=B^TA^TAB=B^TB=I.
> $$
>
> **(c) 构成群。** 因为都是同一个 $A$ 的函数，$pA$ 与 $qA$ 可交换，
> $$
> e^{pA}e^{qA}=e^{(p+q)A},\qquad (e^{pA})^{-1}=e^{-pA}.
> $$
> 单位元为 $e^{0A}=I$。
>
> **(d) 构成群。** 若 $\det A=\det B=1$，则
> $$
> \det(AB)=1,
> $$
> 且 $\det(A^{-1})=1/\det A=1$。单位矩阵的行列式也是 $1$。
> <!-- bilingual-en:start -->
> **(a) Not a group.** Inverses remain positive definite, but products need not be symmetric. The two displayed matrices are positive definite, yet their product is not symmetric and therefore leaves the set.
> **(b) A group.** If $Q^TQ=I$, then $Q^{-1}=Q^T$ is orthogonal. If $A$ and $B$ are orthogonal, then $(AB)^T(AB)=I$.
> **(c) A group.** All members are functions of the same matrix $A$, so $pA$ and $qA$ commute and $e^{pA}e^{qA}=e^{(p+q)A}$. The identity is $e^{0A}=I$, and the inverse of $e^{pA}$ is $e^{-pA}$.
> **(d) A group.** If $\det A=\det B=1$, then $\det(AB)=1$, while $\det(A^{-1})=1/\det A=1$. The identity matrix also has determinant $1$.
> <!-- bilingual-en:end -->

### 3.1.5 边界、反例与易错点
<!-- bilingual-en:start -->
*3.1.5 Boundaries, Counterexamples and common errors*
<!-- bilingual-en:end -->

- 正对角元只是正定的必要条件，不是充分条件；$\begin{bmatrix}1&2\\2&1\end{bmatrix}$ 对角元正，却有特征值 $3,-1$。
- 在本节的实对称前提下，$\det A>0$ **迫使**负特征值的个数为偶数，但不能排除存在两个、四个等负特征值；因此它远不足以推出正定。
- “主元符号与特征值符号数相同”是对实对称矩阵的惯性定律语境；不能任意推广到非对称矩阵。
- 半正定情形不能把“所有顺序主子式非负”当作充分条件；完整半正定判据要求所有主子式非负，而不只顺序主子式。
<!-- bilingual-en:start -->
- The positive diagonal element is only a necessary condition for positive definite, not a sufficient condition; the $\begin{bmatrix}1&2\\2&1\end{bmatrix}$ diagonal element is positive, but has the eigenvalue $3,-1$.
- Under the real symmetry of this section, $\det A>0$**forces the number of negative eigenvalues of**to be even, but it cannot be excluded that there are two or four such negative eigenvalues; therefore, it is far from sufficient to derive positive definiteness.
- For a real symmetric matrix, the law of inertia says that the numbers of positive, negative, and zero pivots match the corresponding counts of eigenvalues. This statement does not extend unchanged to nonsymmetric matrices.
- For positive semidefiniteness, nonnegative leading principal minors are not sufficient. The complete criterion requires every principal minor—not only the leading ones—to be nonnegative.
<!-- bilingual-en:end -->

### 3.1.6 三道自检题
<!-- bilingual-en:start -->
*3.1.6 Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 自检 1（计算）
> 判断 $A=\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$ 是否正定，并给出两种依据。
> <!-- bilingual-en:start -->
> Determine whether $A=\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$ is positive definite, and give two independent justifications.
> <!-- bilingual-en:end -->

> [!success]- 答案
> $\Delta_1=2>0$、$\Delta_2=3>0$，故正定；或特征值为 $1,3$，均正。
> <!-- bilingual-en:start -->
> Since $\Delta_1=2>0$ and $\Delta_2=3>0$, Sylvester's criterion gives positive definiteness. Alternatively, its eigenvalues are $1$ and $3$, both positive.
> <!-- bilingual-en:end -->

> [!question]- 自检 2（证明）
> 若 $A$ 对称正定，证明 $A^{-1}$ 也对称正定。
> <!-- bilingual-en:start -->
> If $A$ is symmetric positive definite, $A^{-1}$ is also symmetric positive definite.
> <!-- bilingual-en:end -->

> [!success]- 答案
> $A=Q\Lambda Q^T$，故 $A^{-1}=Q\Lambda^{-1}Q^T$；$\Lambda^{-1}$ 的对角元 $1/\lambda_i$ 全正。因此 $A^{-1}$ 对称正定。
> <!-- bilingual-en:start -->
> $A=Q\Lambda Q^T$, hence $A^{-1}=Q\Lambda^{-1}Q^T$; the diagonal element $1/\lambda_i$ of $\Lambda^{-1}$ is perfect.  Therefore, the $A^{-1}$ symmetry is positive definite.
> <!-- bilingual-en:end -->

> [!question]- 自检 3（条件诊断）
> “$A$ 的主元全正，所以 $A$ 正定”缺少什么前提？
> <!-- bilingual-en:start -->
> What assumption is missing from the claim “all pivots of $A$ are positive, so $A$ is positive definite”?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 至少要有 $A$ 实对称，并且主元来自不换行、保持对称结构的消元或 $LDL^T$ 分解。
> <!-- bilingual-en:start -->
> At minimum, $A$ must be real symmetric, and the pivots must come from symmetry-preserving elimination without row exchanges, equivalently from an $LDL^T$ factorization.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$$
A=A^T\Longrightarrow\text{实谱与正交特征基}
\Longrightarrow A=Q\Lambda Q^T
\Longrightarrow x^TAx=\sum\lambda_i y_i^2.
$$

正定性把[[特征值、对角化与线性动力系统#特征值与特征向量|特征值]]、主元、顺序主子式、二次型与可逆性连接在一起；下一节把相同思想推广到复数域。
<!-- bilingual-en:start -->
Positive definiteness connects [[特征值、对角化与线性动力系统#特征值与特征向量|eigenvalues]], pivots, leading principal minors, quadratic forms, and invertibility. The next section extends the same ideas to complex vector spaces.
<!-- bilingual-en:end -->

---

## Session 3.2 Complex matrices and fast Fourier transform

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节要回答：进入 $\mathbb C^n$ 后，长度、正交、对称和正交矩阵应怎样改写？离散 Fourier 矩阵为何能被递归分解，从 $O(n^2)$ 降为 $O(n\log n)$？
<!-- bilingual-en:start -->
In this section, you will be asked how the length, orthogonality, symmetry, and orthogonality matrices should be rewritten after entering $\mathbb C^n$.  Why can the discrete Fourier matrix be decomposed recursively from $O(n^2)$ to $O(n\log n)$?
<!-- bilingual-en:end -->

前置知识：复数及其共轭、内积、正交矩阵、矩阵分块。设 $z\in\mathbb C^n$；Fourier 矩阵 $F_n\in\mathbb C^{n\times n}$，下标从 $0$ 到 $n-1$。
<!-- bilingual-en:start -->
Prerequisites: complex numbers and conjugation, inner products, orthogonal matrices, and block matrices. Let $z\in\mathbb C^n$. The Fourier matrix $F_n\in\mathbb C^{n\times n}$ is indexed from $0$ to $n-1$.
<!-- bilingual-en:end -->

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S02_Lecture_Lecture_26_Complex_Matrices_Fast_Fourier_Transform_FFT.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S02_Recitation_Problem_Solving_Complex_Matrices.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.2prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.2sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:start -->
Data portal: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S02_Lecture_Lecture_26_Complex_Matrices_Fast_Fourier_Transform_FFT.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S02_Recitation_Problem_Solving_Complex_Matrices.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.2prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.2sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:end -->

关联卡片：[[正交投影与最小二乘#正交补与最近点|正交性]]、[[对称矩阵与正定二次型#对称矩阵与谱定理|正交矩阵]]、[[特征值、对角化与线性动力系统#对角化与矩阵幂|Fourier 展开]]。
<!-- bilingual-en:start -->
Associated Cards: [[正交投影与最小二乘#正交补与最近点|orthogonal]], [[对称矩阵与正定二次型#对称矩阵与谱定理|orthogonal matrix]], [[特征值、对角化与线性动力系统#对角化与矩阵幂|Fourier Expansion]].
<!-- bilingual-en:end -->

### 3.2.1 复向量的长度与 Hermitian 内积
<!-- bilingual-en:start -->
*3.2.1 Length of Complex Vector and Hermitian Internal Product*
<!-- bilingual-en:end -->

旧公式 $z^Tz$ 不再代表长度平方。例如 $z=(1,i)^T$ 时 $z^Tz=1+i^2=0$。正确公式是
<!-- bilingual-en:start -->
The old formula, $z^Tz$, no longer represents the length squared.  For example, $z^Tz=1+i^2=0$ if $z=(1,i)^T$.  The correct formula is
<!-- bilingual-en:end -->

$$
\langle z,w\rangle=z^*w=\sum_{k=1}^n\overline z_k w_k,
\qquad
\|z\|^2=z^*z=\sum_{k=1}^n|z_k|^2.
$$

它满足：
<!-- bilingual-en:start -->
It satisfies:
<!-- bilingual-en:end -->

1. $\langle z,z\rangle\ge0$，且等号仅在 $z=0$ 时成立；
2. $\langle z,w\rangle=\overline{\langle w,z\rangle}$；
3. 对第二个变量线性，对第一个变量共轭线性。
<!-- bilingual-en:start -->
1. $\langle z,z\rangle\ge0$, and the equal sign only holds if $z=0$;
2. $\langle z,w\rangle=\overline{\langle w,z\rangle}$;
3. Linear for the second variable, conjugate linear for the first variable.
<!-- bilingual-en:end -->

> [!definition] Hermitian 与 unitary
> - $A\in\mathbb C^{n\times n}$ 若 $A=A^*$，称为[[对称矩阵与正定二次型#对称矩阵与谱定理|Hermitian 矩阵]]；它是实对称矩阵的复数推广。
> - $Q\in\mathbb C^{n\times n}$ 若 $Q^*Q=QQ^*=I$，称为[[对称矩阵与正定二次型#对称矩阵与谱定理|unitary 矩阵]]；它是实正交矩阵的复数推广。
> <!-- bilingual-en:start -->
> - $A\in\mathbb C^{n\times n}$ If $A=A^*$, it is called [[对称矩阵与正定二次型#对称矩阵与谱定理|Hermitian matrix]]; it is a complex generalization of a real symmetric matrix.
> - $Q\in\mathbb C^{n\times n}$ If $Q^*Q=QQ^*=I$, it is called [[对称矩阵与正定二次型#对称矩阵与谱定理|unitary matrix]]; it is a complex generalization of a real orthogonal matrix.
> <!-- bilingual-en:end -->

Hermitian 谱定理把 3.1 的所有 $T$ 换为 $*$：特征值为实数，可以选 unitary 特征向量矩阵 $Q$，并写成
<!-- bilingual-en:start -->
The Hermitian spectral theorem replaces all $T$ of 3.1 with $*$: the eigenvalues are real numbers and the unitary eigenvector matrix $Q$ can be chosen and written as
<!-- bilingual-en:end -->

$$
A=Q\Lambda Q^*.
$$

### 3.2.2 Recitation：完整对角化一个 Hermitian 矩阵
<!-- bilingual-en:start -->
*3.2.2 Recitation: Full Diagonalization of a Hermitian Matrix*
<!-- bilingual-en:end -->

取
<!-- bilingual-en:start -->
take
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}2&1-i\\1+i&3\end{bmatrix}=A^*.
$$

特征多项式为
<!-- bilingual-en:start -->
The characteristic polynomial is
<!-- bilingual-en:end -->

$$
\det(A-\lambda I)=(2-\lambda)(3-\lambda)-(1-i)(1+i)
=\lambda^2-5\lambda+4,
$$

故 $\lambda_1=1$、$\lambda_2=4$，确实都是实数。
<!-- bilingual-en:start -->
So $\lambda_1=1$, $\lambda_2=4$, are real numbers.
<!-- bilingual-en:end -->

当 $\lambda_1=1$ 时，可取
<!-- bilingual-en:start -->
When $\lambda_1=1$,
<!-- bilingual-en:end -->

$$
v_1=\begin{bmatrix}1-i\\-1\end{bmatrix};
$$

当 $\lambda_2=4$ 时，可取
<!-- bilingual-en:start -->
When $\lambda_2=4$,
<!-- bilingual-en:end -->

$$
v_2=\begin{bmatrix}1\\1+i\end{bmatrix}.
$$

检查 Hermitian 正交性：
<!-- bilingual-en:start -->
To check Hermitian orthogonality:
<!-- bilingual-en:end -->

$$
v_1^*v_2=\begin{bmatrix}1+i&-1\end{bmatrix}
\begin{bmatrix}1\\1+i\end{bmatrix}=0.
$$

两向量的长度平方都为 $3$，因此
<!-- bilingual-en:start -->
The square of the length of both vectors is $3$, so
<!-- bilingual-en:end -->

$$
Q=\frac1{\sqrt3}\begin{bmatrix}1-i&1\\-1&1+i\end{bmatrix},\qquad
Q^*Q=I,
$$

并有
<!-- bilingual-en:start -->
there is
<!-- bilingual-en:end -->

$$
A=Q\begin{bmatrix}1&0\\0&4\end{bmatrix}Q^*.
$$

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-complex-action.png|780]]

### 3.2.3 离散 Fourier 矩阵
<!-- bilingual-en:start -->
*3.2.3 Discrete Fourier Matrix*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
\omega_n=e^{2\pi i/n},\qquad \omega_n^n=1.
$$

课程使用未归一化 Fourier 矩阵
<!-- bilingual-en:start -->
The course uses the unnormalized Fourier matrix
<!-- bilingual-en:end -->

$$
(F_n)_{jk}=\omega_n^{jk},\qquad j,k=0,1,\ldots,n-1.
$$

第 $k$ 列是离散频率 $k$ 在 $n$ 个采样点上的取值。
<!-- bilingual-en:start -->
Column $k$ records the values of discrete frequency $k$ at the $n$ sample points.
<!-- bilingual-en:end -->

#### 证明：Fourier 列彼此正交
<!-- bilingual-en:start -->
*Proof: The Fourier columns are mutually orthogonal*
<!-- bilingual-en:end -->

第 $k$ 与第 $\ell$ 列的内积为
<!-- bilingual-en:start -->
The inner product of columns $k$ and $\ell$ is
<!-- bilingual-en:end -->

$$
\sum_{j=0}^{n-1}\overline{\omega_n^{jk}}\omega_n^{j\ell}
=\sum_{j=0}^{n-1}\omega_n^{j(\ell-k)}.
$$

- 若 $k=\ell$，每项为 $1$，和为 $n$。
- 若 $k\ne\ell$，令 $r=\omega_n^{\ell-k}\ne1$，但 $r^n=1$。由有限几何级数
  $$
  \sum_{j=0}^{n-1}r^j=\frac{1-r^n}{1-r}=0.
  $$
<!-- bilingual-en:start -->
- If $k=\ell$, every term equals $1$, so the sum is $n$.
- If $k\ne\ell$, let $r=\omega_n^{\ell-k}\ne1$. Since $r^n=1$, the finite geometric-series formula gives zero.
<!-- bilingual-en:end -->

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
F_n^*F_n=nI,\qquad \frac1{\sqrt n}F_n\text{ 是 unitary 矩阵},\qquad
F_n^{-1}=\frac1nF_n^*.
$$

对 $n=4$，$\omega_4=i$：
<!-- bilingual-en:start -->
For $n=4$, $\omega_4=i$:
<!-- bilingual-en:end -->

$$
F_4=
\begin{bmatrix}
1&1&1&1\\
1&i&-1&-i\\
1&-1&1&-1\\
1&-i&-1&i
\end{bmatrix}.
$$

若 $e_0=(1,0,0,0)^T$，则 $F_4e_0=(1,1,1,1)^T$：时间域的单个脉冲包含所有频率，且幅度相同。
<!-- bilingual-en:start -->
If $e_0=(1,0,0,0)^T$, then $F_4e_0=(1,1,1,1)^T$: A single pulse in the time domain contains all frequencies and has the same amplitude.
<!-- bilingual-en:end -->

### 3.2.4 [[特征值、对角化与线性动力系统#对角化与矩阵幂|FFT 的偶奇递归]]
<!-- bilingual-en:start -->
*3.2.4 [[特征值、对角化与线性动力系统#对角化与矩阵幂|Even-odd recursion of FFT]]*
<!-- bilingual-en:end -->

普通矩阵—向量乘法 $F_nx$ 需要约 $n^2$ 次标量运算。对 $n=2m$，先用置换矩阵 $P$ 把输入排成偶数下标、奇数下标：
<!-- bilingual-en:start -->
Ordinary matrix—vector multiplication $F_nx$ requires about $n^2$ scalar operations.  For $n=2m$, the input is first arranged into even subscripts and odd subscripts by permutation matrix $P$:
<!-- bilingual-en:end -->

$$
Px=(x_0,x_2,\ldots,x_{2m-2},x_1,x_3,\ldots,x_{2m-1})^T.
$$

令 $D=\operatorname{diag}(1,\omega_{2m},\ldots,\omega_{2m}^{m-1})$，则
<!-- bilingual-en:start -->
Let $D=\operatorname{diag}(1,\omega_{2m},\ldots,\omega_{2m}^{m-1})$, then
<!-- bilingual-en:end -->

$$
F_{2m}=
\begin{bmatrix}I&D\\I&-D\end{bmatrix}
\begin{bmatrix}F_m&0\\0&F_m\end{bmatrix}P.
$$

原因是 $\omega_{2m}^{2jk}=\omega_m^{jk}$：偶数样本与奇数样本各自形成一个长度 $m$ 的 DFT，$D$ 提供奇数部分的相位修正（twiddle factors），最后用加减法合并。
<!-- bilingual-en:start -->
The reason is that $\omega_{2m}^{2jk}=\omega_m^{jk}$: the even sample and the odd sample form a DFT of length $m$ respectively, $D$ provides the phase correction of the odd part, and finally the addition and subtraction are combined.
<!-- bilingual-en:end -->

若 $T(n)$ 是运算量，则
<!-- bilingual-en:start -->
If $T(n)$ is operand, then
<!-- bilingual-en:end -->

$$
T(n)=2T(n/2)+O(n),
$$

递归 $\log_2n$ 层，得到 $T(n)=O(n\log n)$。这不是近似算法；它与直接乘 $F_n$ 得到完全相同的结果，只是利用结构避免重复计算。
<!-- bilingual-en:start -->
Recursive $\log_2n$ layer to get $T(n)=O(n\log n)$.  This is not an approximation algorithm; it gets exactly the same result as the direct multiplication of $F_n$, only using the structure to avoid double calculation.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-fft-butterfly.png|820]]

### 3.2.5 Homework：完整题解
<!-- bilingual-en:start -->
*3.2.5 Homework: Complete solutions*
<!-- bilingual-en:end -->

> [!question]- Problem 26.1：计算 $F_2$

> [!success]- 解答
> $\omega_2=e^{2\pi i/2}=-1$，所以
> $$
> F_2=\begin{bmatrix}1&1\\1&\omega_2\end{bmatrix}
> =\begin{bmatrix}1&1\\1&-1\end{bmatrix}.
> $$
> 检查：$F_2^*F_2=2I$，所以 $(1/\sqrt2)F_2$ 是 unitary。
> <!-- bilingual-en:start -->
> $\omega_2=e^{2\pi i/2}=-1$, so
> Check: $F_2^*F_2=2I$, so $(1/\sqrt2)F_2$ is unitary.
> <!-- bilingual-en:end -->

> [!question]- Problem 26.2：求 $F_4$ 分解中的 $D$ 与 $P$，并验算
> 求
> $$
> F_4=\begin{bmatrix}I&D\\I&-D\end{bmatrix}
> \begin{bmatrix}F_2&0\\0&F_2\end{bmatrix}P.
> $$
> <!-- bilingual-en:start -->
> seek
> <!-- bilingual-en:end -->

> [!success]- 解答
> 四次单位根依次为 $1,i,-1,-i$，故
> $$
> D=\begin{bmatrix}1&0\\0&i\end{bmatrix}.
> $$
> $P$ 把 $(x_0,x_1,x_2,x_3)^T$ 排成 $(x_0,x_2,x_1,x_3)^T$：
> $$
> P=\begin{bmatrix}
> 1&0&0&0\\
> 0&0&1&0\\
> 0&1&0&0\\
> 0&0&0&1
> \end{bmatrix}.
> $$
> 先算中间两块：
> $$
> \begin{bmatrix}F_2&0\\0&F_2\end{bmatrix}P
> =\begin{bmatrix}
> 1&0&1&0\\1&0&-1&0\\0&1&0&1\\0&1&0&-1
> \end{bmatrix}.
> $$
> 再左乘 butterfly 矩阵，得到
> $$
> \begin{bmatrix}
> 1&1&1&1\\1&i&-1&-i\\1&-1&1&-1\\1&-i&-1&i
> \end{bmatrix}=F_4.
> $$
> <!-- bilingual-en:start -->
> the fourth root is $1,i,-1,-i$, so
> $P$ ranks $(x_0,x_1,x_2,x_3)^T$ as $(x_0,x_2,x_1,x_3)^T$:
> Start with the middle two:
> Multiply the butterfly matrix to the left and get
> <!-- bilingual-en:end -->

### 3.2.6 边界、反例与易错点
<!-- bilingual-en:start -->
*3.2.6 Boundaries, Counterexamples and common errors*
<!-- bilingual-en:end -->

- $A^T=A$ 与 $A^*=A$ 在复数域不同；复数对称矩阵未必有实特征值。
- $F_n$ 本身不是 unitary；$(1/\sqrt n)F_n$ 才是。若采用其他 DFT 符号约定，指数可能写成 $-2\pi i/n$，逆变换的符号会相应交换。
- “实矩阵的复特征值成共轭对”只依赖特征多项式系数为实，不要求矩阵对称。
- FFT 的 $O(n\log n)$ 结论通常假设 $n$ 可递归分解；其他长度也有相应算法，但本课只需掌握二进制分解。
<!-- bilingual-en:start -->
- $A^T=A$ is different from $A^*=A$ in the complex field; complex symmetric matrices do not necessarily have real eigenvalues.
- $F_n$ is not unitary per se; $(1/\sqrt n)F_n$ is.  If other DFT symbolic conventions are used, the exponent may be written as $-2\pi i/n$, and the sign of the inverse transform is exchanged accordingly.
- "Complex eigenvalues of real matrices are conjugate pairs" only depend on the coefficients of the characteristic polynomials as the facts and do not require matrix symmetry.
- The $O(n\log n)$ conclusion of FFT usually assumes that the $n$ is recursive; other lengths also have algorithms, but this lesson only requires knowledge of binary decomposition.
<!-- bilingual-en:end -->

### 3.2.7 三道自检题
<!-- bilingual-en:start -->
*3.2.7 Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 自检 1（概念）
> 为什么 $z^Tz$ 不能作为复向量的长度平方？
> <!-- bilingual-en:start -->
> Why can't $z^Tz$ be the square of the length of a complex vector?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 它可能为零或非实数，即使 $z\ne0$。例如 $(1,i)^T(1,i)=0$；$z^*z=\sum|z_i|^2$ 才满足正定性。
> <!-- bilingual-en:start -->
> It may be zero or non-real, even if $z\ne0$.  For example, $(1,i)^T(1,i)=0$;$z^*z=\sum|z_i|^2$ satisfies positive definiteness.
> <!-- bilingual-en:end -->

> [!question]- 自检 2（计算）
> 求 $F_4^{-1}$。
> <!-- bilingual-en:start -->
> Find $F_4^{-1}$.
> <!-- bilingual-en:end -->

> [!success]- 答案
> 因 $F_4^*F_4=4I$，故 $F_4^{-1}=\frac14F_4^*$。
> <!-- bilingual-en:start -->
> $F_4^{-1}=\frac14F_4^*$ because of $F_4^*F_4=4I$.
> <!-- bilingual-en:end -->

> [!question]- 自检 3（证明）
> 证明 unitary 矩阵保持长度。
> <!-- bilingual-en:start -->
> It is proved that the unitary matrix holds the length.
> <!-- bilingual-en:end -->

> [!success]- 答案
> 若 $Q^*Q=I$，则
> $$
> \|Qx\|^2=(Qx)^*(Qx)=x^*Q^*Qx=x^*x=\|x\|^2.
> $$
> <!-- bilingual-en:start -->
> If $Q^*Q=I$,
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$$
T\mapsto *\Longrightarrow\text{Hermitian/unitary}
\Longrightarrow\text{复正交谱结构}
\Longrightarrow F_n^*F_n=nI
\Longrightarrow\text{FFT 的偶奇分解}.
$$

---

## Session 3.3 Positive definite matrices and minima

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节要回答：正定性为什么等价于一个二次函数沿每个方向都向上？消元、配方、Hessian 与极小值为什么会出现同一组数？
<!-- bilingual-en:start -->
In this section, you will answer: Why is positive qualitatively equivalent to a quadratic function going up in each direction?  Why do elimination, formula, Hessian, and minimum values appear in the same set?
<!-- bilingual-en:end -->

前置知识：3.1 的正定判据、多元微积分中的临界点、$LDL^T$ 分解。设 $A\in\mathbb R^{n\times n}$ 对称，$x\in\mathbb R^n$；二次型（quadratic form）是 $q(x)=x^TAx$。
<!-- bilingual-en:start -->
Prerequisites: the positive-definiteness criteria from Session 3.1, critical points in multivariable calculus, and the $LDL^T$ factorization. Let $A\in\mathbb R^{n\times n}$ be symmetric and $x\in\mathbb R^n$, and define the quadratic form $q(x)=x^TAx$.
<!-- bilingual-en:end -->

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S03_Lecture_Lecture_27_Positive_Definite_Matrices_and_Minima.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S03_Recitation_Problem_Solving_Positive_Definite_Matrices_and_Minima.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.3prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.3sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:start -->
Data portal: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S03_Lecture_Lecture_27_Positive_Definite_Matrices_and_Minima.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S03_Recitation_Problem_Solving_Positive_Definite_Matrices_and_Minima.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.3prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.3sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:end -->

关联卡片：[[对称矩阵与正定二次型#二次型与正定性|正定矩阵]]、[[对称矩阵与正定二次型#二次型与正定性|二次型]]、[[对称矩阵与正定二次型#二次型与正定性|正定性判别]]、[[正交投影与最小二乘#最小二乘与正规方程|最小二乘]]。
<!-- bilingual-en:start -->
Associated cards: [[对称矩阵与正定二次型#二次型与正定性|positive-definite matrices]], [[对称矩阵与正定二次型#二次型与正定性|quadratic forms]], [[对称矩阵与正定二次型#二次型与正定性|tests for positive definiteness]], [[正交投影与最小二乘#最小二乘与正规方程|least squares]].
<!-- bilingual-en:end -->

### 3.3.1 二阶二次型的四个判据
<!-- bilingual-en:start -->
*3.3.1 Four Criterions of Second Order Quadratic Form*
<!-- bilingual-en:end -->

对称矩阵
<!-- bilingual-en:start -->
symmetric matrix
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}a&b\\b&c\end{bmatrix}
$$

对应
<!-- bilingual-en:start -->
mapping
<!-- bilingual-en:end -->

$$
q(x,y)=\begin{bmatrix}x&y\end{bmatrix}A\begin{bmatrix}x\\y\end{bmatrix}
=ax^2+2bxy+cy^2.
$$

$A$ 正定等价于以下任一条件：
<!-- bilingual-en:start -->
$A$ is positive equivalent to any of the following:
<!-- bilingual-en:end -->

1. 两个特征值都正；
2. $a>0$ 且 $ac-b^2>0$；
3. 两个对称消元主元 $a$ 与 $(ac-b^2)/a$ 都正；
4. $q(x,y)>0$ 对所有 $(x,y)\ne(0,0)$ 成立。
<!-- bilingual-en:start -->
1. Both eigenvalues are positive;
2. $a>0$ and $ac-b^2>0$;
3. Both $a$ and $(ac-b^2)/a$ are positive;
4. $q(x,y)>0$ is true for all $(x,y)\ne(0,0)$.
<!-- bilingual-en:end -->

以
<!-- bilingual-en:start -->
with
<!-- bilingual-en:end -->

$$
A_y=\begin{bmatrix}2&6\\6&y\end{bmatrix}
$$

为例，$\Delta_1=2>0$，$\Delta_2=2y-36$，故
<!-- bilingual-en:start -->
For example, $\Delta_1=2>0$, $\Delta_2=2y-36$, and
<!-- bilingual-en:end -->

$$
A_y\succ0\iff y>18.
$$

$y=18$ 是边界：
<!-- bilingual-en:start -->
$y=18$ is the boundary:
<!-- bilingual-en:end -->

$$
x^TA_{18}x=2x_1^2+12x_1x_2+18x_2^2
=2(x_1+3x_2)^2\ge0,
$$

但在 $(x_1,x_2)=(3,-1)$ 处为零，因此只是正半定。
<!-- bilingual-en:start -->
But it's zero at $(x_1,x_2)=(3,-1)$, so it's just a positive semidefinite.
<!-- bilingual-en:end -->

### 3.3.2 配方就是 $LDL^T$
<!-- bilingual-en:start -->
*3.3.2 The formula is $LDL^T$*
<!-- bilingual-en:end -->

课件比较
<!-- bilingual-en:start -->
courseware comparison
<!-- bilingual-en:end -->

$$
q_+(x,y)=2x^2+12xy+20y^2
$$

与
<!-- bilingual-en:start -->
and
<!-- bilingual-en:end -->

$$
q_-(x,y)=2x^2+12xy+7y^2.
$$

逐步配方：
<!-- bilingual-en:start -->
Step-by-step recipe:
<!-- bilingual-en:end -->

$$
2x^2+12xy+20y^2
=2(x+3y)^2+2y^2>0
$$

对非零 $(x,y)$ 成立；而
<!-- bilingual-en:start -->
is true for non-zero $(x,y)$; and
<!-- bilingual-en:end -->

$$
2x^2+12xy+7y^2
=2(x+3y)^2-11y^2
$$

在 $(x,y)=(-3,1)$ 处为负。配方中的系数 $2,2$ 和 $2,-11$，恰好是对应矩阵消元后的主元：
<!-- bilingual-en:start -->
Negative at $(x,y)=(-3,1)$.  The coefficients $2,2$ and $2,-11$ in the formula are exactly the pivots of the corresponding matrix after elimination:
<!-- bilingual-en:end -->

$$
\begin{bmatrix}2&6\\6&20\end{bmatrix}
\longrightarrow
\begin{bmatrix}2&6\\0&2\end{bmatrix},
$$

$$
\begin{bmatrix}2&6\\6&7\end{bmatrix}
\longrightarrow
\begin{bmatrix}2&6\\0&-11\end{bmatrix}.
$$

一般地，若 $A=LDL^T$，则
<!-- bilingual-en:start -->
In general, if $A=LDL^T$, then
<!-- bilingual-en:end -->

$$
x^TAx=(L^Tx)^TD(L^Tx),
$$

这就是高维“配方”：$L^Tx$ 给出新的线性表达式，$D$ 的主元给出每个平方项前的系数。
<!-- bilingual-en:start -->
This is the high-dimensional "recipe": $L^Tx$ gives a new linear expression, and $D$'s pivots give the coefficients before each square term.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-quadratic-bowl.png|820]]

### 3.3.3 [[对称矩阵与正定二次型#二次型与正定性|Hessian]] 与极小值
<!-- bilingual-en:start -->
*3.3.3 [[对称矩阵与正定二次型#二次型与正定性|Hessian]] and Minimums*
<!-- bilingual-en:end -->

对二次函数
<!-- bilingual-en:start -->
Quadratic function
<!-- bilingual-en:end -->

$$
f(x)=\frac12x^TAx-b^Tx+c,\qquad A=A^T,
$$

梯度与 Hessian 为
<!-- bilingual-en:start -->
Gradient vs. Hessian
<!-- bilingual-en:end -->

$$
\nabla f(x)=Ax-b,\qquad \nabla^2f(x)=A.
$$

若 $A\succ0$，则临界点唯一：
<!-- bilingual-en:start -->
If $A\succ0$, the threshold is unique:
<!-- bilingual-en:end -->

$$
x_*=A^{-1}b.
$$

要证明它是唯一全局极小点，不只说“二阶导数为正”，而是令 $h=x-x_*$ 并完整展开：
<!-- bilingual-en:start -->
To prove that it is the only global minimum point, we do not just say "the second derivative is positive", but let $h=x-x_*$ and expand it completely:
<!-- bilingual-en:end -->

$$
\begin{aligned}
f(x_*+h)
&=\frac12(x_*+h)^TA(x_*+h)-b^T(x_*+h)+c\\
&=f(x_*)+h^T(Ax_*-b)+\frac12h^TAh\\
&=f(x_*)+\frac12h^TAh.
\end{aligned}
$$

因为 $A\succ0$，$h\ne0$ 时 $h^TAh>0$，所以 $f(x)>f(x_*)$。这同时证明了局部极小与全局唯一性。
<!-- bilingual-en:start -->
$f(x)>f(x_*)$ because $A\succ0$, $h\ne0$ $h^TAh>0$.  This proves both local minima and global uniqueness.
<!-- bilingual-en:end -->

边界情况：
<!-- bilingual-en:start -->
Boundary Condition:
<!-- bilingual-en:end -->

- $A\succeq0$ 时函数是凸的，但可能沿零空间方向平坦，极小点可能不唯一；若 $b$ 在 $A$ 的列空间之外，甚至可能无下界。
- $A$ 有负特征值时，沿对应特征向量 $q$，$f(tq)$ 的二次项为 $(\lambda/2)t^2$，会向 $-\infty$ 下降。
- $A$ 同时有正、负特征值时：若 $b=0$，纯二次函数在原点呈马鞍形；更一般地，若 $A$ 可逆，则唯一临界点 $x_*=A^{-1}b$ 呈马鞍形。不能在 $b\ne0$ 时无条件把原点称为临界点。
<!-- bilingual-en:start -->
- If $A\succeq0$, the function is convex but may be flat along nullspace directions, so a minimizer need not be unique. If $b\notin C(A)$, the function may even be unbounded below.
- If $A$ has a negative eigenvalue, then along its eigenvector $q$ the quadratic term in $f(tq)$ is $(\lambda/2)t^2$, which tends to $-\infty$.
- If $A$ has both positive and negative eigenvalues, then for $b=0$ the pure quadratic has a saddle at the origin. More generally, if $A$ is invertible, its unique critical point $x_*=A^{-1}b$ is a saddle. When $b\ne0$, the origin cannot automatically be called a critical point.
<!-- bilingual-en:end -->

#### 三维课件例子
<!-- bilingual-en:start -->
*A three-dimensional example from the course materials*
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}2&-1&0\\-1&2&-1\\0&-1&2\end{bmatrix}.
$$

顺序主子式为
<!-- bilingual-en:start -->
The leading principal minors are
<!-- bilingual-en:end -->

$$
\Delta_1=2,\qquad \Delta_2=3,\qquad \Delta_3=4,
$$

因此 $A\succ0$。相应主元为
<!-- bilingual-en:start -->
So $A\succ0$.  The corresponding pivot is
<!-- bilingual-en:end -->

$$
d_1=2,\qquad d_2=\frac32,\qquad d_3=\frac43.
$$

二次型
<!-- bilingual-en:start -->
quadratic
<!-- bilingual-en:end -->

$$
x^TAx=2x_1^2+2x_2^2+2x_3^2-2x_1x_2-2x_2x_3
$$

的等值面 $x^TAx=1$ 是椭球；特征向量给出主轴方向，半轴长度与 $1/\sqrt{\lambda_i}$ 成正比。
<!-- bilingual-en:start -->
The iso-surface $x^TAx=1$ is ellipsoid; the eigenvector gives the principal axis direction, and the length of the semi-axis is proportional to $1/\sqrt{\lambda_i}$.
<!-- bilingual-en:end -->

### 3.3.4 Recitation：带参数的三种判别
<!-- bilingual-en:start -->
*3.3.4 Recitation: Three discriminations with parameters*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
B(c)=\begin{bmatrix}
2&-1&-1\\
-1&2&-1\\
-1&-1&2+c
\end{bmatrix}.
$$

顺序主子式：
<!-- bilingual-en:start -->
Leading principal minors:
<!-- bilingual-en:end -->

$$
\Delta_1=2,\qquad \Delta_2=3,\qquad \Delta_3=3c.
$$

所以 $c>0$ 时正定。对称消元得到主元
<!-- bilingual-en:start -->
Therefore the matrix is positive definite exactly when $c>0$. Symmetry-preserving elimination gives the pivots
<!-- bilingual-en:end -->

$$
2,\quad \frac32,\quad c.
$$

同时，完整配方是
<!-- bilingual-en:start -->
At the same time, the complete formula is
<!-- bilingual-en:end -->

$$
\begin{aligned}
\begin{bmatrix}x&y&z\end{bmatrix}B(c)
\begin{bmatrix}x\\y\\z\end{bmatrix}
&=2\left(x-\frac y2-\frac z2\right)^2
+\frac32(y-z)^2+cz^2.
\end{aligned}
$$

因此：
<!-- bilingual-en:start -->
Therefore:
<!-- bilingual-en:end -->

- $c>0$：三个平方项系数都正，且对应三角变换可逆，所以正定；
- $c=0$：表达式非负，但 $(x,y,z)=(1,1,1)$ 使它为零，所以正半定但不正定；
- $c<0$：取 $x=y=z=1$，前两个平方项为零，剩下 $c<0$，故不半正定。
<!-- bilingual-en:start -->
- If $c>0$, all three squared terms have positive coefficients and the associated triangular change of variables is invertible, so the form is positive definite.
- If $c=0$, the expression is nonnegative, but $(x,y,z)=(1,1,1)$ makes it zero; the form is positive semidefinite but not positive definite.
- If $c<0$, setting $x=y=z=1$ makes the first two squared terms vanish and leaves $c<0$, so the form is not positive semidefinite.
<!-- bilingual-en:end -->

### 3.3.5 Homework：完整题解
<!-- bilingual-en:start -->
*3.3.5 Homework: Complete solutions*
<!-- bilingual-en:end -->

> [!question]- Problem 27.1：证明两个正定矩阵乘积的特征值仍为正
> $A,B$ 均为实对称正定，$AB$ 未必对称。由 $ABx=\lambda x$ 出发，证明 $\lambda>0$。
> <!-- bilingual-en:start -->
> Let $A$ and $B$ be real symmetric positive-definite matrices. Although $AB$ need not be symmetric, start from $ABx=\lambda x$ and prove that $\lambda>0$.
> <!-- bilingual-en:end -->

> [!success]- 解答
> 因 $B$ 可逆，$x\ne0$ 时 $Bx\ne0$。特征向量事先可能是复向量，所以使用 Hermitian 内积。由 $ABx=\lambda x$，左乘 $(Bx)^*$：
> $$
> (Bx)^*ABx=\lambda (Bx)^*x=\lambda x^*Bx.
> $$
> 实对称正定矩阵对非零复向量也满足 $z^*Az>0$：把 $z=p+iq$ 展开可得 $z^*Az=p^TAp+q^TAq$。因此左边 $(Bx)^*A(Bx)>0$，分母 $x^*Bx>0$。于是
> $$
> \lambda=\frac{(Bx)^*A(Bx)}{x^*Bx}>0.
> $$
> 这个商本身是实数，所以也证明了 $AB$ 的特征值为实且正。另一种结构证明是 $AB$ 与 $B^{1/2}AB^{1/2}$ 相似，而后者对称正定。
> <!-- bilingual-en:start -->
> Because $B$ is invertible, $x\ne0$ implies $Bx\ne0$. An eigenvector may be complex, so use the Hermitian inner product. Starting from $ABx=\lambda x$, multiply on the left by $(Bx)^*$.
> A real symmetric positive-definite matrix also satisfies $z^*Az>0$ for every nonzero complex vector $z$: writing $z=p+iq$ gives $z^*Az=p^TAp+q^TAq$. Hence $(Bx)^*A(Bx)>0$ and $x^*Bx>0$, so
> $\lambda=\dfrac{(Bx)^*A(Bx)}{x^*Bx}>0$.
> The quotient is real, proving that every eigenvalue of $AB$ is real and positive. Structurally, one may also note that $AB$ is similar to the symmetric positive-definite matrix $B^{1/2}AB^{1/2}$.
> <!-- bilingual-en:end -->

> [!question]- Problem 27.2：求矩阵的二次型并判断符号
> 对
> $$
> A=\begin{bmatrix}1&5\\7&9\end{bmatrix},
> $$
> 求 $[x\ y]A[x\ y]^T$，判断它恒正、恒负还是可正可负。
> <!-- bilingual-en:start -->
> For the given matrix $A$, compute $[x\ y]A[x\ y]^T$ and determine whether it is always positive, always negative, or can take both signs.
> <!-- bilingual-en:end -->

> [!success]- 解答
> 逐步相乘：
> $$
> A\begin{bmatrix}x\\y\end{bmatrix}
> =\begin{bmatrix}x+5y\\7x+9y\end{bmatrix},
> $$
> 因而
> $$
> q(x,y)=x(x+5y)+y(7x+9y)=x^2+12xy+9y^2.
> $$
> 取 $(x,y)=(1,0)$，$q=1>0$；取 $(2,-2)$，
> $$
> q=4-48+36=-8<0.
> $$
> 所以可正可负。
>
> 注意 $A$ 不对称，二次型实际只由
> $$
> \frac{A+A^T}{2}=\begin{bmatrix}1&6\\6&9\end{bmatrix}
> $$
> 决定；其行列式 $9-36=-27<0$，直接显示一正一负两个方向。不能把非对称 $A$ 的“正定判据”不加说明地直接套用。
> <!-- bilingual-en:start -->
> Multiply in two steps. This gives $q(x,y)=x^2+12xy+9y^2$.
> At $(x,y)=(1,0)$, $q=1>0$; at $(2,-2)$, $q=-8<0$. Thus the form takes both positive and negative values.
> The matrix $A$ is not symmetric, and the quadratic form is determined by its symmetric part $(A+A^T)/2$. That symmetric matrix has determinant $9-36=-27<0$, directly revealing one positive and one negative direction. Positive-definiteness criteria for symmetric matrices cannot be applied to a nonsymmetric $A$ without this clarification.
> <!-- bilingual-en:end -->

### 3.3.6 边界、反例与易错点
<!-- bilingual-en:start -->
*3.3.6 Boundaries, Counterexamples and common errors*
<!-- bilingual-en:end -->

- Hessian 正定是临界点为严格局部极小的充分条件；若 Hessian 仅半正定，二阶检验可能无结论，必须看高阶项。
- 对一般 $C^2$ 函数，$\nabla^2f(x_*)\succ0$ 只先给出局部极小；二次函数或全局 Hessian 半正定时才能直接推全局凸性。
- “所有顺序主子式非负”不足以判正半定；$c=0$ 的例子可由结构直接验证，但一般半正定要检查所有主子式或特征值。
<!-- bilingual-en:start -->
Positive definiteness of the Hessian is sufficient for a critical point to be a strict local minimum. If the Hessian is only positive semidefinite, the second-order test may be inconclusive and higher-order terms must be examined.
- For general $C^2$ functions, $\nabla^2f(x_*)\succ0$ only gives local minima; quadratic functions or global Hessian semipositive timing can directly infer global convexity.
- Nonnegative leading principal minors alone are not sufficient for positive semidefiniteness. The case $c=0$ can be verified directly from the structure, but in general one must check all principal minors or the eigenvalues.
<!-- bilingual-en:end -->

### 3.3.7 三道自检题
<!-- bilingual-en:start -->
*3.3.7 Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 自检 1（配方）
> 把 $3x^2+6xy+5y^2$ 配成平方和并判断正定性。
> <!-- bilingual-en:start -->
> Formulate the $3x^2+6xy+5y^2$ as a sum of squares and determine the positive definiteness.
> <!-- bilingual-en:end -->

> [!success]- 答案
> $$
> 3x^2+6xy+5y^2=3(x+y)^2+2y^2>0
> $$
> 对非零 $(x,y)$ 成立，因此对应矩阵正定。
> <!-- bilingual-en:start -->
> It is true for non-zero $(x,y)$, so the corresponding matrix is positive definite.
> <!-- bilingual-en:end -->

> [!question]- 自检 2（半正定边界）
> 判断 $A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$ 是正定、半正定还是不定，并说明“顺序主子式非负”为何在这里不能被误读成正定。
> <!-- bilingual-en:start -->
> Determine whether $A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$ is positive definite, positive semidefinite, or indefinite, and explain why nonnegative leading principal minors must not be misread as evidence of positive definiteness here.
> <!-- bilingual-en:end -->

> [!success]- 答案
> $x^TAx=(x_1+2x_2)^2\ge0$，故 $A\succeq0$；但 $A( -2,1)^T=0$，存在非零零空间方向，故不正定。其顺序主子式为 $1,0$；“非负”在这个例子中可验证半正定，却不能作为一般矩阵只检查顺序主子式的充分判据。
> <!-- bilingual-en:start -->
> Since $x^TAx=(x_1+2x_2)^2\ge0$, we have $A\succeq0$. But $A(-2,1)^T=0$, so the nullspace contains a nonzero direction and $A$ is not positive definite. Its leading principal minors are $1$ and $0$. Their nonnegativity is consistent with positive semidefiniteness in this example, but checking only leading principal minors is not a sufficient test for a general matrix.
> <!-- bilingual-en:end -->

> [!question]- 自检 3（极小值）
> 若 $A\succ0$，求 $f(x)=\tfrac12x^TAx-b^Tx$ 的最小点与最小值表达式。
> <!-- bilingual-en:start -->
> If $A\succ0$, calculate the minimum point and minimum value expression of $f(x)=\tfrac12x^TAx-b^Tx$.
> <!-- bilingual-en:end -->

> [!success]- 答案
> $x_*=A^{-1}b$；代入得
> $$
> f(x_*)=-\frac12b^TA^{-1}b.
> $$
> <!-- bilingual-en:start -->
> $x_*=A^{-1}b$;substitution
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$$
A=LDL^T\Longleftrightarrow\text{完成平方}
\Longleftrightarrow x^TAx\text{ 的曲率}
\Longleftrightarrow\text{Hessian 与极小值}.
$$

---

## Session 3.4 Similar matrices and Jordan form

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节先补齐 Lecture 28 开头关于正定矩阵封闭性质与 $A^TA$ 的结论，再回答：相似矩阵为何表示“同一个线性变换的不同坐标”？当特征向量不足时，Jordan 链怎样补足基？Jordan 块怎样影响矩阵幂和微分方程长期行为？
<!-- bilingual-en:start -->
This section first completes Lecture 28's results on closure properties of positive-definite matrices and on $A^TA$. It then asks: Why do similar matrices represent the same linear transformation in different coordinates? When ordinary eigenvectors do not form a basis, how do Jordan chains complete one? How do Jordan blocks affect matrix powers and the long-run behavior of differential equations?
<!-- bilingual-en:end -->

前置知识：对角化 $A=S\Lambda S^{-1}$、特征空间、矩阵幂。设 $A,B\in\mathbb F^{n\times n}$。Jordan 形要求特征多项式在底层域 $\mathbb F$ 上分裂；在 $\mathbb C$ 上总能分裂，在 $\mathbb R$ 上则未必，例如非平凡二维旋转没有实 Jordan 对角形。
<!-- bilingual-en:start -->
Prerequisites are diagonalization $A=S\Lambda S^{-1}$, eigenspaces, and matrix powers. Let $A,B\in\mathbb F^{n\times n}$. Jordan form requires the characteristic polynomial to split over the underlying field $\mathbb F$. It always splits over $\mathbb C$, but not necessarily over $\mathbb R$; for example, a nontrivial planar rotation has no real Jordan form with real eigenvalues on the diagonal.
<!-- bilingual-en:end -->

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S04_Recitation_Problem_Solving_Similar_Matrices.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.4prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.4sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:start -->
Data portal: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S04_Recitation_Problem_Solving_Similar_Matrices.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.4prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.4sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:end -->

关联卡片：[[对称矩阵与正定二次型#二次型与正定性|正定矩阵]]、[[线性变换与换基#换基与相似|相似矩阵]]、[[特征值、对角化与线性动力系统#对角化与矩阵幂|对角化]]、[[特征值、对角化与线性动力系统#Jordan 结构的边界|Jordan 形]]、[[特征值、对角化与线性动力系统#对角化与矩阵幂|矩阵指数]]。
<!-- bilingual-en:start -->
Associated cards: [[对称矩阵与正定二次型#二次型与正定性|positive-definite matrices]], [[线性变换与换基#换基与相似|similarity transformations]], [[特征值、对角化与线性动力系统#对角化与矩阵幂|diagonalization]], [[特征值、对角化与线性动力系统#Jordan 结构的边界|Jordan form]], and [[特征值、对角化与线性动力系统#对角化与矩阵幂|matrix powers]].
<!-- bilingual-en:end -->

### 3.4.1 Lecture 28 开头：正定矩阵的封闭性质与 $A^TA$
<!-- bilingual-en:start -->
*3.4.1 Lecture 28 Beginning: Closure Properties of Positive Definite Matrices and $A^TA$*
<!-- bilingual-en:end -->

这一段属于官方 Session 3.4，而不是上一节的二次型例题。以下每个结论都保留假设；尤其“正定”默认矩阵为实对称方阵。
<!-- bilingual-en:start -->
This material belongs to the official Session 3.4, not to the quadratic-form example in the preceding section. Every conclusion below retains its stated assumptions; in particular, “positive definite” refers by default to a real symmetric square matrix.
<!-- bilingual-en:end -->

#### 正定矩阵之和仍正定
<!-- bilingual-en:start -->
*The sum of positive definite matrices is still positive definite*
<!-- bilingual-en:end -->

若 $A\succ0$、$B\succ0$ 且尺寸相同，则 $A+B$ 对称，并且对任意 $x\ne0$，
<!-- bilingual-en:start -->
If $A\succ0$ and $B\succ0$ have the same size, then $A+B$ is symmetric and, for every $x\ne0$,
<!-- bilingual-en:end -->

$$
x^T(A+B)x=x^TAx+x^TBx>0+0=0.
$$

故 $A+B\succ0$。这里不能把“和”替换为“乘积”：$AB$ 未必对称。
<!-- bilingual-en:start -->
Hence $A+B\succ0$. The statement does not remain true if “sum” is replaced by “product”: $AB$ need not be symmetric.
<!-- bilingual-en:end -->

#### 正定矩阵之逆仍正定
<!-- bilingual-en:start -->
*The Inverse of Positive Definite Matrix is Still Positive Definite*
<!-- bilingual-en:end -->

由谱定理，$A=Q\Lambda Q^T$，其中 $\lambda_i>0$。于是
<!-- bilingual-en:start -->
From the spectral theorem, $A=Q\Lambda Q^T$, where $\lambda_i>0$.  therefore
<!-- bilingual-en:end -->

$$
A^{-1}=Q\Lambda^{-1}Q^T,
\qquad
\Lambda^{-1}=\operatorname{diag}(1/\lambda_1,\ldots,1/\lambda_n).
$$

所有 $1/\lambda_i$ 仍为正数，所以 $A^{-1}\succ0$。
<!-- bilingual-en:start -->
All $1/\lambda_i$ are still positive, so $A^{-1}\succ0$.
<!-- bilingual-en:end -->

#### $C^TC$ 总半正定，满列秩时才正定
<!-- bilingual-en:start -->
*$C^TC$ is always positive semidefinite and is positive definite exactly when $C$ has full column rank*
<!-- bilingual-en:end -->

对任意 $C\in\mathbb R^{m\times n}$ 与 $x\in\mathbb R^n$，
<!-- bilingual-en:start -->
For any $C\in\mathbb R^{m\times n}$ and $x\in\mathbb R^n$,
<!-- bilingual-en:end -->

$$
x^TC^TCx=(Cx)^T(Cx)=\|Cx\|^2\ge0,
$$

故 $C^TC\succeq0$。进一步，
<!-- bilingual-en:start -->
So, $C^TC\succeq0$.  further,
<!-- bilingual-en:end -->

$$
\begin{aligned}
C^TC\succ0
&\iff \|Cx\|^2>0\quad\text{对每个 }x\ne0\\
&\iff Cx=0\text{ 只有零解}\\
&\iff N(C)=\{0\}\\
&\iff \operatorname{rank}(C)=n.
\end{aligned}
$$

因此 $C^TC$ 可逆当且仅当 $C$ 满列秩。此时正规方程 $C^TC\hat x=C^Tb$ 有唯一解；列相关时仍可用 SVD 或伪逆描述全部最小二乘解。
<!-- bilingual-en:start -->
Thus $C^TC$ is invertible if and only if $C$ has full column rank. In that case the normal equation $C^TC\hat x=C^Tb$ has a unique solution. When the columns are dependent, all least-squares solutions can instead be described using the SVD or pseudoinverse.
<!-- bilingual-en:end -->

### 3.4.2 相似关系保存什么
<!-- bilingual-en:start -->
*3.4.2 What similarity holds*
<!-- bilingual-en:end -->

> [!definition] 相似矩阵
> 若存在可逆 $M\in\mathbb F^{n\times n}$ 使
> $$
> B=M^{-1}AM,
> $$
> 则称 $A$ 与 $B$ 相似。$M$ 的列通常是新基在旧坐标中的表示。
> <!-- bilingual-en:start -->
> If there is a invertible $M\in\mathbb F^{n\times n}$
> It is also known that $A$ is similar to $B$.  The column for $M$ is usually the representation of the new basis in the old coordinates.
> <!-- bilingual-en:end -->

相似变换保存：
<!-- bilingual-en:start -->
Similar Transformation Save:
<!-- bilingual-en:end -->

- 特征多项式、特征值及其代数重数；
- 每个特征值的几何重数；
- 秩、行列式、迹；
- 最小多项式与 Jordan 块尺寸；
- 对任意多项式 $p$，$p(B)=M^{-1}p(A)M$。
<!-- bilingual-en:start -->
- Characteristic polynomials, eigenvalues and their algebraic multiplicities;
- Geometric multiplicity of each eigenvalue;
- rank, determinant, footprint;
- Minimum polynomial and Jordan block size;
- For any polynomial $p$, $p(B)=M^{-1}p(A)M$.
<!-- bilingual-en:end -->

#### 证明特征值保存
<!-- bilingual-en:start -->
*proof eigenvalue preservation*
<!-- bilingual-en:end -->

若 $Ax=\lambda x$，令 $y=M^{-1}x\ne0$。则
<!-- bilingual-en:start -->
If $Ax=\lambda x$, let $y=M^{-1}x\ne0$.
<!-- bilingual-en:end -->

$$
By=M^{-1}AM(M^{-1}x)=M^{-1}Ax=\lambda M^{-1}x=\lambda y.
$$

所以 $\lambda$ 也是 $B$ 的特征值。反向使用 $A=MBM^{-1}$ 得到完全相同的结论。
<!-- bilingual-en:start -->
Thus $\lambda$ is also an eigenvalue of $B$. Applying the same argument in the reverse direction with $A=MBM^{-1}$ proves equality of the spectra.
<!-- bilingual-en:end -->

#### 证明多项式保持相似
<!-- bilingual-en:start -->
*proving polynomial similarity*
<!-- bilingual-en:end -->

先看幂：
<!-- bilingual-en:start -->
Begin with Power:
<!-- bilingual-en:end -->

$$
B^2=(M^{-1}AM)(M^{-1}AM)=M^{-1}A^2M,
$$

中间的 $MM^{-1}$ 消去。归纳得 $B^k=M^{-1}A^kM$，于是
<!-- bilingual-en:start -->
The $MM^{-1}$ in the middle goes away.  $B^k=M^{-1}A^kM$, so
<!-- bilingual-en:end -->

$$
p(B)=\sum_{k=0}^rc_kB^k
=M^{-1}\left(\sum_{k=0}^rc_kA^k\right)M
=M^{-1}p(A)M.
$$

相同论证还能通过收敛幂级数给出 $e^B=M^{-1}e^AM$。
<!-- bilingual-en:start -->
The same argument can also give $e^B=M^{-1}e^AM$ by converging power series.
<!-- bilingual-en:end -->

### 3.4.3 对角化与重复特征值
<!-- bilingual-en:start -->
*3.4.3 Diagonalizing and Repeating Eigenvalues*
<!-- bilingual-en:end -->

若 $A$ 有 $n$ 个线性无关特征向量，取 $S=[x_1\ \cdots\ x_n]$，则
<!-- bilingual-en:start -->
If $A$ has $n$ linear independent eigenvectors and $S=[x_1\ \cdots\ x_n]$, then
<!-- bilingual-en:end -->

$$
AS=S\Lambda,\qquad S^{-1}AS=\Lambda.
$$

有 $n$ 个互不相同特征值时必有 $n$ 个独立特征向量，所以必可对角化。但特征值重复时，代数重数不保证几何重数足够。例如
<!-- bilingual-en:start -->
If a matrix has $n$ distinct eigenvalues, it has $n$ linearly independent eigenvectors and is therefore diagonalizable. With repeated eigenvalues, however, algebraic multiplicity alone does not guarantee enough eigenvectors. For example,
<!-- bilingual-en:end -->

$$
\begin{bmatrix}4&0\\0&4\end{bmatrix}
$$

有二维特征空间，而
<!-- bilingual-en:start -->
has a two-dimensional eigenspace, whereas
<!-- bilingual-en:end -->

$$
\begin{bmatrix}4&1\\0&4\end{bmatrix}
$$

只有一维特征空间；二者不能相似。
<!-- bilingual-en:start -->
has only a one-dimensional eigenspace. The two matrices therefore cannot be similar.
<!-- bilingual-en:end -->

### 3.4.4 Jordan 链与 Jordan 标准形
<!-- bilingual-en:start -->
*3.4.4 Jordan Chains and Jordan Standardization*
<!-- bilingual-en:end -->

对特征值 $\lambda$，大小为 $k$ 的 Jordan 块是
<!-- bilingual-en:start -->
For the eigenvalue $\lambda$, a Jordan block of size $k$ is
<!-- bilingual-en:end -->

$$
J_k(\lambda)=
\begin{bmatrix}
\lambda&1&0&\cdots&0\\
0&\lambda&1&\ddots&\vdots\\
\vdots&\ddots&\ddots&\ddots&0\\
0&\cdots&0&\lambda&1\\
0&\cdots&\cdots&0&\lambda
\end{bmatrix}
=\lambda I+N,
$$

其中 $N^k=0$。对应基向量形成广义特征向量链：
<!-- bilingual-en:start -->
where $N^k=0$.  The corresponding basis vectors form a generalized eigenvector chain:
<!-- bilingual-en:end -->

$$
(A-\lambda I)v_1=0,\qquad
(A-\lambda I)v_2=v_1,\qquad\ldots,\qquad
(A-\lambda I)v_k=v_{k-1}.
$$

在特征多项式分裂的条件下，存在可逆 $S$ 使 $S^{-1}AS$ 为 Jordan 块的直和。课程重点是理解块结构与计算，不从头证明整个存在性定理。
<!-- bilingual-en:start -->
Under the condition of characteristic polynomial splitting, there exists a invertible $S$ such that $S^{-1}AS$ is the straight sum of Jordan blocks.  The focus of the course is on understanding block structure and computation without proving the entire existence theorem from scratch.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-jordan-chain.png|820]]

#### Jordan 块的幂和指数
<!-- bilingual-en:start -->
*Power sum exponent of Jordan block*
<!-- bilingual-en:end -->

因为 $\lambda I$ 与 $N$ 可交换，
<!-- bilingual-en:start -->
Because $\lambda I$ and $N$ are interchangeable,
<!-- bilingual-en:end -->

$$
J_k(\lambda)^m
=\sum_{j=0}^{\min(m,k-1)}\binom mj\lambda^{m-j}N^j.
$$

矩阵指数为
<!-- bilingual-en:start -->
The matrix exponential is
<!-- bilingual-en:end -->

$$
e^{tJ_k(\lambda)}
=e^{\lambda t}e^{tN}
=e^{\lambda t}\sum_{j=0}^{k-1}\frac{t^j}{j!}N^j.
$$

因此长期行为不只由 $e^{\lambda t}$ 决定：Jordan 块大小 $k$ 还带来最高 $t^{k-1}$ 的多项式因子。
<!-- bilingual-en:start -->
So the long-term behavior is not only determined by $e^{\lambda t}$: the Jordan block size $k$ also brings a polynomial factor of up to $t^{k-1}$.
<!-- bilingual-en:end -->

> [!warning] 稳定性的重要边界
> 若 $\operatorname{Re}\lambda<0$，指数衰减最终压过任意固定次数多项式；若 $\operatorname{Re}\lambda>0$，解指数增长。若 $\operatorname{Re}\lambda=0$，非平凡 Jordan 块会产生多项式增长。尤其“所有特征值纯虚，所以解有界”只在这些纯虚特征值对应 Jordan 块全为 $1\times1$（即相应部分可对角化）时成立。
> <!-- bilingual-en:start -->
> If $\operatorname{Re}\lambda<0$, exponential decay dominates every fixed-degree polynomial; if $\operatorname{Re}\lambda>0$, the solution grows exponentially. If $\operatorname{Re}\lambda=0$, a nontrivial Jordan block produces polynomial growth. Therefore, the claim “all eigenvalues are purely imaginary, so every solution is bounded” is valid only when all Jordan blocks for those eigenvalues have size $1\times1$, equivalently when that part of the matrix is diagonalizable.
> <!-- bilingual-en:end -->

### 3.4.5 课件与 Recitation 例子
<!-- bilingual-en:start -->
*3.4.5 Courseware and Recitation Examples*
<!-- bilingual-en:end -->

课件用
<!-- bilingual-en:start -->
courseware
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}2&1\\1&2\end{bmatrix}
$$

说明同一个相似类有许多矩阵。$A$ 的特征值是 $3,1$，故相似于 $\operatorname{diag}(3,1)$。任取可逆 $M$，$M^{-1}AM$ 仍有相同谱结构，但条目可能完全不同。
<!-- bilingual-en:start -->
It shows that there are many matrices in the same similar class.  The eigenvalue of $A$ is $3,1$, so it is similar to $\operatorname{diag}(3,1)$.  For any invertible $M$, $M^{-1}AM$ still have the same spectral structure, but the entries may be completely different.
<!-- bilingual-en:end -->

Recitation 的核心判断：
<!-- bilingual-en:start -->
Recitation's core judgment:
<!-- bilingual-en:end -->

1. 若 $A\sim B$，则 $2A^3+A-3I\sim2B^3+B-3I$，因为多项式保持相似。
2. 两个 $3\times3$ 矩阵若都有互异特征值 $1,0,-1$，则都相似于同一对角矩阵，故彼此相似。
3. 两个 Jordan 形即使特征值相同，若某个 $\dim N(J-\lambda I)$ 不同，就不相似。更一般地，$\dim N((A-\lambda I)^j)$ 的序列能恢复 Jordan 块尺寸。
<!-- bilingual-en:start -->
1. If $A\sim B$, then $2A^3+A-3I\sim2B^3+B-3I$, because the polynomials remain similar.
2. If two $3\times3$ matrices each have the three distinct eigenvalues $1,0,-1$, both are similar to the same diagonal matrix and hence to each other.
3. Two Jordan forms with the same eigenvalues are not similar if, for some $\lambda$, their values of $\dim N(J-\lambda I)$ differ. More generally, the sequence $\dim N((A-\lambda I)^j)$ determines the Jordan block sizes.
<!-- bilingual-en:end -->

### 3.4.6 Homework：完整题解
<!-- bilingual-en:start -->
*3.4.6 Homework: Complete solutions*
<!-- bilingual-en:end -->

> [!question]- Problem 28.1：证明两个零特征值 Jordan 矩阵不相似
> 设
> $$
> J=\begin{bmatrix}0&1&0&0\\0&0&0&0\\0&0&0&1\\0&0&0&0\end{bmatrix},\quad
> K=\begin{bmatrix}0&1&0&0\\0&0&1&0\\0&0&0&0\\0&0&0&0\end{bmatrix}.
> $$
> 对一般 $M=(m_{ij})$，证明 $JM=MK$ 会迫使 $M$ 不可逆。
> <!-- bilingual-en:start -->
> if
> For a general matrix $M=(m_{ij})$, prove that $JM=MK$ forces $M$ to be singular.
> <!-- bilingual-en:end -->

> [!success]- 解答
> 左乘 $J$ 会把 $M$ 的第 2 行放到第 1 行、第 4 行放到第 3 行，其余行变零：
> $$
> JM=\begin{bmatrix}
> m_{21}&m_{22}&m_{23}&m_{24}\\
> 0&0&0&0\\
> m_{41}&m_{42}&m_{43}&m_{44}\\
> 0&0&0&0
> \end{bmatrix}.
> $$
> 右乘 $K$ 后，各列为 $0$、$M$ 的第 1 列、第 2 列、$0$：
> $$
> MK=\begin{bmatrix}
> 0&m_{11}&m_{12}&0\\
> 0&m_{21}&m_{22}&0\\
> 0&m_{31}&m_{32}&0\\
> 0&m_{41}&m_{42}&0
> \end{bmatrix}.
> $$
> 比较元素得到 $m_{21}=m_{41}=0$，又由第二行得 $m_{21}=m_{22}=0$，第四行得 $m_{41}=m_{42}=0$，第一、三行进一步给出 $m_{11}=m_{31}=0$。因此 $M$ 的第一列全为零，$M$ 不可逆。
>
> 若 $J\sim K$，应存在可逆 $M$ 使 $K=M^{-1}JM$，等价于 $JM=MK$；与上述结论矛盾。故不相似。结构上，$J$ 的块尺寸为 $2+2$，$K$ 为 $3+1$。
> <!-- bilingual-en:start -->
> Left multiplication by $J$ places row two of $M$ into row one and row four into row three, while the remaining rows become zero.
> After right multiplication by $K$, the columns are zero, column one of $M$, column two of $M$, and zero, as displayed above.
> Comparing entries gives $m_{21}=m_{41}=0$. The second row then gives $m_{21}=m_{22}=0$, the fourth gives $m_{41}=m_{42}=0$, and the first and third give $m_{11}=m_{31}=0$. Thus the first column of $M$ is zero, so $M$ is singular.
> If $J\sim K$, an invertible matrix $M$ would satisfy $K=M^{-1}JM$, equivalently $JM=MK$. The calculation forces $M$ to be singular, a contradiction, so the matrices are not similar. Structurally, $J$ has Jordan block sizes $2+2$, whereas $K$ has sizes $3+1$.
> <!-- bilingual-en:end -->

> [!question]- Problem 28.2：解释五个相似性判断

> [!success]- 解答
> **(a)** 若 $A=M^{-1}BM$，则
> $$
> A^2=M^{-1}BM M^{-1}BM=M^{-1}B^2M,
> $$
> 所以 $A^2\sim B^2$。
>
> **(b)** 逆命题不成立。取
> $$
> A=\begin{bmatrix}0&0\\0&0\end{bmatrix},\qquad
> B=\begin{bmatrix}0&1\\0&0\end{bmatrix}.
> $$
> 则 $A^2=B^2=0$，但 $A$ 的秩为 $0$、$B$ 的秩为 $1$，故 $A\not\sim B$。
>
> **(c)** $\operatorname{diag}(3,4)$ 与 $\begin{bmatrix}3&1\\0&4\end{bmatrix}$ 都有两个互异特征值 $3,4$，因此都可对角化且彼此相似。可显式取
> $$
> M=\begin{bmatrix}1&1\\0&1\end{bmatrix},\qquad
> M^{-1}\begin{bmatrix}3&1\\0&4\end{bmatrix}M
> =\begin{bmatrix}3&0\\0&4\end{bmatrix}.
> $$
>
> **(d)** $3I$ 与 $\begin{bmatrix}3&1\\0&3\end{bmatrix}$ 不相似。前者特征空间维数为 $2$，后者只有 $1$；并且 $M^{-1}(3I)M=3I$ 对任何可逆 $M$ 都不变。
>
> **(e)** 设 $P$ 交换前两坐标，则 $P^{-1}=P$。先交换 $A$ 的前两行是 $PA$，再交换前两列是 $PAP=P^{-1}AP$，所以新矩阵与 $A$ 相似。
> <!-- bilingual-en:start -->
> **(a)**If $A=M^{-1}BM$
> So, $A^2\sim B^2$.
> **(b)** The converse is false.
> then $A^2=B^2=0$, but the rank of $A$ is $0$, and the rank of $B$ is $1$, so $A\not\sim B$.
> **(c)**$\operatorname{diag}(3,4)$ and $\begin{bmatrix}3&1\\0&4\end{bmatrix}$ both have two distinct eigenvalues $3,4$, so they are diagonalizable and similar to each other.  Explicitly Available
> **(d)**$3I$ is not similar to $\begin{bmatrix}3&1\\0&3\end{bmatrix}$.  The dimension of the former is $2$, and the latter is only $1$. Moreover, $M^{-1}(3I)M=3I$ is invariant to any invertible $M$.
> **(e)**Let $P$ exchange the first two coordinates, and then $P^{-1}=P$.  The first two rows of $A$ are swapped for $PA$, and then the first two columns are swapped for $PAP=P^{-1}AP$, so the new matrix is similar to $A$.
> <!-- bilingual-en:end -->

### 3.4.7 边界、反例与易错点
<!-- bilingual-en:start -->
*3.4.7 Boundaries, Counterexamples and common errors*
<!-- bilingual-en:end -->

- “特征值相同”通常不推出相似；还需相同 Jordan 块结构。只有特征值互异从而都可对角化时，才可直接推出。
- 相似与合同（congruence）不同：$M^{-1}AM$ 描述同一线性算子换基，$M^TAM$ 描述同一二次型换坐标。
- Jordan 形在浮点计算中极不稳定；它主要提供理论分类，不是大规模数值算法首选。
- 实矩阵若有非实特征值，必须到 $\mathbb C$ 上使用普通 Jordan 形，或在 $\mathbb R$ 上使用 $2\times2$ 实块。
<!-- bilingual-en:start -->
- Having the same eigenvalues does not generally imply similarity; the Jordan block structure must also agree. A direct conclusion is possible when all eigenvalues are distinct, because both matrices are then diagonalizable.
- Similarity and congruence are different: $M^{-1}AM$ represents a change of basis for the same linear operator, while $M^TAM$ represents a change of coordinates for the same quadratic form.
- Jordan form is extremely unstable in floating-point computation. It is chiefly a theoretical classification and is not the preferred tool for large-scale numerical algorithms.
- For a real matrix with nonreal eigenvalues, either work over $\mathbb C$ with ordinary Jordan form or use real $2\times2$ blocks over $\mathbb R$.
<!-- bilingual-en:end -->

### 3.4.8 三道自检题
<!-- bilingual-en:start -->
*3.4.8 Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 自检 1（结构）
> $J=\begin{bmatrix}2&1\\0&2\end{bmatrix}$ 的 $J^m$ 是什么？
> <!-- bilingual-en:start -->
> What is $J=\begin{bmatrix}2&1\\0&2\end{bmatrix}$'s $J^m$?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 写成 $J=2I+N$、$N^2=0$：
> $$
> J^m=2^mI+m2^{m-1}N
> =\begin{bmatrix}2^m&m2^{m-1}\\0&2^m\end{bmatrix}.
> $$
> <!-- bilingual-en:start -->
> Written as $J=2I+N$, $N^2=0$:
> <!-- bilingual-en:end -->

> [!question]- 自检 2（稳定性）
> $A=\begin{bmatrix}i&1\\0&i\end{bmatrix}$ 的 $e^{tA}$ 是否有界？
> <!-- bilingual-en:start -->
> Is $e^{tA}$ bounded for $A=\begin{bmatrix}i&1\\0&i\end{bmatrix}$?
> <!-- bilingual-en:end -->

> [!success]- 答案
> $$
> e^{tA}=e^{it}\begin{bmatrix}1&t\\0&1\end{bmatrix},
> $$
> 模长含线性因子 $t$，故不有界；纯虚谱不足以保证有界。
> <!-- bilingual-en:start -->
> The moduli are not bounded because they contain a linear factor $t$. The pure imaginary spectrum is not sufficient to guarantee boundedness.
> <!-- bilingual-en:end -->

> [!question]- 自检 3（判断）
> 迹和行列式都相同的两个 $2\times2$ 矩阵一定相似吗？
> <!-- bilingual-en:start -->
> Do the two $2\times2$ matrices that have the same trace and determinant have to be similar?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 不一定。$3I$ 与 $\begin{bmatrix}3&1\\0&3\end{bmatrix}$ 迹、行列式相同，却因几何重数不同而不相似。
> <!-- bilingual-en:start -->
> Not necessarily.  $3I$ is the same as $\begin{bmatrix}3&1\\0&3\end{bmatrix}$ trace and determinant, but not similar due to different geometric multiplicity.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$$
\text{换基}\Longrightarrow B=M^{-1}AM
\Longrightarrow\text{谱与多项式不变量}
\Longrightarrow\begin{cases}
\Lambda,&\text{特征向量够}\,;\\
J,&\text{用广义特征向量补足}.
\end{cases}
$$

---

## Session 3.5 Singular value decomposition

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节要回答：一个任意矩阵 $A\in\mathbb R^{m\times n}$ 如何在输入空间与输出空间分别选择最佳的正交坐标，使它只剩沿互相垂直方向的伸缩？
<!-- bilingual-en:start -->
In this section, we should answer: How does an arbitrary matrix $A\in\mathbb R^{m\times n}$ choose the best orthogonal coordinates in the input space and the best orthogonal coordinates in the output space, so that it only has to be stretched in the vertical direction to each other?
<!-- bilingual-en:end -->

前置知识：谱定理、四个基本子空间、$A^TA$ 的半正定性。尺寸必须始终检查：
<!-- bilingual-en:start -->
Prerequisites: the spectral theorem, the four fundamental subspaces, and positive semidefiniteness of $A^TA$. Dimensions must always be checked:
<!-- bilingual-en:end -->

$$
A_{m\times n}=U_{m\times m}\Sigma_{m\times n}V^T_{n\times n}.
$$

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S05_Lecture_Lecture_29_Singular_Value_Decomposition.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S05_Recitation_Problem_Solving_Computing_the_Singular_Value_Decomposition.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.5prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.5sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:start -->
Data portal: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S05_Lecture_Lecture_29_Singular_Value_Decomposition.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S05_Recitation_Problem_Solving_Computing_the_Singular_Value_Decomposition.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.5prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.5sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:end -->

关联卡片：[[奇异值分解与低秩近似#SVD 的三层结构|奇异值分解]]、[[对称矩阵与正定二次型#对称矩阵与谱定理|谱分解]]、[[正交投影与最小二乘#正交补与最近点|正交性]]、[[线性方程组与四个基本子空间#基、维数与秩|矩阵秩]]。
<!-- bilingual-en:start -->
Associated Cards: [[奇异值分解与低秩近似#SVD 的三层结构|singular value decomposition]], [[对称矩阵与正定二次型#对称矩阵与谱定理|spectral decomposition]], [[正交投影与最小二乘#正交补与最近点|orthogonal]], [[线性方程组与四个基本子空间#基、维数与秩|matrix rank]].
<!-- bilingual-en:end -->

### 3.5.1 SVD 定理与几何意义
<!-- bilingual-en:start -->
*3.5.1 SVD Theorem and Geometric Significance*
<!-- bilingual-en:end -->

> [!theorem] 奇异值分解（SVD）
> 对每个 $A\in\mathbb R^{m\times n}$，存在正交矩阵 $U\in\mathbb R^{m\times m}$、$V\in\mathbb R^{n\times n}$，以及仅在主对角线上可能非零的 $\Sigma\in\mathbb R^{m\times n}$，使
> $$
> A=U\Sigma V^T.
> $$
> 非零对角元按
> $$
> \sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0
> $$
> 排列，称为[[奇异值分解与低秩近似#SVD 的三层结构|奇异值（singular value）]]；$r=\operatorname{rank}(A)$。
> <!-- bilingual-en:start -->
> For each $A\in\mathbb R^{m\times n}$, there are orthogonal matrices $U\in\mathbb R^{m\times m}$, $V\in\mathbb R^{n\times n}$, and $\Sigma\in\mathbb R^{m\times n}$ that may be non-zero only on the major diagonal, such that
> non-zero diagonal primitives
> Arrange, known as [[奇异值分解与低秩近似#SVD 的三层结构|singular value]];$r=\operatorname{rank}(A)$.
> <!-- bilingual-en:end -->

等价地，若 $v_i$ 是 $V$ 的第 $i$ 列、$u_i$ 是 $U$ 的第 $i$ 列，则
<!-- bilingual-en:start -->
Equivalently, if $v_i$ is the $i$ column of $V$ and $u_i$ is the $i$ column of $U$
<!-- bilingual-en:end -->

$$
Av_i=\sigma_i u_i\quad(1\le i\le r),
$$

而 $Av_i=0$（$i>r$）。因此 $V^T$ 先把输入转到右奇异向量坐标，$\Sigma$ 沿正交坐标轴伸缩或压扁，$U$ 再把结果旋转/反射到输出空间。
<!-- bilingual-en:start -->
$Av_i=0$ ($i>r$).  So $V^T$ turns the input to the right singular vector coordinate, $\Sigma$ stretches or flattens along the orthogonal coordinate axis, and $U$ rotates/reflects the result to the output space.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-svd-geometry.png|860]]

单位球经 $A$ 映成椭球：轴方向为 $u_i$，半轴长度为 $\sigma_i$；零奇异值把相应输入方向压到零。
<!-- bilingual-en:start -->
The unit sphere is mapped into an ellipsoid by $A$: the axis direction is $u_i$, the semiaxis length is $\sigma_i$; and the corresponding input direction is pressed to zero by zero singular value.
<!-- bilingual-en:end -->

### 3.5.2 [[奇异值分解与低秩近似#SVD 的三层结构|SVD 的课程级存在性证明]]
<!-- bilingual-en:start -->
*3.5.2 [[奇异值分解与低秩近似#SVD 的三层结构|Proof of the Course-level Existence of SVD]]*
<!-- bilingual-en:end -->

**目标。** 从已经证明的实对称谱定理构造 $U,\Sigma,V$。
<!-- bilingual-en:start -->
**Objective.**Construct a $U,\Sigma,V$ from the proved real symmetric spectral theorem.
<!-- bilingual-en:end -->

**第一步：对角化 $A^TA$。**
<!-- bilingual-en:start -->
**Step 1: Diagonalize $A^TA$.**
<!-- bilingual-en:end -->

$A^TA\in\mathbb R^{n\times n}$ 对称半正定，因为
<!-- bilingual-en:start -->
$A^TA\in\mathbb R^{n\times n}$ symmetric semipositive definite because
<!-- bilingual-en:end -->

$$
x^TA^TAx=\|Ax\|^2\ge0.
$$

谱定理给出标准正交特征基 $v_1,\ldots,v_n$：
<!-- bilingual-en:start -->
The spectral theorem gives the orthonormal eigenbasis $v_1,\ldots,v_n$:
<!-- bilingual-en:end -->

$$
A^TAv_i=\lambda_i v_i,\qquad \lambda_i\ge0.
$$

按 $\lambda_1\ge\cdots\ge\lambda_n$ 排列，定义
<!-- bilingual-en:start -->
$\lambda_1\ge\cdots\ge\lambda_n$, defining
<!-- bilingual-en:end -->

$$
\sigma_i=\sqrt{\lambda_i}\ge0.
$$

**第二步：由右奇异向量构造左奇异向量。**
<!-- bilingual-en:start -->
**Step 2: The left singular vector is constructed from the right singular vector.**
<!-- bilingual-en:end -->

对 $\sigma_i>0$，定义
<!-- bilingual-en:start -->
for $\sigma_i>0$, defining
<!-- bilingual-en:end -->

$$
u_i=\frac{Av_i}{\sigma_i}.
$$

验证长度：
<!-- bilingual-en:start -->
Validate length:
<!-- bilingual-en:end -->

$$
\|u_i\|^2
=\frac{v_i^TA^TAv_i}{\sigma_i^2}
=\frac{\lambda_i v_i^Tv_i}{\lambda_i}=1.
$$

验证不同 $u_i$ 正交：若 $i\ne j$，
<!-- bilingual-en:start -->
Verify different $u_i$ orthogonality: If $i\ne j$,
<!-- bilingual-en:end -->

$$
u_i^Tu_j
=\frac{v_i^TA^TAv_j}{\sigma_i\sigma_j}
=\frac{\lambda_jv_i^Tv_j}{\sigma_i\sigma_j}=0.
$$

因此非零奇异值对应的 $u_1,\ldots,u_r$ 是标准正交组，且按定义 $Av_i=\sigma_i u_i$。
<!-- bilingual-en:start -->
Therefore the vectors $u_1,\ldots,u_r$ corresponding to nonzero singular values form an orthonormal set, and by definition $Av_i=\sigma_i u_i$.
<!-- bilingual-en:end -->

**第三步：处理零空间与补齐基。**
<!-- bilingual-en:start -->
**Step three: handle the nullspace and complete the bases.**
<!-- bilingual-en:end -->

$\lambda_i=0$ 当且仅当
<!-- bilingual-en:start -->
$\lambda_i=0$ if and only if
<!-- bilingual-en:end -->

$$
0=v_i^TA^TAv_i=\|Av_i\|^2,
$$

即 $Av_i=0$。所以 $v_{r+1},\ldots,v_n$ 构成 $N(A)$ 的标准正交基。再把 $u_1,\ldots,u_r$ 补成 $\mathbb R^m$ 的标准正交基；补入的 $u_{r+1},\ldots,u_m$ 可选为 $N(A^T)$ 的标准正交基。
<!-- bilingual-en:start -->
$Av_i=0$. Therefore $v_{r+1},\ldots,v_n$ form an orthonormal basis of $N(A)$. Complete $u_1,\ldots,u_r$ to an orthonormal basis of $\mathbb R^m$ by choosing $u_{r+1},\ldots,u_m$ as an orthonormal basis of $N(A^T)$.
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
V=[v_1\ \cdots\ v_n],\qquad U=[u_1\ \cdots\ u_m],
$$

并把 $\sigma_i$ 放进 $\Sigma$，便有 $AV=U\Sigma$；右乘 $V^T=V^{-1}$ 得
<!-- bilingual-en:start -->
Placing the $\sigma_i$ in $\Sigma$ gives $AV=U\Sigma$. Multiplying on the right by $V^T=V^{-1}$ yields
<!-- bilingual-en:end -->

$$
A=U\Sigma V^T.
$$

> [!note] 证明层级说明
> 这个证明把 SVD 的存在性完全归约到实对称谱定理。若要从实数公理开始完整证明谱定理，还需补入代数学基本定理或等价的紧致性论证；MIT 18.06SC 把谱定理作为本课程已建立的核心结果。
> <!-- bilingual-en:start -->
> This proof reduces the existence of SVD completely to the real symmetric spectral theorem.  In order to prove the spectral theorem completely from the real number axiom, the basic theorem of algebra or the equivalent compactness argument should be added; MIT 18.06SC has made the spectral theorem the core result of this course.
> <!-- bilingual-en:end -->

### 3.5.3 四个基本子空间在 SVD 中的归位
<!-- bilingual-en:start -->
*3.5.3 The homing of the four basic subspaces in SVD*
<!-- bilingual-en:end -->

若 $\operatorname{rank}(A)=r$，则
<!-- bilingual-en:start -->
If $\operatorname{rank}(A)=r$,
<!-- bilingual-en:end -->

$$
\begin{aligned}
C(A^T)&=\operatorname{span}(v_1,\ldots,v_r),\\
N(A)&=\operatorname{span}(v_{r+1},\ldots,v_n),\\
C(A)&=\operatorname{span}(u_1,\ldots,u_r),\\
N(A^T)&=\operatorname{span}(u_{r+1},\ldots,u_m).
\end{aligned}
$$

其中 $A$ 在行空间与列空间之间给出一一对应：$v_i\mapsto\sigma_i u_i$；在零空间上则全部映到零。这正是伪逆能够“只逆可逆部分”的原因。
<!-- bilingual-en:start -->
Restricted to the row space, $A$ gives a one-to-one correspondence with the column space via $v_i\mapsto\sigma_i u_i$; on the nullspace it maps every vector to zero. This is why the pseudoinverse reverses only the invertible part of the transformation.
<!-- bilingual-en:end -->

SVD 还给出秩一展开：
<!-- bilingual-en:start -->
SVD also gives the rank-one expansion:
<!-- bilingual-en:end -->

$$
A=\sum_{i=1}^r\sigma_i u_iv_i^T.
$$

每一项 $u_iv_i^T$ 先取输入沿 $v_i$ 的分量，再沿 $u_i$ 输出。
<!-- bilingual-en:start -->
Each $u_iv_i^T$ takes a component of the input along the $v_i$ and outputs it along the $u_i$.
<!-- bilingual-en:end -->

### 3.5.4 课件例题：按[[奇异值分解与低秩近似#SVD 的三层结构|标准流程计算 SVD]]
<!-- bilingual-en:start -->
*3.5.4 Course example: Compute an SVD by the [[奇异值分解与低秩近似#SVD 的三层结构|standard workflow]]*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}4&4\\-3&3\end{bmatrix}.
$$

**Step 1：计算 $A^TA$。**
<!-- bilingual-en:start -->
**Step 1: Calculate $A^TA$.**
<!-- bilingual-en:end -->

$$
A^TA=
\begin{bmatrix}4&-3\\4&3\end{bmatrix}
\begin{bmatrix}4&4\\-3&3\end{bmatrix}
=\begin{bmatrix}25&7\\7&25\end{bmatrix}.
$$

其标准正交特征向量与特征值为
<!-- bilingual-en:start -->
The orthonormal eigenvectors and eigenvalues are
<!-- bilingual-en:end -->

$$
v_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},\quad \lambda_1=32;
\qquad
v_2=\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix},\quad \lambda_2=18.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\sigma_1=4\sqrt2,\qquad \sigma_2=3\sqrt2.
$$

**Step 2：用 $u_i=Av_i/\sigma_i$ 固定符号。**
<!-- bilingual-en:start -->
**Step 2: Fix the signs using $u_i=Av_i/\sigma_i$.**
<!-- bilingual-en:end -->

$$
Av_1=\frac1{\sqrt2}\begin{bmatrix}8\\0\end{bmatrix}
=4\sqrt2\begin{bmatrix}1\\0\end{bmatrix},
$$

所以 $u_1=(1,0)^T$。又
<!-- bilingual-en:start -->
So $u_1=(1,0)^T$.
<!-- bilingual-en:end -->

$$
Av_2=\frac1{\sqrt2}\begin{bmatrix}0\\-6\end{bmatrix}
=3\sqrt2\begin{bmatrix}0\\-1\end{bmatrix},
$$

所以 $u_2=(0,-1)^T$。最终
<!-- bilingual-en:start -->
So, $u_2=(0,-1)^T$.  final
<!-- bilingual-en:end -->

$$
U=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\quad
\Sigma=\begin{bmatrix}4\sqrt2&0\\0&3\sqrt2\end{bmatrix},\quad
V=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}.
$$

检查 $U^TU=V^TV=I$，并直接相乘验证 $U\Sigma V^T=A$。
<!-- bilingual-en:start -->
Check $U^TU=V^TV=I$ and verify $U\Sigma V^T=A$ by multiplying it directly.
<!-- bilingual-en:end -->

> [!warning] SVD 的符号配对
> $A^TA$ 的特征向量可乘 $-1$，但 $u_i,v_i$ 的符号必须成对协调，使 $Av_i=\sigma_i u_i$。先随意求 $U$、$V$ 再拼起来，最容易出现一列符号错误。
> <!-- bilingual-en:start -->
> The eigenvectors of $A^TA$ can be multiplied by $-1$, but the symbols of $u_i,v_i$ must be coordinated in pairs so that $Av_i=\sigma_i u_i$.  $U$ and $V$ are the easiest to get in a row of symbol errors.
> <!-- bilingual-en:end -->

### 3.5.5 Recitation 例题：$C=\begin{bmatrix}5&5\\-1&7\end{bmatrix}$
<!-- bilingual-en:start -->
*3.5.5 Recitation Example: $C=\begin{bmatrix}5&5\\-1&7\end{bmatrix}$*
<!-- bilingual-en:end -->

$$
C^TC=\begin{bmatrix}26&18\\18&74\end{bmatrix}.
$$

特征多项式
<!-- bilingual-en:start -->
characteristic polynomial
<!-- bilingual-en:end -->

$$
\lambda^2-100\lambda+1600=(\lambda-20)(\lambda-80),
$$

所以奇异值为 $2\sqrt5,4\sqrt5$。按从大到小排列可取
<!-- bilingual-en:start -->
So the singular value is $2\sqrt5,4\sqrt5$.  Sort Largest to Smallest
<!-- bilingual-en:end -->

$$
v_1=\frac1{\sqrt{10}}\begin{bmatrix}1\\3\end{bmatrix},\quad \sigma_1=4\sqrt5;
\qquad
v_2=\frac1{\sqrt{10}}\begin{bmatrix}-3\\1\end{bmatrix},\quad \sigma_2=2\sqrt5.
$$

由 $u_i=Cv_i/\sigma_i$：
<!-- bilingual-en:start -->
By $u_i=Cv_i/\sigma_i$:
<!-- bilingual-en:end -->

$$
u_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},\qquad
u_2=\frac1{\sqrt2}\begin{bmatrix}-1\\1\end{bmatrix}.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
C=
\frac1{\sqrt2}\begin{bmatrix}1&-1\\1&1\end{bmatrix}
\begin{bmatrix}4\sqrt5&0\\0&2\sqrt5\end{bmatrix}
\frac1{\sqrt{10}}\begin{bmatrix}1&3\\-3&1\end{bmatrix}.
$$

最后一个矩阵是 $V^T$，不是 $V$；这是尺寸和转置检查的重点。
<!-- bilingual-en:start -->
The last matrix is $V^T$, not $V$; this is the focus of the size and transpose checks.
<!-- bilingual-en:end -->

### 3.5.6 Homework：完整题解
<!-- bilingual-en:start -->
*3.5.6 Homework: Complete solutions*
<!-- bilingual-en:end -->

> [!question]- Problem 29.1：验证 Fibonacci 矩阵的奇异值
> 对
> $$
> A=\begin{bmatrix}1&1\\1&0\end{bmatrix},
> $$
> 验证
> $$
> \Sigma=\begin{bmatrix}\frac{1+\sqrt5}{2}&0\\0&\frac{\sqrt5-1}{2}\end{bmatrix}.
> $$
> <!-- bilingual-en:start -->
> Yes
> verification
> <!-- bilingual-en:end -->

> [!success]- 解答
> 因 $A=A^T$，
> $$
> A^TA=A^2=\begin{bmatrix}2&1\\1&1\end{bmatrix}.
> $$
> 特征多项式为
> $$
> \lambda^2-3\lambda+1,
> $$
> 故
> $$
> \lambda_{1,2}=\frac{3\pm\sqrt5}{2}.
> $$
> 奇异值是正平方根。逐项验证：
> $$
> \left(\frac{1+\sqrt5}{2}\right)^2
> =\frac{3+\sqrt5}{2},
> $$
> $$
> \left(\frac{\sqrt5-1}{2}\right)^2
> =\frac{3-\sqrt5}{2}.
> $$
> 二者均非负且按降序排列，因此给出的 $\Sigma$ 正确。
> <!-- bilingual-en:start -->
> Because $A=A^T$,
> The characteristic polynomial is
> therefore
> The singular value is a positive square root.  Validation by Item:
> Both are non-negative and in descending order, so the $\Sigma$ given is correct.
> <!-- bilingual-en:end -->

> [!question]- Problem 29.2：已知 $A$ 的列正交，直接写出 SVD
> 设 $A=[w_1\ \cdots\ w_n]$ 的列彼此正交，且 $\|w_i\|=\sigma_i>0$。计算 $A^TA$，并给出 $U,\Sigma,V$。
> <!-- bilingual-en:start -->
> directly
> Let the rows of $A=[w_1\ \cdots\ w_n]$ be orthogonal to each other and $\|w_i\|=\sigma_i>0$.  The $A^TA$ is calculated and the $U,\Sigma,V$ is given.
> <!-- bilingual-en:end -->

> [!success]- 解答
> $A^TA$ 的第 $(i,j)$ 项是 $w_i^Tw_j$，故
> $$
> A^TA=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_n^2).
> $$
> 因它已经在标准基下对角化，可取
> $$
> \widetilde U_0=\begin{bmatrix}w_1/\sigma_1&\cdots&w_n/\sigma_n\end{bmatrix},
> \qquad
> \Sigma_0=\operatorname{diag}(\sigma_1,\ldots,\sigma_n),
> \qquad V_0=I
> $$
> 给出薄（thin）、未排序的分解 $A=\widetilde U_0\Sigma_0V_0^T$。这里非零奇异方向满足
> $$
> u_i=\frac{w_i}{\sigma_i}.
> $$
> 这些列已经标准正交。若坚持全文“奇异值按降序”约定，令置换矩阵 $P\in\mathbb R^{n\times n}$ 把 $\sigma_i$ 排成降序，则薄 SVD 可写为
> $$
> \widetilde U=\widetilde U_0P,\qquad
> \Sigma=P^T\Sigma_0P,\qquad
> V=P,
> $$
> 从而 $\widetilde U\Sigma V^T=\widetilde U_0\Sigma_0=A$。若需要完整 SVD 且 $m>n$，最后再把 $\widetilde U$ 补成 $m\times m$ 正交矩阵，并把 $\Sigma$ 补成 $m\times n$。若某列长度为零，则它属于零空间，需要把零奇异值单独处理，不能除以 $\sigma_i=0$。
> <!-- bilingual-en:start -->
> Item $(i,j)$ of $A^TA$ is $w_i^Tw_j$, so
> Since it is already diagonal in the standard basis, we obtain
> Give a thin, unsorted decomposition $A=\widetilde U_0\Sigma_0V_0^T$.  Here the non-zero singular direction satisfies
> These columns are orthonormal.  If the convention of "Singular values in descending order" is adhered to in the full text, and permutation matrix $P\in\mathbb R^{n\times n}$ orders $\sigma_i$ in descending order, thin SVD can be written as
> Hence $\widetilde U\Sigma V^T=\widetilde U_0\Sigma_0=A$. If a full SVD is required and $m>n$, extend $\widetilde U$ to an $m\times m$ orthogonal matrix and pad $\Sigma$ to size $m\times n$. If a column is zero, it lies in the nullspace; the corresponding zero singular value must be handled separately and must never be used as a divisor.
> <!-- bilingual-en:end -->

### 3.5.7 边界、反例与易错点
<!-- bilingual-en:start -->
*3.5.7 Boundaries, Counterexamples and common errors*
<!-- bilingual-en:end -->

- 奇异值永远非负；负号应吸收到 $u_i$ 或 $v_i$，不能写进 $\Sigma$。
- $A^TA$ 与 $AA^T$ 的非零特征值相同，都是 $\sigma_i^2$；零特征值个数可能因 $m,n$ 不同而不同。
- 特征值只适用于方阵；SVD 适用于任意矩阵。
- 对称矩阵的奇异值是特征值的绝对值。只有对称正半定时才可直接令 $U=V=Q$、$\Sigma=\Lambda$。
- 重复奇异值对应的奇异向量不唯一，但相应子空间唯一。
<!-- bilingual-en:start -->
- Singular values are never negative; negative signs should be absorbed into $u_i$ or $v_i$ and cannot be written into $\Sigma$.
- $A^TA$ and $AA^T$ have the same non-zero eigenvalues and are both $\sigma_i^2$; the number of zero eigenvalues may vary depending on $m,n$.
- Eigenvalues are only applicable to square matrices; SVD is applicable to arbitrary matrices.
- The singular value of a symmetric matrix is the absolute value of the eigenvalue.  Only symmetric positive half-time can directly make $U=V=Q$, $\Sigma=\Lambda$.
- The singular vector corresponding to the repeated singular value is not unique, but the corresponding subspace is unique.
<!-- bilingual-en:end -->

### 3.5.8 三道自检题
<!-- bilingual-en:start -->
*3.5.8 Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 自检 1（尺寸）
> $A\in\mathbb R^{4\times2}$ 的完整 SVD 中，$U,\Sigma,V$ 的尺寸分别是什么？
> <!-- bilingual-en:start -->
> What is the size of the $U,\Sigma,V$ in the full SVD of the $A\in\mathbb R^{4\times2}$?
> <!-- bilingual-en:end -->

> [!success]- 答案
> $U:4\times4$，$\Sigma:4\times2$，$V:2\times2$。
> <!-- bilingual-en:start -->
> $U:4\times4$,$\Sigma:4\times2$,$V:2\times2$.
> <!-- bilingual-en:end -->

> [!question]- 自检 2（子空间）
> 哪些奇异向量张成 $N(A)$？
> <!-- bilingual-en:start -->
> Which singular vectors are $N(A)$?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 与零奇异值对应的右奇异向量，即 $v_{r+1},\ldots,v_n$。
> <!-- bilingual-en:start -->
> The right singular vector corresponding to zero singular value is $v_{r+1},\ldots,v_n$.
> <!-- bilingual-en:end -->

> [!question]- 自检 3（证明）
> 证明 $\|A\|_2=\max_{\|x\|=1}\|Ax\|=\sigma_1$。
> <!-- bilingual-en:start -->
> Proof $\|A\|_2=\max_{\|x\|=1}\|Ax\|=\sigma_1$.
> <!-- bilingual-en:end -->

> [!success]- 答案
> 写 $x=Vy$，则 $\|y\|=1$，且
> $$
> \|Ax\|^2=\|U\Sigma y\|^2=\sum_i\sigma_i^2y_i^2\le\sigma_1^2.
> $$
> 取 $x=v_1$ 时等号成立。
> <!-- bilingual-en:start -->
> Write $x=Vy$, then $\|y\|=1$, and
> The equal sign holds when $x=v_1$ is taken.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$$
A^TA=V\Sigma^T\Sigma V^T
\Longrightarrow Av_i=\sigma_i u_i
\Longrightarrow A=U\Sigma V^T
\Longrightarrow\text{四子空间与秩一展开}.
$$

---

## Session 3.6 Linear transformations and their matrices

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节要回答：矩阵为什么不是线性变换本身，而是线性变换在选定输入基和输出基下的坐标表示？怎样从基向量的像逐列构造矩阵？
<!-- bilingual-en:start -->
In this section, we will answer: Why is the matrix not a linear transformation per se, but a coordinate representation of the linear transformation in selected input and output bases?  How to construct the matrix from the image of the basis vector column by column?
<!-- bilingual-en:end -->

前置知识：向量空间、基、坐标向量、矩阵乘法。设 $T:V\to W$，$\dim V=n$、$\dim W=m$；选定基后，$[T]$ 的尺寸是 $m\times n$。
<!-- bilingual-en:start -->
Prerequisite knowledge: vector space, basis, coordinate vector, matrix multiplication.  Suppose $T:V\to W$, $\dim V=n$, $\dim W=m$; the size of $[T]$ is $m\times n$ when the basis is selected.
<!-- bilingual-en:end -->

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.6sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S06_Lecture_Lecture_30_Linear_Transformations_and_their_Matrices.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S06_Recitation_Problem_Solving_Linear_Transformations.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.6prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.6sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:start -->
Data portal: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.6sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S06_Lecture_Lecture_30_Linear_Transformations_and_their_Matrices.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S06_Recitation_Problem_Solving_Linear_Transformations.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.6prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.6sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:end -->

关联卡片：[[线性变换与换基#线性变换与矩阵表示|线性变换]]、[[线性方程组与四个基本子空间#基、维数与秩|基]]、[[线性变换与换基#换基与相似|换基]]。
<!-- bilingual-en:start -->
Associated cards: [[线性变换与换基#线性变换与矩阵表示|linear transformations]], [[线性方程组与四个基本子空间#基、维数与秩|bases]], [[线性变换与换基#换基与相似|change of basis]].
<!-- bilingual-en:end -->

### 3.6.1 定义与立刻可推出的性质
<!-- bilingual-en:start -->
*3.6.1 Definition and immediate consequences*
<!-- bilingual-en:end -->

> [!definition] 线性变换
> $T:V\to W$ 称为线性变换（linear transformation），若对所有 $v,w\in V$ 与标量 $c,d\in\mathbb F$，
> $$
> T(cv+dw)=cT(v)+dT(w).
> $$
> <!-- bilingual-en:start -->
> $T:V\to W$ is called linear transformation, if for all $v,w\in V$ and scalar $c,d\in\mathbb F$,
> <!-- bilingual-en:end -->

把 $c=d=1$ 得加法保持。要得到齐次性，在定义中令 $d=0$：
<!-- bilingual-en:start -->
Keep the addition of $c=d=1$.  For homogeneity, let $d=0$:
<!-- bilingual-en:end -->

$$
T(cv)=cT(v)+0T(w)=cT(v).
$$

再令 $c=0$，得到
<!-- bilingual-en:start -->
And then $c=0$, get
<!-- bilingual-en:end -->

$$
T(0)=T(0\cdot v)=0T(v)=0.
$$

所以平移 $T(v)=v+v_0$（$v_0\ne0$）不线性；长度函数 $T(v)=\|v\|$ 也不线性，因为负标量破坏齐次性。
<!-- bilingual-en:start -->
So the translation $T(v)=v+v_0$ ($v_0\ne0$) is non-linear, and the length function $T(v)=\|v\|$ is also non-linear, because the negative scalar breaks the homogeneity.
<!-- bilingual-en:end -->

典型线性变换包括：投影、过原点的旋转/反射、微分、积分（在合适函数空间上）、转置，以及任何固定矩阵乘法 $T(x)=Ax$。
<!-- bilingual-en:start -->
Typical linear transformations include projection, rotation or reflection about the origin, differentiation, integration on a suitable function space, the transpose map, and multiplication by any fixed matrix $T(x)=Ax$.
<!-- bilingual-en:end -->

### 3.6.2 为什么基向量的像决定整个变换
<!-- bilingual-en:start -->
*3.6.2 Why the images of basis vectors determine the whole transformation*
<!-- bilingual-en:end -->

设 $\mathcal B=(v_1,\ldots,v_n)$ 是 $V$ 的基。任意 $v\in V$ 有唯一展开
<!-- bilingual-en:start -->
Let $\mathcal B=(v_1,\ldots,v_n)$ be the basis of $V$.  Any $v\in V$ has a unique expansion
<!-- bilingual-en:end -->

$$
v=c_1v_1+\cdots+c_nv_n.
$$

线性性给出
<!-- bilingual-en:start -->
linearity
<!-- bilingual-en:end -->

$$
T(v)=c_1T(v_1)+\cdots+c_nT(v_n).
$$

因此只需知道 $n$ 个基向量的像，就知道 $T$ 在所有向量上的作用。
<!-- bilingual-en:start -->
So we only need to know the image of the $n$ basis vectors, and we can know the function of $T$ on all vectors.
<!-- bilingual-en:end -->

再选 $W$ 的基 $\mathcal C=(w_1,\ldots,w_m)$，把每个像展开：
<!-- bilingual-en:start -->
Then choose $W$'s basis $\mathcal C=(w_1,\ldots,w_m)$ and expand each image:
<!-- bilingual-en:end -->

$$
T(v_j)=a_{1j}w_1+\cdots+a_{mj}w_m.
$$

定义
<!-- bilingual-en:start -->
defined
<!-- bilingual-en:end -->

$$
[T]_{\mathcal C\leftarrow\mathcal B}
=\begin{bmatrix}
a_{11}&\cdots&a_{1n}\\
\vdots&&\vdots\\
a_{m1}&\cdots&a_{mn}
\end{bmatrix}.
$$

第 $j$ 列正是 $[T(v_j)]_{\mathcal C}$。于是对任意 $v$，
<!-- bilingual-en:start -->
The $j$ column is exactly $[T(v_j)]_{\mathcal C}$.  So any $v$,
<!-- bilingual-en:end -->

$$
[T(v)]_{\mathcal C}
=[T]_{\mathcal C\leftarrow\mathcal B}[v]_{\mathcal B}.
$$

这个箭头记号明确区分输入基与输出基，可以避免“到底乘哪个换基矩阵”的混乱。
<!-- bilingual-en:start -->
This arrow mark clearly distinguishes the input basis from the output basis, and can avoid the confusion of "which basis matrix to multiply".
<!-- bilingual-en:end -->

### 3.6.3 课件例子
<!-- bilingual-en:start -->
*3.6.3 Courseware Examples*
<!-- bilingual-en:end -->

#### 例 1：逆时针旋转 $45^\circ$
<!-- bilingual-en:start -->
*Example 1: Rotate $45^\circ$ CCW*
<!-- bilingual-en:end -->

Lecture 30 从旋转说明“矩阵的列是基向量的像”。标准基下
<!-- bilingual-en:start -->
Lecture 30 explains from rotation that "columns of a matrix are images of a basis vector."  standard basis
<!-- bilingual-en:end -->

$$
R=\frac1{\sqrt2}
\begin{bmatrix}
1&-1\\
1&1
\end{bmatrix}.
$$

第一列是 $R e_1=(1/\sqrt2,1/\sqrt2)^T$，第二列是 $R e_2=(-1/\sqrt2,1/\sqrt2)^T$；它们正是两个标准基向量旋转 $45^\circ$ 后的坐标。
<!-- bilingual-en:start -->
The first column is $R e_1=(1/\sqrt2,1/\sqrt2)^T$, and the second column is $R e_2=(-1/\sqrt2,1/\sqrt2)^T$; they are the coordinates of the two standard basis vectors after they are rotated by $45^\circ$.
<!-- bilingual-en:end -->

#### 补充例：反射
<!-- bilingual-en:start -->
*Supplementary example: Reflection*
<!-- bilingual-en:end -->

标准基下
<!-- bilingual-en:start -->
standard basis
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}1&0\\0&-1\end{bmatrix}
$$

把 $(x,y)$ 映到 $(x,-y)$，即关于 $x$ 轴反射。两个标准基向量分别映到 $e_1$ 与 $-e_2$，恰好形成矩阵两列。
<!-- bilingual-en:start -->
The $(x,y)$ is mapped to the $(x,-y)$, i.e., the reflection on the $x$ axis.  The two standard basis vectors are mapped to $e_1$ and $-e_2$, respectively, forming two columns of the matrix.
<!-- bilingual-en:end -->

#### 例 2：投影到 $45^\circ$ 直线
<!-- bilingual-en:start -->
*Example 2: Projection to a $45^\circ$ Line*
<!-- bilingual-en:end -->

令 $a=(1,1)^T$。标准基下正交投影矩阵是
<!-- bilingual-en:start -->
Get $a=(1,1)^T$.  Orthogonal projection matrix in standard basis is
<!-- bilingual-en:end -->

$$
P=\frac{aa^T}{a^Ta}
=\frac12\begin{bmatrix}1&1\\1&1\end{bmatrix}.
$$

若改用标准正交基
<!-- bilingual-en:start -->
If we use the orthonormal basis instead
<!-- bilingual-en:end -->

$$
q_1=\frac1{\sqrt2}(1,1)^T,\qquad q_2=\frac1{\sqrt2}(1,-1)^T,
$$

同一个变换的矩阵变成
<!-- bilingual-en:start -->
The Matrix of the Same Transformation becomes
<!-- bilingual-en:end -->

$$
\begin{bmatrix}1&0\\0&0\end{bmatrix}.
$$

变换没变，只有坐标变简单了。
<!-- bilingual-en:start -->
The transformation itself has not changed; only its coordinate representation has become simpler.
<!-- bilingual-en:end -->

#### 例 3：微分算子
<!-- bilingual-en:start -->
*Example 3: Differential Operator*
<!-- bilingual-en:end -->

令 $V=P_2$ 为次数不超过 $2$ 的多项式空间，输入基 $(1,x,x^2)$；输出落在 $P_1$，取基 $(1,x)$。微分算子 $D(p)=p'$ 满足
<!-- bilingual-en:start -->
Let $V=P_2$ be the space of polynomials of degree at most $2$, with input basis $(1,x,x^2)$. The output lies in $P_1$, for which we use basis $(1,x)$. The differentiation operator $D(p)=p'$ satisfies
<!-- bilingual-en:end -->

$$
D(1)=0,\qquad D(x)=1,\qquad D(x^2)=2x.
$$

故
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
[D]_{(1,x)\leftarrow(1,x,x^2)}
=\begin{bmatrix}0&1&0\\0&0&2\end{bmatrix}.
$$

尺寸 $2\times3$ 与“输入维数 3、输出维数 2”完全一致。
<!-- bilingual-en:start -->
The dimension $2\times3$ is exactly the same as the "input dimension 3, output dimension 2".
<!-- bilingual-en:end -->

### 3.6.4 复合、逆、核与像
<!-- bilingual-en:start -->
*3.6.4 Composite, inverse, nuclear and imaging*
<!-- bilingual-en:end -->

若 $T_1:V\to W$、$T_2:W\to Z$，并且中间空间使用同一基，则
<!-- bilingual-en:start -->
If $T_1:V\to W$, $T_2:W\to Z$, and the intermediate space uses the same basis, then
<!-- bilingual-en:end -->

$$
[T_2\circ T_1]=[T_2][T_1].
$$

顺序与函数复合相反：先做 $T_1$，矩阵写在右边。
<!-- bilingual-en:start -->
The order is the opposite of function composition: $T_1$ first, with the matrix written on the right.
<!-- bilingual-en:end -->

若 $T:V\to W$ 双射，则 $\dim V=\dim W$，矩阵方阵可逆，而且
<!-- bilingual-en:start -->
If $T:V\to W$ is a bijection, then $\dim V=\dim W$, its matrix representation is square and invertible, and
<!-- bilingual-en:end -->

$$
[T^{-1}]=[T]^{-1}.
$$

在固定基下，
<!-- bilingual-en:start -->
On a fixed basis,
<!-- bilingual-en:end -->

$$
N([T])\leftrightarrow\ker T,\qquad C([T])\leftrightarrow\operatorname{im}T.
$$

秩—零度定理因此可写成
<!-- bilingual-en:start -->
The rank–nullity theorem can therefore be written as
<!-- bilingual-en:end -->

$$
\dim\ker T+\dim\operatorname{im}T=\dim V.
$$

### 3.6.5 Recitation：转置算子在两组基下
<!-- bilingual-en:start -->
*3.6.5 Recitation: The transpose operator in two different bases*
<!-- bilingual-en:end -->

在 $M_{2\times2}(\mathbb R)$ 上定义 $T(A)=A^T$。因为
<!-- bilingual-en:start -->
Define $T(A)=A^T$ on $M_{2\times2}(\mathbb R)$.  because
<!-- bilingual-en:end -->

$$
(A+B)^T=A^T+B^T,\qquad (cA)^T=cA^T,
$$

所以 $T$ 线性；又 $(A^T)^T=A$，故 $T^{-1}=T$。
<!-- bilingual-en:start -->
So $T$ linear; and $(A^T)^T=A$, so $T^{-1}=T$.
<!-- bilingual-en:end -->

标准基
<!-- bilingual-en:start -->
standard basis
<!-- bilingual-en:end -->

$$
E_{11},E_{12},E_{21},E_{22}
$$

下，$T(E_{11})=E_{11}$、$T(E_{12})=E_{21}$、$T(E_{21})=E_{12}$、$T(E_{22})=E_{22}$，所以
<!-- bilingual-en:start -->
$T(E_{11})=E_{11}$, $T(E_{12})=E_{21}$, $T(E_{21})=E_{12}$, $T(E_{22})=E_{22}$, so
<!-- bilingual-en:end -->

$$
[T]_{\text{std}}=
\begin{bmatrix}
1&0&0&0\\
0&0&1&0\\
0&1&0&0\\
0&0&0&1
\end{bmatrix}.
$$

改用三个对称矩阵加一个斜对称矩阵组成的基
<!-- bilingual-en:start -->
Instead, a basis consisting of three symmetric matrices plus a skew symmetric matrix is used
<!-- bilingual-en:end -->

$$
w_1=E_{11},\quad w_2=E_{22},\quad
w_3=E_{12}+E_{21},\quad w_4=E_{12}-E_{21},
$$

则前三个满足 $T(w_i)=w_i$，最后一个满足 $T(w_4)=-w_4$，故
<!-- bilingual-en:start -->
the first three satisfy $T(w_i)=w_i$, the last satisfy $T(w_4)=-w_4$, therefore
<!-- bilingual-en:end -->

$$
[T]_{\mathcal W}=\operatorname{diag}(1,1,1,-1).
$$

这也揭示了 $M_{2\times2}$ 分解为对称子空间与斜对称子空间。
<!-- bilingual-en:start -->
This also reveals that $M_{2\times2}$ decomposes into symmetric and skew-symmetric subspaces.
<!-- bilingual-en:end -->

### 3.6.6 Homework：完整题解
<!-- bilingual-en:start -->
*3.6.6 Homework: Complete solutions*
<!-- bilingual-en:end -->

> [!question]- Problem 30.1：极坐标中的径向放大是否线性？
> $T(r,\theta)=(2r,\theta)$：判断线性，改写为直角坐标，求矩阵。
> <!-- bilingual-en:start -->
> $T(r,\theta)=(2r,\theta)$: Judging the linearity, rewriting it to rectangular coordinates, and calculating the matrix.
> <!-- bilingual-en:end -->

> [!success]- 解答
> 保持方向并把到原点距离乘 $2$，在向量语言中就是 $T(v)=2v$。因此
> $$
> T(v+w)=2(v+w)=T(v)+T(w),\qquad T(cv)=2cv=cT(v),
> $$
> 所以线性。
>
> 直角坐标下 $T(x,y)=(2x,2y)$。长度检查：
> $$
> \|(2x,2y)\|=\sqrt{4x^2+4y^2}=2\sqrt{x^2+y^2}.
> $$
> 标准基下矩阵为
> $$
> [T]=\begin{bmatrix}2&0\\0&2\end{bmatrix}=2I.
> $$
> 极坐标表达看似非线性，是因为 $(r,\theta)$ 不是向量空间的线性坐标；判断线性必须回到向量加法和数乘。
> <!-- bilingual-en:start -->
> Preserving direction while doubling the distance from the origin means $T(v)=2v$ in vector notation. Therefore
> $$
> T(v+w)=T(v)+T(w),\qquad T(cv)=cT(v),
> $$
> so $T$ is linear. In Cartesian coordinates, $T(x,y)=(2x,2y)$, and
> $$
> \|(2x,2y)\|=2\sqrt{x^2+y^2}.
> $$
> In the standard basis, $[T]=2I$. The polar-coordinate formula may look nonlinear only because $(r,\theta)$ are not linear coordinates on the vector space; linearity must be checked using vector addition and scalar multiplication.
> <!-- bilingual-en:end -->

> [!question]- Problem 30.2：给出固定零向量但不线性的变换

> [!success]- 解答
> 例如
> $$
> T(x,y)=(x,y^2).
> $$
> 它满足 $T(0,0)=(0,0)$，但通常
> $$
> T(c(x,y))=(cx,c^2y^2)\ne(cx,cy^2)=cT(x,y).
> $$
> 取 $c=2,y=1$ 即可看出失败。另一个例子是 $T(x,y)=(x,|y|)$：它保持零，却不满足 $T(-v)=-T(v)$。
> <!-- bilingual-en:start -->
> For example
> It satisfies $T(0,0)=(0,0)$, but typically
> Taking $c=2$ and $y=1$ exhibits the failure. Another example is $T(x,y)=(x,|y|)$: it maps zero to zero but does not satisfy $T(-v)=-T(v)$.
> <!-- bilingual-en:end -->

### 3.6.7 边界、反例与易错点
<!-- bilingual-en:start -->
*3.6.7 Boundaries, Counterexamples and common errors*
<!-- bilingual-en:end -->

- $T(0)=0$ 是线性的必要条件，不是充分条件。
- “矩阵的列是基向量的像”只有在输入基、输出基都已说明时才完整。
- 同一个线性变换在不同基下矩阵不同；反过来，同一矩阵若作用在不同坐标约定下，也可能代表不同几何变换。
- 仿射映射 $x\mapsto Ax+b$ 在 $b\ne0$ 时不是线性映射，但可在增广坐标中表示。
<!-- bilingual-en:start -->
- $T(0)=0$ is a necessary and not a sufficient condition for linearity.
- “The columns of a matrix are the images of basis vectors” is complete only after both the input and output bases have been specified.
- The same linear transformation has different matrices in different bases. Conversely, the same numerical matrix can represent different geometric transformations under different coordinate conventions.
- The affine map $x\mapsto Ax+b$ is not linear when $b\ne0$, but can be represented in augmented coordinates.
<!-- bilingual-en:end -->

### 3.6.8 三道自检题
<!-- bilingual-en:start -->
*3.6.8 Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 自检 1（判断）
> $T(f)=f(0)$ 从 $P_2$ 到 $\mathbb R$ 是否线性？标准基下矩阵是什么？
> <!-- bilingual-en:start -->
> Is $T(f)=f(0)$ a linear map from $P_2$ to $\mathbb R$? What is its matrix in the standard basis?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 线性；$T(1)=1,T(x)=T(x^2)=0$，故矩阵是 $[1\ 0\ 0]$。
> <!-- bilingual-en:start -->
> Linear; $T(1)=1,T(x)=T(x^2)=0$, so the matrix is $[1\ 0\ 0]$.
> <!-- bilingual-en:end -->

> [!question]- 自检 2（列构造）
> 若 $T(e_1)=(1,2)^T$、$T(e_2)=(-1,3)^T$，标准基矩阵是什么？
> <!-- bilingual-en:start -->
> If $T(e_1)=(1,2)^T$ and $T(e_2)=(-1,3)^T$, what is the matrix of $T$ in the standard basis?
> <!-- bilingual-en:end -->

> [!success]- 答案
> $$
> [T]=\begin{bmatrix}1&-1\\2&3\end{bmatrix}.
> $$

> [!question]- 自检 3（复合）
> $T_1$ 的矩阵为 $A$，$T_2$ 的矩阵为 $B$，先做 $T_1$ 再做 $T_2$ 的矩阵是什么？
> <!-- bilingual-en:start -->
> The matrix of $T_1$ is $A$, the matrix of $T_2$ is $B$. What is the matrix of $T_1$ and $T_2$?
> <!-- bilingual-en:end -->

> [!success]- 答案
> $BA$，前提是 $A$ 的输出坐标基与 $B$ 的输入坐标基一致。
> <!-- bilingual-en:start -->
> $BA$, provided that the output coordinate of $A$ is consistent with the input coordinate of $B$.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$$
\text{线性}\Longrightarrow\text{基向量的像决定全部}
\Longrightarrow [T]_{\mathcal C\leftarrow\mathcal B}
\Longrightarrow\text{矩阵乘法表示复合}.
$$

---

## Session 3.7 Change of basis and image compression

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节要回答：同一个向量如何在两组基之间换坐标？同一个线性算子的矩阵为何通过相似变换改变？为什么换到 Fourier、wavelet 或 SVD 基后，图像可以只保留少量系数？
<!-- bilingual-en:start -->
This section asks how to convert the coordinates of one vector between two bases, why the matrices of the same linear operator are related by similarity, and why images may have only a few significant coefficients in Fourier, wavelet, or SVD bases.
<!-- bilingual-en:end -->

前置知识：基与坐标、线性变换、相似矩阵、正交矩阵、SVD。以下先在 $n$ 维空间讨论方阵换基，再说明压缩中的高维向量和低秩矩阵。
<!-- bilingual-en:start -->
Prerequisites: bases and coordinates, linear transformations, similarity, orthogonal matrices, and SVD. This section first studies change-of-basis matrices in an $n$-dimensional space, then explains how basis sparsity and low-rank matrix structure support compression.
<!-- bilingual-en:end -->

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.7sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S07_Lecture_Lecture_31_Change_of_Basis_Image_Compression.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S07_Recitation_Problem_Solving_Change_of_Basis.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.7prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.7sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:start -->
Data portal: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.7sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S07_Lecture_Lecture_31_Change_of_Basis_Image_Compression.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S07_Recitation_Problem_Solving_Change_of_Basis.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.7prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.7sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:end -->

关联卡片：[[线性变换与换基#换基与相似|换基]]、[[线性变换与换基#换基与相似|相似矩阵]]、[[奇异值分解与低秩近似#SVD 的三层结构|奇异值分解]]、[[特征值、对角化与线性动力系统#对角化与矩阵幂|Fourier 展开]]。
<!-- bilingual-en:start -->
Associated cards: [[线性变换与换基#换基与相似|change of basis]], [[线性变换与换基#换基与相似|similarity transformations]], [[奇异值分解与低秩近似#SVD 的三层结构|singular value decomposition]], [[特征值、对角化与线性动力系统#对角化与矩阵幂|diagonalization]].
<!-- bilingual-en:end -->

### 3.7.1 向量换基：先说明矩阵的方向
<!-- bilingual-en:start -->
*3.7.1 Changing coordinates of a vector: state the matrix direction first*
<!-- bilingual-en:end -->

设旧基为 $\mathcal E=(e_1,\ldots,e_n)$，新基为 $\mathcal W=(w_1,\ldots,w_n)$。把新基向量用旧基坐标作列，得到
<!-- bilingual-en:start -->
Let the old basis be $\mathcal E=(e_1,\ldots,e_n)$ and the new basis be $\mathcal W=(w_1,\ldots,w_n)$. Placing the old-coordinate vectors of the new basis vectors in columns gives
<!-- bilingual-en:end -->

$$
W=\begin{bmatrix}[w_1]_{\mathcal E}&\cdots&[w_n]_{\mathcal E}\end{bmatrix}.
$$

若 $c=[x]_{\mathcal W}$，则
<!-- bilingual-en:start -->
If $c=[x]_{\mathcal W}$,
<!-- bilingual-en:end -->

$$
x=c_1w_1+\cdots+c_nw_n,
$$

所以旧基坐标满足
<!-- bilingual-en:start -->
So the old coordinates are satisfied
<!-- bilingual-en:end -->

$$
[x]_{\mathcal E}=W[x]_{\mathcal W}. \tag{1}
$$

反向为
<!-- bilingual-en:start -->
Reverse To
<!-- bilingual-en:end -->

$$
[x]_{\mathcal W}=W^{-1}[x]_{\mathcal E}. \tag{2}
$$

式 (1)–(2) 对任意两组基都成立；但仅知道几何基 $\mathcal W$ 标准正交，并不能在任意旧基 $\mathcal E$ 的坐标中推出 $W^{-1}=W^T$。若 $\mathcal E$ 与 $\mathcal W$ 都是在同一个内积下的标准正交基，则坐标矩阵 $W$ 为 orthogonal（复数域为 unitary），此时
<!-- bilingual-en:start -->
Formulas (1)–(2) hold for any two bases. Knowing only that the geometric basis $\mathcal W$ is orthonormal does not imply $W^{-1}=W^T$ when its coordinates are expressed in an arbitrary old basis $\mathcal E$. If both $\mathcal E$ and $\mathcal W$ are orthonormal with respect to the same inner product, then the coordinate matrix $W$ is orthogonal (unitary over the complex field), and in that case
<!-- bilingual-en:end -->

$$
W^{-1}=W^T\quad\text{（复数域为 }W^{-1}=W^*\text{）}.
$$

不依赖旧坐标选择的内在说法是：只要 $\mathcal W$ 本身为标准正交基，向量的 $\mathcal W$-坐标就由
<!-- bilingual-en:start -->
The intrinsic statement, independent of the old coordinate system, is that whenever $\mathcal W$ itself is an orthonormal basis, the $\mathcal W$-coordinates of a vector are determined by
<!-- bilingual-en:end -->

$$
c_i=\langle w_i,x\rangle
$$

给出；在标准实坐标中是 $c_i=w_i^Tx$，在标准复坐标中是 $c_i=w_i^*x$。
<!-- bilingual-en:start -->
Given; $c_i=w_i^Tx$ in standard real coordinates and $c_i=w_i^*x$ in standard complex coordinates.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-change-of-basis.png|820]]

> [!warning] 主动变换与被动换基
> $Wx$ 若把 $W$ 看作一个线性算子，是主动地移动向量；式 (1) 中 $W$ 是坐标翻译器，几何向量并未改变。两者数值公式可能相同，含义不同。
> <!-- bilingual-en:start -->
> If $Wx$ regards $W$ as a linear operator, it is an active moving vector; in formula (1), $W$ is a coordinate translator, and the geometric vector does not change.  The two may have the same numerical formula but different meanings.
> <!-- bilingual-en:end -->

### 3.7.2 线性算子的换基公式
<!-- bilingual-en:start -->
*3.7.2 Basis Conversion Formula for Linear Operators*
<!-- bilingual-en:end -->

设同一个 $T:V\to V$ 在旧基 $\mathcal E$ 下矩阵为 $A$，在新基 $\mathcal W$ 下矩阵为 $B$。从新坐标 $c=[x]_{\mathcal W}$ 出发：
<!-- bilingual-en:start -->
Suppose that the same $T:V\to V$ matrix is $A$ in the old basis $\mathcal E$ and $B$ in the new basis $\mathcal W$.  From the new coordinates $c=[x]_{\mathcal W}$:
<!-- bilingual-en:end -->

1. 用 $Wc$ 转为旧坐标；
2. 用 $A$ 作用，得到 $AWc$；
3. 用 $W^{-1}$ 转回新坐标。
<!-- bilingual-en:start -->
1. Use $Wc$ to change to the old coordinates;
2. $AWc$ was obtained by the action of $A$;
3. Use $W^{-1}$ to return to the new coordinates.
<!-- bilingual-en:end -->

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
Bc=W^{-1}AWc
$$

对所有 $c$ 成立，即
<!-- bilingual-en:start -->
for all $c$, that is,
<!-- bilingual-en:end -->

$$
B=W^{-1}AW.
$$

这就是相似变换的坐标意义。若 $W$ 的列是 $A$ 的特征向量，则 $B=\Lambda$；若列是广义特征向量，则 $B=J$。
<!-- bilingual-en:start -->
This is the coordinate meaning of similarity transformation.  $B=\Lambda$ if the column of $W$ is the eigenvector of $A$, and $B=J$ if the column is the generalized eigenvector.
<!-- bilingual-en:end -->

### 3.7.3 图像为什么是向量，压缩为什么是换基
<!-- bilingual-en:start -->
*3.7.3 Why images are vectors and compression is a change of basis*
<!-- bilingual-en:end -->

一幅 $512\times512$ 灰度图可以看成 $\mathbb R^{512^2}$ 中的向量：每个标准基向量只改变一个像素。标准基适合记录，却不适合压缩，因为自然图像的相邻像素往往高度相关。
<!-- bilingual-en:start -->
A $512\times512$ grayscale image can be viewed as a vector in $\mathbb R^{512^2}$: each standard basis vector changes only one pixel.  The standard basis is suitable for recording, but not for compression because the neighboring pixels of natural images are often highly correlated.
<!-- bilingual-en:end -->

换到 Fourier、离散余弦或 wavelet 基后，平滑区域、边缘和局部细节会集中到少数系数：
<!-- bilingual-en:start -->
After switching to the Fourier, discrete cosine, or wavelet basis, smooth regions, edges, and local details are concentrated in a small number of coefficients:
<!-- bilingual-en:end -->

$$
x=Wc,\qquad c=W^{-1}x.
$$

压缩流程是
<!-- bilingual-en:start -->
The compression process is
<!-- bilingual-en:end -->

$$
x\xrightarrow{W^{-1}}c
\xrightarrow{\text{小系数置零/量化}}\widehat c
\xrightarrow{W}\widehat x.
$$

好的压缩基同时满足：
<!-- bilingual-en:start -->
A useful compression basis should satisfy both of the following:
<!-- bilingual-en:end -->

1. $W$ 与 $W^{-1}$ 乘法快；
2. 典型信号的系数稀疏或快速衰减；
3. 丢弃小系数造成的视觉误差可控。
<!-- bilingual-en:start -->
1. $W$ and $W^{-1}$ multiplication fast;
2. Coefficient sparse or fast attenuation of typical signal;
3. The visual error caused by discarding small coefficients can be controlled.
<!-- bilingual-en:end -->

Fourier 基擅长全局频率；Haar wavelet 同时具有尺度和位置局部性。JPEG 实际主要使用分块离散余弦变换，而不是直接使用复 Fourier 矩阵；课程用 Fourier 思想解释其结构。
<!-- bilingual-en:start -->
The Fourier basis is good at global frequency; the Haar wavelet has both scale and location locality.  In practice, JPEG mainly uses the block discrete cosine transform rather than the complex Fourier matrix; the course explains its structure with the idea of Fourier.
<!-- bilingual-en:end -->

### 3.7.4 课程外补充：SVD 的[[奇异值分解与低秩近似#低秩近似与压缩|低秩近似与压缩]]
<!-- bilingual-en:start -->
*3.7.4 Out-of-Course Replenishment: [[奇异值分解与低秩近似#低秩近似与压缩|Low-rank Approximation and Compression]] for SVD*
<!-- bilingual-en:end -->

> [!note] 与 Lecture 31 的边界
> 官方 Lecture 31 的图像压缩主线是 Fourier/Haar 等固定基中的坐标稀疏化。下面的 Eckart–Young 低秩近似是基于 SVD 的自然延伸，用来补充“矩阵图像”的另一种压缩视角，不把它误记为本讲采用的证明或算法。
> <!-- bilingual-en:start -->
> The main line of image compression in the official Lecture 31 is coordinate sparse in a fixed basis such as Fourier/Haar.  The following Eckart-Young low rank approximation is a natural extension of SVD, which is used to complement another compression perspective of "matrix images", without misrepresenting it as the proof or algorithm used in this paper.
> <!-- bilingual-en:end -->

若把灰度图本身看成矩阵 $A\in\mathbb R^{m\times n}$，SVD 给出
<!-- bilingual-en:start -->
If the grayscale image itself is seen as a matrix $A\in\mathbb R^{m\times n}$, SVD gives
<!-- bilingual-en:end -->

$$
A=\sum_{i=1}^r\sigma_i u_iv_i^T.
$$

对 $0\le k<r$，保留前 $k$ 项得到秩至多 $k$ 的近似
<!-- bilingual-en:start -->
For $0\le k<r$, retaining the previous $k$ term yields an approximation of rank up to $k$
<!-- bilingual-en:end -->

$$
A_k=\sum_{i=1}^k\sigma_i u_iv_i^T.
$$

Eckart–Young 定理说明，在所有秩至多 $k$ 的矩阵中，$A_k$ 同时最小化谱范数误差与 Frobenius 范数误差：
<!-- bilingual-en:start -->
Eckart-Young's theorem shows that $A_k$ minimizes both the spectral norm error and the $k$ norm error in all matrices of rank up to Frobenius:
<!-- bilingual-en:end -->

$$
\|A-A_k\|_2=\sigma_{k+1},
$$

$$
\|A-A_k\|_F^2=\sum_{i=k+1}^r\sigma_i^2.
$$

若 $k\ge r$，则 $A_k=A$，上述两种误差都为零。Eckart–Young 的完整极小化证明通常放在后续数值线性代数课程；这里把结论作为拓展而不冒充 Lecture 31 的内容。存储 $A_k$ 约需 $k(m+n+1)$ 个数；只有当它显著小于 $mn$ 时才真正节省空间。
<!-- bilingual-en:start -->
If $k\ge r$, $A_k=A$, both errors are zero.  Complete minimization proofs for Eckart-Young are usually presented in a subsequent course on numerical linear algebra; here the conclusion is taken as an extension without pretending to be the content of Lecture 31.  It takes about $k(m+n+1)$ to store $A_k$; it only really saves space if it is significantly smaller than $mn$.
<!-- bilingual-en:end -->

### 3.7.5 Recitation：插值基、换基与微分矩阵
<!-- bilingual-en:start -->
*3.7.5 Recitation: Interpolation, Substitution and Differential Matrices*
<!-- bilingual-en:end -->

在 $P_2$ 中，设 $w_1,w_2,w_3$ 满足
<!-- bilingual-en:start -->
In $P_2$, let $w_1,w_2,w_3$ satisfy
<!-- bilingual-en:end -->

$$
w_j(-1)=\delta_{1j},\qquad w_j(0)=\delta_{2j},\qquad w_j(1)=\delta_{3j}.
$$

也就是说，$w_1,w_2,w_3$ 是节点 $-1,0,1$ 的 Lagrange 基。
<!-- bilingual-en:start -->
That is, $w_1,w_2,w_3$ is the Lagrange basis of the node $-1,0,1$.
<!-- bilingual-en:end -->

#### (a) 不显式求基也能求坐标
<!-- bilingual-en:start -->
*(a) Coordinates can be obtained without explicit basis*
<!-- bilingual-en:end -->

若 $y(x)=-x+5$，则
<!-- bilingual-en:start -->
If $y(x)=-x+5$,
<!-- bilingual-en:end -->

$$
y(-1)=6,\qquad y(0)=5,\qquad y(1)=4.
$$

对 $y=\alpha w_1+\beta w_2+\gamma w_3$ 在三个节点取值，立刻得
<!-- bilingual-en:start -->
For $y=\alpha w_1+\beta w_2+\gamma w_3$ to take the value in three nodes, immediately
<!-- bilingual-en:end -->

$$
[y]_{\mathcal W}=\begin{bmatrix}6\\5\\4\end{bmatrix},
\qquad y=6w_1+5w_2+4w_3.
$$

#### (b) 两个换基矩阵
<!-- bilingual-en:start -->
*(b) Two change-of-basis matrices*
<!-- bilingual-en:end -->

对 $p(x)=a+bx+cx^2$，其新基坐标就是三点函数值：
<!-- bilingual-en:start -->
For $p(x)=a+bx+cx^2$, its new basis coordinate is the function value of three points:
<!-- bilingual-en:end -->

$$
[p]_{\mathcal W}
=\begin{bmatrix}p(-1)\\p(0)\\p(1)\end{bmatrix}
=E\begin{bmatrix}a\\b\\c\end{bmatrix},
$$

其中
<!-- bilingual-en:start -->
where
<!-- bilingual-en:end -->

$$
E=\begin{bmatrix}
1&-1&1\\
1&0&0\\
1&1&1
\end{bmatrix}.
$$

反向矩阵为
<!-- bilingual-en:start -->
The inverse matrix is
<!-- bilingual-en:end -->

$$
E^{-1}=\begin{bmatrix}
0&1&0\\
-\frac12&0&\frac12\\
\frac12&-1&\frac12
\end{bmatrix}.
$$

它的列给出
<!-- bilingual-en:start -->
It's listed
<!-- bilingual-en:end -->

$$
w_1=-\frac12x+\frac12x^2,\quad
w_2=1-x^2,\quad
w_3=\frac12x+\frac12x^2.
$$

#### (c) 微分算子的换基
<!-- bilingual-en:start -->
*(c) Changing basis for the differentiation operator*
<!-- bilingual-en:end -->

在标准基 $(1,x,x^2)$ 下，若输出仍放在 $P_2$ 中，
<!-- bilingual-en:start -->
In the standard basis $(1,x,x^2)$, if the output is still represented in $P_2$,
<!-- bilingual-en:end -->

$$
D_{\mathcal E}=\begin{bmatrix}0&1&0\\0&0&2\\0&0&0\end{bmatrix}.
$$

在 $\mathcal W$ 下：
<!-- bilingual-en:start -->
Under $\mathcal W$:
<!-- bilingual-en:end -->

$$
D_{\mathcal W}=ED_{\mathcal E}E^{-1}
=\begin{bmatrix}
-\frac32&2&-\frac12\\
-\frac12&0&\frac12\\
\frac12&-2&\frac32
\end{bmatrix}.
$$

这个三矩阵乘法恰好对应“新坐标 → 旧坐标 → 微分 → 新坐标”。
<!-- bilingual-en:start -->
This trimatrix multiplication corresponds exactly to "new coordinates→old coordinates→differential→new coordinates".
<!-- bilingual-en:end -->

### 3.7.6 Homework：完整题解
<!-- bilingual-en:start -->
*3.7.6 Homework: Complete solutions*
<!-- bilingual-en:end -->

> [!question]- Problem 31.1：验证 Haar wavelet 基正交并归一化

> [!success]- 解答
> 课程给出的 $\mathbb R^8$ Haar 向量可写为
> $$
> \begin{aligned}
> h_0&=(1,1,1,1,1,1,1,1),\\
> h_1&=(1,1,1,1,-1,-1,-1,-1),\\
> h_2&=(1,1,-1,-1,0,0,0,0),\\
> h_3&=(0,0,0,0,1,1,-1,-1),\\
> h_4&=(1,-1,0,0,0,0,0,0),\\
> h_5&=(0,0,1,-1,0,0,0,0),\\
> h_6&=(0,0,0,0,1,-1,0,0),\\
> h_7&=(0,0,0,0,0,0,1,-1).
> \end{aligned}
> $$
> 不同尺度的向量要么支撑不相交，内积为零；要么一个向量在另一个向量的支撑上拥有相同数量的 $+1$、$-1$，相消为零。因此两两正交。
>
> 长度为
> $$
> \|h_0\|=\|h_1\|=\sqrt8,\quad
> \|h_2\|=\|h_3\|=2,\quad
> \|h_4\|=\cdots=\|h_7\|=\sqrt2.
> $$
> 所以标准正交基是
> $$
> \frac{h_0}{\sqrt8},\ \frac{h_1}{\sqrt8},
> \frac{h_2}{2},\ \frac{h_3}{2},\
> \frac{h_4}{\sqrt2},\ldots,\frac{h_7}{\sqrt2}.
> $$
> 八个非零正交向量自动线性无关，在八维空间中因此构成基。
> <!-- bilingual-en:start -->
> The $\mathbb R^8$ Haar vector given in the course can be written as
> Vectors with different scales either have disjoint support and zero inner product, or one vector has the same number of $+1$, $-1$ on the support of another vector and the cancelation is zero.  So the two of them are orthogonal.
> Length is
> So the orthonormal basis is
> The eight non-zero orthogonal vectors are automatically linearly independent and form a basis in the eight-dimensional space.
> <!-- bilingual-en:end -->

> [!question]- Problem 31.2：为 $M_{2\times2}(\mathbb R)$ 给出两组基，并比较用途

> [!success]- 解答
> 最直接的标准基是
> $$
> E_{11}=\begin{bmatrix}1&0\\0&0\end{bmatrix},\quad
> E_{12}=\begin{bmatrix}0&1\\0&0\end{bmatrix},\quad
> E_{21}=\begin{bmatrix}0&0\\1&0\end{bmatrix},\quad
> E_{22}=\begin{bmatrix}0&0\\0&1\end{bmatrix}.
> $$
> 第二组可选
> $$
> B_1=I,\quad
> B_2=\begin{bmatrix}0&1\\1&0\end{bmatrix},\quad
> B_3=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\quad
> B_4=\begin{bmatrix}0&1\\-1&0\end{bmatrix}.
> $$
> 把矩阵按条目摊平成 $\mathbb R^4$ 向量，可见两组都线性无关，故都是基。
>
> - 描述对角矩阵时，标准基只需 $E_{11},E_{22}$；第二组只需 $B_1,B_3$，两者都方便。
> - 描述上/下三角矩阵时，标准基最直接，只需三个相应基向量。
> - 描述对称矩阵时，第二组更好：对称矩阵恰好是 $\operatorname{span}(B_1,B_2,B_3)$，斜对称分量 $B_4$ 的系数为零。
>
> “哪组基更好”没有绝对答案；标准是目标对象能否用少数、易计算的坐标表示。
> <!-- bilingual-en:start -->
> The first displayed set is the standard basis; the second is an alternative basis. Flattening each $2\times2$ matrix into a vector in $\mathbb R^4$ shows that each set is linearly independent, so both are bases.
> - For diagonal matrices, the standard basis needs only $E_{11},E_{22}$, while the second basis needs only $B_1,B_3$; both are convenient.
> - For upper- or lower-triangular matrices, the standard basis is the most direct because only the three corresponding matrix units are needed.
> - For symmetric matrices, the second basis is better: the symmetric subspace is exactly $\operatorname{span}(B_1,B_2,B_3)$, and the coefficient of the skew-symmetric component $B_4$ is zero.
> There is no universally best basis; the useful basis is the one that represents the objects of interest with few, easily computed coordinates.
> <!-- bilingual-en:end -->

### 3.7.7 边界、反例与易错点
<!-- bilingual-en:start -->
*3.7.7 Boundaries, Counterexamples and common errors*
<!-- bilingual-en:end -->

- 写换基公式前先写清“矩阵的列是哪组基用哪组坐标表示”，不要只背 $W$ 或 $W^{-1}$。
- 若 $W$ 的列是新基在旧基中的坐标，则 $W^{-1}=W^T$ 要求这个坐标矩阵本身正交；例如新旧两组基都标准正交。只说“新基在几何上标准正交”而旧基任意，还不够。
- 阈值置零是有损压缩；可逆换基本身不损失信息。
- SVD 低秩近似对给定矩阵最优，但计算完整 SVD 可能昂贵；JPEG 使用固定快速基而不是每张小块都求全局 SVD。
<!-- bilingual-en:start -->
- Before writing a change-of-basis formula, state the basis in which the columns of the matrix are expressed; do not write only "$W$" or "$W^{-1}$" without this context.
- If the columns of $W$ are the new basis vectors expressed in the old coordinates, then $W^{-1}=W^T$ requires the coordinate matrix itself to be orthogonal—for example, both old and new bases may be orthonormal. It is not enough that the new basis is geometrically orthonormal when the old basis is arbitrary.
- Threshold zeroing is lossy compression; invertible fundamentals do not lose information.
- The SVD low-rank approximation is optimal for a given matrix, but calculating the full SVD can be expensive; JPEG uses fixed fast bases instead of each small block to find the global SVD.
<!-- bilingual-en:end -->

### 3.7.8 三道自检题
<!-- bilingual-en:start -->
*3.7.8 Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 自检 1（方向）
> $W$ 的列是新基向量的旧坐标。已知旧坐标 $x$，怎样求新坐标？
> <!-- bilingual-en:start -->
> The column of $W$ is the old coordinate of the new basis vector.  If the old coordinate $x$ is known, how do we find the new coordinate?
> <!-- bilingual-en:end -->

> [!success]- 答案
> $[x]_{\mathcal W}=W^{-1}[x]_{\mathcal E}$。
> <!-- bilingual-en:start -->
> $[x]_{\mathcal W}=W^{-1}[x]_{\mathcal E}$.
> <!-- bilingual-en:end -->

> [!question]- 自检 2（算子换基）
> 旧基矩阵是 $A$，新基矩阵列组成 $W$。新基下算子矩阵是什么？
> <!-- bilingual-en:start -->
> The operator matrix in the old basis is $A$, and the change-of-basis matrix is $W$. What is the operator matrix in the new basis?
> <!-- bilingual-en:end -->

> [!success]- 答案
> $W^{-1}AW$。
> <!-- bilingual-en:start -->
> $W^{-1}AW$.
> <!-- bilingual-en:end -->

> [!question]- 自检 3（误差）
> 截断 SVD 到秩 $k$ 后，谱范数误差是多少？
> <!-- bilingual-en:start -->
> What is the spectral norm error after truncating SVD to rank $k$?
> <!-- bilingual-en:end -->

> [!success]- 答案
> $\sigma_{k+1}$；若 $k\ge r$，误差为零。
> <!-- bilingual-en:start -->
> $\sigma_{k+1}$; if $k\ge r$, the error is zero.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$$
x=Wc\Longrightarrow c=W^{-1}x
\Longrightarrow B=W^{-1}AW
\Longrightarrow\text{选择稀疏/低秩坐标进行压缩}.
$$

---

## Session 3.8 Left and right inverses and pseudoinverse

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节要回答：长方矩阵何时有左逆或右逆？秩亏时怎样利用 SVD 定义在行空间—列空间之间真正可逆的部分？伪逆为何同时给出最小二乘解和最小范数解？
<!-- bilingual-en:start -->
This section asks: when does a rectangular matrix have a left or right inverse? How does the SVD isolate the invertible correspondence between row space and column space when the matrix is rank-deficient? Why does the pseudoinverse produce both least-squares solutions and minimum-norm solutions?
<!-- bilingual-en:end -->

前置知识：秩、四个基本子空间、最小二乘、SVD。始终设 $A\in\mathbb R^{m\times n}$、$\operatorname{rank}(A)=r$。
<!-- bilingual-en:start -->
Prerequisite knowledge: rank, four basic subspaces, least squares, SVD.  Always set $A\in\mathbb R^{m\times n}$, $\operatorname{rank}(A)=r$.
<!-- bilingual-en:end -->

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S08_Lecture_Lecture_33_Left_and_Right_Inverses_Pseudoinverse.pdf#page=1|Lecture 33 transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S08_Recitation_Problem_Solving_Pseudoinverses.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.8prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.8sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:start -->
Data portal: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S08_Lecture_Lecture_33_Left_and_Right_Inverses_Pseudoinverse.pdf#page=1|Lecture 33 transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S08_Recitation_Problem_Solving_Pseudoinverses.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.8prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.8sol.pdf#page=1|Official solution p.1]]
<!-- bilingual-en:end -->

关联卡片：[[广义逆与最小范数解#左逆、右逆与可逆|左逆]]、[[广义逆与最小范数解#左逆、右逆与可逆|右逆]]、[[广义逆与最小范数解#Moore–Penrose 伪逆|伪逆]]、[[正交投影与最小二乘#最小二乘与正规方程|最小二乘]]、[[正交投影与最小二乘#投影矩阵|正交投影]]。
<!-- bilingual-en:start -->
Associated cards: [[广义逆与最小范数解#左逆、右逆与可逆|left inverse]], [[广义逆与最小范数解#左逆、右逆与可逆|right inverse]], [[广义逆与最小范数解#Moore–Penrose 伪逆|pseudoinverse]], [[正交投影与最小二乘#最小二乘与正规方程|least squares]], and [[正交投影与最小二乘#投影矩阵|orthogonal projection]].
<!-- bilingual-en:end -->

### 3.8.1 左逆与满列秩
<!-- bilingual-en:start -->
*3.8.1 Left inverses and full column rank*
<!-- bilingual-en:end -->

若存在 $L\in\mathbb R^{n\times m}$ 使
<!-- bilingual-en:start -->
If there exists $L\in\mathbb R^{n\times m}$ such that
<!-- bilingual-en:end -->

$$
LA=I_n,
$$

则 $L$ 是 $A$ 的左逆。若 $Ax=0$，左乘 $L$ 得 $x=0$，所以 $N(A)=\{0\}$，即 $r=n$；因此必须 $m\ge n$。
<!-- bilingual-en:start -->
then $L$ is a left inverse of $A$. If $Ax=0$, multiplying on the left by $L$ gives $x=0$, so $N(A)=\{0\}$ and $r=n$. Consequently $m\ge n$.
<!-- bilingual-en:end -->

反之，若 $r=n$，则 $A^TA$ 正定可逆，并且
<!-- bilingual-en:start -->
Conversely, if $r=n$, then $A^TA$ is positive definite invertible, and
<!-- bilingual-en:end -->

$$
L=(A^TA)^{-1}A^T
$$

满足
<!-- bilingual-en:start -->
satisfied
<!-- bilingual-en:end -->

$$
LA=(A^TA)^{-1}A^TA=I_n.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
A\text{ 有左逆}\iff\operatorname{rank}(A)=n.
$$

此时 $Ax=b$ 至多一个解，但 $b\notin C(A)$ 时无解。$AL=A(A^TA)^{-1}A^T$ 通常不是 $I_m$，而是投影到 $C(A)$。
<!-- bilingual-en:start -->
In this case $Ax=b$ has at most one solution, but it has no solution when $b\notin C(A)$. The product $AL=A(A^TA)^{-1}A^T$ is generally not $I_m$; it is the orthogonal projector onto $C(A)$.
<!-- bilingual-en:end -->

### 3.8.2 右逆与满行秩
<!-- bilingual-en:start -->
*3.8.2 Right inverses and full row rank*
<!-- bilingual-en:end -->

若存在 $R\in\mathbb R^{n\times m}$ 使
<!-- bilingual-en:start -->
If there exists $R\in\mathbb R^{n\times m}$ such that
<!-- bilingual-en:end -->

$$
AR=I_m,
$$

则 $R$ 是 $A$ 的右逆。这说明每个 $b\in\mathbb R^m$ 都等于 $A(Rb)$，故 $C(A)=\mathbb R^m$，即 $r=m$；因此必须 $n\ge m$。
<!-- bilingual-en:start -->
Then $R$ is a right inverse of $A$. Every $b\in\mathbb R^m$ equals $A(Rb)$, so $C(A)=\mathbb R^m$ and $r=m$. Consequently $n\ge m$ is necessary.
<!-- bilingual-en:end -->

反之，若 $r=m$，则 $AA^T$ 正定可逆，且
<!-- bilingual-en:start -->
Conversely, if $r=m$, then $AA^T$ is positive definite invertible, and
<!-- bilingual-en:end -->

$$
R=A^T(AA^T)^{-1}
$$

满足 $AR=I_m$。所以
<!-- bilingual-en:start -->
satisfies $AR=I_m$. Therefore,
<!-- bilingual-en:end -->

$$
A\text{ 有右逆}\iff\operatorname{rank}(A)=m.
$$

此时 $Ax=b$ 对每个 $b$ 都可解，但若 $n>m$，因为 $\dim N(A)=n-m>0$，解不唯一。
<!-- bilingual-en:start -->
Then $Ax=b$ is solvable for every $b$. If $n>m$, however, $\dim N(A)=n-m>0$, so the solution is not unique.
<!-- bilingual-en:end -->

> [!note] 双边逆为何只属于满秩方阵
> 若同一个 $A$ 同时有左逆和右逆，则 $n\le m$ 且 $m\le n$，所以 $m=n=r$；此时左右逆相等并等于通常的 $A^{-1}$。
> <!-- bilingual-en:start -->
> If the same matrix $A$ has both a left inverse and a right inverse, then $n\le m$ and $m\le n$, so $m=n=r$. The left and right inverses then coincide and equal the ordinary inverse $A^{-1}$.
> <!-- bilingual-en:end -->

### 3.8.3 SVD 定义伪逆
<!-- bilingual-en:start -->
*3.8.3 Defining the pseudoinverse through the SVD*
<!-- bilingual-en:end -->

设完整 SVD 为
<!-- bilingual-en:start -->
Set full SVD to
<!-- bilingual-en:end -->

$$
A=U\Sigma V^T,\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_r,0,\ldots)
$$

（按 $m\times n$ 尺寸理解）。定义 $\Sigma^+\in\mathbb R^{n\times m}$：把每个非零 $\sigma_i$ 换成 $1/\sigma_i$，转置矩形形状，零仍保留为零。Moore–Penrose 伪逆为
<!-- bilingual-en:start -->
Interpret $\Sigma$ as an $m\times n$ matrix. Define $\Sigma^+\in\mathbb R^{n\times m}$ by replacing each nonzero singular value $\sigma_i$ with $1/\sigma_i$, transposing the rectangular shape, and leaving every zero entry at zero. The Moore–Penrose pseudoinverse is
<!-- bilingual-en:end -->

$$
A^+=V\Sigma^+U^T.
$$

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-pseudoinverse.png|840]]

它的作用是
<!-- bilingual-en:start -->
Its role is
<!-- bilingual-en:end -->

$$
u_i\mapsto \frac1{\sigma_i}v_i\quad(i\le r),
$$

并把 $N(A^T)$ 中的分量映为零。于是
<!-- bilingual-en:start -->
and map the components in $N(A^T)$ to zero.  therefore
<!-- bilingual-en:end -->

$$
A^+A=V
\begin{bmatrix}I_r&0\\0&0\end{bmatrix}V^T
$$

是投影到 $C(A^T)$（行空间）的正交投影，而
<!-- bilingual-en:start -->
is an orthogonal projection onto $C(A^T)$ (row space), and
<!-- bilingual-en:end -->

$$
AA^+=U
\begin{bmatrix}I_r&0\\0&0\end{bmatrix}U^T
$$

是投影到 $C(A)$ 的正交投影。
<!-- bilingual-en:start -->
is the orthogonal projection projected to the $C(A)$.
<!-- bilingual-en:end -->

#### 为什么行空间到列空间的映射可逆
<!-- bilingual-en:start -->
*Why the Mapping of Row Space to Column Space Is Invertible*
<!-- bilingual-en:end -->

限制映射
<!-- bilingual-en:start -->
limit mapping
<!-- bilingual-en:end -->

$$
A:C(A^T)\to C(A)
$$

是双射。先证明满射：任取 $y\in C(A)$，由列空间定义存在 $z\in\mathbb R^n$ 使 $y=Az$。利用基本子空间的正交直和
<!-- bilingual-en:start -->
This restricted map is a bijection. To prove surjectivity, take any $y\in C(A)$. By definition of the column space, there is some $z\in\mathbb R^n$ such that $y=Az$. Using the orthogonal direct-sum decomposition of the fundamental subspaces,
<!-- bilingual-en:end -->

$$
\mathbb R^n=C(A^T)\oplus N(A),
$$

唯一分解 $z=z_r+z_0$，其中 $z_r\in C(A^T)$、$z_0\in N(A)$。于是
<!-- bilingual-en:start -->
Unique split $z=z_r+z_0$ where $z_r\in C(A^T)$, $z_0\in N(A)$.  therefore
<!-- bilingual-en:end -->

$$
y=Az=A z_r+A z_0=A z_r,
$$

所以 $y$ 确实有一个来自定义域 $C(A^T)$ 的原像；这才完成满射证明。
<!-- bilingual-en:start -->
Thus $y$ has a preimage in the restricted domain $C(A^T)$, completing the proof of surjectivity.
<!-- bilingual-en:end -->

再证明单射：若 $x,y\in C(A^T)$ 且 $Ax=Ay$，则
<!-- bilingual-en:start -->
To prove injectivity, suppose $x,y\in C(A^T)$ and $Ax=Ay$. Then
<!-- bilingual-en:end -->

$$
A(x-y)=0,
$$

所以 $x-y\in N(A)$；同时行空间对子法封闭，故 $x-y\in C(A^T)$。而
<!-- bilingual-en:start -->
So $x-y\in N(A)$;The simultaneous row space is closed to sublaws, so $x-y\in C(A^T)$. And
<!-- bilingual-en:end -->

$$
C(A^T)\perp N(A),
$$

两者交集只有零，因此 $x-y=0$。伪逆就是这个限制映射的逆，并在左零空间上定义为零。
<!-- bilingual-en:start -->
Their intersection contains only the zero vector, so $x-y=0$. The pseudoinverse is the inverse of the restricted map $A:C(A^T)\to C(A)$ and is defined to be zero on the left nullspace $N(A^T)$.
<!-- bilingual-en:end -->

### 3.8.4 Moore–Penrose 四条件
<!-- bilingual-en:start -->
*3.8.4 Moore-Penrose Quad Condition*
<!-- bilingual-en:end -->

$A^+$ 是唯一满足下列四式的矩阵：
<!-- bilingual-en:start -->
$A^+$ is the only matrix that satisfies the following four expressions:
<!-- bilingual-en:end -->

$$
AA^+A=A,\qquad A^+AA^+=A^+,
$$

$$
(AA^+)^T=AA^+,\qquad (A^+A)^T=A^+A.
$$

前两式说明在可逆部分上来回不会改变；后两式保证两个乘积是正交投影，而不是任意斜投影。用 SVD 代入后，四式都归结为对角矩阵 $\Sigma,\Sigma^+$ 的逐项关系。
<!-- bilingual-en:start -->
The first two identities say that moving through the invertible part and back changes nothing. The last two ensure that the products are orthogonal projections rather than arbitrary oblique projections. Substituting the SVD reduces all four identities to entrywise relations between the diagonal matrices $\Sigma$ and $\Sigma^+$.
<!-- bilingual-en:end -->

### 3.8.5 伪逆同时解决两类“最佳解”
<!-- bilingual-en:start -->
*3.8.5 How the pseudoinverse solves two kinds of “best solution”*
<!-- bilingual-en:end -->

对任意 $b\in\mathbb R^m$，令
<!-- bilingual-en:start -->
For any $b\in\mathbb R^m$, let
<!-- bilingual-en:end -->

$$
\hat x=A^+b.
$$

则
<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->

$$
A\hat x=AA^+b=P_{C(A)}b,
$$

所以残差 $b-A\hat x$ 与 $C(A)$ 正交，$\hat x$ 是最小二乘解。若最小二乘解不唯一，所有解相差一个 $z\in N(A)$；而 $\hat x\in C(A^T)$ 且 $C(A^T)\perp N(A)$，故
<!-- bilingual-en:start -->
Thus the residual $b-A\hat x$ is orthogonal to $C(A)$ and $\hat x$ is a least-squares solution. If least-squares solutions are not unique, every other solution has the form $\hat x+z$ with $z\in N(A)$. Since $\hat x\in C(A^T)$ and $C(A^T)\perp N(A)$,
<!-- bilingual-en:end -->

$$
\|\hat x+z\|^2=\|\hat x\|^2+\|z\|^2\ge\|\hat x\|^2.
$$

所以 $A^+b$ 还是所有最小二乘解中范数最小者。
<!-- bilingual-en:start -->
So $A^+b$ is also the least norm of all the least squares solutions.
<!-- bilingual-en:end -->

特殊情况：
<!-- bilingual-en:start -->
Special circumstances:
<!-- bilingual-en:end -->

$$
A^+=(A^TA)^{-1}A^T\quad\text{若 }r=n,
$$

$$
A^+=A^T(AA^T)^{-1}\quad\text{若 }r=m,
$$

$$
A^+=A^{-1}\quad\text{若 }r=m=n.
$$

### 3.8.6 Recitation：$A=[1\ 2]$

$A\in\mathbb R^{1\times2}$，$AA^T=[5]$，所以它满行秩并且
<!-- bilingual-en:start -->
$A\in\mathbb R^{1\times2}$, $AA^T=[5]$, so it's all over the line and
<!-- bilingual-en:end -->

$$
A^+=A^T(AA^T)^{-1}
=\frac15\begin{bmatrix}1\\2\end{bmatrix}.
$$

两个乘积为
<!-- bilingual-en:start -->
The two products are
<!-- bilingual-en:end -->

$$
AA^+=1,
$$

$$
A^+A=\frac15\begin{bmatrix}1&2\\2&4\end{bmatrix}.
$$

后者投影到 $C(A^T)=\operatorname{span}((1,2)^T)$：
<!-- bilingual-en:start -->
The latter projects to the $C(A^T)=\operatorname{span}((1,2)^T)$:
<!-- bilingual-en:end -->

- 若 $x\in N(A)=\operatorname{span}((-2,1)^T)$，则 $A^+Ax=0$；
- 若 $x\in C(A^T)$，则 $A^+Ax=x$。
<!-- bilingual-en:start -->
- If $x\in N(A)=\operatorname{span}((-2,1)^T)$, $A^+Ax=0$;
- If $x\in C(A^T)$, $A^+Ax=x$.
<!-- bilingual-en:end -->

这里右下角必须是 $4/5$；任何写成 $1/5$ 的版本都会使投影矩阵不幂等，是可通过 $P^2=P$ 发现的算术错误。
<!-- bilingual-en:start -->
The lower right corner here must be $4/5$; any version written as $1/5$ will make the projection matrix irequal, an arithmetic error that can be found via $P^2=P$.
<!-- bilingual-en:end -->

### 3.8.7 Homework：完整题解
<!-- bilingual-en:start -->
*3.8.7 Homework: Complete solutions*
<!-- bilingual-en:end -->

> [!question]- Problem 32.1：求一个右逆
> 对
> $$
> A=\begin{bmatrix}1&0&1\\0&1&0\end{bmatrix},
> $$
> 求右逆。
> <!-- bilingual-en:start -->
> Yes
> Find the right inverse.
> <!-- bilingual-en:end -->

> [!success]- 解答
> $A$ 的两行独立，满行秩 $r=m=2$，可用
> $$
> R=A^T(AA^T)^{-1}.
> $$
> 逐步计算
> $$
> A^T=\begin{bmatrix}1&0\\0&1\\1&0\end{bmatrix},\qquad
> AA^T=\begin{bmatrix}2&0\\0&1\end{bmatrix},
> $$
> $$
> (AA^T)^{-1}=\begin{bmatrix}1/2&0\\0&1\end{bmatrix}.
> $$
> 因而
> $$
> R=\begin{bmatrix}1/2&0\\0&1\\1/2&0\end{bmatrix}.
> $$
> 验算：
> $$
> AR=\begin{bmatrix}1&0\\0&1\end{bmatrix}=I_2.
> $$
> 右逆不唯一；例如改变第一列为 $(t,0,1-t)^T$ 仍可得到第一标准基向量。
> <!-- bilingual-en:start -->
> The two rows of $A$ are independent, so $A$ has full row rank: $r=m=2$. Therefore a right inverse exists.
> Solving column by column gives
> hence
> Verification:
> The right inverse is not unique. For example, replacing its first column by $(t,0,1-t)^T$ still maps that column to the first standard basis vector.
> <!-- bilingual-en:end -->

> [!question]- Problem 32.2：秩一方阵的左右逆与伪逆
> 对
> $$
> A=\begin{bmatrix}4&3\\8&6\end{bmatrix},
> $$
> 判断并求适当的逆。
> <!-- bilingual-en:start -->
> For
> determine which inverses exist and compute the appropriate one.
> <!-- bilingual-en:end -->

> [!success]- 解答
> 第二行是第一行的两倍，故 $\operatorname{rank}(A)=1<2$、$\det A=0$。它既非满列秩，也非满行秩，所以没有左逆或右逆。若存在 $AB=I$ 或 $BA=I$，取行列式会得到 $0=1$，矛盾。
>
> 伪逆总存在。把 $A$ 写成秩一外积：
> $$
> A=\begin{bmatrix}1\\2\end{bmatrix}\begin{bmatrix}4&3\end{bmatrix}.
> $$
> 两向量长度分别是 $\sqrt5$、$5$，故唯一非零奇异值为 $5\sqrt5=\sqrt{125}$，并可取
> $$
> u_1=\frac1{\sqrt5}\begin{bmatrix}1\\2\end{bmatrix},\qquad
> v_1=\frac15\begin{bmatrix}4\\3\end{bmatrix}.
> $$
> 因此
> $$
> A^+=\frac1{\sigma_1}v_1u_1^T
> =\frac1{125}\begin{bmatrix}4&8\\3&6\end{bmatrix}.
> $$
> 检查投影：
> $$
> AA^+=\frac15\begin{bmatrix}1&2\\2&4\end{bmatrix}=P_{C(A)},
> $$
> $$
> A^+A=\frac1{25}\begin{bmatrix}16&12\\12&9\end{bmatrix}=P_{C(A^T)}.
> $$
> 两者均对称且幂等，符合 Moore–Penrose 条件。
> <!-- bilingual-en:start -->
> The second row is twice the first, so $\operatorname{rank}(A)=1<2$ and $\det A=0$. The matrix has neither full column rank nor full row rank, so it has neither a left inverse nor a right inverse. If $AB=I$ or $BA=I$ held, taking determinants would give $0=1$, a contradiction.
> The pseudoinverse always exists. Write $A$ as a rank-one outer product.
> The two vectors have norms $\sqrt5$ and $5$, so the only nonzero singular value is $5\sqrt5=\sqrt{125}$, with the corresponding normalized singular vectors shown above.
> Therefore the displayed formula gives $A^+$.
> Checking the two products yields the displayed orthogonal projections.
> Both products are symmetric and idempotent, as required by the Moore–Penrose conditions.
> <!-- bilingual-en:end -->

### 3.8.8 边界、反例与易错点
<!-- bilingual-en:start -->
*3.8.8 Boundaries, Counterexamples and common errors*
<!-- bilingual-en:end -->

- 左逆要求满列秩，右逆要求满行秩；记忆时看单位矩阵尺寸：$LA=I_n$、$AR=I_m$。
- $(A^TA)^{-1}A^T$ 只有在 $A$ 满列秩时才存在，不能拿来定义所有矩阵的伪逆。
- 小奇异值取倒数会放大噪声；理论伪逆存在不代表数值计算稳定。实际常用截断 SVD 或正则化。
- $A^+A$ 与 $AA^+$ 通常不是单位矩阵，而是两个不同空间上的正交投影。
<!-- bilingual-en:start -->
- A left inverse requires full column rank, whereas a right inverse requires full row rank. The dimensions of the identity matrices—$LA=I_n$ and $AR=I_m$—make the distinction easy to remember.
- $(A^TA)^{-1}A^T$ exists only when $A$ has full column rank; it cannot define the pseudoinverse for every matrix.
The results show that the noise is amplified by the reciprocal of small singular value, and the existence of theoretical pseudoinverse does not indicate numerical stability.  In practice, truncation SVD or regularization is commonly used.
- $A^+A$ and $AA^+$ are usually not unit matrices but orthogonal projections on two different spaces.
<!-- bilingual-en:end -->

### 3.8.9 三道自检题
<!-- bilingual-en:start -->
*3.8.9 Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 自检 1（存在性）
> $A\in\mathbb R^{3\times5}$ 能有左逆吗？能有右逆吗？
> <!-- bilingual-en:start -->
> $A\in\mathbb R^{3\times5}$, can you make a left reversal?  Can there be a right reversal?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 不可能有左逆，因为秩至多 $3<5$；若秩为 $3$，则有右逆。
> <!-- bilingual-en:start -->
> There can be no left inversion because the rank is at most $3<5$; if the rank is $3$, there is a right inversion.
> <!-- bilingual-en:end -->

> [!question]- 自检 2（投影）
> $A^+A$ 投影到哪里？其零空间是什么？
> <!-- bilingual-en:start -->
> Where does the $A^+A$ project?  What is its nullspace?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 投影到行空间 $C(A^T)$，零空间为 $N(A)$。
> <!-- bilingual-en:start -->
> Projection to the row space $C(A^T)$, nullspace is $N(A)$.
> <!-- bilingual-en:end -->

> [!question]- 自检 3（最小范数）
> 为什么 $A^+b$ 在所有最小二乘解中范数最小？
> <!-- bilingual-en:start -->
> Why is the norm of $A^+b$ minimum in all least squares solutions?
> <!-- bilingual-en:end -->

> [!success]- 答案
> $A^+b\in C(A^T)$；其他最小二乘解为 $A^+b+z$、$z\in N(A)$。两空间正交，所以平方范数增加 $\|z\|^2$。
> <!-- bilingual-en:start -->
> $A^+b\in C(A^T)$, while every other least-squares solution has the form $A^+b+z$ with $z\in N(A)$. These two subspaces are orthogonal, so the squared norm increases by $\|z\|^2$.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$$
\text{满列秩}\to\text{左逆},\qquad
\text{满行秩}\to\text{右逆},
$$

$$
A=U\Sigma V^T\Longrightarrow A^+=V\Sigma^+U^T
\Longrightarrow\text{两个投影、最小二乘与最小范数}.
$$

---

## Session 3.9 Exam 3 review

### 本节问题、官方范围与资料
<!-- bilingual-en:start -->
*Issues, official scope and information for this section*
<!-- bilingual-en:end -->

Session 3.9 对应 Lecture 32 的 Exam 3 review。官方明确说明：**Exam 3 的主要范围截至 Session 3.5 SVD**；Session 3.6 线性变换、3.7 换基/压缩和 3.8 伪逆主要进入 Final Exam。不过前三单元知识会交叉，因此复习必须能把早先的特征值、微分方程、投影和 Markov 矩阵与本单元结构连接起来。
<!-- bilingual-en:start -->
Session 3.9 corresponds to Lecture 32, the Exam 3 review. The official scope states that **Exam 3 primarily covers material through Session 3.5 on the SVD**; Sessions 3.6 on linear transformations, 3.7 on change of basis and compression, and 3.8 on the pseudoinverse are mainly assessed on the final exam. Because ideas from the first three units interact, the review must connect earlier work on eigenvalues, differential equations, projections, and Markov matrices to the structures in this unit.
<!-- bilingual-en:end -->

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.9sum.pdf#page=1|Review summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S09_Lecture_Exam_3_Review.pdf#page=1|Lecture 32 transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S09_Recitation_Exam_3_Problem_Solving.pdf#page=1|Review recitation p.1]]
<!-- bilingual-en:start -->
Data portal: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.9sum.pdf#page=1|Review summary p.1]] [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S09_Lecture_Exam_3_Review.pdf#page=1|Lecture 32 transcript p.1]] [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S09_Recitation_Exam_3_Problem_Solving.pdf#page=1|Review recitation p.1]]
<!-- bilingual-en:end -->

### 3.9.1 一张表完成题目分流
<!-- bilingual-en:start -->
*3.9.1 A one-table guide for choosing an approach*
<!-- bilingual-en:end -->

| 看到的结构 | 立刻检查 | 首选表示 |
|---|---|---|
| $A=A^T$ | 实特征值、正交特征基 | $A=Q\Lambda Q^T$ |
| $A\succ0$ | $x^TAx$、特征值、主元、顺序主子式 | $Q\Lambda Q^T$ 或 $LDL^T$ |
| $A^T=-A$ | 纯虚或零特征值、长度保持流 | unitary 谱分解、$e^{tA}$ |
| $B=M^{-1}AM$ | 相同谱、Jordan 块、幂与指数 | 相似标准形 |
| 任意 $A_{m\times n}$ | $A^TA$、秩、四子空间 | $A=U\Sigma V^T$ |
| 投影 $P^2=P$ | 特征值 $0,1$ | 正交投影时 $P=P^T$ |
| $u'=Au$ | $e^{tA}$、特征值实部、Jordan 块 | $u(t)=e^{tA}u(0)$ |
<!-- bilingual-en:start -->
|Seen Structure|Check Now|Preferred Representation|
|---|---|---|
| $A=A^T$ | Real eigenvalues and an orthonormal eigenbasis | $A=Q\Lambda Q^T$ |
| $A\succ0$ | $x^TAx$, eigenvalues, pivots, leading principal minors | $Q\Lambda Q^T$ or $LDL^T$ |
| $A^T=-A$ | Purely imaginary or zero eigenvalues; length-preserving flow | Unitary spectral decomposition, $e^{tA}$ |
| $B=M^{-1}AM$ | Same spectrum, Jordan blocks, powers, and exponentials | A canonical form under similarity |
| Any $A_{m\times n}$ | $A^TA$, rank, four subspaces | $A=U\Sigma V^T$ |
| Projection $P^2=P$ | Eigenvalues $0$ and $1$ | For an orthogonal projection, also $P=P^T$ |
| $u'=Au$ | $e^{tA}$, real parts of eigenvalues, and Jordan blocks | $u(t)=e^{tA}u(0)$ |
<!-- bilingual-en:end -->

### 3.9.2 Review Problem 1：斜对称微分方程
<!-- bilingual-en:start -->
*3.9.2 Review Problem 1: A skew-symmetric differential equation*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
u'=Au,\qquad
A=\begin{bmatrix}0&-1&0\\1&0&-1\\0&1&0\end{bmatrix},\qquad A^T=-A.
$$

#### (a) 特征值与解的形态
<!-- bilingual-en:start -->
*(a) Eigenvalues and the form of the solution*
<!-- bilingual-en:end -->

$$
\det(A-\lambda I)=-\lambda^3-2\lambda
=-\lambda(\lambda^2+2),
$$

故
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\lambda_1=0,\qquad \lambda_{2,3}=\pm\sqrt2\,i.
$$

可取特征向量
<!-- bilingual-en:start -->
One possible choice of eigenvectors is
<!-- bilingual-en:end -->

$$
x_1=\begin{bmatrix}1\\0\\1\end{bmatrix},\quad
x_2=\begin{bmatrix}-1\\\sqrt2\,i\\1\end{bmatrix},\quad
x_3=\begin{bmatrix}-1\\-\sqrt2\,i\\1\end{bmatrix}
=\overline{x_2}.
$$

所以复形式通解为
<!-- bilingual-en:start -->
So the general solution of complex form is
<!-- bilingual-en:end -->

$$
u(t)=c_1x_1+c_2e^{\sqrt2it}x_2+c_3e^{-\sqrt2it}x_3.
$$

因 $A$ 实且初值实，取 $c_1\in\mathbb R$、$c_3=\overline{c_2}$，则后两项互为共轭：
<!-- bilingual-en:start -->
Because $A$ is real and the initial value is real, take $c_1\in\mathbb R$, $c_3=\overline{c_2}$, then the latter two are conjugated with each other:
<!-- bilingual-en:end -->

$$
c_2e^{\sqrt2it}x_2+c_3e^{-\sqrt2it}x_3
=2\operatorname{Re}\!\left(c_2e^{\sqrt2it}x_2\right),
$$

所以最终 $u(t)$ 为实。这里特意选择 $x_3=\overline{x_2}$，避免把特征向量额外乘 $-1$ 后遗漏系数中的负号。
<!-- bilingual-en:start -->
Thus the final solution $u(t)$ is real. We deliberately choose $x_3=\overline{x_2}$; multiplying an eigenvector by an extra factor of $-1$ would require a corresponding sign change in its coefficient, which is easy to overlook.
<!-- bilingual-en:end -->

#### (b) 周期与有界性
<!-- bilingual-en:start -->
*(b) Periodicity and boundedness*
<!-- bilingual-en:end -->

非零频率为 $\sqrt2$，因此所有解都满足共同周期
<!-- bilingual-en:start -->
The non-zero frequency is $\sqrt2$, so all solutions satisfy the common period
<!-- bilingual-en:end -->

$$
T=\frac{2\pi}{\sqrt2}=\pi\sqrt2.
$$

只要解含非零振荡分量，这也是它的基本周期；若初值完全落在零特征空间，解为常向量，只能说任意正数都是周期，并不存在最小的正基本周期。
<!-- bilingual-en:start -->
If the initial value of the solution is completely in the zero eigenspace and the solution is a constant vector, it can only be said that any positive number is a period, and there is no minimum positive fundamental period.
<!-- bilingual-en:end -->

这里 $A$ 是实斜对称，亦是[[对称矩阵与正定二次型#对称矩阵与谱定理|normal matrix]]；它可 unitary 对角化，没有非平凡 Jordan 块，所以纯虚谱确实给出有界振荡。不能把这个结论推广到带 Jordan 块的任意纯虚谱矩阵。
<!-- bilingual-en:start -->
Here $A$ is real and skew-symmetric, and is therefore also a [[对称矩阵与正定二次型#对称矩阵与谱定理|normal matrix]]. It is unitarily diagonalizable and has no nontrivial Jordan blocks, so its purely imaginary spectrum does produce bounded oscillations. This conclusion does not extend to arbitrary matrices with purely imaginary spectra and nontrivial Jordan blocks.
<!-- bilingual-en:end -->

#### (c) 正交性与共轭
<!-- bilingual-en:start -->
*(c) Orthogonality and conjugation*
<!-- bilingual-en:end -->

斜 Hermitian 矩阵 $A^*=-A$ 的不同特征值对应特征向量在 Hermitian 内积下正交。计算复向量内积时必须使用 $x_i^*x_j$。对上面选取的向量逐项验算：
<!-- bilingual-en:start -->
For a skew-Hermitian matrix $A^*=-A$, eigenvectors corresponding to distinct eigenvalues are orthogonal under the Hermitian inner product. Inner products of complex vectors must be computed as $x_i^*x_j$. For the vectors chosen above, check each pair directly:
<!-- bilingual-en:end -->

$$
x_1^*x_2=-1+0+1=0,
\qquad
x_1^*x_3=-1+0+1=0,
$$

并且
<!-- bilingual-en:start -->
and
<!-- bilingual-en:end -->

$$
x_2^*x_3
=(-1)(-1)+(-\sqrt2\,i)(-\sqrt2\,i)+1\cdot1
=1-2+1=0.
$$

这完成了 Review Problem 1(c) 要求的显式验证，而不只引用一般定理。
<!-- bilingual-en:start -->
This completes the explicit validation required by Review Problem 1(c), and does not refer only to general theorems.
<!-- bilingual-en:end -->

#### (d) 计算 $e^{tA}$
<!-- bilingual-en:start -->
*(d) Calculation of $e^{tA}$*
<!-- bilingual-en:end -->

若 $A=S\Lambda S^{-1}$，则
<!-- bilingual-en:start -->
If $A=S\Lambda S^{-1}$,
<!-- bilingual-en:end -->

$$
e^{tA}=Se^{t\Lambda}S^{-1},\qquad
e^{t\Lambda}=\operatorname{diag}(1,e^{\sqrt2it},e^{-\sqrt2it}).
$$

另一个结构检查是：
<!-- bilingual-en:start -->
Another structural check is:
<!-- bilingual-en:end -->

$$
\frac{d}{dt}\|u(t)\|^2
=u^T(A^T+A)u=0,
$$

所以流保持长度。
<!-- bilingual-en:start -->
So the flow remains long.
<!-- bilingual-en:end -->

### 3.9.3 Review Problem 2：由谱数据识别矩阵类别
<!-- bilingual-en:start -->
*3.9.3 Review Problem 2: Identifying Matrix Categories from Spectral Data*
<!-- bilingual-en:end -->

已知 $3\times3$ 矩阵 $A$ 的特征值为 $0,c,2$，对应特征向量
<!-- bilingual-en:start -->
The eigenvalue of the known $3\times3$ matrix $A$ is $0,c,2$, corresponding to the eigenvector
<!-- bilingual-en:end -->

$$
x_1=\begin{bmatrix}1\\1\\1\end{bmatrix},\quad
x_2=\begin{bmatrix}1\\-1\\0\end{bmatrix},\quad
x_3=\begin{bmatrix}1\\1\\-2\end{bmatrix}.
$$

三向量两两正交，因而一定独立。
<!-- bilingual-en:start -->
Three vectors are orthogonal in two, so they must be independent.
<!-- bilingual-en:end -->

1. **何时可对角化？** 对所有 $c$ 都可，因为已经给出三个独立特征向量，即使特征值重合也不影响这组基的独立性。
2. **何时对称？** 若 $c\in\mathbb R$，把特征向量归一化组成正交 $Q$，则 $A=Q\operatorname{diag}(0,c,2)Q^T$ 对称。若 $c$ 非实，则不可能是实对称矩阵。
3. **何时正定？** 永不正定，因为固定有零特征值。$c\ge0$ 时是正半定。
4. **是否可能是 Markov 矩阵？** 不可能；Markov 矩阵的谱半径为 $1$，而这里有特征值 $2$。
5. **$P=A/2$ 何时可能是投影？** 投影特征值必须为 $0$ 或 $1$，故 $c/2\in\{0,1\}$，即 $c=0$ 或 $c=2$。又已有正交特征基，所以此时确实为正交投影。
<!-- bilingual-en:start -->
1. **For which values of $c$ is diagonalisation possible?** For every $c$, because three linearly independent eigenvectors are already given. Coincident eigenvalues do not destroy the independence of this eigenbasis.
2. When is**symmetric?**If $c\in\mathbb R$, normalizes the eigenvectors into orthogonal $Q$, then $A=Q\operatorname{diag}(0,c,2)Q^T$ symmetry.  If $c$ is not real, it cannot be a real symmetric matrix.
3. When will**be finalized?**never positive, because zero eigenvalues are fixed.  $c\ge0$ is positive semidefinite.
4. Is**a possible Markov matrix?**Impossible; The spectral radius of the Markov matrix is $1$, and here the eigenvalue $2$.
5. When might**$P=A/2$ be a projection?**The projection eigenvalue must be $0$ or $1$, so $c/2\in\{0,1\}$ is $c=0$ or $c=2$.  There are also orthogonal eigenbases, so it is the orthogonal projection.
<!-- bilingual-en:end -->

### 3.9.4 Review Problem 3：只看 $\Sigma$ 读出结构
<!-- bilingual-en:start -->
*3.9.4 Review Problem 3: Read the structure directly from $\Sigma$*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
If
<!-- bilingual-en:end -->

$$
\Sigma=\begin{bmatrix}3&0\\0&2\end{bmatrix},
$$

且 $U,V$ 都是 $2\times2$ 正交矩阵，则 $A=U\Sigma V^T$ 是可逆 $2\times2$ 矩阵，秩为 $2$。
<!-- bilingual-en:start -->
Because $U$ and $V$ are $2\times2$ orthogonal matrices, $A=U\Sigma V^T$ is an invertible $2\times2$ matrix of rank $2$.
<!-- bilingual-en:end -->

- $\operatorname{diag}(3,-5)$ 不能作为 SVD 的 $\Sigma$，因为奇异值必须非负。
- 若 $\Sigma=\operatorname{diag}(3,0)$，则 $\operatorname{rank}(A)=1$，$\dim N(A)=1$；$V$ 的第二列张成 $N(A)$，$U$ 的第二列张成 $N(A^T)$。
<!-- bilingual-en:start -->
- $\operatorname{diag}(3,-5)$ cannot be a $\Sigma$ for SVD because singular values must be non-negative.
- If $\Sigma=\operatorname{diag}(3,0)$, the second row of $\operatorname{rank}(A)=1$,$\dim N(A)=1$;$V$ is set to $N(A)$, and the second row of $U$ is set to $N(A^T)$.
<!-- bilingual-en:end -->

### 3.9.5 Review Problem 4：矩阵同时对称且正交
<!-- bilingual-en:start -->
*3.9.5 Review Problem 4: Matrices are Symmetric and Orthogonal*
<!-- bilingual-en:end -->

若 $A=A^T$ 且 $A^TA=I$，则 $A^2=I$。因此每个特征值满足
<!-- bilingual-en:start -->
If $A=A^T$ and $A^TA=I$, $A^2=I$.  So each eigenvalue satisfies
<!-- bilingual-en:end -->

$$
\lambda^2=1,
$$

即 $\lambda=\pm1$。由此：
<!-- bilingual-en:start -->
$\lambda=\pm1$.  From this:
<!-- bilingual-en:end -->

- $A$ 未必正定，例如 $\operatorname{diag}(1,-1)$；
- 特征值可以重复；
- $A$ 可正交对角化并一定可逆；
- $P=(A+I)/2$ 是投影，因为
  $$
  P^2=\frac14(A^2+2A+I)=\frac12(A+I)=P,
  $$
  且 $P^T=P$，所以是正交投影。
<!-- bilingual-en:start -->
- $A$ need not be positive definite; for example, $\operatorname{diag}(1,-1)$ is not;
- Eigenvalues can be repeated;
- $A$ can be orthogonally diagonalized and must be invertible;
- $P=(A+I)/2$ is a projection because
  and $P^T=P$, so it's an orthogonal projection.
<!-- bilingual-en:end -->

### 3.9.6 Recitation：投影、旋转与反射的谱
<!-- bilingual-en:start -->
*3.9.6 Recitation: Spectrum for Projection, Rotation and Reflection*
<!-- bilingual-en:end -->

令 $a=(3,4)^T$。
<!-- bilingual-en:start -->
Get $a=(3,4)^T$.
<!-- bilingual-en:end -->

1. 投影
   $$
   P=\frac{aa^T}{a^Ta}=\frac1{25}\begin{bmatrix}9&12\\12&16\end{bmatrix}
   $$
   有特征对：$a\leftrightarrow1$，$(-4,3)^T\leftrightarrow0$。
2. 旋转
   $$
   Q=\begin{bmatrix}0.6&-0.8\\0.8&0.6\end{bmatrix}
   $$
   有特征值 $0.6\pm0.8i$，对应复特征向量可取 $(1,\mp i)^T$。非平凡平面旋转没有实特征方向。
3. 反射 $R=2P-I$ 与 $P$ 有同一组特征向量；特征值由 $\lambda_R=2\lambda_P-1$ 得 $1,-1$。投影方向保持，垂直方向反向。
<!-- bilingual-en:start -->
1. Projection
   Characteristic pairs: $a\leftrightarrow1$, $(-4,3)^T\leftrightarrow0$.
2. Rotate
   There is an eigenvalue $0.6\pm0.8i$, and the corresponding complex eigenvector can be $(1,\mp i)^T$.Non-trivial plane rotation has no real characteristic direction.
3. The reflection $R=2P-I$ and $P$ have the same set of eigenvectors; the eigenvalue is $\lambda_R=2\lambda_P-1$ to $1,-1$.The projection direction is held and the vertical direction is reversed.
<!-- bilingual-en:end -->

### 3.9.7 三道自检题
<!-- bilingual-en:start -->
*3.9.7 Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 自检 1（谱分类）
> 一个实对称矩阵所有特征值都在 $\{0,1\}$，它一定是什么矩阵？
> <!-- bilingual-en:start -->
> All the eigenvalues of a real symmetric matrix are in $\{0,1\}$. What matrix is it?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 正交投影矩阵。谱分解给 $P^2=Q\Lambda^2Q^T=Q\Lambda Q^T=P$。
> <!-- bilingual-en:start -->
> orthogonal projection matrix.  The spectrum is decomposed to $P^2=Q\Lambda^2Q^T=Q\Lambda Q^T=P$.
> <!-- bilingual-en:end -->

> [!question]- 自检 2（SVD）
> 若 $A\in\mathbb R^{5\times3}$ 有两个非零奇异值，四个基本子空间维数是什么？
> <!-- bilingual-en:start -->
> If $A\in\mathbb R^{5\times3}$ has two non-zero singular values, what are the dimensions of the four elementary subspaces?
> <!-- bilingual-en:end -->

> [!success]- 答案
> $\dim C(A)=\dim C(A^T)=2$，$\dim N(A)=3-2=1$，$\dim N(A^T)=5-2=3$。
> <!-- bilingual-en:start -->
> $\dim C(A)=\dim C(A^T)=2$,$\dim N(A)=3-2=1$,$\dim N(A^T)=5-2=3$.
> <!-- bilingual-en:end -->

> [!question]- 自检 3（微分方程）
> “$A$ 的特征值实部都不正，所以 $e^{tA}$ 有界”是否总正确？
> <!-- bilingual-en:start -->
> Is the following claim always true: “Every eigenvalue of $A$ has nonpositive real part, so $e^{tA}$ is bounded”?
> <!-- bilingual-en:end -->

> [!success]- 答案
> 不正确。实部为零的特征值若有非平凡 Jordan 块，会产生 $t,t^2,\ldots$ 的多项式增长。若所有实部严格负则指数衰减；实部为零时还需半单/可对角化条件。
> <!-- bilingual-en:start -->
> No. A nontrivial Jordan block associated with an eigenvalue of zero real part produces polynomial growth in $t,t^2,\ldots$. Strictly negative real parts give exponential decay; eigenvalues on the imaginary axis additionally require semisimplicity (equivalently, no nontrivial Jordan blocks there) for boundedness.
> <!-- bilingual-en:end -->

### Review 知识链
<!-- bilingual-en:start -->
*Review Knowledge Chain*
<!-- bilingual-en:end -->

$$
\text{特殊矩阵结构}
\Longrightarrow\text{谱限制}
\Longrightarrow\text{分解}
\Longrightarrow\text{幂、指数、投影、稳定性与子空间}.
$$

---

## Unit 3 Exam 完整题解
<!-- bilingual-en:start -->
*Complete Unit 3 Exam solutions*
<!-- bilingual-en:end -->

> [!info] 考试材料
> 原题：[[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex3.pdf#page=1|Unit 3 Exam p.1]]；官方答案：[[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex3s.pdf#page=1|Official solutions p.1]]。原题 PDF 的部分文本层乱码，本节按页面视觉内容与官方答案逐项核对。
> <!-- bilingual-en:start -->
> Original exam: [[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex3.pdf#page=1|Unit 3 Exam p.1]]; official solutions: [[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex3s.pdf#page=1|Official solutions p.1]]. Part of the original PDF's text layer is scrambled, so this section cross-checks the visually rendered pages against the official solutions item by item.
> <!-- bilingual-en:end -->

### Exam Problem 1（34 分）：SVD、谱分解与矩阵类别
<!-- bilingual-en:start -->
*Exam Problem 1 (34 points): SVD, spectral decomposition and matrix categories*
<!-- bilingual-en:end -->

> [!question] 题目
> (a) 方阵 $A$ 的全部 $n$ 个奇异值都等于 $1$。它必属于哪些基本矩阵类别：singular、symmetric、orthogonal、positive definite/semidefinite、diagonal？
>
> (b) $H$ 的标准正交列是 $B$ 的特征向量：
> $$
> H=\frac12\begin{bmatrix}
> 1&1&-1&-1\\
> 1&-1&-1&1\\
> 1&1&1&1\\
> 1&-1&1&-1
> \end{bmatrix},\qquad H^{-1}=H^T,
> $$
> $B$ 的特征值依次为 $0,1,2,3$。把 $B$ 写成三个具体矩阵的乘积；再把 $C=(B+I)^{-1}$ 写成三个矩阵的乘积。
>
> (c) 从 (a) 的类别列表中分别判断 $B,C$ 属于哪些类。
> <!-- bilingual-en:start -->
> (a) All the $n$ singular values of the square $A$ are equal to $1$.  What basic matrix categories must it belong to: singular, symmetric, orthogonal, positive definite/semidefinite, diagonal?
> (b) The orthonormal column of $H$ is the eigenvector of $B$:
> The eigenvalue of $B$ is $0,1,2,3$ in turn.  We can write $B$ as the product of three concrete matrices, and write $C=(B+I)^{-1}$ as the product of three matrices.
> (c) Determine which classes $B,C$ belongs to from the list of categories in (a).
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> **(a) 目标：从奇异值判断 $A^TA$。**
>
> SVD 为 $A=U\Sigma V^T$，而全部奇异值为 $1$，所以 $\Sigma=I$。于是
> $$
> A^TA=V\Sigma^TU^TU\Sigma V^T=VII V^T=I.
> $$
> 因而 $A$ 必为 **orthogonal matrix**，并且必可逆，所以不 singular。
>
> 其他类别都不必成立。反例取非平凡旋转
> $$
> \begin{bmatrix}0&-1\\1&0\end{bmatrix};
> $$
> 它奇异值全为 $1$，但不对称、不对角，也不是正定/半正定矩阵。因此唯一必选类别是 orthogonal。
>
> **(b) 目标：使用正交谱分解。**
>
> 令
> $$
> \Lambda=\operatorname{diag}(0,1,2,3).
> $$
> 因 $H$ 的列是按此顺序排列的标准正交特征向量，
> $$
> \boxed{B=H\Lambda H^T}.
> $$
> 这已经是三个题目指定的具体矩阵的乘积。
>
> $B+I$ 与 $B$ 有相同特征向量，特征值分别变成 $1,2,3,4$：
> $$
> B+I=H(\Lambda+I)H^T.
> $$
> 每个特征值都非零，所以可逆；利用 $(H^T)^{-1}=H$，
> $$
> \boxed{
> C=(B+I)^{-1}
> =H\operatorname{diag}\left(1,\frac12,\frac13,\frac14\right)H^T}.
> $$
>
> **(c) 分类。**
>
> $B$ 有正交谱分解且谱为 $0,1,2,3$，所以：
>
> - 有零特征值，故 singular；
> - $B=H\Lambda H^T$，故 symmetric；
> - 所有特征值非负，故 positive semidefinite；
> - 因有零特征值，不 positive definite；
> - 不必且实际并非 orthogonal 或 diagonal。
>
> $C$ 的特征值 $1,1/2,1/3,1/4$ 全正，且 $C=HC_\Lambda H^T$，所以 symmetric positive definite。它可逆，不 singular；特征值模长不全为 $1$，故不 orthogonal；标准基下也不 diagonal。
>
> **验点。** $BC$ 不必是 $I$，因为 $C$ 是 $(B+I)^{-1}$；正确检查是
> $$
> (B+I)C=H(\Lambda+I)(\Lambda+I)^{-1}H^T=I.
> $$
> <!-- bilingual-en:start -->
> **(a) Use the singular values to determine $A^TA$.**
> The SVD is $A=U\Sigma V^T$. Since every singular value equals $1$, $\Sigma=I$, and the displayed calculation gives $A^TA=I$. Therefore $A$ must be orthogonal and hence invertible, so it is not singular.
> None of the other listed properties is forced. A nontrivial rotation is a counterexample: all its singular values are $1$, yet it is neither symmetric nor diagonal, and its quadratic form is neither positive definite nor positive semidefinite. Thus “orthogonal” is the only category that must apply.
> **(b) Use the orthogonal spectral decomposition.**
> Set $\Lambda=\operatorname{diag}(0,1,2,3)$. Since the columns of $H$ are orthonormal eigenvectors in that order, $B=H\Lambda H^T$, already a product of the three requested concrete matrices.
> The matrices $B+I$ and $B$ have the same eigenvectors, while their eigenvalues are shifted to $1,2,3,4$. All are nonzero, so $B+I$ is invertible; using $(H^T)^{-1}=H$ gives the displayed formula for $C=(B+I)^{-1}$.
> **(c) Classify $B$ and $C$.**
> The matrix $B$ has orthogonal spectral decomposition and spectrum $0,1,2,3$. Hence it is singular, symmetric, and positive semidefinite, but not positive definite. It need not—and here does not—belong to the orthogonal or diagonal categories.
> The eigenvalues of $C$ are $1,1/2,1/3,1/4$, and $C=HC_\Lambda H^T$, so $C$ is symmetric positive definite and invertible. It is not orthogonal because its eigenvalues do not all have modulus $1$, and it is not diagonal in the standard basis.
> **Check.** We need not have $BC=I$, because $C=(B+I)^{-1}$. The correct identity is the displayed equation $(B+I)C=I$.
> <!-- bilingual-en:end -->

### Exam Problem 2（33 分）：对角化、矩阵幂与 $A^TA$
<!-- bilingual-en:start -->
*Exam Problem 2 (33 points): Diagonalization, Matrix Power and $A^TA$*
<!-- bilingual-en:end -->

> [!question] 题目
> 设
> $$
> A=\begin{bmatrix}-1&2&4\\0&0&5\\0&0&1\end{bmatrix}.
> $$
> (a) 求三个特征值和一个特征向量矩阵 $S$。
>
> (b) 解释为什么 $A^{1001}=A$；$A^{1000}$ 是否等于 $I$？写出 $e^{At}$ 的三个对角元。
>
> (c) 已知
> $$
> A^TA=\begin{bmatrix}1&-2&-4\\-2&4&8\\-4&8&42\end{bmatrix}.
> $$
> 不直接求根，说明它有几个正、零、负特征值；它与 $A$ 是否有相同特征向量？
> <!-- bilingual-en:start -->
> Let $A$ be the matrix displayed above.
> (a) Find its three eigenvalues and an eigenvector matrix $S$.
> (b) Explain why $A^{1001}=A$. Is $A^{1000}=I$? State the three diagonal entries of $e^{At}$.
> (c) For the displayed matrix $A^TA$, determine—without solving its characteristic equation—the numbers of positive, zero, and negative eigenvalues, and decide whether it has the same eigenvectors as $A$.
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> **(a) 求谱与 $S$。**
>
> $A$ 为上三角矩阵，特征值就是对角元：
> $$
> \lambda_1=-1,\qquad \lambda_2=0,\qquad \lambda_3=1.
> $$
>
> 对 $\lambda=-1$，解 $(A+I)x=0$，可取
> $$
> x_1=\begin{bmatrix}1\\0\\0\end{bmatrix}.
> $$
>
> 对 $\lambda=0$，解 $Ax=0$。第三行给 $x_3=0$，第一行给 $-x_1+2x_2=0$，可取
> $$
> x_2=\begin{bmatrix}2\\1\\0\end{bmatrix}.
> $$
>
> 对 $\lambda=1$，解 $(A-I)x=0$。第二行给 $-x_2+5x_3=0$，故 $x_2=5x_3$；第一行给 $-2x_1+2x_2+4x_3=0$，故 $x_1=7x_3$。取 $x_3=1$：
> $$
> x_3=\begin{bmatrix}7\\5\\1\end{bmatrix}.
> $$
>
> 三个特征值互异，特征向量独立，所以
> $$
> S=\begin{bmatrix}1&2&7\\0&1&5\\0&0&1\end{bmatrix},\qquad
> A=S\operatorname{diag}(-1,0,1)S^{-1}.
> $$
>
> **(b) 矩阵幂与指数。**
>
> 对任何正整数 $k$，
> $$
> A^k=S\operatorname{diag}((-1)^k,0^k,1^k)S^{-1}.
> $$
> 当 $k=1001$ 为奇数时，对角阵仍为 $\operatorname{diag}(-1,0,1)$，所以
> $$
> \boxed{A^{1001}=A}.
> $$
> 当 $k=1000$ 时，
> $$
> A^{1000}=S\operatorname{diag}(1,0,1)S^{-1}\ne I,
> $$
> 因为中间特征方向仍被映为零；更快的检查是 $A$ singular，所以 $A^{1000}$ 也 singular，不可能等于可逆的 $I$。
>
> 矩阵指数为
> $$
> e^{At}=S\operatorname{diag}(e^{-t},1,e^t)S^{-1}.
> $$
> 由于 $S$、$S^{-1}$ 都是单位对角的上三角矩阵，$e^{At}$ 的三个对角元为
> $$
> \boxed{e^{-t},\ 1,\ e^t}.
> $$
> 也可由上三角矩阵幂级数逐项看出。
>
> **(c) $A^TA$ 的惯性与特征向量。**
>
> 对任意 $x$，$x^TA^TAx=\|Ax\|^2\ge0$，故 $A^TA$ 没有负特征值。$A$ 的前两列相关，第三列不在它们的张成中，所以 $\operatorname{rank}(A)=2$。又
> $$
> \operatorname{rank}(A^TA)=\operatorname{rank}(A)=2.
> $$
> 因而 $3\times3$ 的 $A^TA$ 有两个正特征值、一个零特征值、零个负特征值。
>
> 它不与 $A$ 共享整组特征向量。例如 $e_1$ 是 $A$ 对应 $-1$ 的特征向量，但
> $$
> A^TAe_1=\begin{bmatrix}1\\-2\\-4\end{bmatrix}
> $$
> 不是 $e_1$ 的倍数。一般而言，$A$ 的特征向量与 $A^TA$ 的右奇异向量是不同概念；只有 normal 等特殊情形才可能一致。
> <!-- bilingual-en:start -->
> **(a) Spectrum and $S$.**
> Since $A$ is upper triangular, its eigenvalues are its diagonal entries: $-1,0,1$.
> For $\lambda=-1$, solving $(A+I)x=0$ gives the choice $x_1=(1,0,0)^T$.
> For $\lambda=0$, solving $Ax=0$ gives $x_3=0$ and $x_1=2x_2$, so take $x_2=(2,1,0)^T$.
> For $\lambda=1$, solving $(A-I)x=0$ gives $x_2=5x_3$ and $x_1=7x_3$, so take $x_3=(7,5,1)^T$.
> The three eigenvalues are distinct, so these eigenvectors are linearly independent and form the columns of the displayed matrix $S$, with $A=S\operatorname{diag}(-1,0,1)S^{-1}$.
> **(b) Matrix powers and the matrix exponential.**
> For every positive integer $k$, $A^k=S\operatorname{diag}((-1)^k,0^k,1^k)S^{-1}$. For the odd exponent $1001$, the diagonal factor is again $\operatorname{diag}(-1,0,1)$, so $A^{1001}=A$.
> For $k=1000$, the middle eigendirection is still sent to zero, so $A^{1000}=S\operatorname{diag}(1,0,1)S^{-1}\ne I$. More quickly, $A$ is singular, hence every positive power of $A$ is singular and cannot equal the invertible identity matrix.
> The exponential is $e^{At}=S\operatorname{diag}(e^{-t},1,e^t)S^{-1}$. Because both $S$ and $S^{-1}$ are unit upper triangular, the diagonal entries of $e^{At}$ are $e^{-t},1,e^t$. The same result follows term by term from the power series of an upper triangular matrix.
> **(c) Inertia and eigenvectors of $A^TA$.**
> For every $x$, $x^TA^TAx=\|Ax\|^2\ge0$, so $A^TA$ has no negative eigenvalues. The first two columns of $A$ are linearly dependent, while the third is outside their span, so $\operatorname{rank}(A)=\operatorname{rank}(A^TA)=2$. Therefore the $3\times3$ matrix $A^TA$ has two positive eigenvalues, one zero eigenvalue, and no negative eigenvalues.
> It does not share a complete eigenbasis with $A$. For example, $e_1$ is an eigenvector of $A$ for $\lambda=-1$, but the displayed vector $A^TAe_1$ is not a scalar multiple of $e_1$. In general, eigenvectors of $A$ and right singular vectors—eigenvectors of $A^TA$—are different objects; they may coincide in special cases such as normal matrices.
> <!-- bilingual-en:end -->

### Exam Problem 3（33 分）：正交特征基中的求逆与解方程
<!-- bilingual-en:start -->
*Exam Problem 3(33 points): Inverse and Solution Equation in Orthogonal Eigenbasis*
<!-- bilingual-en:end -->

> [!question] 题目
> $A\in\mathbb R^{n\times n}$ 有标准正交特征向量 $q_1,\ldots,q_n$ 和正特征值 $\lambda_1,\ldots,\lambda_n$，即 $Aq_j=\lambda_jq_j$。
>
> (a) $A^{-1}$ 的特征值和特征向量是什么？证明。
>
> (b) 若 $b=c_1q_1+\cdots+c_nq_n$，利用正交性快速求 $c_1$。
>
> (c) 若 $A^{-1}b=d_1q_1+\cdots+d_nq_n$，快速求 $d_1$。
> <!-- bilingual-en:start -->
> $A\in\mathbb R^{n\times n}$ has the orthonormal eigenvector $q_1,\ldots,q_n$ and the positive eigenvalue $\lambda_1,\ldots,\lambda_n$, namely $Aq_j=\lambda_jq_j$.
> (a) What are the eigenvalues and eigenvectors of $A^{-1}$?  Proof.
> (b) If $b=c_1q_1+\cdots+c_nq_n$, use orthogonality to get $c_1$ quickly.
> (c) If $A^{-1}b=d_1q_1+\cdots+d_nq_n$, seek $d_1$ promptly.
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> **前置检查。** 所有 $\lambda_j>0$，所以 $A$ 无零特征值并可逆。标准正交特征向量矩阵 $Q=[q_1\ \cdots\ q_n]$ 满足 $Q^TQ=I$，且
> $$
> A=Q\Lambda Q^T,\qquad \Lambda=\operatorname{diag}(\lambda_j).
> $$
>
> **(a) 求逆的特征对。**
>
> 从
> $$
> Aq_j=\lambda_jq_j
> $$
> 左乘 $A^{-1}$：
> $$
> q_j=\lambda_jA^{-1}q_j.
> $$
> 再除以非零的 $\lambda_j$：
> $$
> A^{-1}q_j=\frac1{\lambda_j}q_j.
> $$
> 所以 $A^{-1}$ 与 $A$ 有相同特征向量，特征值变成倒数 $1/\lambda_j$。等价地，
> $$
> A^{-1}=Q\Lambda^{-1}Q^T.
> $$
>
> **(b) 用内积抽取系数。**
>
> 左乘 $q_1^T$：
> $$
> q_1^Tb=c_1q_1^Tq_1+\sum_{j=2}^nc_jq_1^Tq_j.
> $$
> 标准正交性给 $q_1^Tq_1=1$、$q_1^Tq_j=0$，所以
> $$
> \boxed{c_1=q_1^Tb}.
> $$
> 若只知正交而未归一化，则应为 $c_1=(q_1^Tb)/(q_1^Tq_1)$。
>
> **(c) 求解系数。**
>
> 对展开式逐项作用 $A^{-1}$：
> $$
> A^{-1}b
> =\sum_{j=1}^nc_jA^{-1}q_j
> =\sum_{j=1}^n\frac{c_j}{\lambda_j}q_j.
> $$
> 因此
> $$
> \boxed{d_1=\frac{c_1}{\lambda_1}
> =\frac{q_1^Tb}{\lambda_1}}.
> $$
>
> **验点与意义。** 把 $x=A^{-1}b$ 代回：
> $$
> Ax=\sum_j\lambda_j\frac{c_j}{\lambda_j}q_j
> =\sum_jc_jq_j=b.
> $$
> 这说明在特征向量坐标中，解 $Ax=b$ 只是把第 $j$ 个坐标除以 $\lambda_j$。很小的 $\lambda_j$ 会显著放大该方向上的误差，这与 SVD 中小奇异值导致病态完全平行。
> <!-- bilingual-en:start -->
> **Pre-check.** Every $\lambda_j$ is positive, so $A$ has no zero eigenvalue and is invertible. The orthonormal eigenvector matrix $Q=[q_1\ \cdots\ q_n]$ satisfies $Q^TQ=I$ and gives the spectral decomposition shown above.
> **(a) Eigenpairs of the inverse.** From $Aq_j=\lambda_jq_j$, multiply by $A^{-1}$ and divide by the nonzero scalar $\lambda_j$ to obtain $A^{-1}q_j=\lambda_j^{-1}q_j$. Thus $A^{-1}$ has the same eigenvectors as $A$, with reciprocal eigenvalues; equivalently, use the displayed spectral formula for $A^{-1}$.
> **(b) Extract a coordinate by an inner product.** Multiplying the expansion of $b$ by $q_1^T$ and using orthonormality—$q_1^Tq_1=1$ and $q_1^Tq_j=0$ for $j\ne1$—gives $c_1=q_1^Tb$. If the basis were merely orthogonal rather than normalized, the denominator would be $q_1^Tq_1$.
> **(c) Solve coefficient by coefficient.** Applying $A^{-1}$ divides the $j$th eigenvector coordinate by $\lambda_j$, yielding the displayed expression for $x$ and $d_1=q_1^Tb/\lambda_1$.
> **Check and interpretation.** Substitution verifies that $Ax=b$. In eigenvector coordinates, solving $Ax=b$ simply divides each coordinate by its eigenvalue. A very small $\lambda_j$ strongly amplifies error in that direction, exactly paralleling the ill-conditioning caused by small singular values in the SVD.
> <!-- bilingual-en:end -->

---

## 本单元最终检查表
<!-- bilingual-en:start -->
*Final Checklist in this module*
<!-- bilingual-en:end -->

### 理论与证明
<!-- bilingual-en:start -->
*theory and proof*
<!-- bilingual-en:end -->

- [ ] 能用 $x^*x>0$ 正确证明实对称/Hermitian 矩阵的特征值为实，不再误用复数情形的 $x^Tx$。
- [ ] 能证明不同特征值的特征向量正交，并写出 $A=Q\Lambda Q^T$ 或 $Q\Lambda Q^*$。
- [ ] 能说明正定的各判据及其全部前提，尤其是对称性与不换行的 $LDL^T$ 主元。
- [ ] 能把配方、$LDL^T$、Hessian 和唯一极小值连成一条推导。
- [ ] 能区分相似与合同，并说明 Jordan 形的域/分裂假设。
- [ ] 能由 $e^{tJ}$ 解释 Jordan 块的多项式因子，以及纯虚谱不自动保证有界。
- [ ] 能从谱定理逐步构造 SVD，并说明四个基本子空间分别由哪些奇异向量张成。
- [ ] 能从基向量的像构造线性变换矩阵，并写清输入/输出基。
- [ ] 能从 $x=Wc$ 推出向量换基与算子换基公式。
- [ ] 能证明左右逆的秩条件，以及伪逆的两个投影与最小范数性质。
<!-- bilingual-en:start -->
- [ ] Can prove that the eigenvalues of a real symmetric or Hermitian matrix are real using $x^*x>0$, without incorrectly using $x^Tx$ for complex vectors.
- [ ] Can prove that eigenvectors corresponding to distinct eigenvalues are orthogonal and write $A=Q\Lambda Q^T$ or $A=Q\Lambda Q^*$.
- [ ] Can state the equivalent criteria for positive definiteness together with all prerequisites, especially symmetry and the no-row-exchange condition behind the pivots in $LDL^T$.
- [ ] Can connect completing the square, $LDL^T$, the Hessian, and uniqueness of a minimum in one derivation.
- [ ] Can distinguish similarity from congruence and explain the field and splitting assumptions behind Jordan form.
- [ ] Can use $e^{tJ}$ to explain the polynomial factors produced by Jordan blocks, and explain why a purely imaginary spectrum alone does not guarantee bounded solutions.
- [ ] Can construct an SVD step by step from the spectral theorem and identify which singular vectors span each of the four fundamental subspaces.
- [ ] Can construct the matrix of a linear transformation from the images of basis vectors and state the input and output bases clearly.
- [ ] Can derive the change-of-coordinate formulas for vectors and operators from $x=Wc$.
- [ ] Can prove the rank conditions for left and right inverses, and the two projection and minimum-norm properties of the pseudoinverse.
<!-- bilingual-en:end -->

### 计算技能
<!-- bilingual-en:start -->
*computing skills*
<!-- bilingual-en:end -->

- [ ] 会用特征值、顺序主子式、主元和二次型至少两种方法判断正定。
- [ ] 会对角化 Hermitian 矩阵并正确使用共轭转置。
- [ ] 会写 $F_4$，解释 Fourier 列正交与 FFT 偶奇分解。
- [ ] 会识别相似不变量，并用几何重数/Jordan 块排除错误相似关系。
- [ ] 会按“$A^TA\to V,\Sigma\to u_i=Av_i/\sigma_i$”稳定地计算 SVD。
- [ ] 会由 $\Sigma$ 直接读秩、零空间维数、谱范数和四子空间。
- [ ] 会在两组基之间换向量坐标和变换矩阵。
- [ ] 会计算满列秩左逆、满行秩右逆以及 SVD 伪逆。
- [ ] 已独立重做三道 Unit 3 Exam，并用定义或代回检查最终答案。
<!-- bilingual-en:start -->
- [ ] I can establish positive definiteness by at least two methods: eigenvalues, leading principal minors, pivots, or the quadratic form.
- [ ] The Hermitian matrix is diagonalized and the conjugate transpose is used properly.
- [ ] Writes $F_4$ explaining the Fourier column orthogonal and FFT even-odd decomposition.
- [ ] identifies similarity invariants and excludes erroneous similarity relationships with geometric multiplicity/Jordan blocks.
- [ ] The SVD is calculated consistently as "$A^TA\to V,\Sigma\to u_i=Av_i/\sigma_i$".
- [ ] Rank, nullspace dimension, spectral norm, and four subspaces are read directly by $\Sigma$.
- [ ] I can convert vector coordinates and transformation matrices between two bases.
- [ ] I can compute the left inverse for a full-column-rank matrix, the right inverse for a full-row-rank matrix, and the SVD pseudoinverse.
- [ ] Redid the three Unit 3 Exam independently and checked the final answer with a definition or substitution.
<!-- bilingual-en:end -->

## 全单元知识链
<!-- bilingual-en:start -->
*whole unit knowledge chain*
<!-- bilingual-en:end -->

$$
\boxed{
\begin{array}{c}
\text{对称/Hermitian}\to\text{正交谱分解}\to\text{正定与二次型}\\[2mm]
\text{一般方阵}\to\text{相似与 Jordan}\to\text{幂、指数与稳定性}\\[2mm]
\text{任意矩阵}\to\text{SVD}\to\text{四子空间、低秩近似与伪逆}\\[2mm]
\text{抽象线性变换}\to\text{选基得到矩阵}\to\text{换基与压缩}
\end{array}}
$$
