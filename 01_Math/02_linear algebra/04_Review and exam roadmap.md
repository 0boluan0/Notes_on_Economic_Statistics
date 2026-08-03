---
aliases:
  - MIT 18.06SC Final Course Review
  - MIT 18.06SC Final Exam
  - 线性代数期末复习与题解
tags:
  - 线性代数
  - mit-ocw
  - course-note
  - exam-review
---

# MIT 18.06SC Final Course Review and Final Exam

> [!info] 课程来源与本页定位
> 本页对应 MIT OCW 18.06SC Fall 2011 的 **Final Course Review** 与 **Final Exam**。它不是公式清单，而是把前三个 Unit 的结构压缩成一套可以实际用于判断、计算与验算的复习系统，并完整解答本地期末试卷的九道题。
>
> - 课程总览：[[01_Math/02_linear algebra/00_课程总览|MIT 18.06SC course map]]
> - Unit I：[[线性方程组与四个基本子空间]]
> - Unit II：[[正交投影与最小二乘]]
> - Unit III：[[对称矩阵与正定二次型|对称矩阵、正定性]]与[[奇异值分解与低秩近似|奇异值分解]]
> - 资料索引：[[MIT_OCW_18.06SC_PDF/index|MIT 18.06SC PDF index]]
> <!-- bilingual-en:start -->
> This page covers the **Final Course Review** and **Final Exam** for MIT OCW 18.06SC Fall 2011. It is not merely a formula sheet: it condenses the structure of the first three units into a practical system for choosing methods, carrying out calculations, and checking results, and it provides complete solutions to the nine questions in the locally stored final exam.
> - Course Overview: [[01_Math/02_linear algebra/00_课程总览|MIT 18.06SC course map]]
> - Unit I:[[线性方程组与四个基本子空间|Linear Equations and Four Basic Subspaces]]
> - Unit II: [[正交投影与最小二乘|Orthogonal Projections and Least Squares]]
> - Unit III: [[对称矩阵与正定二次型|Symmetric matrices and positive definiteness]], and [[奇异值分解与低秩近似|singular value decomposition]]
> - Resource index: [[MIT_OCW_18.06SC_PDF/index|MIT 18.06SC PDF index]]
> <!-- bilingual-en:end -->

## 本页怎么使用
<!-- bilingual-en:start -->
*How to use this page*
<!-- bilingual-en:end -->

第一次复习时，依次完成以下四步：
<!-- bilingual-en:start -->
When you first review, complete the following four steps in turn:
<!-- bilingual-en:end -->

1. 不看公式，先口述“全课程的五条结构链”。
2. 阅读“题型入口”，练习从题目措辞选择正确工具。
3. 完成 Final Course Review 的五组代表题。
4. 合上答案，限时完成 Final Exam，再逐题对照本页的完整题解。
<!-- bilingual-en:start -->

&nbsp;
**1.** Without looking at the formulas, explain the "five structural chains of the course" aloud.<br>
**2.** Read the "problem-type entry points" and practice selecting the right tool from the wording of each question.<br>
**3.** Complete the five sets of representative questions for Final Course Review.<br>
**4.** Hide the answers, complete the Final Exam under timed conditions, and then compare your work with the full solutions on this page, question by question.<br>
<!-- bilingual-en:end -->

如果某道题算错，不要只改最后一个数；应判断错误属于哪一层：
<!-- bilingual-en:start -->
If you get a question wrong, do not merely correct the final number; identify the level at which the error occurred:
<!-- bilingual-en:end -->

- **对象层错误**：矩阵尺寸、向量所在空间或底层域判断错误；
- **条件层错误**：遗漏方阵、满秩、对称、正定或可对角化条件；
- **方法层错误**：把消元、投影、特征分解或 SVD 用在了错误的问题上；
- **计算层错误**：行变换、符号、矩阵乘法或特征多项式算错；
- **验算层错误**：没有代回、检查秩、检查正交性或比较维数。
<!-- bilingual-en:start -->
- **Object-level error**: misidentifying matrix dimensions, the vector space involved, or the underlying field;
- **Condition-level error**: overlooking a requirement such as squareness, full rank, symmetry, positive definiteness, or diagonalizability;
- **Method-level error**: applying elimination, projection, eigendecomposition, or SVD to the wrong kind of problem;
- **Computational error**: making a mistake in row operations, signs, matrix multiplication, or the characteristic polynomial;
- **Verification error**: failing to substitute the result back, check rank or orthogonality, or compare dimensions.
<!-- bilingual-en:end -->

## 全课程的五条结构链
<!-- bilingual-en:start -->
*Five Structural Chains for the Full Course*
<!-- bilingual-en:end -->

### 1. 方程与四个基本子空间
<!-- bilingual-en:start -->
*1. Equations and the four fundamental subspaces*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
A\in\mathbb F^{m\times n},\qquad x\in\mathbb F^n,\qquad b\in\mathbb F^m,
$$

其中 $\mathbb F=\mathbb R$ 或 $\mathbb C$。[[线性方程组与四个基本子空间|线性方程组的解结构]]是
<!-- bilingual-en:start -->
where $\mathbb F=\mathbb R$ or $\mathbb C$. The [[线性方程组与四个基本子空间|solution structure of a linear system]] is
<!-- bilingual-en:end -->

$$
Ax=b\text{ 可解}
\iff b\in C(A),
$$

且一旦存在一个特解 $x_p$，全部解为
<!-- bilingual-en:start -->
And once there is a particular solution $x_p$, all the solutions are
<!-- bilingual-en:end -->

$$
x=x_p+x_n,\qquad x_n\in N(A).
$$

因此：
<!-- bilingual-en:start -->
Therefore:
<!-- bilingual-en:end -->

- [[线性方程组与四个基本子空间|列空间]] $C(A)\subseteq\mathbb F^m$ 控制哪些右端 $b$ 可以到达；
- [[线性方程组与四个基本子空间|零空间]] $N(A)\subseteq\mathbb F^n$ 控制解是否唯一；
- 行空间 $C(A^*)\subseteq\mathbb F^n$ 与 $N(A)$ 正交；
- 左零空间 $N(A^*)\subseteq\mathbb F^m$ 与 $C(A)$ 正交。
<!-- bilingual-en:start -->
- The [[线性方程组与四个基本子空间|column space]] $C(A)\subseteq\mathbb F^m$ determines which right-hand sides $b$ are attainable;
- [[线性方程组与四个基本子空间|null space]] $N(A)\subseteq\mathbb F^n$ controls whether the solution is unique;
- The row space $C(A^*)\subseteq\mathbb F^n$ is orthogonal to $N(A)$;
- The left null space $N(A^*)\subseteq\mathbb F^m$ is orthogonal to $C(A)$.
<!-- bilingual-en:end -->

若 $\operatorname{rank}(A)=r$，则
<!-- bilingual-en:start -->
If $\operatorname{rank}(A)=r$,
<!-- bilingual-en:end -->

$$
\begin{aligned}
\dim C(A)&=r, & \dim N(A)&=n-r,\\
\dim C(A^*)&=r, & \dim N(A^*)&=m-r.
\end{aligned}
$$

这里实矩阵的 $A^*=A^T$；复矩阵必须使用共轭转置。
<!-- bilingual-en:start -->
For real matrices, $A^*=A^T$; for complex matrices, the conjugate transpose must be used.
<!-- bilingual-en:end -->

### 2. 消元、逆与分解
<!-- bilingual-en:start -->
*2. Elimination, inversion and decomposition*
<!-- bilingual-en:end -->

高斯消元把 $A$ 化为阶梯形或简化阶梯形矩阵，用于读取主元、自由变量与解结构。行变换保持方程组的解集，但通常改变列空间中的具体列向量，所以：
<!-- bilingual-en:start -->
Gaussian elimination reduces $A$ to row-echelon or reduced row-echelon form, from which pivots, free variables, and the solution structure can be read. Row operations preserve the solution set of the system but generally alter the actual column vectors, so:
<!-- bilingual-en:end -->

- 从 RREF 读取主元位置、行空间与零空间；
- 列空间的基必须回到**原矩阵**选主元列；
- 若方阵每列都有主元，则 [[广义逆与最小范数解#左逆、右逆与可逆|逆矩阵]]存在；
- 无换行消元可组织成 [[线性方程组与四个基本子空间|LU 分解]] $A=LU$；有换行时应写 $PA=LU$。
<!-- bilingual-en:start -->
- Read pivot positions, the row space, and the null space from the RREF;
- To obtain a basis for the column space, take the corresponding pivot columns from the **original matrix**;
- If a square matrix has a pivot in every column, its [[广义逆与最小范数解#左逆、右逆与可逆|inverse]] exists;
- Elimination without row exchanges gives the [[线性方程组与四个基本子空间|LU factorization]] $A=LU$; with row exchanges, write $PA=LU$.
<!-- bilingual-en:end -->

可逆方阵的核心等价链为
<!-- bilingual-en:start -->
For a square matrix, the central chain of equivalences is
<!-- bilingual-en:end -->

$$
A^{-1}\text{ 存在}
\iff N(A)=\{0\}
\iff \operatorname{rank}(A)=n
\iff \det A\ne0.
$$

### 3. 正交、投影与最小二乘
<!-- bilingual-en:start -->
*3. Orthogonality, projection and least squares*
<!-- bilingual-en:end -->

当 $Ax=b$ 无解时，不再寻找不存在的精确解，而寻找使误差最小的 $\hat x$：
<!-- bilingual-en:start -->
When $Ax=b$ has no exact solution, seek the coefficient vector $\hat x$ that minimizes the error:
<!-- bilingual-en:end -->

$$
\min_x\|Ax-b\|^2.
$$

最近点 $p=A\hat x$ 是 $b$ 在 $C(A)$ 上的[[正交投影与最小二乘|正交投影]]，残差
<!-- bilingual-en:start -->
$p=A\hat x$ is the [[正交投影与最小二乘|orthogonal projection]] of $b$ onto $C(A)$, and the residual
<!-- bilingual-en:end -->

$$
e=b-p
$$

必须垂直于整个列空间，因此
<!-- bilingual-en:start -->
must be orthogonal to the entire column space. Therefore,
<!-- bilingual-en:end -->

$$
A^T(b-A\hat x)=0
\iff A^TA\hat x=A^Tb.
$$

只有当 $A$ 满列秩时，才能进一步写
<!-- bilingual-en:start -->
Only when $A$ has full column rank can we further write
<!-- bilingual-en:end -->

$$
\hat x=(A^TA)^{-1}A^Tb,
\qquad
P=A(A^TA)^{-1}A^T.
$$

若 $A=QR$，其中 $Q^TQ=I$ 且 $R$ 可逆，则
<!-- bilingual-en:start -->
If $A=QR$, where $Q^TQ=I$ and $R$ is invertible, then
<!-- bilingual-en:end -->

$$
P=QQ^T,
\qquad
\hat x=R^{-1}Q^Tb.
$$

### 4. 行列式、特征值与动力系统
<!-- bilingual-en:start -->
*4. Determinants, eigenvalues, and dynamical systems*
<!-- bilingual-en:end -->

[[行列式#行列式的结构含义|行列式]]同时编码三件事：
<!-- bilingual-en:start -->
The [[行列式#行列式的结构含义|determinant]] encodes three things at once:
<!-- bilingual-en:end -->

- $\det A=0$ 表示空间被压扁，矩阵奇异；
- $|\det A|$ 是有向体积缩放因子的绝对值；
- $\det A$ 等于全部特征值之积（按代数重数计）。
<!-- bilingual-en:start -->
- $\det A=0$ means that the transformation collapses the space in at least one direction, so the matrix is singular;
- $|\det A|$ is the absolute volume-scaling factor;
- $\det A$ equals the product of all eigenvalues, counted with algebraic multiplicity.
<!-- bilingual-en:end -->

[[特征值、对角化与线性动力系统#特征值与特征向量|特征值]]与[[特征值、对角化与线性动力系统#特征值与特征向量|特征向量]]满足
<!-- bilingual-en:start -->
An [[特征值、对角化与线性动力系统#特征值与特征向量|eigenvalue]] and its [[特征值、对角化与线性动力系统#特征值与特征向量|eigenvector]] satisfy
<!-- bilingual-en:end -->

$$
Av=\lambda v,\qquad v\ne0.
$$

若 $A=S\Lambda S^{-1}$，则
<!-- bilingual-en:start -->
If $A=S\Lambda S^{-1}$,
<!-- bilingual-en:end -->

$$
A^k=S\Lambda^kS^{-1},
\qquad
e^{At}=Se^{\Lambda t}S^{-1}.
$$

长期行为由模最大的特征值控制，但还必须检查：该特征值是否唯一、是否存在非平凡 Jordan 块，以及初值在对应广义特征空间中是否有非零分量。
<!-- bilingual-en:start -->
Long-run behavior is governed by the eigenvalues of largest modulus, but you must also check whether the dominant eigenvalue is unique, whether nontrivial Jordan blocks are present, and whether the initial state has a nonzero component in the corresponding generalized eigenspace.
<!-- bilingual-en:end -->

### 5. 对称、正定、SVD 与伪逆
<!-- bilingual-en:start -->
*5. Symmetry, positive definiteness, SVD, and the pseudoinverse*
<!-- bilingual-en:end -->

实对称矩阵可正交对角化：
<!-- bilingual-en:start -->
Every real symmetric matrix can be orthogonally diagonalized:
<!-- bilingual-en:end -->

$$
A=Q\Lambda Q^T,
\qquad Q^TQ=I.
$$

若所有特征值为正，则 $A$ 是[[对称矩阵与正定二次型#二次型与正定性|正定矩阵]]，并且
<!-- bilingual-en:start -->
If all eigenvalues are positive, then $A$ is a [[对称矩阵与正定二次型#二次型与正定性|positive-definite matrix]], and
<!-- bilingual-en:end -->

$$
x^TAx>0\qquad(x\ne0).
$$

任意 $m\times n$ 矩阵都具有[[奇异值分解与低秩近似#SVD 的三层结构|奇异值分解]]
<!-- bilingual-en:start -->
Every $m\times n$ matrix has a [[奇异值分解与低秩近似#SVD 的三层结构|singular value decomposition]]
<!-- bilingual-en:end -->

$$
A=U\Sigma V^T.
$$

它把输入空间中的正交方向 $v_i$ 映为输出空间中的正交方向 $u_i$：
<!-- bilingual-en:start -->
It maps the orthogonal direction $v_i$ in the input space to the orthogonal direction $u_i$ in the output space:
<!-- bilingual-en:end -->

$$
Av_i=\sigma_i u_i.
$$

非零奇异值的个数就是秩。[[广义逆与最小范数解#Moore–Penrose 伪逆|Moore--Penrose 伪逆]]为
<!-- bilingual-en:start -->
The number of non-zero singular values is the rank.  [[广义逆与最小范数解#Moore–Penrose 伪逆|Moore—Penrose pseudoinverse]] is
<!-- bilingual-en:end -->

$$
A^+=V\Sigma^+U^T,
$$

并统一给出相容系统的最小范数解与不相容系统的最小二乘解。
<!-- bilingual-en:start -->
This formula gives both the minimum-norm solution of a consistent system and the least-squares solution of an inconsistent system.
<!-- bilingual-en:end -->

## 题型入口：看到什么就问什么
<!-- bilingual-en:start -->
*Problem-type entry points: what to ask when you recognize a cue*
<!-- bilingual-en:end -->

| 题目线索 | 首先检查 | 首选工具 | 必做验算 |
|---|---|---|---|
| “有解、唯一、自由变量” | $m,n,r$ 与 $b\in C(A)$ | elimination / RREF | 代回 $Ax=b$ |
| “四个基本子空间” | 每个空间位于 $\mathbb F^m$ 还是 $\mathbb F^n$ | pivot columns / RREF | 维数和正交性 |
| “closest、best fit、error” | $b$ 是否在 $C(A)$ | projection / normal equations | $A^T(b-A\hat x)=0$ |
| “orthonormal basis” | 向量是否独立 | Gram--Schmidt / QR | $Q^TQ=I$ |
| “volume、singular、cofactor” | 是否方阵 | determinant | 行操作或特征值交叉检查 |
| “powers、steady state、ODE” | 是否可对角化、谱半径 | eigen / Jordan / $e^{At}$ | 代回 $Av=\lambda v$ |
| “symmetric、minimum” | 是否实对称 | spectral theorem / quadratic form | 特征值、主子式或主元 |
| “rectangular、compression” | 秩与形状 | SVD / pseudoinverse | $AA^+A=A$ |
<!-- bilingual-en:start -->
| Wording in the question | Check first | Preferred tool | Required verification |
|---|---|---|---|
| "solvable, unique, free variables" | $m,n,r$ and $b\in C(A)$ | elimination / RREF | Substitute into $Ax=b$ |
| "four fundamental subspaces" | Whether each space lies in $\mathbb F^m$ or $\mathbb F^n$ | pivot columns / RREF | Dimensions and orthogonality |
| "closest, best fit, error" | whether $b$ is on $C(A)$ | projection / normal equations | $A^T(b-A\hat x)=0$ |
| "orthonormal basis" | Whether the vectors are linearly independent | Gram--Schmidt / QR | $Q^TQ=I$ |
| "volume, singular, cofactor" | Whether the matrix is square | determinant | Cross-check with row operations or eigenvalues |
| "powers, steady state, ODE" | Diagonalizability and spectral radius | eigendecomposition / Jordan form / $e^{At}$ | Substitute into $Av=\lambda v$ |
| "symmetric, minimum" | Whether the matrix is real symmetric | spectral theorem / quadratic form | Eigenvalues, principal minors, or pivots |
| "rectangular, compression" | Rank and dimensions | SVD / pseudoinverse | $AA^+A=A$ |
<!-- bilingual-en:end -->

## Final Course Review：五组代表题
<!-- bilingual-en:start -->
*Final Course Review: Five sets of questions*
<!-- bilingual-en:end -->

### Review 1：从可解性反推尺寸与秩
<!-- bilingual-en:start -->
*Review 1: Reverse Dimensions and Ranks from Solvability*
<!-- bilingual-en:end -->

设 $A\in\mathbb R^{m\times n}$、$\operatorname{rank}(A)=r$，并且
<!-- bilingual-en:start -->
$A\in\mathbb R^{m\times n}$, $\operatorname{rank}(A)=r$, and
<!-- bilingual-en:end -->

$$
Ax=\begin{bmatrix}1\\0\\0\end{bmatrix}
$$

无解，而
<!-- bilingual-en:start -->
No solution, but
<!-- bilingual-en:end -->

$$
Ax=\begin{bmatrix}0\\1\\0\end{bmatrix}
$$

恰有一个解。
<!-- bilingual-en:start -->
There's just one solution.
<!-- bilingual-en:end -->

**第一步：确定 $m$。** 两个右端都有三个分量，所以 $Ax\in\mathbb R^3$，从而 $m=3$。
<!-- bilingual-en:start -->
**Step 1: Determine $m$.** Both right-hand sides have three components, so $Ax\in\mathbb R^3$ and therefore $m=3$.
<!-- bilingual-en:end -->

**第二步：利用无解信息。** 第一个右端不在 $C(A)$，故 $C(A)\ne\mathbb R^3$，于是 $r<3$。第二个右端能够到达，说明 $r\ge1$。
<!-- bilingual-en:start -->
**Step 2: Use the inconsistent system.** The first right-hand side is not in $C(A)$, so $C(A)\ne\mathbb R^3$ and hence $r<3$. The second right-hand side is attainable, so $r\ge1$.
<!-- bilingual-en:end -->

**第三步：利用唯一性。** 若一个相容系统恰有一个解，则不能存在非零 $z\in N(A)$，否则 $x+z$ 也是另一解。因此 $N(A)=\{0\}$，由秩—零度定理得
<!-- bilingual-en:start -->
**Step 3: Use uniqueness.** If a consistent system has exactly one solution, there can be no nonzero $z\in N(A)$; otherwise $x+z$ would be another solution. Thus $N(A)=\{0\}$, and the rank--nullity theorem gives
<!-- bilingual-en:end -->

$$
n-r=0\Longrightarrow n=r.
$$

综上
<!-- bilingual-en:start -->
Combining these facts,
<!-- bilingual-en:end -->

$$
m=3,
\qquad
1\le n=r<3.
$$

最简单例子是
<!-- bilingual-en:start -->
The simplest example is
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}0\\1\\0\end{bmatrix}.
$$

它的列空间是 $y$ 轴，第二个右端唯一可达，第一个右端不可达。
<!-- bilingual-en:start -->
Its column space is the $y$-axis. The second right-hand side is attained by a unique coefficient, whereas the first is unattainable.
<!-- bilingual-en:end -->

进一步，$A^TA$ 必可逆，因为 $A$ 满列秩；$AA^T$ 只有秩 $r<3$，所以不可能正定，只能是半正定。对任意 $c\in\mathbb R^n$，$A^Ty=c$ 都可解，因为 $A^T$ 满行秩；又因
<!-- bilingual-en:start -->
Furthermore, $A^TA$ is invertible because $A$ has full column rank. By contrast, $AA^T$ has rank $r<3$, so it cannot be positive definite and is only positive semidefinite. For every $c\in\mathbb R^n$, the system $A^Ty=c$ is solvable because $A^T$ has full row rank; moreover,
<!-- bilingual-en:end -->

$$
\dim N(A^T)=3-r>0,
$$

每个相容系统都有无穷多解。
<!-- bilingual-en:start -->
every such consistent system has infinitely many solutions.
<!-- bilingual-en:end -->

### Review 2：列组合、零空间与正交投影
<!-- bilingual-en:start -->
*Review 2: Column combinations, the null space, and orthogonal projection*
<!-- bilingual-en:end -->

若 $A=[v_1\ v_2\ v_3]$，则
<!-- bilingual-en:start -->
If $A=[v_1\ v_2\ v_3]$,
<!-- bilingual-en:end -->

$$
Ax=v_1-v_2+v_3
$$

的一个解立即是
<!-- bilingual-en:start -->
One solution is immediately
<!-- bilingual-en:end -->

$$
x=\begin{bmatrix}1\\-1\\1\end{bmatrix}.
$$

如果 $v_1-v_2+v_3=0$，则这个非零 $x$ 属于 $N(A)$，系统 $Ax=0$ 不可能只有零解。注意这里是 $N(A)$，不是 $N(A^T)$。
<!-- bilingual-en:start -->
If $v_1-v_2+v_3=0$, then this nonzero $x$ belongs to $N(A)$, so the homogeneous system $Ax=0$ has a nontrivial solution. Notice that the relevant space is $N(A)$, not $N(A^T)$.
<!-- bilingual-en:end -->

若 $v_1,v_2,v_3$ 正交归一，则 $v_3$ 在 $\operatorname{span}(v_1,v_2)$ 上的投影为
<!-- bilingual-en:start -->
If $v_1,v_2,v_3$ are orthonormal, the projection of $v_3$ onto $\operatorname{span}(v_1,v_2)$ is
<!-- bilingual-en:end -->

$$
(v_1^Tv_3)v_1+(v_2^Tv_3)v_2=0.
$$

因此距 $v_3$ 最近的组合是 $0v_1+0v_2$。
<!-- bilingual-en:start -->
So the closest combination to $v_3$ is $0v_1+0v_2$.
<!-- bilingual-en:end -->

### Review 3：Markov 矩阵的稳态
<!-- bilingual-en:start -->
*Review 3: The steady state of a Markov matrix*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}
0.2&0.4&0.3\\
0.4&0.2&0.3\\
0.4&0.4&0.4
\end{bmatrix}.
$$

每列元素和为 $1$，所以 $1$ 是特征值。又因前两列之和等于第三列的两倍，列向量相关，故 $0$ 是特征值。迹为 $0.8$，第三个特征值为
<!-- bilingual-en:start -->
Every column sums to $1$, so $1$ is an eigenvalue. The sum of the first two columns equals twice the third column, so the columns are linearly dependent and $0$ is also an eigenvalue. Since the trace is $0.8$, the third eigenvalue is
<!-- bilingual-en:end -->

$$
0.8-1-0=-0.2.
$$

解 $(A-I)q=0$ 可得稳态方向
<!-- bilingual-en:start -->
Steady-state direction can be obtained by solving $(A-I)q=0$
<!-- bilingual-en:end -->

$$
q=\begin{bmatrix}3\\3\\4\end{bmatrix}.
$$

若初始状态总量为 $10$，则总量在列随机矩阵作用下保持不变，而 $q$ 的分量和正好也是 $10$。由于其余两个特征值的模都小于 $1$，
<!-- bilingual-en:start -->
If the entries of the initial state sum to $10$, that total is preserved by this column-stochastic matrix, and the entries of $q$ also sum to $10$. Since the other two eigenvalues have modulus less than $1$,
<!-- bilingual-en:end -->

$$
\lim_{k\to\infty}A^ku^{(0)}
=\begin{bmatrix}3\\3\\4\end{bmatrix},
$$

前提是初始状态的分量和为 $10$。
<!-- bilingual-en:start -->
provided that the entries of the initial state sum to $10$.
<!-- bilingual-en:end -->

### Review 4：按结构构造矩阵
<!-- bilingual-en:start -->
*Review 4: Constructing matrices by structure*
<!-- bilingual-en:end -->

1. 投影到 $a=(4,-3)^T$ 所张成直线：
<!-- bilingual-en:start -->

&nbsp;
**1.** Projection onto the line spanned by $a=(4,-3)^T$:<br>
<!-- bilingual-en:end -->

   $$
   P=\frac{aa^T}{a^Ta}
   =\frac1{25}\begin{bmatrix}16&-12\\-12&9\end{bmatrix}.
   $$

   验算：$P^T=P$、$P^2=P$、$Pa=a$。
<!-- bilingual-en:start -->
Checking: $P^T=P$, $P^2=P$, $Pa=a$.
<!-- bilingual-en:end -->

2. 给定特征对
<!-- bilingual-en:start -->

&nbsp;
**2.** Given the eigenpairs<br>
<!-- bilingual-en:end -->

   $$
   \lambda_1=0,
   \quad x_1=\begin{bmatrix}1\\2\end{bmatrix},
   \qquad
   \lambda_2=3,
   \quad x_2=\begin{bmatrix}2\\1\end{bmatrix},
   $$

   令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

   $$
   S=\begin{bmatrix}1&2\\2&1\end{bmatrix},
   \quad
   \Lambda=\begin{bmatrix}0&0\\0&3\end{bmatrix}.
   $$

   则
<!-- bilingual-en:start -->
Then,
<!-- bilingual-en:end -->

   $$
   A=S\Lambda S^{-1}
   =\begin{bmatrix}4&-2\\2&-1\end{bmatrix}.
   $$

3. 任意非对称实矩阵都不可能写成 $B^TB$，因为 $B^TB$ 必对称。例如
<!-- bilingual-en:start -->

&nbsp;
**3.** Any asymmetric real matrix cannot be written as $B^TB$ because $B^TB$ must be symmetric.  For example<br>
<!-- bilingual-en:end -->

   $$
   \begin{bmatrix}0&0\\1&0\end{bmatrix}.
   $$

### Review 5：直线拟合与左零空间
<!-- bilingual-en:start -->
*Review 5: Line fitting and the left null space*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix},
\qquad
b=\begin{bmatrix}3\\4\\1\end{bmatrix}.
$$

正规方程给出
<!-- bilingual-en:start -->
The normal equations give
<!-- bilingual-en:end -->

$$
\hat x=\begin{bmatrix}\hat c\\\hat d\end{bmatrix}
=\begin{bmatrix}11/3\\-1\end{bmatrix}.
$$

所以投影为
<!-- bilingual-en:start -->
So the projection is
<!-- bilingual-en:end -->

$$
p=A\hat x
=\begin{bmatrix}11/3\\8/3\\5/3\end{bmatrix}.
$$

对应的最小二乘直线是
<!-- bilingual-en:start -->
The corresponding least squares line is
<!-- bilingual-en:end -->

$$
y=\frac{11}{3}-t.
$$

若希望最小二乘解为零，就必须令 $b$ 与 $C(A)$ 正交，即 $b\in N(A^T)$。解
<!-- bilingual-en:start -->
For the least-squares solution to be zero, $b$ must be orthogonal to $C(A)$; equivalently, $b\in N(A^T)$. Solving
<!-- bilingual-en:end -->

$$
A^Tb=0
$$

可取
<!-- bilingual-en:start -->
one possible choice is
<!-- bilingual-en:end -->

$$
b=\begin{bmatrix}1\\-2\\1\end{bmatrix}.
$$

## Final Exam：作答约定
<!-- bilingual-en:start -->
*Final Exam: Answer Conventions*
<!-- bilingual-en:end -->

以下题目依据本地试卷准确转述。每题解答均包含结构判断、计算、条件与验算。考试中的 $T$ 均针对实矩阵；若推广到复矩阵，应改用 $*$。
<!-- bilingual-en:start -->
The following questions are transcribed faithfully from the locally stored exam. Each solution includes structural reasoning, calculation, required conditions, and verification. The symbol $T$ in the exam refers to the transpose of a real matrix; for complex matrices, replace it with $*$.
<!-- bilingual-en:end -->

### Question 1：由零空间特殊解重建 RREF
<!-- bilingual-en:start -->
*Question 1: Reconstructing an RREF from special solutions of the null space*
<!-- bilingual-en:end -->

> [!question] 题目
> 设 $A$ 是 $3\times4$ 矩阵，$Ax=0$ 恰有两个特殊解
> $$
> z_1=\begin{bmatrix}1\\1\\1\\0\end{bmatrix},
> \qquad
> z_2=\begin{bmatrix}-2\\-1\\0\\1\end{bmatrix}.
> $$
> 1. 求 $A$ 的简化行阶梯形 $R$。
> 2. 求四个基本子空间的维数，并尽可能给出基。
> <!-- bilingual-en:start -->
> Let $A$ be a $3\times4$ matrix for which $Ax=0$ has exactly the following two special solutions.
> **1.** Find the reduced row-echelon form $R$ of $A$.<br>
> **2.** Find the dimensions of the four fundamental subspaces and give bases wherever possible.<br>
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> 两个“特殊解”分别对应把某个自由变量设为 $1$、其余自由变量设为 $0$。观察 $z_1,z_2$ 可知第 3、4 个变量是自由变量，第 1、2 个变量是主元变量。因此 $r=2$，且
> $$
> R=\begin{bmatrix}
> 1&0&\alpha&\beta\\
> 0&1&\gamma&\delta\\
> 0&0&0&0
> \end{bmatrix}.
> $$
> 由 $Rz_1=0$：
> $$
> 1+\alpha=0,
> \qquad
> 1+\gamma=0,
> $$
> 所以 $\alpha=\gamma=-1$。由 $Rz_2=0$，第一、二行分别给出
> $$
> -2+\beta=0,
> \qquad
> -1+\delta=0.
> $$
> 从而 $\beta=2,\delta=1$，
> $$
> \boxed{R=\begin{bmatrix}1&0&-1&2\\0&1&-1&1\\0&0&0&0\end{bmatrix}}.
> $$
> 验算：直接计算可得 $Rz_1=Rz_2=0$。
>
> 由 $m=3,n=4,r=2$：
> $$
> \dim C(A)=2,
> \quad
> \dim N(A)=2,
> \quad
> \dim C(A^T)=2,
> \quad
> \dim N(A^T)=1.
> $$
> 零空间的基已给出：
> $$
> \mathcal B_{N(A)}=\{z_1,z_2\}.
> $$
> 行空间不因行变换而改变，故可取 $R$ 的两个非零行为基：
> $$
> \mathcal B_{C(A^T)}=
> \left\{
> \begin{bmatrix}1\\0\\-1\\2\end{bmatrix},
> \begin{bmatrix}0\\1\\-1\\1\end{bmatrix}
> \right\}.
> $$
> 列空间可确定由原矩阵 $A$ 的第 1、2 列构成一组基，但题目没有给出 $A$ 的具体列，不能写成数值向量。左零空间同样只能确定维数为 $1$；需要知道 $A$ 的列后才能写出具体基。
> <!-- bilingual-en:start -->
> Each "special solution" is obtained by setting one free variable to $1$ and all other free variables to $0$. From $z_1$ and $z_2$, the third and fourth variables are free while the first and second are pivot variables. Thus $r=2$, and
> Using $Rz_1=0$ gives
> so $\alpha=\gamma=-1$. Using $Rz_2=0$, the first and second rows give
> hence $\beta=2$ and $\delta=1$, so
> Verification: direct multiplication gives $Rz_1=Rz_2=0$.
> Since $m=3$, $n=4$, and $r=2$,
> A basis for the null space has already been supplied:
> Row operations preserve the row space, so the two nonzero rows of $R$ form a basis:
> A basis for the column space consists of columns 1 and 2 of the original matrix $A$. Because the problem does not provide those columns explicitly, no numerical basis vectors can be written down. Likewise, only the dimension $1$ of the left null space can be determined; an explicit basis would require the actual columns of $A$.
> <!-- bilingual-en:end -->

### Question 2：上三角矩阵的逆、特征向量矩阵与 SVD
<!-- bilingual-en:start -->
*Question 2: The inverse, eigenvector matrix, and SVD of an upper-triangular matrix*
<!-- bilingual-en:end -->

> [!question] 题目
> 设
> $$
> U=\begin{bmatrix}a&b&c\\0&d&e\\0&0&f\end{bmatrix},
> $$
> 其中 $a,b,c,d,e,f$ 均非零。
> 1. 求 $U^{-1}$。
> 2. 若 $U$ 的列是矩阵 $A$ 的一组特征向量，证明 $A$ 也是上三角矩阵。
> 3. 解释为什么这里的 $U$ 不可能是 SVD $A=U\Sigma V^T$ 中的左奇异向量矩阵。
> <!-- bilingual-en:start -->
> Let
> where $a,b,c,d,e,f$ are all nonzero.
> **1.** Find $U^{-1}$.<br>
> **2.** If the columns of $U$ form a complete set of eigenvectors of a matrix $A$, prove that $A$ is also upper triangular.<br>
> **3.** Explain why this $U$ cannot be the matrix of left singular vectors in an SVD $A=U\Sigma V^T$.<br>
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> 由于 $a,d,f\ne0$，$U$ 可逆。设
> $$
> U^{-1}=\begin{bmatrix}x&y&z\\0&s&t\\0&0&w\end{bmatrix}.
> $$
> 由 $UU^{-1}=I$ 逐项比较：
> $$
> ax=1\Rightarrow x=\frac1a,
> \quad
> ds=1\Rightarrow s=\frac1d,
> \quad
> fw=1\Rightarrow w=\frac1f.
> $$
> 第一行第二列给出
> $$
> ay+bs=0
> \Rightarrow
> y=-\frac{b}{ad}.
> $$
> 第二行第三列给出
> $$
> dt+ew=0
> \Rightarrow
> t=-\frac{e}{df}.
> $$
> 第一行第三列给出
> $$
> az+bt+cw=0,
> $$
> 所以
> $$
> z=-\frac1a\left(-\frac{be}{df}+\frac cf\right)
> =\frac{be-cd}{adf}.
> $$
> 因而
> $$
> \boxed{
> U^{-1}=\begin{bmatrix}
> 1/a&-b/(ad)&(be-cd)/(adf)\\
> 0&1/d&-e/(df)\\
> 0&0&1/f
> \end{bmatrix}}.
> $$
> 上三角矩阵的乘积仍为上三角矩阵，上式也说明可逆上三角矩阵的逆仍上三角。
>
> 若 $U$ 的列为 $A$ 的完整特征向量组，则
> $$
> AU=U\Lambda
> \Longrightarrow
> A=U\Lambda U^{-1}.
> $$
> $U,\Lambda,U^{-1}$ 都是上三角矩阵，所以 $A$ 上三角。
>
> SVD 中的左奇异向量矩阵必须满足 $U^TU=I$，即列向量正交归一。题设矩阵的前两列点积为
> $$
> \begin{bmatrix}a\\0\\0\end{bmatrix}^T
> \begin{bmatrix}b\\d\\0\end{bmatrix}=ab\ne0,
> $$
> 因此这些列不正交，不可能充当 SVD 的 $U$。这里两个字母 $U$ 只是记号相同，角色不同。
> <!-- bilingual-en:start -->
> Because $a,d,f\ne0$, $U$ is invertible.
> Comparing entries in $UU^{-1}=I$ gives
> The entry in row 1, column 2 gives
> The entry in row 2, column 3 gives
> The entry in row 1, column 3 gives
> so
> therefore
> A product of upper-triangular matrices is upper triangular. The formula above also shows that the inverse of an invertible upper-triangular matrix is upper triangular.
> If the columns of $U$ form a complete set of eigenvectors of $A$, then
> $U$, $\Lambda$, and $U^{-1}$ are all upper triangular, so $A$ is upper triangular.
> In an SVD, the matrix of left singular vectors must satisfy $U^TU=I$; that is, its columns must be orthonormal. Here the dot product of the first two columns is
> Thus these columns are not orthogonal and cannot serve as the $U$ in an SVD. The two uses of the letter $U$ share only the notation, not the role.
> <!-- bilingual-en:end -->

### Question 3：拼接矩阵的秩与零度
<!-- bilingual-en:start -->
*Question 3: Rank and nullity of concatenated matrices*
<!-- bilingual-en:end -->

> [!question] 题目
> 1. 若 $A,B$ 行数相同，比较 $\operatorname{rank}(A)$ 与 $\operatorname{rank}[A\ B]$。
> 2. 若 $B=A^2$，再次比较二者。
> 3. 若 $A$ 是秩为 $r$ 的 $m\times n$ 矩阵，求 $N(A)$ 与 $N([A\ A])$ 的维数。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** If the number of rows in $A,B$ is the same, compare $\operatorname{rank}(A)$ with $\operatorname{rank}[A\ B]$.<br>
> **2.** If $B=A^2$, compare the two again.<br>
> **3.** If $A$ is a $m\times n$ matrix with rank $r$, calculate the dimensions of $N(A)$ and $N([A\ A])$.<br>
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> $A$ 的每一列也都是 $[A\ B]$ 的列，因此
> $$
> C(A)\subseteq C([A\ B]).
> $$
> 所以只能普遍断言
> $$
> \boxed{\operatorname{rank}(A)\le\operatorname{rank}([A\ B])}.
> $$
> $B$ 可能增加新方向，也可能完全不增加。
>
> 若 $B=A^2$，则 $A$ 必须为方阵。$A^2$ 的第 $j$ 列为 $A(Ae_j)$，是 $A$ 的列向量的线性组合。因此
> $$
> C(A^2)\subseteq C(A),
> $$
> 拼接 $A^2$ 不会增加列空间：
> $$
> \boxed{\operatorname{rank}([A\ A^2])=\operatorname{rank}(A)=r}.
> $$
>
> 秩—零度定理给出
> $$
> \boxed{\dim N(A)=n-r}.
> $$
> $[A\ A]$ 是 $m\times2n$ 矩阵，重复列不增加秩，故
> $$
> \boxed{\dim N([A\ A])=2n-r}.
> $$
> 还可直接看见一族零空间向量：对任意 $z\in\mathbb R^n$，$(z,-z)^T$ 都满足 $[A\ A](z,-z)^T=0$。
> <!-- bilingual-en:start -->
> Each column of $A$ is also a column of $[A\ B]$, so
> Therefore, the only universally valid conclusion is
> $B$ may add new directions, or it may add none.
> If $B=A^2$, then $A$ must be square. The $j$th column of $A^2$ is $A(Ae_j)$, which is a linear combination of the columns of $A$. Hence
> Appending $A^2$ does not enlarge the column space:
> The rank--nullity theorem gives
> $[A\ A]$ is an $m\times2n$ matrix. Repeating the columns does not increase the rank, so
> A family of null-space vectors is also visible directly: for any $z\in\mathbb R^n$, $(z,-z)^T$ satisfies $[A\ A](z,-z)^T=0$.
> <!-- bilingual-en:end -->

### Question 4：满列秩、$A^TA$ 与单侧逆
<!-- bilingual-en:start -->
*Question 4: Full column rank, $A^TA$, and a one-sided inverse*
<!-- bilingual-en:end -->

> [!question] 题目
> 设 $A$ 是 $5\times3$ 矩阵，且 $Ax=0$ 仅在 $x=0$ 时成立。
> 1. 说明 $A$ 的列向量有什么性质。
> 2. 证明 $A^TAx=0$ 也只有零解。
> 3. 令 $B=(A^TA)^{-1}A^T$，说明它是哪一侧的逆，并解释为什么不是双侧逆。
> <!-- bilingual-en:start -->
> Let $A$ be a $5\times3$ matrix and $Ax=0$ be true only when $x=0$.
> **1.** State what this implies about the columns of $A$.<br>
> **2.** Prove that $A^TAx=0$ also has only the zero solution.<br>
> **3.** Let $B=(A^TA)^{-1}A^T$. Determine on which side it is an inverse of $A$, and explain why it is not a two-sided inverse.<br>
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> 条件正是
> $$
> N(A)=\{0\}.
> $$
> 因为 $A$ 有三列，秩—零度定理给出 $3-r=0$，所以 $r=3$，列向量线性无关。
>
> 假设 $A^TAx=0$。左乘 $x^T$：
> $$
> 0=x^TA^TAx=(Ax)^T(Ax)=\|Ax\|^2.
> $$
> 欧氏范数平方只有在 $Ax=0$ 时才为零；由题设进一步得到 $x=0$。因此 $N(A^TA)=\{0\}$，$3\times3$ 方阵 $A^TA$ 可逆。事实上这还证明了 $A^TA$ 正定：
> $$
> x^TA^TAx=\|Ax\|^2>0\quad(x\ne0).
> $$
>
> 尺寸为
> $$
> B=(A^TA)^{-1}A^T\in\mathbb R^{3\times5}.
> $$
> 计算
> $$
> BA=(A^TA)^{-1}A^TA=I_3,
> $$
> 因此 $B$ 是 $A$ 的左逆。另一方面
> $$
> AB=A(A^TA)^{-1}A^T
> $$
> 是投影到三维子空间 $C(A)\subset\mathbb R^5$ 的 $5\times5$ 投影矩阵，秩只有 $3$，不可能等于秩为 $5$ 的 $I_5$。
>
> <!-- bilingual-en:start -->
> The condition is precisely
> Because $A$ has three columns, the rank--nullity theorem gives $3-r=0$. Thus $r=3$, so the columns are linearly independent.
> Assume $A^TAx=0$.  Left-Multiply $x^T$:
> The squared Euclidean norm is zero only when $Ax=0$, and the hypothesis then implies $x=0$. Therefore $N(A^TA)=\{0\}$, so the $3\times3$ matrix $A^TA$ is invertible. In fact, this also proves that $A^TA$ is positive definite:
> The dimensions are
> Compute
> Therefore, $B$ is a left inverse of $A$. On the other hand,
> is the $5\times5$ orthogonal projection matrix onto the three-dimensional subspace $C(A)\subset\mathbb R^5$. Its rank is only $3$, so it cannot equal $I_5$, whose rank is $5$.
> <!-- bilingual-en:end -->

> [!warning] 官方答案笔误
> 官方解答最后一句把投影矩阵误写成了 $BA$；由尺寸即可发现应为 $AB$。
> <!-- bilingual-en:start -->
> The official solution mistakenly writes the projection matrix as $BA$ in its final sentence. The matrix dimensions show that it must be $AB$.
> <!-- bilingual-en:end -->

### Question 5：Rayleigh 商的最大值
<!-- bilingual-en:start -->
*Question 5: Maximizing the Rayleigh quotient*
<!-- bilingual-en:end -->

> [!question] 题目
> 设 $A$ 是 $3\times3$ 实对称正定矩阵，$Aq_i=\lambda_iq_i$，$q_i$ 正交归一且
> $$
> 0<\lambda_1<\lambda_2<\lambda_3.
> $$
> 若 $x=c_1q_1+c_2q_2+c_3q_3$，计算 $x^Tx$、$x^TAx$，并确定 $x^TAx/x^Tx$ 何时最大。
> <!-- bilingual-en:start -->
> Let $A$ be a $3\times3$ real symmetric positive-definite matrix, with $Aq_i=\lambda_iq_i$, where the $q_i$ are orthonormal and
> If $x=c_1q_1+c_2q_2+c_3q_3$, compute $x^Tx$ and $x^TAx$, and determine when $x^TAx/x^Tx$ is maximized.
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> 由 $q_i^Tq_j=\delta_{ij}$，所有交叉项消失：
> $$
> x^Tx=c_1^2+c_2^2+c_3^2.
> $$
> 又因为 $Aq_i=\lambda_iq_i$，
> $$
> Ax=c_1\lambda_1q_1+c_2\lambda_2q_2+c_3\lambda_3q_3,
> $$
> 从而
> $$
> x^TAx=c_1^2\lambda_1+c_2^2\lambda_2+c_3^2\lambda_3.
> $$
> 对 $x\ne0$，Rayleigh 商为
> $$
> \rho_A(x)=\frac{x^TAx}{x^Tx}
> =\frac{c_1^2\lambda_1+c_2^2\lambda_2+c_3^2\lambda_3}
> {c_1^2+c_2^2+c_3^2}.
> $$
> 它是三个特征值的加权平均，权重 $c_i^2/(c_1^2+c_2^2+c_3^2)$ 非负且和为 $1$，所以
> $$
> \lambda_1\le\rho_A(x)\le\lambda_3.
> $$
> 最大值 $\lambda_3$ 当且仅当 $c_1=c_2=0$、$c_3\ne0$，即 $x$ 是最大特征值对应特征向量 $q_3$ 的非零倍数。
> <!-- bilingual-en:start -->
> Since $q_i^Tq_j=\delta_{ij}$, all cross terms vanish:
> Because $Aq_i=\lambda_iq_i$,
> hence
> For $x\ne0$, the Rayleigh quotient is
> It is the weighted average of the three eigenvalues, the weight $c_i^2/(c_1^2+c_2^2+c_3^2)$ is non-negative and the sum is $1$, so
> The maximum $\lambda_3$ is attained if and only if $c_1=c_2=0$ and $c_3\ne0$; equivalently, $x$ is a nonzero multiple of the eigenvector $q_3$ associated with the largest eigenvalue.
> <!-- bilingual-en:end -->

### Question 6：Gram--Schmidt、QR 与投影矩阵
<!-- bilingual-en:start -->
*Question 6: Gram--Schmidt, QR, and projection matrices*
<!-- bilingual-en:end -->

> [!question] 题目
> 给定线性无关向量 $u,v$：
> 1. 求由 $u,v$ 线性组合得到且垂直于 $u$ 的非零向量 $w$。
> 2. 对 $A=[u\ v]$ 求 $A=QR$。
> 3. 只用 $Q$ 写出投影到 $\operatorname{span}(u,v)$ 的矩阵。
> <!-- bilingual-en:start -->
> Given linearly independent vectors $u$ and $v$:
> **1.** Find a nonzero vector $w$, formed as a linear combination of $u$ and $v$, that is orthogonal to $u$.<br>
> **2.** For $A=[u\ v]$, find a factorization $A=QR$.<br>
> **3.** Using only $Q$, write the matrix that projects onto $\operatorname{span}(u,v)$.<br>
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> 从 $v$ 中减去它在 $u$ 上的投影：
> $$
> \boxed{w=v-\frac{u^Tv}{u^Tu}u}.
> $$
> 验算：
> $$
> u^Tw=u^Tv-\frac{u^Tv}{u^Tu}u^Tu=0.
> $$
> 因为 $u,v$ 独立，$w\ne0$。
>
> 令
> $$
> q_1=\frac{u}{\|u\|},
> \qquad
> q_2=\frac{w}{\|w\|},
> \qquad
> Q=[q_1\ q_2].
> $$
> 则 $Q^TQ=I_2$。第一列满足 $u=\|u\|q_1$。第二列分解为
> $$
> v=(q_1^Tv)q_1+(q_2^Tv)q_2
> =\frac{u^Tv}{\|u\|}q_1+\|w\|q_2.
> $$
> 因此
> $$
> \boxed{
> R=\begin{bmatrix}
> \|u\|&u^Tv/\|u\|\\
> 0&\|w\|
> \end{bmatrix}},
> \qquad A=QR.
> $$
>
> 由于 $Q$ 的列已经是该平面的标准正交基，投影矩阵直接为
> $$
> \boxed{P=QQ^T}.
> $$
> 也可从一般公式验证：
> $$
> \begin{aligned}
> A(A^TA)^{-1}A^T
> &=QR(R^TQ^TQR)^{-1}R^TQ^T\\
> &=QR(R^TR)^{-1}R^TQ^T\\
> &=QRR^{-1}(R^T)^{-1}R^TQ^T\\
> &=QQ^T.
> \end{aligned}
> $$
> <!-- bilingual-en:start -->
> Subtract the projection of $v$ onto $u$ from $v$:
> Verification:
> Because $u$ and $v$ are linearly independent, $w\ne0$.
> Let
> Then $Q^TQ=I_2$. The first column satisfies $u=\|u\|q_1$, while the second column decomposes as
> Therefore,
> Because the columns of $Q$ already form an orthonormal basis for the plane, the projection matrix is simply
> You can also verify from a general formula:
> <!-- bilingual-en:end -->

### Question 7：循环置换矩阵的谱与行列式
<!-- bilingual-en:start -->
*Question 7: Spectrum and Determinant of Cyclic Permutation Matrices*
<!-- bilingual-en:end -->

> [!question] 题目
> 设
> $$
> C=\begin{bmatrix}
> 0&0&0&1\\
> 1&0&0&0\\
> 0&1&0&0\\
> 0&0&1&0
> \end{bmatrix}.
> $$
> 求 $C,C^2$ 的特征值，求其逆，并计算 $\det C$、$\det(C+I)$、$\det(C+2I)$。
> <!-- bilingual-en:start -->
> Let
> Find the eigenvalues of $C$ and $C^2$, find their inverses, and compute $\det C$, $\det(C+I)$, and $\det(C+2I)$.
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> $C$ 把坐标循环移动一格，连续作用四次回到原位：
> $$
> C^4=I.
> $$
> 若 $Cv=\lambda v$，则
> $$
> v=C^4v=\lambda^4v,
> $$
> 因 $v\ne0$，有 $\lambda^4=1$。四个特征值为
> $$
> \boxed{1,-1,i,-i}.
> $$
> $C^2$ 的特征值为这些数的平方：
> $$
> \boxed{1,1,-1,-1}.
> $$
>
> 置换矩阵正交，所以
> $$
> \boxed{C^{-1}=C^T=C^3},
> \qquad
> \boxed{(C^2)^{-1}=C^2}.
> $$
> 特征值之积给出
> $$
> \det C=(1)(-1)(i)(-i)=-1.
> $$
> $C+\alpha I$ 与 $C$ 有相同特征向量，特征值变为 $\lambda_j+\alpha$。因此
> $$
> \det(C+I)=2\cdot0\cdot(1+i)(1-i)=0,
> $$
> $$
> \det(C+2I)=3\cdot1\cdot(2+i)(2-i)=3\cdot5=15.
> $$
> <!-- bilingual-en:start -->
> $C$ cyclically shifts the coordinates by one position, so four successive applications return every vector to its original position:
> If $Cv=\lambda v$,
> Since $v\ne0$, we have $\lambda^4=1$. The four eigenvalues are
> The eigenvalues of $C^2$ are the squares of these numbers:
> A permutation matrix is orthogonal, so
> Taking the product of the eigenvalues gives
> $C+\alpha I$ has the same eigenvectors as $C$, and each eigenvalue becomes $\lambda_j+\alpha$. Therefore,
> <!-- bilingual-en:end -->

### Question 8：满列秩矩阵的最小二乘与投影
<!-- bilingual-en:start -->
*Question 8: Least Squares and Projections of Full-column Rank Matrices*
<!-- bilingual-en:end -->

> [!question] 题目
> 设矩形矩阵 $A$ 的列向量线性无关。
> 1. 写出最小二乘解 $\hat x$ 与投影 $p=A\hat x$。
> 2. 说明 $p$、$e=b-p$ 分别位于哪个基本子空间。
> 3. 对
> $$
> A=\begin{bmatrix}1&0\\3&0\\0&-1\\0&-3\end{bmatrix}
> $$
> 求投影到 $C(A)$ 的矩阵。
> <!-- bilingual-en:start -->
> Suppose the columns of a rectangular matrix $A$ are linearly independent.
> **1.** Write the least-squares solution $\hat x$ and the projection $p=A\hat x$.<br>
> **2.** State which fundamental subspaces contain $p$ and $e=b-p$.<br>
> **3.** For the matrix shown below, find the matrix that projects onto $C(A)$.<br>
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> 目标是最小化 $\|Ax-b\|^2$。最近点的残差必须垂直于所有列：
> $$
> A^T(b-A\hat x)=0.
> $$
> 得正规方程
> $$
> A^TA\hat x=A^Tb.
> $$
> 由于 $A$ 满列秩，$A^TA$ 可逆：
> $$
> \boxed{\hat x=(A^TA)^{-1}A^Tb},
> $$
> $$
> \boxed{p=A\hat x=A(A^TA)^{-1}A^Tb}.
> $$
> $p$ 是列向量的线性组合，所以 $p\in C(A)$；残差满足 $A^Te=0$，所以
> $$
> e\in N(A^T).
> $$
>
> 本题两列正交，且长度平方都是 $10$：
> $$
> A^TA=\begin{bmatrix}10&0\\0&10\end{bmatrix}=10I_2.
> $$
> 因而
> $$
> P=A(A^TA)^{-1}A^T=\frac1{10}AA^T.
> $$
> 展开得到
> $$
> \boxed{
> P=\frac1{10}\begin{bmatrix}
> 1&3&0&0\\
> 3&9&0&0\\
> 0&0&1&3\\
> 0&0&3&9
> \end{bmatrix}}.
> $$
> 验算：$P^T=P$、$P^2=P$、$\operatorname{rank}(P)=2$。
> <!-- bilingual-en:start -->
> The goal is to minimize $\|Ax-b\|^2$.  The residuals of the closest point must be perpendicular to all columns:
> This gives the normal equations
> Because $A$ has full column rank, $A^TA$ is invertible:
> $p$ is a linear combination of the columns, so $p\in C(A)$; the residual satisfies $A^Te=0$, so
> The two columns in this problem are orthogonal, and both have squared norm $10$:
> Hence,
> Expanding gives
> Checking: $P^T=P$, $P^2=P$, $\operatorname{rank}(P)=2$.
> <!-- bilingual-en:end -->

### Question 9：三对角行列式递推与增长率
<!-- bilingual-en:start -->
*Question 9: Tridiagonal determinant recursion and growth rate*
<!-- bilingual-en:end -->

> [!question] 题目
> 令 $A_n$ 为主对角线为 $3$、上对角线为 $2$、下对角线为 $1$ 的 $n\times n$ 三对角矩阵，并记 $D_n=\det A_n$。
> 1. 求 $D_2,D_3$。
> 2. 推导 $D_n=aD_{n-1}+bD_{n-2}$。
> 3. 用对应二阶矩阵的特征值判断 $D_n$ 的增长率，并求 $D_5$。
> <!-- bilingual-en:start -->
> Let $A_n$ be the $n\times n$ tridiagonal matrix with $3$ on the main diagonal, $2$ on the superdiagonal, and $1$ on the subdiagonal, and define $D_n=\det A_n$.
> **1.** Find $D_2$ and $D_3$.<br>
> **2.** Derive $D_n=aD_{n-1}+bD_{n-2}$.<br>
> **3.** Use the eigenvalues of the corresponding $2\times2$ matrix to determine the growth rate of $D_n$, and compute $D_5$.<br>
> <!-- bilingual-en:end -->

> [!success]- 完整解答
> 直接计算：
> $$
> D_1=3,
> \qquad
> D_2=\begin{vmatrix}3&2\\1&3\end{vmatrix}=9-2=7.
> $$
> 对 $A_3$ 沿第一行展开：
> $$
> D_3=3D_2-2\begin{vmatrix}1&2\\0&3\end{vmatrix}
> =3\cdot7-2\cdot3=15.
> $$
> 一般地沿第一行展开。第一项留下同型的 $A_{n-1}$；第二项的余子式第一列只有顶端的 $1$ 非零，再展开一次留下 $A_{n-2}$。代数余子式符号使第二项为负，因此
> $$
> \boxed{D_n=3D_{n-1}-2D_{n-2}}.
> $$
> 写成一阶系统：
> $$
> \begin{bmatrix}D_n\\D_{n-1}\end{bmatrix}
> =\begin{bmatrix}3&-2\\1&0\end{bmatrix}
> \begin{bmatrix}D_{n-1}\\D_{n-2}\end{bmatrix}.
> $$
> 转移矩阵的特征方程为
> $$
> \lambda^2-3\lambda+2=(\lambda-1)(\lambda-2),
> $$
> 所以主导增长率为 $2^n$。由初值还可求出精确公式
> $$
> \boxed{D_n=2^{n+1}-1}.
> $$
> 因而
> $$
> \boxed{D_5=2^6-1=63}.
> $$
> 递推验算：$D_4=3\cdot15-2\cdot7=31$，$D_5=3\cdot31-2\cdot15=63$。
>
> <!-- bilingual-en:start -->
> Direct calculation:
> Expand $A_3$ along its first row:
> In general, expand along the first row. The first term leaves a matrix of the same form, $A_{n-1}$. In the minor for the second term, the first column has only one nonzero entry, the topmost $1$; expanding again leaves $A_{n-2}$. The cofactor sign makes the second term negative, so
> Write as a first-order system:
> The characteristic equation of the transition matrix is
> So the dominant growth rate is $2^n$.  An exact formula can also be derived from the initial value
> thus
> Recursive checking: $D_4=3\cdot15-2\cdot7=31$, $D_5=3\cdot31-2\cdot15=63$.
> <!-- bilingual-en:end -->

> [!warning] 官方答案笔误
> 官方答案前面正确写出 $D_n=3D_{n-1}-2D_{n-2}$，最后计算 $D_5$ 时却把负号误写为正号并得到 $207$。这与原矩阵、前述递推和直接计算均矛盾，正确答案是 $63$。
> <!-- bilingual-en:start -->
> The official solution first states the correct recurrence $D_n=3D_{n-1}-2D_{n-2}$, but then changes the minus sign to a plus sign when computing $D_5$ and obtains $207$. This contradicts the original matrix, the recurrence, and direct computation; the correct answer is $63$.
> <!-- bilingual-en:end -->

## Final Exam 错误诊断表
<!-- bilingual-en:start -->
*Final Exam Troubleshooting Table*
<!-- bilingual-en:end -->

| 题号 | 最常见错误 | 为什么错 | 快速修正 |
|---|---|---|---|
| 1 | 从 RREF 的主元列写 $C(A)$ 的具体基 | 行变换改变列向量 | 回到原矩阵取对应主元列 |
| 2 | 认为所有特征向量矩阵都正交 | 一般可对角化不等于正交可对角化 | 计算列内积 |
| 3 | 断言拼接矩阵秩一定严格增加 | 新列可能已在原列空间 | 比较列空间包含关系 |
| 4 | 把左逆、右逆的乘法顺序写反 | 两个乘积尺寸不同 | 先写每个矩阵尺寸 |
| 5 | 直接微分 Rayleigh 商 | 忽略正交特征基带来的加权平均结构 | 先在特征基展开 |
| 6 | Gram--Schmidt 后忘记归一化 | 得到的列只正交、不正交归一 | 检查 $Q^TQ=I$ |
| 7 | 把 $C+\alpha I$ 的特征值写成 $\alpha\lambda$ | 加单位阵是谱平移 | 代入 $(C+\alpha I)v$ |
| 8 | 未检查满列秩便求 $(A^TA)^{-1}$ | $A^TA$ 可能奇异 | 先检查 $N(A)=\{0\}$ |
| 9 | 余子式展开丢失负号 | 第二项带 $(-1)^{1+2}$ | 用 $D_2,D_3$ 检验递推 |
<!-- bilingual-en:start -->
| Question | Most common error | Why it is wrong | Quick correction |
|---|---|---|---|
| 1 | Using pivot columns of the RREF as a concrete basis for $C(A)$ | Row operations change the column vectors | Return to the original matrix and select its pivot columns |
| 2 | Assuming every eigenvector matrix is orthogonal | Diagonalizability does not imply orthogonal diagonalizability | Compute inner products between columns |
| 3 | Claiming that concatenation must strictly increase rank | The new columns may already lie in the original column space | Compare the two column spaces by inclusion |
| 4 | Reversing the multiplication order for left and right inverses | The two products even have different dimensions | Write down all matrix dimensions first |
| 5 | Differentiating the Rayleigh quotient immediately | This hides the weighted-average structure induced by an orthonormal eigenbasis | Expand in the eigenbasis first |
| 6 | Forgetting to normalize after Gram--Schmidt | The resulting columns are orthogonal but not orthonormal | Check $Q^TQ=I$ |
| 7 | Writing the eigenvalues of $C+\alpha I$ as $\alpha\lambda$ | Adding a scalar multiple of the identity shifts the spectrum | Substitute into $(C+\alpha I)v$ |
| 8 | Computing $(A^TA)^{-1}$ without checking full column rank | $A^TA$ may be singular | First check that $N(A)=\{0\}$ |
| 9 | Losing the minus sign in the cofactor expansion | The second term carries $(-1)^{1+2}$ | Test the recurrence against $D_2$ and $D_3$ |
<!-- bilingual-en:end -->

## 三道全课程自检
<!-- bilingual-en:start -->
*Three whole-course self-checks*
<!-- bilingual-en:end -->

1. 对 $A\in\mathbb R^{m\times n}$，说明 $N(A)=\{0\}$ 分别对列、$A^TA$、最小二乘解和左逆意味着什么。
2. 比较 eigen-decomposition 与 SVD：各自需要什么条件，输入输出方向分别是什么？
3. 若一个计算结果声称 $P$ 是正交投影矩阵，至少应检查哪三条性质？
<!-- bilingual-en:start -->

&nbsp;
**1.** For $A\in\mathbb R^{m\times n}$, explain what $N(A)=\{0\}$ means for columns, $A^TA$, least squares solution, and left inverse.<br>
**2.** Comparing eigen-decomposition and SVD: What are the conditions and the input and output directions?<br>
**3.** If a calculation indicates that $P$ is an orthogonal projection matrix, at least which three properties should be examined?<br>
<!-- bilingual-en:end -->

> [!success]- 自检答案
> 1. $N(A)=\{0\}$ 等价于列独立、$r=n$；于是 $A^TA$ 正定可逆，每个最小二乘问题有唯一系数解，并存在左逆 $(A^TA)^{-1}A^T$。
> 2. 特征分解 $A=S\Lambda S^{-1}$ 要求方阵且有完整特征向量组，描述同一空间中的不变方向；SVD 对任意矩形矩阵存在，$v_i$ 是输入方向、$u_i$ 是输出方向，二者由 $Av_i=\sigma_i u_i$ 相连。
> 3. 检查 $P^T=P$、$P^2=P$，并检查 $C(P)$ 是否等于目标子空间；还可检查特征值只能是 $0,1$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $N(A)=\{0\}$ is equivalent to linear independence of the columns and to $r=n$. Consequently, $A^TA$ is positive definite and invertible, every least-squares problem has a unique coefficient vector, and the left inverse $(A^TA)^{-1}A^T$ exists.<br>
> **2.** The eigendecomposition $A=S\Lambda S^{-1}$ requires a square matrix with a complete set of eigenvectors and describes invariant directions within one space. An SVD exists for every rectangular matrix; $v_i$ is an input direction, $u_i$ is an output direction, and they are related by $Av_i=\sigma_i u_i$.<br>
> **3.** Check $P^T=P$ and $P^2=P$, and verify that $C(P)$ is the intended target subspace. You can also check that every eigenvalue is either $0$ or $1$.<br>
> <!-- bilingual-en:end -->

## 本地材料

- [[MIT_OCW_18.06SC_PDF/01_Exams/MIT18_06SCF11_FinalRevsum.pdf#page=1|Final Course Review summary（pp.1–7）]]
- [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/FINAL_Lecture_Final_Course_Review.pdf#page=1|Final Course Review lecture transcript（pp.1–20）]]
- [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/FINAL_Recitation_Final_Exam_Problem_Solving.pdf#page=1|Final Exam problem-solving transcript（pp.1–4）]]
- [[MIT_OCW_18.06SC_PDF/01_Exams/MIT18_06SCF11_final_ex.pdf#page=1|Final Exam problems（pp.1–10）]]
- [[MIT_OCW_18.06SC_PDF/01_Exams/MIT18_06SCF11_final_exs.pdf#page=1|Final Exam official solutions（pp.1–14）]]

**知识链：**$Ax=b$ 与四子空间 → 正交投影与最小二乘 → 行列式与谱 → 对称正定结构 → SVD 与伪逆 → 统一的结构判断、计算和验算。
<!-- bilingual-en:start -->
**Knowledge chain:** $Ax=b$ and the four fundamental subspaces → orthogonal projections and least squares → determinants and spectra → symmetric positive-definite structure → SVD and the pseudoinverse → unified structural reasoning, computation, and verification.
<!-- bilingual-en:end -->
