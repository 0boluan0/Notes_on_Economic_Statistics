---
aliases:
  - MIT 18.06SC Unit I
  - Ax=b 与四个基本子空间
tags:
  - 线性代数
  - mit-ocw
  - course-note
---

# Ax = b and the Four Subspaces

> [!abstract] 本单元真正要解决的问题
> 给定 $A\in\mathbb F^{m\times n}$、$x\in\mathbb F^n$、$b\in\mathbb F^m$（本单元通常取 $\mathbb F=\mathbb R$），怎样判断 $Ax=b$ 是否有解、解是否唯一，并把所有解写完整？算法上用消元；结构上用列空间、零空间、秩、基与四个基本子空间。本篇的最终目标不是“会做行运算”，而是能解释每一步行运算揭示了什么空间结构。
> <!-- bilingual-en:start -->
> Given $A\in\mathbb F^{m\times n}$, $x\in\mathbb F^n$, and $b\in\mathbb F^m$ (usually with $\mathbb F=\mathbb R$ in this unit), how can we decide whether $Ax=b$ is solvable, whether its solution is unique, and how to describe every solution? Elimination supplies the algorithm; column space, nullspace, rank, bases, and the four fundamental subspaces supply the structure. The aim is not merely to carry out row operations, but to understand what each operation reveals about the underlying spaces.
> <!-- bilingual-en:end -->

## 课程来源、约定与导航
<!-- bilingual-en:start -->
*Course Source, Engagement and Navigation*
<!-- bilingual-en:end -->

- 官方课程：MIT OCW 18.06SC *Linear Algebra, Fall 2011*；总入口见 [[00_MIT OCW 18.06SC course map|课程总览]]。
- 本地资料索引：[[MIT_OCW_18.06SC_PDF/index|MIT 18.06SC PDF 索引]]。
- 本篇严格按官网逻辑顺序写：Geometry → Overview → Elimination → Inverse → LU → Vector Spaces → Column/Null Spaces → $Ax=0$ → $Ax=b$ → Basis/Dimension → Four Subspaces → Matrix Spaces → Graphs → Review → Exam 1。
- **编号提醒**：官网第二讲 Overview 的本地 summary 是 `Ses1.13sum.pdf`；官网第三至第十三讲依次使用本地 `Ses1.2–Ses1.12` 资料。
- 尺寸检查规则：若 $A$ 是 $m\times n$，则它有 $m$ 行、$n$ 列；$Ax$ 只有在 $x\in\mathbb F^n$ 时有定义，结果属于 $\mathbb F^m$。
- 除非特别说明，向量均写成列向量。$C(A)$、$N(A)$、$C(A^T)$、$N(A^T)$ 分别表示列空间、零空间、行空间和左零空间。
<!-- bilingual-en:start -->
- Official course: MIT OCW 18.06SC *Linear Algebra, Fall 2011*; see [[00_MIT OCW 18.06SC course map|Course Overview]] for the main entrance.
- Local Data Index: [[MIT_OCW_18.06SC_PDF/index|MIT 18.06SC PDF Index]].
- This story is written in strict order on the official website: Geometry → Overview → Elimination → Inverse → LU → Vector Spaces → Column/Null Spaces → $Ax=0$ → $Ax=b$ → Basis/Dimension → Four Subspaces → Matrix Spaces → Graphs → Review → Exam 1.
-**Code alert**: Lecture 2 on the official website, Overview's local summary is `Ses1.13sum.pdf`; Lecture 3 to 13 on the official website, use local `Ses1.2–Ses1.12` data in turn.
- Dimension check rule: If $A$ is $m\times n$, it has $m$ rows, $n$ columns; $Ax$ is only defined when $x\in\mathbb F^n$, and the result is $\mathbb F^m$.
- Vectors are written as column vectors unless otherwise specified.  $C(A)$, $N(A)$, $C(A^T)$, $N(A^T)$ represent column space, nullspace, row space, and left nullspace, respectively.
<!-- bilingual-en:end -->

### Session 目录

1. [[#Session 1.1 The geometry of linear equations|线性方程的几何图像]]
2. [[#Session 1.2 An overview of key ideas|全课程结构预览]]
3. [[#Session 1.3 Elimination with matrices|矩阵消元]]
4. [[#Session 1.4 Multiplication and inverse matrices|矩阵乘法与逆矩阵]]
5. [[#Session 1.5 Factorization into A = LU|LU 分解]]
6. [[#Session 1.6 Transposes, permutations, vector spaces|转置、置换与向量空间]]
7. [[#Session 1.7 Column space and nullspace|列空间与零空间]]
8. [[#Session 1.8 Solving Ax = 0: pivot variables and special solutions|齐次系统与特殊解]]
9. [[#Session 1.9 Solving Ax = b: row reduced form R|非齐次系统与完整解]]
10. [[#Session 1.10 Independence, basis, and dimension|线性无关、基与维数]]
11. [[#Session 1.11 The four fundamental subspaces|四个基本子空间]]
12. [[#Session 1.12 Matrix spaces, rank 1, and small world graphs|矩阵空间与秩一矩阵]]
13. [[#Session 1.13 Graphs, networks, and incidence matrices|图、网络与关联矩阵]]
14. [[#Session 1.14 Exam 1 review|Exam 1 复习]]
15. [[#Exam 1 完整题解|Exam 1 完整题解]]

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-row-column-picture.png|760]]

## Session 1.1 The geometry of linear equations

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：同一个方程组为什么既能看成几何对象的交，又能看成列向量的线性组合？
<!-- bilingual-en:start -->
**Question**: Why can the same system of equations be regarded as both the intersection of geometric objects and the linear combination of column vectors?
<!-- bilingual-en:end -->

**前置知识**：二元一次方程、向量加法与数乘。本节首次建立 [[线性代数 Course Atlas|线性代数]] 的三个视角。
<!-- bilingual-en:start -->
**Prerequisites**: systems of two linear equations, vector addition, and scalar multiplication. This section introduces the course's three complementary views of [[线性代数 Course Atlas|linear algebra]].
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.1sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S01_Lecture_The_Geometry_of_Linear_Equations.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S01_Recitation_Geometry_of_Linear_Algebra.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.1prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.1sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.1sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S01_Lecture_The_Geometry_of_Linear_Equations.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S01_Recitation_Geometry_of_Linear_Algebra.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.1prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.1sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 从 row picture 到 column picture
<!-- bilingual-en:start -->
*1. From row picture to column picture*
<!-- bilingual-en:end -->

考虑课堂中的系统
<!-- bilingual-en:start -->
Consider systems in the classroom
<!-- bilingual-en:end -->

$$
\begin{cases}
2x-y=0,\\
-x+2y=3.
\end{cases}
$$

**行图像（row picture）**把每一个方程看成 $xy$ 平面中的直线。第一条是 $y=2x$，第二条是 $y=(x+3)/2$；交点同时满足两个方程。联立得
<!-- bilingual-en:start -->
The **row picture** treats each equation as a line in the $xy$-plane. The first is $y=2x$ and the second is $y=(x+3)/2$; their intersection satisfies both equations. Solving them simultaneously gives
<!-- bilingual-en:end -->

$$
2x=\frac{x+3}{2}\Longrightarrow 4x=x+3\Longrightarrow x=1,
\qquad y=2.
$$

因此交点是 $(1,2)$。在 $m$ 个方程、$n$ 个未知数的一般情形中，每一行在 $\mathbb R^n$ 中给出一个超平面；解集是这些超平面的交。
<!-- bilingual-en:start -->
So the intersection is $(1,2)$.  In the general case of $m$ equations and $n$ unknowns, each row gives a hyperplane in $\mathbb R^n$; the solution set is the intersection of these hyperplanes.
<!-- bilingual-en:end -->

**列图像（column picture）**把同一系统改写成
<!-- bilingual-en:start -->
**column picture**rewrites the same system as
<!-- bilingual-en:end -->

$$
x\begin{bmatrix}2\\-1\end{bmatrix}
+y\begin{bmatrix}-1\\2\end{bmatrix}
=\begin{bmatrix}0\\3\end{bmatrix}.
$$

未知数 $x,y$ 不再只是平面坐标，而是两列的组合系数。代入 $(x,y)=(1,2)$：
<!-- bilingual-en:start -->
The unknown $x,y$ is no longer just a planar coordinate, but a combination of two columns.  Introduce $(x,y)=(1,2)$:
<!-- bilingual-en:end -->

$$
\begin{bmatrix}2\\-1\end{bmatrix}
+2\begin{bmatrix}-1\\2\end{bmatrix}
=\begin{bmatrix}0\\3\end{bmatrix}.
$$

这引出 [[线性方程组与四个基本子空间#四个基本子空间|列空间]]：$Ax=b$ 有解，当且仅当 $b$ 属于 $A$ 的列向量所张成的空间。
<!-- bilingual-en:start -->
This leads to a [[线性方程组与四个基本子空间#四个基本子空间|column space]]:$Ax=b$ solution if and only if the $b$ belongs to the space spanned by the column vector of $A$.
<!-- bilingual-en:end -->

### 2. 矩阵乘向量的两种等价读法
<!-- bilingual-en:start -->
*2. Two kinds of equivalent reading methods of matrix multiplication vector*
<!-- bilingual-en:end -->

设
<!-- bilingual-en:start -->
if
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}a_1&a_2&\cdots&a_n\end{bmatrix}\in\mathbb R^{m\times n},
\qquad x=\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix}\in\mathbb R^n.
$$

按列读：
<!-- bilingual-en:start -->
Read by column:
<!-- bilingual-en:end -->

$$
Ax=x_1a_1+\cdots+x_na_n\in\mathbb R^m.
$$

按行读：若 $r_i^T$ 是第 $i$ 行，则
<!-- bilingual-en:start -->
Read By Row: If $r_i^T$ is row $i$, then
<!-- bilingual-en:end -->

$$
Ax=\begin{bmatrix}r_1^Tx\\\vdots\\r_m^Tx\end{bmatrix}.
$$

前者强调“输出是哪些列的组合”，后者强调“每个方程如何约束输入”。二者是同一次矩阵乘法，不是两个不同定义。
<!-- bilingual-en:start -->
The former emphasizes "which columns the output is a combination of" and the latter emphasizes "how each equation constrains the input".  The two are the same matrix multiplication, not two different definitions.
<!-- bilingual-en:end -->

> [!proof] 为什么列组合公式必然成立
> **目标**：证明矩阵乘向量等于列的线性组合。
>
> **构造**：比较等式两侧的第 $i$ 个分量。矩阵乘法定义给出
> $$
> (Ax)_i=\sum_{j=1}^n a_{ij}x_j.
> $$
> 第 $j$ 列 $a_j$ 的第 $i$ 个分量是 $a_{ij}$，所以
> $$
> (x_1a_1+\cdots+x_na_n)_i
> =\sum_{j=1}^n x_ja_{ij}.
> $$
> 两式逐分量相等。
>
> **边界与尺寸**：每个 $a_j\in\mathbb R^m$，所以线性组合仍在 $\mathbb R^m$；系数恰有 $n$ 个，与 $x$ 的长度一致。
>
> **结论**：$Ax=\sum_jx_ja_j$。
> <!-- bilingual-en:start -->
> **Target**: Prove that the matrix multiplier vector is equal to a linear combination of columns.
> **Construct**: Compares the $i$ component on either side of the equation.  The definition of matrix multiplication is given
> the $j$ column The $i$ component of $a_j$ is $a_{ij}$, so
> The two formulas are equal one by one.
> **Boundary and Dimensions**: Each $a_j\in\mathbb R^m$, so the linear combination is still $\mathbb R^m$; the coefficients are exactly $n$, consistent with the length of $x$.
> **Conclusion**: $Ax=\sum_jx_ja_j$.
> <!-- bilingual-en:end -->

### 3. 三种解的几何命运
<!-- bilingual-en:start -->
*3. Geometric fate of the three solutions*
<!-- bilingual-en:end -->

- **唯一解**：$b$ 可由列生成，而且系数表示唯一。
- **无解**：$b$ 不在列空间中。
- **无穷多解**：$b$ 可达，但存在非零向量 $z$ 使 $Az=0$；若 $Ax_p=b$，则 $A(x_p+tz)=b$。
<!-- bilingual-en:start -->
- **Unique solution**: $b$ lies in the column space and the columns are independent, so the coefficient vector is unique.
- **No solution**: $b$ does not lie in the column space.
- **Infinitely many solutions**: $b$ is attainable, but some nonzero vector $z$ satisfies $Az=0$; if $Ax_p=b$, then $A(x_p+tz)=b$ for every scalar $t$.
<!-- bilingual-en:end -->

这里已经预告了 [[线性方程组与四个基本子空间#四个基本子空间|零空间]] 和 [[线性方程组与四个基本子空间#可解性与完整解|线性方程组解结构]]。注意：方阵并不自动可逆；列必须既独立又张成整个输出空间。
<!-- bilingual-en:start -->
This already anticipates the [[线性方程组与四个基本子空间#四个基本子空间|nullspace]] and the [[线性方程组与四个基本子空间#可解性与完整解|solution structure of a linear system]]. A square matrix is not automatically invertible: its columns must be linearly independent and must span the entire output space.
<!-- bilingual-en:end -->

### 4. Recitation 代表例题
<!-- bilingual-en:start -->
*4. Recitation representative example*
<!-- bilingual-en:end -->

Recitation 使用
<!-- bilingual-en:start -->
Recitation using
<!-- bilingual-en:end -->

$$
\begin{cases}
2x+y=3,\\
x-2y=-1.
\end{cases}
$$

第二式给 $x=2y-1$，代入第一式：$2(2y-1)+y=3$，所以 $5y=5$、$y=1$、$x=1$。列图像为
<!-- bilingual-en:start -->
The second model is given to $x=2y-1$, and the first model is given to $2(2y-1)+y=3$, so $5y=5$, $y=1$, $x=1$.  Column image is
<!-- bilingual-en:end -->

$$
x\begin{bmatrix}2\\1\end{bmatrix}
+y\begin{bmatrix}1\\-2\end{bmatrix}
=\begin{bmatrix}3\\-1\end{bmatrix}.
$$

右侧恰是两列各取一份。验证时既要代回两条原方程，也可直接做一次矩阵乘法。
<!-- bilingual-en:start -->
On the right is two columns, one for each.  In the verification, the two original equations should be substituted back, and the matrix multiplication can also be done directly.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 1.1：判断三个向量的相关性
> **题目转述**：给定
> $$
> w_1=\begin{bmatrix}1\\2\\3\end{bmatrix},\quad
> w_2=\begin{bmatrix}4\\5\\6\end{bmatrix},\quad
> w_3=\begin{bmatrix}7\\8\\9\end{bmatrix},
> $$
> 找到非全零的 $x_1,x_2,x_3$ 使 $x_1w_1+x_2w_2+x_3w_3=0$，判断相关性并说明它们位于什么几何对象中。
>
> **解答**：观察相邻向量差相同：$w_2-w_1=w_3-w_2=(3,3,3)^T$，因此
> $$
> w_1-2w_2+w_3=0.
> $$
> 系数 $(1,-2,1)$ 非全零，故三向量线性相关。它们都在由 $w_1,w_2$ 张成的、经过原点的平面中；以它们为列的 $3\times3$ 矩阵没有三条独立方向，因此不可逆。
> <!-- bilingual-en:start -->
> **Title Restatement**: Given
> Locate a non-all-zero $x_1,x_2,x_3$ to make $x_1w_1+x_2w_2+x_3w_3=0$, determine the correlation, and explain what geometry they are in.
> **Answer**: Observe the same difference between adjacent vectors: $w_2-w_1=w_3-w_2=(3,3,3)^T$, so
> The coefficients $(1,-2,1)$ are not all zero, so the three vectors are linearly dependent. They lie in the plane through the origin spanned by $w_1$ and $w_2$; the $3\times3$ matrix having these vectors as columns therefore lacks three independent directions and is singular.
> <!-- bilingual-en:end -->

> [!question]- Problem 1.2：矩阵乘向量
> **题目转述**：计算
> $$
> \begin{bmatrix}1&2&0\\2&0&3\\4&1&1\end{bmatrix}
> \begin{bmatrix}3\\-2\\1\end{bmatrix}.
> $$
>
> **解答**：尺寸为 $(3\times3)(3\times1)=3\times1$。逐行点乘：
> $$
> \begin{bmatrix}
> 1(3)+2(-2)+0(1)\\
> 2(3)+0(-2)+3(1)\\
> 4(3)+1(-2)+1(1)
> \end{bmatrix}
> =\begin{bmatrix}-1\\9\\11\end{bmatrix}.
> $$
> 也可读成 $3a_1-2a_2+a_3$，结果相同。
> <!-- bilingual-en:start -->
> **Restatement**: Compute the matrix–vector product.
> **Answer**: The dimensions are $(3\times3)(3\times1)=3\times1$. Taking the dot product of each row with the vector gives the displayed result. Equivalently, form the column combination $3a_1-2a_2+a_3$; both views give the same vector.
> <!-- bilingual-en:end -->

> [!question]- Problem 1.3：矩阵乘法的尺寸
> **题目转述**：判断“$3\times2$ 矩阵 $A$ 乘 $2\times3$ 矩阵 $B$ 得到 $3\times3$ 矩阵 $AB$”是否正确。
>
> **解答**：正确。一般地，
> $$
> A_{m\times n}B_{n\times p}=(AB)_{m\times p}.
> $$
> 内侧尺寸 $n$ 必须相等，外侧尺寸 $m,p$ 成为结果的尺寸。
> <!-- bilingual-en:start -->
> **Title**: Determine whether "$3\times2$ matrix $A$ multiplied by $2\times3$ matrix $B$ to get $3\times3$ matrix $AB$" is correct.
> **Answer**: Correct.  In general,
> The inner dimensions $n$ must agree, and the outer dimensions $m,p$ determine the size of the product.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- 行图像位于未知数空间 $\mathbb R^n$；列图像位于输出空间 $\mathbb R^m$。当 $m\ne n$ 时，这两个空间甚至维数不同。
- “列数多”不等于“张成空间大”；重复列会带来冗余。
- $Ax=0$ 永远至少有零解；“无解”只可能发生在非齐次系统 $Ax=b$ 中。
<!-- bilingual-en:start -->
- The row image is in the unknown space $\mathbb R^n$; the column image is in the output space $\mathbb R^m$.  When $m\ne n$, the two spaces have different dimensions.
- More columns do not necessarily produce a larger span; repeated or dependent columns add redundancy.
- $Ax=0$ always has at least the zero solution; only a nonhomogeneous system $Ax=b$ can be inconsistent.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 尺寸题
> 若 $A\in\mathbb R^{4\times3}$，那么 $x$、$b$ 分别属于什么空间？
>
> **答案**：$x\in\mathbb R^3$，$b=Ax\in\mathbb R^4$。
> <!-- bilingual-en:start -->
> If $A\in\mathbb R^{4\times3}$, then what space does $x$ and $b$ belong to?
> **Answer**: $x\in\mathbb R^3$, $b=Ax\in\mathbb R^4$.
> <!-- bilingual-en:end -->

> [!question]- 2. 结构题
> 若 $Az=0$ 且 $z\ne0$，为什么 $Ax=b$ 不可能有唯一解？
>
> **答案**：只要有一个特解 $x_p$，便有 $A(x_p+tz)=b$；不同 $t$ 给出不同解。若没有特解，则是无解，也不是唯一解。
> <!-- bilingual-en:start -->
> If $Az=0$ and $z\ne0$, why can't $Ax=b$ have a unique solution?
> **Answer:** If a particular solution $x_p$ exists, then $A(x_p+tz)=b$ for every scalar $t$; distinct values of $t$ give distinct solutions. If no particular solution exists, the system is inconsistent, so uniqueness is not even at issue.
> <!-- bilingual-en:end -->

> [!question]- 3. 计算题
> 把 $\begin{bmatrix}1&-1\\2&3\end{bmatrix}(4,1)^T$ 同时写成列组合并算出结果。
>
> **答案**：$4(1,2)^T+1(-1,3)^T=(3,11)^T$。
> <!-- bilingual-en:start -->
> Write $\begin{bmatrix}1&-1\\2&3\end{bmatrix}(4,1)^T$ as a column combination and calculate the results.
> **Answer**: $4(1,2)^T+1(-1,3)^T=(3,11)^T$.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

方程交点 → 列向量组合 → $b\in C(A)$ 决定存在性 → $N(A)$ 决定唯一性 → 下一步用消元系统地找出这些结构。
<!-- bilingual-en:start -->
The intersection of the equations → the combination of the column vectors → $b\in C(A)$ determines the existence → $N(A)$ determines the uniqueness → the next step is to systematically identify these structures by elimination.
<!-- bilingual-en:end -->

## Session 1.2 An overview of key ideas

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：消元、子空间、正交、特征值和 SVD 为什么不是彼此无关的技巧？
<!-- bilingual-en:start -->
**Question**: Why are elimination, subspace, orthogonality, eigenvalue and SVD not mutually independent techniques?
<!-- bilingual-en:end -->

**前置知识**：能从 Session 1.1 读懂 $Ax=b$ 的行图像和列图像。
<!-- bilingual-en:start -->
**Prerequisites**: $Ax=b$ row and column images are readable from Session 1.1.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.13sum.pdf#page=1|overview summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S02_Lecture_An_Overview_of_Linear_Algebra.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S02_Recitation_An_Overview_of_Key_Ideas.pdf#page=1|recitation transcript p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.13sum.pdf#page=1|overview summary p.1]] [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S02_Lecture_An_Overview_of_Linear_Algebra.pdf#page=1|lecture transcript p.1]] [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S02_Recitation_An_Overview_of_Key_Ideas.pdf#page=1|recitation transcript p.1]]
<!-- bilingual-en:end -->

### 1. 整门课围绕一个输入—输出问题
<!-- bilingual-en:start -->
*1. The whole course revolves around one input—output*
<!-- bilingual-en:end -->

矩阵 $A\in\mathbb R^{m\times n}$ 是线性映射
<!-- bilingual-en:start -->
Matrix $A\in\mathbb R^{m\times n}$ is a linear mapping
<!-- bilingual-en:end -->

$$
A:\mathbb R^n\longrightarrow\mathbb R^m.
$$

课程不断追问四件事：
<!-- bilingual-en:start -->
Four things are being asked:
<!-- bilingual-en:end -->

1. 哪些输入被送到 $0$？答案是 $N(A)$。
2. 哪些输出可以被达到？答案是 $C(A)$。
3. 不可达时，哪个可达输出离目标最近？答案由正交投影与最小二乘给出。
4. 哪些输入方向在作用后只改变长度、不改变方向，或对应最自然的输入—输出方向？答案通向特征向量和 SVD。
<!-- bilingual-en:start -->
1. Which inputs are sent to $0$?  The answer is $N(A)$.
2. Which outputs can be achieved?  The answer is $C(A)$.
3. When unreachable, which reachable output is the closest to the target?  The answer is given by orthogonal projection and least squares.
4. Which input directions are merely rescaled rather than rotated, and which paired input–output directions are most natural for the transformation? These questions lead to eigenvectors and the SVD.
<!-- bilingual-en:end -->

### 2. 差分矩阵展示可逆与奇异
<!-- bilingual-en:start -->
*2. A difference matrix illustrates invertibility and singularity*
<!-- bilingual-en:end -->

普通差分矩阵
<!-- bilingual-en:start -->
ordinary difference matrix
<!-- bilingual-en:end -->

$$
D=\begin{bmatrix}
1&0&0\\
-1&1&0\\
0&-1&1
\end{bmatrix}
$$

把位置 $x=(x_1,x_2,x_3)^T$ 变为差分 $(x_1,x_2-x_1,x_3-x_2)^T$。它是下三角矩阵且三个对角元均为 $1$，因此可以从差分逐步恢复 $x$。
<!-- bilingual-en:start -->
Change the position $x=(x_1,x_2,x_3)^T$ into the differential $(x_1,x_2-x_1,x_3-x_2)^T$.  It is a lower triangular matrix and all three diagonal elements are $1$, so $x$ can be recovered from the difference step by step.
<!-- bilingual-en:end -->

若首尾相连得到循环差分矩阵
<!-- bilingual-en:start -->
Connecting the two ends gives the cyclic difference matrix
<!-- bilingual-en:end -->

$$
C=\begin{bmatrix}
-1&1&0\\
0&-1&1\\
1&0&-1
\end{bmatrix},
$$

则 $C(1,1,1)^T=0$。常数位移被完全丢失；并且每个输出 $b=Cx$ 都满足 $b_1+b_2+b_3=0$。这同时展示了奇异矩阵的两面：存在非零零空间方向，并且列空间不能覆盖整个输出空间。
<!-- bilingual-en:start -->
$C(1,1,1)^T=0$. Constant displacements are therefore lost completely, and every output $b=Cx$ satisfies $b_1+b_2+b_3=0$. This displays both sides of singularity: the nullspace contains a nonzero direction, and the column space does not cover the whole output space.
<!-- bilingual-en:end -->

### 3. 课程地图
<!-- bilingual-en:start -->
*3. Course map*
<!-- bilingual-en:end -->

- **Elimination**：把方程组变成容易读的形状。
- **Rank / basis / dimension**：数清独立方向。
- **Four fundamental subspaces**：统一输入端与输出端的可达、丢失结构。
- **Orthogonality / least squares**：处理无精确解时的最佳近似。
- **Determinant / eigenvalues**：描述体积缩放和反复作用。
- **SVD**：为任意 $m\times n$ 矩阵找最自然的正交输入、输出方向。
<!-- bilingual-en:start -->
-**Elimination**: Make the equations easy to read.
-**Rank / basis / dimension**: Count independent directions.
- **Four fundamental subspaces**: unify the reachable and lost directions on the input and output sides.
- **Orthogonality / least squares**: provide the best approximation when no exact solution exists.
- **Determinant / eigenvalues**: describe volume scaling and repeated application of a matrix.
- **SVD**: finds the most natural orthogonal input and output directions for any $m\times n$ matrix.
<!-- bilingual-en:end -->

其中 [[线性方程组与四个基本子空间#基、维数与秩|秩]] 是贯穿全课的有效维数。若 $A$ 有 $n$ 列，秩—零度关系为
<!-- bilingual-en:start -->
[[线性方程组与四个基本子空间#基、维数与秩|Rank]] is the effective dimension used throughout the course. If $A$ has $n$ columns, the rank–nullity relation is
<!-- bilingual-en:end -->

$$
\operatorname{rank}(A)+\dim N(A)=n.
$$

本式将在 Session 1.10 完整证明。
<!-- bilingual-en:start -->
This identity is proved in full in Session 1.10.
<!-- bilingual-en:end -->

### 4. Recitation 反向推理例题
<!-- bilingual-en:start -->
*4. Recitation reverse reasoning problem*
<!-- bilingual-en:end -->

已知 $A$ 有三列，且
<!-- bilingual-en:start -->
$A$ is known to have three columns, and
<!-- bilingual-en:end -->

$$
Ax=b\quad\text{的全部解为}\quad
x=\begin{bmatrix}0\\1\\1\end{bmatrix}
+t\begin{bmatrix}0\\2\\1\end{bmatrix},
\qquad b=\begin{bmatrix}1\\4\\1\\1\end{bmatrix}.
$$

尺寸先行：$x\in\mathbb R^3$、$b\in\mathbb R^4$，故 $A\in\mathbb R^{4\times3}$。记列为 $c_1,c_2,c_3$。特解给出
<!-- bilingual-en:start -->
Size First: $x\in\mathbb R^3$, $b\in\mathbb R^4$, so $A\in\mathbb R^{4\times3}$.  It's listed as $c_1,c_2,c_3$.  give out
<!-- bilingual-en:end -->

$$
c_2+c_3=b,
$$

齐次方向给出
<!-- bilingual-en:start -->
Homogeneous direction given
<!-- bilingual-en:end -->

$$
2c_2+c_3=0.
$$

> [!note] Transcript 勘误
> 本地 transcript 的题首把齐次方向写成 $(1,2,1)^T$，后续计算却使用 $(0,2,1)^T$。原始 MIT Fall 1999 Quiz 1 Q4 与官方解答都确认后者正确。因此
> $$
> c_2+c_3=b,\qquad 2c_2+c_3=0
> $$
> 给出 $c_2=-b,c_3=2b$。又因全部解只有一个自由方向，nullity $=1$、rank $=2$；而 $c_2,c_3$ 只张成 $\operatorname{span}(b)$，故 $c_1$ 必须不是 $b$ 的倍数，才能使列空间达到二维。
> <!-- bilingual-en:start -->
> Local transcript's title begins with the homogeneous direction as $(1,2,1)^T$, but subsequent calculations use $(0,2,1)^T$.  The original MIT Fall 1999 Quiz 1 Q4 and the official answer both confirm the latter is correct.  therefore
> Give $c_2=-b,c_3=2b$.  Because all solutions have only one free direction, nullity $=1$, rank $=2$, and $c_2,c_3$ is only $\operatorname{span}(b)$, $c_1$ must not be a multiple of $b$ to make the column space reach two-dimensional.
> <!-- bilingual-en:end -->

### 5. 为什么“结构先于计算”
<!-- bilingual-en:start -->
*5. Why "structure precedes calculation"*
<!-- bilingual-en:end -->

同一道题可有很多行运算路线，但秩、零空间维数、可解条件不会随路线改变。可靠解题顺序是：
<!-- bilingual-en:start -->
The same problem may admit many computational routes, but rank, nullity, and consistency conditions do not depend on the route. A reliable solution sequence is:
<!-- bilingual-en:end -->

1. 写尺寸；
2. 判断题目问存在性、唯一性还是参数化；
3. 再选择消元、子空间或分解；
4. 最后代回或做维数检查。
<!-- bilingual-en:start -->
1. Write size;
2. To judge the existence, uniqueness or parameterization of the question;
3. Choosing elimination, subspace or decomposition;
4. Last generation or dimension check.
<!-- bilingual-en:end -->

### 易错点与边界
<!-- bilingual-en:start -->
*Fault-prone points and boundaries*
<!-- bilingual-en:end -->

- $N(A)$ 在输入空间 $\mathbb R^n$，$C(A)$ 在输出空间 $\mathbb R^m$，不能相加或直接比较，除非 $m=n$ 且另有语境。
- 秩下降同时影响存在性与唯一性，但两者不是同一句话：存在性由 $b\in C(A)$ 决定，唯一性由 $N(A)=\{0\}$ 决定。
- Overview 是官方第二讲；不要按本地 `Ses1.13` 文件名把它放到图论之后。
<!-- bilingual-en:start -->
- $N(A)$ in input space $\mathbb R^n$, $C(A)$ in output space $\mathbb R^m$, cannot be added or compared directly, unless $m=n$ and there is another context.
- Rank decline affects both existence and uniqueness, but they are not the same sentence: existence is determined by $b\in C(A)$ and uniqueness by $N(A)=\{0\}$.
- Overview is the official second lecture; do not put it after the graph by the local `Ses1.13` filename.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 为什么循环差分矩阵不可能可逆？
> 因为 $C\mathbf1=0$ 且 $\mathbf1\ne0$，映射不是一一对应。
> <!-- bilingual-en:start -->
> Because $C\mathbf1=0$ and $\mathbf1\ne0$, the mapping is not one-to-one.
> <!-- bilingual-en:end -->

> [!question]- 2. 若 $A\in\mathbb R^{5\times3}$ 且 nullity $=1$，rank 是多少？
> 由 rank-nullity，$r=3-1=2$。
> <!-- bilingual-en:start -->
> By rank-nullity, $r=3-1=2$.
> <!-- bilingual-en:end -->

> [!question]- 3. 若 $N(A)=\{0\}$，是否保证每个 $b\in\mathbb R^m$ 都可解？
> 不保证。它只保证“至多一个解”；还需 $C(A)=\mathbb R^m$ 才保证存在。高矩阵可有独立列但不能覆盖全部 $\mathbb R^m$。
> <!-- bilingual-en:start -->
> No promises.  It only guarantees "at most one solution"; it also requires $C(A)=\mathbb R^m$ to guarantee existence.  High matrices can have independent columns but cannot overwrite all $\mathbb R^m$.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$Ax=b$ → 输入端 $N(A)$ 与输出端 $C(A)$ → rank 计数有效方向 → 正交、特征值与 SVD 将在后续单元继续刻画这些方向。
<!-- bilingual-en:start -->
$Ax=b$ → the input-side nullspace $N(A)$ and output-side column space $C(A)$ → rank counts effective directions → later units use orthogonality, eigenvalues, and the SVD to characterize those directions more precisely.
<!-- bilingual-en:end -->

## Session 1.3 Elimination with matrices

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：怎样用不改变解集的操作，把 $Ax=b$ 变成可回代的上三角系统？
<!-- bilingual-en:start -->
**Question**: How can we transform $Ax=b$ into an upper triangular system that can be solved by back-substitution, without changing its solution set?
<!-- bilingual-en:end -->

**前置知识**：矩阵表示、矩阵乘向量、方程组等价。
<!-- bilingual-en:start -->
**Prerequisites**: matrix representations, matrix-vector multiplication, and equivalent systems of equations.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.2sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S03_Lecture_Elimination_with_Matrices.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S03_Recitation_Recitation_Elimination_with_Matrices.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.2prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.2sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.2sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S03_Lecture_Elimination_with_Matrices.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S03_Recitation_Recitation_Elimination_with_Matrices.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.2prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.2sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 三种基本行操作
<!-- bilingual-en:start -->
*1. Three basic row operations*
<!-- bilingual-en:end -->

[[线性方程组与四个基本子空间#消元、主元与 LU|高斯消元（Gaussian elimination）]]对增广矩阵 $[A\mid b]$ 反复使用三类可逆行操作：
<!-- bilingual-en:start -->
[[线性方程组与四个基本子空间#消元、主元与 LU|Gaussian elimination]] repeatedly uses three types of invertible operations on the augmented matrix $[A\mid b]$:
<!-- bilingual-en:end -->

1. 交换两行；
2. 一行乘非零标量；
3. 一行加上另一行的任意倍数。
<!-- bilingual-en:start -->
1. Exchange two lines;
2. Multiply one row by a non-zero scalar;
3. One row plus any multiple of the other.
<!-- bilingual-en:end -->

> [!proof] 行操作为什么保持解集
> **目标**：说明每种行操作前后方程组等价。
>
> **构造与依据**：交换方程只改变书写顺序；把方程乘 $c\ne0$ 可再乘 $1/c$ 恢复；以 $R_i\leftarrow R_i-kR_j$ 替换第 $i$ 行，可用逆操作 $R_i\leftarrow R_i+kR_j$ 恢复。
>
> **边界**：一行乘 $0$ 不可逆，会丢掉原方程，因此不允许。
>
> **结论**：三种操作均可逆，故新旧系统有完全相同的解集；对应的系数矩阵彼此[[线性方程组与四个基本子空间#消元、主元与 LU|行等价（row-equivalent）]]。
> <!-- bilingual-en:start -->
> **Goal:** prove that the systems before and after each operation are equivalent.
> **Argument:** Swapping two equations changes only their order. Multiplying an equation by $c\ne0$ is reversed by multiplying by $1/c$. Replacing $R_i$ by $R_i-kR_j$ is reversed by $R_i\leftarrow R_i+kR_j$. Since each operation is reversible, it neither loses nor introduces solutions.
> **Boundary:** Multiplying a row by $0$ is not invertible: it destroys the original equation and is therefore not an allowed elementary row operation.
> **Conclusion**:The three operations are invertible, so the old and new systems have the same solution set;the corresponding coefficient matrices are [[线性方程组与四个基本子空间#消元、主元与 LU|row-equivalent]] to each other.
> <!-- bilingual-en:end -->

行操作也可以写成左乘[[线性方程组与四个基本子空间#消元、主元与 LU|初等矩阵（elementary matrix）]]。若
<!-- bilingual-en:start -->
Row operations can also be written as left times [[线性方程组与四个基本子空间#消元、主元与 LU|elementary matrix]]. If
<!-- bilingual-en:end -->

$$
E_{21}=\begin{bmatrix}1&0\\-3&1\end{bmatrix},
$$

则 $E_{21}A$ 把 $A$ 的第二行替换为 $R_2-3R_1$。左乘改变行，右乘改变列，不能混淆。
<!-- bilingual-en:start -->
$E_{21}A$ replaces the second row of $A$ by $R_2-3R_1$. Left multiplication performs row operations; right multiplication performs column operations. Do not confuse the two.
<!-- bilingual-en:end -->

### 2. 主元、换行与消元失败
<!-- bilingual-en:start -->
*2. Pivots, row exchanges, and elimination failure*
<!-- bilingual-en:end -->

每一步用一个非零[[线性方程组与四个基本子空间#消元、主元与 LU|pivot（主元）]]消去其下方元素。若预定主元是 $0$：
<!-- bilingual-en:start -->
Each step eliminates the elements below it with a non-zero [[线性方程组与四个基本子空间#消元、主元与 LU|pivot]].  If the intended pivot is $0$:
<!-- bilingual-en:end -->

- 下方有非零数：交换行，将非零数换上来；
- 整列从当前位置向下全为 $0$：这一列没有主元，后来对应自由变量；
- 若增广列出现 $[0\ \cdots\ 0\mid c]$ 且 $c\ne0$：系统不相容。
<!-- bilingual-en:start -->
- If a nonzero entry appears below, exchange rows to move that entry into the pivot position;
- If every entry below the current position in a column is zero, that column contains no pivot at this stage and the corresponding variable will be free;
- If the augmented matrix contains $[0\ \cdots\ 0\mid c]$ with $c\ne0$, the system is inconsistent.
<!-- bilingual-en:end -->

主元个数就是矩阵的秩。主元的具体数值会随行缩放改变，但主元个数不会。
<!-- bilingual-en:start -->
The number of pivots is the rank of the matrix.  The exact number of pivots varies by scaling, but the number of pivots does not.
<!-- bilingual-en:end -->

### 3. Recitation 完整消元例题
<!-- bilingual-en:start -->
*3. Recitation complete elimination problem*
<!-- bilingual-en:end -->

求解
<!-- bilingual-en:start -->
solving
<!-- bilingual-en:end -->

$$
\begin{cases}
x-y-z+u=0,\\
2x+2z=8,\\
-y-2z=-8,\\
3x-3y-2z+4u=7.
\end{cases}
$$

增广矩阵尺寸是 $4\times5$：
<!-- bilingual-en:start -->
The size of the augmented matrix is $4\times5$:
<!-- bilingual-en:end -->

$$
\left[\begin{array}{rrrr|r}
1&-1&-1&1&0\\
2&0&2&0&8\\
0&-1&-2&0&-8\\
3&-3&-2&4&7
\end{array}\right].
$$

先做 $R_2\leftarrow R_2-2R_1$、$R_4\leftarrow R_4-3R_1$：
<!-- bilingual-en:start -->
Start with $R_2\leftarrow R_2-2R_1$, $R_4\leftarrow R_4-3R_1$:
<!-- bilingual-en:end -->

$$
\left[\begin{array}{rrrr|r}
1&-1&-1&1&0\\
0&2&4&-2&8\\
0&-1&-2&0&-8\\
0&0&1&1&7
\end{array}\right].
$$

再做 $R_3\leftarrow R_3+\tfrac12R_2$：
<!-- bilingual-en:start -->
$R_3\leftarrow R_3+\tfrac12R_2$:
<!-- bilingual-en:end -->

$$
\left[\begin{array}{rrrr|r}
1&-1&-1&1&0\\
0&2&4&-2&8\\
0&0&0&-1&-4\\
0&0&1&1&7
\end{array}\right].
$$

第三个预定主元为 $0$，交换 $R_3,R_4$ 得上三角系统。由下往上回代：
<!-- bilingual-en:start -->
The third pivot is $0$, and the exchange $R_3,R_4$ is triangulated.  Descending from Bottom:
<!-- bilingual-en:end -->

$$
-u=-4\Rightarrow u=4,
$$
$$
z+u=7\Rightarrow z=3,
$$
$$
2y+4z-2u=8\Rightarrow 2y+12-8=8\Rightarrow y=2,
$$
$$
x-y-z+u=0\Rightarrow x=1.
$$

代回原四式全部成立，故解为 $(1,2,3,4)^T$。
<!-- bilingual-en:start -->
The solution is $(1,2,3,4)^T$.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-elimination-lu.png|760]]

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 2.1：二元系统的消元、主元与回代
> **题目转述**：求解
> $$
> 2x+3y=5,\qquad 6x+15y=12,
> $$
> 写成矩阵形式，说明消元倍数与主元，并验算。
>
> **解答**：
> $$
> \left[\begin{array}{cc|c}2&3&5\\6&15&12\end{array}\right]
> \xrightarrow{R_2-3R_1}
> \left[\begin{array}{cc|c}2&3&5\\0&6&-3\end{array}\right].
> $$
> 主元为 $2,6$。回代：$6y=-3\Rightarrow y=-\tfrac12$；
> $$
> 2x+3(-\tfrac12)=5
> \Rightarrow2x=\frac{13}{2}
> \Rightarrow x=\frac{13}{4}.
> $$
> 验算第二式：$6(13/4)+15(-1/2)=78/4-30/4=12$。
> <!-- bilingual-en:start -->
> **Restatement**: Write the system in matrix form, identify the elimination multiplier and pivots, solve, and check the result.
> **Answer**:
> The pivots are $2$ and $6$. Back-substitution gives $6y=-3\Rightarrow y=-\tfrac12$, followed by $x=13/4$. Checking the second equation gives $6(13/4)+15(-1/2)=78/4-30/4=12$.
> <!-- bilingual-en:end -->

> [!question]- Problem 2.2：Pascal 矩阵的消元矩阵
> **题目转述**：找下三角矩阵 $E$，把
> $$
> P=\begin{bmatrix}
> 1&0&0&0\\1&1&0&0\\1&2&1&0\\1&3&3&1
> \end{bmatrix}
> $$
> 化为第一列已清零的较小 Pascal 矩阵；再找 $M$ 使 $MP=I$。
>
> **解答**：相邻行相减由
> $$
> E=\begin{bmatrix}
> 1&0&0&0\\-1&1&0&0\\0&-1&1&0\\0&0&-1&1
> \end{bmatrix}
> $$
> 实现。继续对第二、第三列重复相邻行相减，三个消元矩阵的乘积为
> $$
> M=\begin{bmatrix}
> 1&0&0&0\\
> -1&1&0&0\\
> 1&-2&1&0\\
> -1&3&-3&1
> \end{bmatrix}.
> $$
> 直接乘法给 $MP=I$，所以 $M=P^{-1}$。每个乘积均为 $(4\times4)(4\times4)$，尺寸合法。
> <!-- bilingual-en:start -->
> **Restatement**: Find the lower triangular elimination matrix $E$ that reduces the Pascal matrix to a smaller Pascal matrix after clearing the first column below its leading entry; then find $M$ such that $MP=I$.
> **Solution**: Subtract each row from the row immediately below it. Apply the same adjacent-row subtraction successively to the second and third columns. The product of the three elimination matrices is
> $M$. Direct multiplication gives $MP=I$, so $M=P^{-1}$. Every multiplication is between two $4\times4$ matrices, so the dimensions are compatible.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- 行操作必须同时作用于 $b$；只消 $A$ 会得到另一个方程组。
- “无主元”不一定无解：它可能只意味着自由变量；只有矛盾行才表示无解。
- 初等矩阵写在左边；$AE$ 一般执行的是列操作。
- 数值计算中若主元很小，实际算法会做 pivoting；本课先关注精确代数结构。
<!-- bilingual-en:start -->
- Row operations must also act on $b$; only $A$ gets another set of equations.
- "No pivot" does not necessarily mean no solution: it may mean only free variables; only contradictory lines mean no solution.
- The elementary matrix is written on the left; $AE$ typically performs column operations.
- In numerical calculations, if the pivot is small, the actual algorithm will be pivoting; this lesson first focuses on exact algebraic structures.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 哪个矩阵实现 $R_3\leftarrow R_3+2R_1$？
> **答案**：$3\times3$ 单位矩阵的 $(3,1)$ 元改为 $2$，即 $E=I+2e_3e_1^T$。
> <!-- bilingual-en:start -->
> **Answer**: Change the $(3,1)$ element of the $3\times3$ identity matrix to $2$, or $E=I+2e_3e_1^T$.
> <!-- bilingual-en:end -->

> [!question]- 2. 行最简过程中出现 $[0\ 0\ 0\mid5]$ 表示什么？
> **答案**：方程 $0=5$，系统不相容，无解。
> <!-- bilingual-en:start -->
> **Answer**: Equation $0=5$, System incompatible, No solution.
> <!-- bilingual-en:end -->

> [!question]- 3. 为什么交换两行不改变解？
> **答案**：只是交换两个必须同时满足的方程的书写顺序，且交换操作自身就是逆操作。
> <!-- bilingual-en:start -->
> **Answer**: It only changes the order in which two simultaneously required equations are written, and the row swap is its own inverse.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

可逆行操作 → 上三角形 → 主元与 rank → 回代；下一节把行操作写成矩阵乘法，并研究对所有 $b$ 一次性求解的逆矩阵。
<!-- bilingual-en:start -->
Invertible row operations → upper triangular form → pivots and rank → back-substitution. The next section represents row operations by matrix multiplication and studies the inverse, which solves the system for every right-hand side $b$.
<!-- bilingual-en:end -->

## Session 1.4 Multiplication and inverse matrices

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：矩阵乘法如何表示变换复合？什么时候存在能撤销 $A$ 的矩阵？
<!-- bilingual-en:start -->
**Problem**: How does matrix multiplication represent transform composition?  When is there a matrix that revokes $A$?
<!-- bilingual-en:end -->

**前置知识**：矩阵乘向量、初等矩阵与消元。
<!-- bilingual-en:start -->
**Prerequisite knowledge**: matrix multiplication vector, elementary matrix and elimination.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.3sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S04_Lecture_Multiplication_and_Inverse_Matrices.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S04_Recitation_Inverse_Matrices.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.3prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.3sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.3sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S04_Lecture_Multiplication_and_Inverse_Matrices.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S04_Recitation_Inverse_Matrices.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.3prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.3sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 矩阵乘法的四种读法
<!-- bilingual-en:start -->
*1. Four readings of matrix multiplication*
<!-- bilingual-en:end -->

设 $A\in\mathbb R^{m\times n}$、$B\in\mathbb R^{n\times p}$，则 $AB\in\mathbb R^{m\times p}$。
<!-- bilingual-en:start -->
Let $A\in\mathbb R^{m\times n}$, $B\in\mathbb R^{n\times p}$, then $AB\in\mathbb R^{m\times p}$.
<!-- bilingual-en:end -->

1. **行乘列**：$(AB)_{ij}=\sum_{k=1}^n a_{ik}b_{kj}$。
2. **按列**：$AB$ 的第 $j$ 列是 $A$ 乘 $B$ 的第 $j$ 列。
3. **按行**：$AB$ 的第 $i$ 行是 $A$ 的第 $i$ 行乘 $B$。
4. **外积和**：
   $$
   AB=\sum_{k=1}^n A_{:k}B_{k:},
   $$
   每项都是列向量乘行向量的秩至多一矩阵。
<!-- bilingual-en:start -->
1.**Line multiplication column**: $(AB)_{ij}=\sum_{k=1}^n a_{ik}b_{kj}$.
2.**By column**: The $j$ column of $AB$ is the $A$ multiplied by the $B$ column of $j$.
3.**By line**: $AB$ Line $i$ is $A$ Line $i$ Ride $B$.
4.**External product sum**:
   Each term is a rank of column vectors multiplied by at most one matrix of row vectors.
<!-- bilingual-en:end -->

矩阵乘法表示变换复合：$ABx=A(Bx)$，因此先做 $B$、后做 $A$。它满足结合律与分配律，但一般不满足交换律。
<!-- bilingual-en:start -->
The matrix multiplication represents the transform composition: $ABx=A(Bx)$, so do $B$ first, then $A$.  It satisfies the combination law and the distribution law, but does not satisfy the exchange law generally.
<!-- bilingual-en:end -->

### 2. 逆矩阵
<!-- bilingual-en:start -->
*2. Inverse matrix*
<!-- bilingual-en:end -->

若方阵 $A\in\mathbb R^{n\times n}$ 存在 $A^{-1}$ 使
<!-- bilingual-en:start -->
If $A^{-1}$ is present in the square $A\in\mathbb R^{n\times n}$,
<!-- bilingual-en:end -->

$$
A^{-1}A=AA^{-1}=I_n,
$$

则称 $A$ 可逆，$A^{-1}$ 是 [[线性方程组与四个基本子空间#可解性与完整解|逆矩阵]]。于是 $Ax=b$ 的唯一解为 $x=A^{-1}b$。
<!-- bilingual-en:start -->
$A$ is invertible, and $A^{-1}$ is [[线性方程组与四个基本子空间#可解性与完整解|inverse matrix]].  So the only solution of $Ax=b$ is $x=A^{-1}b$.
<!-- bilingual-en:end -->

> [!proof] 逆矩阵若存在则唯一
> **目标**：若 $B$、$C$ 都是 $A$ 的双侧逆，证明 $B=C$。
>
> **变形**：
> $$
> B=BI=B(AC)=(BA)C=IC=C.
> $$
> 每一步只用单位矩阵定义和结合律。
>
> **边界**：此论证要求尺寸均为 $n\times n$，并使用双侧逆。
>
> **结论**：逆矩阵唯一。
> <!-- bilingual-en:start -->
> **Target**: If both $B$ and $C$ are $A$'s two-sided inverse, prove $B=C$.
> **Transformation**:
> In each step, only the identity matrix is used to define the combination law.
> **Boundary**: This argument requires that the dimensions be $n\times n$ and that the two-sided inversion be used.
> **Conclusion**: The inverse matrix is unique.
> <!-- bilingual-en:end -->

### 3. Gauss–Jordan 求逆的原理
<!-- bilingual-en:start -->
*3. Principle of Gauss-Jordan's inversion*
<!-- bilingual-en:end -->

对增广矩阵 $[A\mid I]$ 做相同行操作，相当于左乘一串初等矩阵 $E_k\cdots E_1$。若左侧最终成为 $I$，则
<!-- bilingual-en:start -->
Do the same row operation to the augmented matrix $[A\mid I]$, equivalent to left multiplying a string of elementary matrix $E_k\cdots E_1$.  If the left side ultimately becomes $I$, then
<!-- bilingual-en:end -->

$$
E_k\cdots E_1A=I,
$$

故 $E_k\cdots E_1=A^{-1}$，右侧同时变成 $A^{-1}$：
<!-- bilingual-en:start -->
So $E_k\cdots E_1=A^{-1}$, the right side becomes $A^{-1}$:
<!-- bilingual-en:end -->

$$
[A\mid I]\longrightarrow[I\mid A^{-1}].
$$

若左侧无法产生 $n$ 个主元，$A$ 是 [[线性方程组与四个基本子空间#可解性与完整解|奇异矩阵]]，逆不存在。
<!-- bilingual-en:start -->
If $n$ primitives cannot be generated on the left, $A$ is [[线性方程组与四个基本子空间#可解性与完整解|singular matrix]] and non-existent.
<!-- bilingual-en:end -->

### 4. 可逆等价链
<!-- bilingual-en:start -->
*4. Invertible equivalence chains*
<!-- bilingual-en:end -->

对 $n\times n$ 方阵，下列命题等价：
<!-- bilingual-en:start -->
For the $n\times n$ matrix, the following propositions are equivalent:
<!-- bilingual-en:end -->

- $A$ 可逆；
- 消元有 $n$ 个主元；
- $Ax=0$ 只有零解；
- 列向量线性无关；
- 列空间为 $\mathbb R^n$；
- 对每个 $b$，$Ax=b$ 有唯一解。
<!-- bilingual-en:start -->
- $A$ is invertible;
- elimination produces $n$ pivots;
- $Ax=0$ has only the zero solution;
- the columns of $A$ are linearly independent;
- the column space is $\mathbb R^n$;
- for every $b$, the system $Ax=b$ has a unique solution.
<!-- bilingual-en:end -->

这组结论集中见 [[线性方程组与四个基本子空间#基、维数与秩|可逆矩阵等价链]]。其逻辑核心是：无非零丢失方向保证一一性，覆盖整个输出空间保证满射；同维有限维空间中二者等价。
<!-- bilingual-en:start -->
See the [[线性方程组与四个基本子空间#基、维数与秩|invertible-matrix equivalence chain]] for the complete set of statements. The central idea is that having no nonzero direction mapped to zero gives injectivity, while reaching the entire output space gives surjectivity; for finite-dimensional spaces of equal dimension, these two properties are equivalent.
<!-- bilingual-en:end -->

### 5. Recitation 的参数矩阵
<!-- bilingual-en:start -->
*5. Parameter matrix of Recitation*
<!-- bilingual-en:end -->

Recitation 计算
<!-- bilingual-en:start -->
Recitation Computing
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}
a&b&b\\
a&a&b\\
a&a&a
\end{bmatrix}
$$

的可逆条件和逆矩阵。先用行差制造主元：
<!-- bilingual-en:start -->
the invertible condition and the inverse matrix.  To manufacture a pivot with line differences:
<!-- bilingual-en:end -->

$$
R_3\leftarrow R_3-R_2=(0,0,a-b),
$$

$$
R_2\leftarrow R_2-R_1=(0,a-b,0),
$$

第一主元是 $a$，后两个主元是 $a-b$。若使用 Unit II 才会系统学习的行列式语言，也可把这一结果写成 $\det A=a(a-b)^2$。
<!-- bilingual-en:start -->
The first pivot is $a$, and the second two pivots are $a-b$.  This result can also be written as $\det A=a(a-b)^2$ if the determinant language is learned by Unit II.
<!-- bilingual-en:end -->

因此恰在
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
a\ne0,\qquad a\ne b
$$

时可逆。继续对 $[A\mid I]$ 做 Gauss–Jordan，得到
<!-- bilingual-en:start -->
Invertible.  Keep Gauss-Jordan on $[A\mid I]$, get
<!-- bilingual-en:end -->

$$
A^{-1}=\begin{bmatrix}
\dfrac1{a-b}&0&-\dfrac{b}{a(a-b)}\\[6pt]
-\dfrac1{a-b}&\dfrac1{a-b}&0\\[6pt]
0&-\dfrac1{a-b}&\dfrac1{a-b}
\end{bmatrix}.
$$

例如第一行第一列的乘积为 $a/(a-b)-b/(a-b)=1$，第一行第二列为 $b/(a-b)-b/(a-b)=0$；其余位置同理可验证 $AA^{-1}=I$。这里的关键不是记住公式，而是先识别每个可能为零的主元，再在合法条件下相除。
<!-- bilingual-en:start -->
For example, the product of the first column of the first row is $a/(a-b)-b/(a-b)=1$, and the second column of the first row is $b/(a-b)-b/(a-b)=0$; the rest of the same location verifies $AA^{-1}=I$.  The key here is not to memorize the formula, but to identify each pivot that may be zero before dividing it under legitimate conditions.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 3.1：验证左分配律
> **题目转述**：给定
> $$
> A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
> B=\begin{bmatrix}1&0\\0&0\end{bmatrix},\quad
> C=\begin{bmatrix}0&0\\5&6\end{bmatrix},
> $$
> 比较 $AB+AC$ 与 $A(B+C)$。
>
> **解答**：
> $$
> AB=\begin{bmatrix}1&0\\3&0\end{bmatrix},\qquad
> AC=\begin{bmatrix}10&12\\20&24\end{bmatrix},
> $$
> 所以
> $$
> AB+AC=\begin{bmatrix}11&12\\23&24\end{bmatrix}.
> $$
> 又
> $$
> B+C=\begin{bmatrix}1&0\\5&6\end{bmatrix},\qquad
> A(B+C)=\begin{bmatrix}11&12\\23&24\end{bmatrix}.
> $$
> 因而 $AB+AC=A(B+C)$。
> <!-- bilingual-en:start -->
> **Title Restatement**: Given
> Compare $AB+AC$ with $A(B+C)$.
> **Answer**:
> therefore
> also
> So $AB+AC=A(B+C)$.
> <!-- bilingual-en:end -->

> [!question]- Problem 3.2：符号上三角矩阵的逆
> **题目转述**：用 Gauss–Jordan 求
> $$
> U=\begin{bmatrix}1&a&b\\0&1&c\\0&0&1\end{bmatrix}
> $$
> 的逆。
>
> **解答**：从第三列向上清零。先 $R_1\leftarrow R_1-aR_2$、$R_2\leftarrow R_2-cR_3$，再 $R_1\leftarrow R_1-(b-ac)R_3$。得到
> $$
> U^{-1}=\begin{bmatrix}
> 1&-a&ac-b\\
> 0&1&-c\\
> 0&0&1
> \end{bmatrix}.
> $$
> 验算第一行第三列：$1(ac-b)+a(-c)+b(1)=0$；其余非对角元同样为 $0$，故 $UU^{-1}=I$。原官方解中末行把结果标成 `L^{-1}` 是排版笔误，此处应为 $U^{-1}$。
> <!-- bilingual-en:start -->
> **Title Restatement**: Gauss-Jordan
> Reverse.
> **Answer**: Zero from third column up.  $R_1\leftarrow R_1-aR_2$, $R_2\leftarrow R_2-cR_3$, $R_1\leftarrow R_1-(b-ac)R_3$.  get
> The first row, third column: $1(ac-b)+a(-c)+b(1)=0$; the remaining non-diagonal elements are also $0$, so $UU^{-1}=I$.  The final line of the original official text marked the result as `L^{-1}` is a typesetting error, which should be $U^{-1}$.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- $AB\ne BA$ 一般成立；甚至 $AB$ 有定义时 $BA$ 也可能无定义。
- $(AB)^{-1}=B^{-1}A^{-1}$，顺序必须反转，因为撤销复合要先撤销最后执行的 $A$。
- $(A+B)^{-1}$ 一般不等于 $A^{-1}+B^{-1}$。
- 求逆只适用于方阵；解一般矩阵系统应使用消元和子空间语言。
<!-- bilingual-en:start -->
- $AB\ne BA$ is generally true; even if $AB$ is defined, $BA$ may be undefined.
- $(AB)^{-1}=B^{-1}A^{-1}$, the order must be reversed because undoing a composite requires undoing the last executed $A$ first.
- $(A+B)^{-1}$ is not usually equal to $A^{-1}+B^{-1}$.
- Inversion is only suitable for square matrix; elimination and subspace language should be used for solving general matrix system.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若 $A$ 为 $2\times3$、$B$ 为 $3\times4$，$AB$ 的尺寸是什么？$BA$ 呢？
> **答案**：$AB$ 为 $2\times4$；$BA$ 的内侧尺寸 $4$ 与 $2$ 不合，未定义。
> <!-- bilingual-en:start -->
> **Answer**: $AB$ is $2\times4$; $BA$'s inside dimensions, $4$, do not fit $2$ and are undefined.
> <!-- bilingual-en:end -->

> [!question]- 2. 证明若 $A$ 可逆且 $Ax=0$，则 $x=0$。
> **答案**：左乘 $A^{-1}$：$x=I x=A^{-1}Ax=A^{-1}0=0$。
> <!-- bilingual-en:start -->
> **Answer**: Left-multiply $A^{-1}$:$x=I x=A^{-1}Ax=A^{-1}0=0$.
> <!-- bilingual-en:end -->

> [!question]- 3. 为什么 $[A\mid I]$ 左侧出现零行就不能得到逆？
> **答案**：零行表示主元数少于 $n$，无法经可逆行操作把左侧变为有 $n$ 个主元的 $I$。
> <!-- bilingual-en:start -->
> **Answer:** A zero row shows that the left block has fewer than $n$ pivots. No sequence of invertible row operations can then turn it into $I_n$, which has $n$ pivots.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

矩阵乘法 = 变换复合 → 初等矩阵 = 可逆行操作 → Gauss–Jordan 同时求出撤销变换的 $A^{-1}$ → 下一节把整串消元压缩为 $A=LU$。
<!-- bilingual-en:start -->
matrix multiplication = transform complex → elementary matrix = invertible operation → Gauss-Jordan and $A^{-1}$ to cancel the transform → the next section compresses the whole series of elimination to $A=LU$.
<!-- bilingual-en:end -->

## Session 1.5 Factorization into A = LU

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：消元产生的一整串操作，怎样保存为一次可复用的矩阵分解？
<!-- bilingual-en:start -->
**Problem**: How to save the whole series of operations generated by elimination as a reusable matrix factorization?
<!-- bilingual-en:end -->

**前置知识**：初等矩阵、上三角回代、逆矩阵。
<!-- bilingual-en:start -->
**Prerequisites**: elementary matrices, back-substitution in an upper triangular system, and inverse matrices.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S05_Lecture_Factorization_into_A_LU.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S05_Recitation_LU_Decomposition.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.4prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.4sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S05_Lecture_Factorization_into_A_LU.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S05_Recitation_LU_Decomposition.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.4prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.4sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 从消元到 [[线性方程组与四个基本子空间#消元、主元与 LU|LU 分解]]
<!-- bilingual-en:start -->
*1. From elimination to [[线性方程组与四个基本子空间#消元、主元与 LU|LU decomposition]]*
<!-- bilingual-en:end -->

设消元不需要换行。若初等矩阵依次满足
<!-- bilingual-en:start -->
Suppose elimination requires no row exchanges. If the successive elementary matrices satisfy
<!-- bilingual-en:end -->

$$
E_k\cdots E_2E_1A=U,
$$

其中 $U$ 为上三角矩阵，则
<!-- bilingual-en:start -->
where $U$ is the upper triangular matrix, then
<!-- bilingual-en:end -->

$$
A=E_1^{-1}E_2^{-1}\cdots E_k^{-1}U=LU.
$$

$L$ 是单位下三角矩阵；在没有换行的标准消元中，$L$ 的下三角位置直接保存消元倍数。
<!-- bilingual-en:start -->
$L$ is unit lower triangular. In standard elimination without row exchanges, the entries below the diagonal of $L$ store the elimination multipliers directly.
<!-- bilingual-en:end -->

> [!proof] 为什么倍数可以直接放进 $L$
> **目标**：说明消元倍数而不是它们的负数出现在 $L$ 中。
>
> **构造**：令 $e_i$ 表示第 $i$ 个标准基列向量。若 $E_{ij}=I-\ell_{ij}e_ie_j^T$ 实现 $R_i\leftarrow R_i-\ell_{ij}R_j$，由于 $(e_ie_j^T)^2=0$（$i\ne j$），
> $$
> E_{ij}^{-1}=I+\ell_{ij}e_ie_j^T.
> $$
> 因而撤销消元时，下三角位置是 $+\ell_{ij}$。
>
> **逐步依据**：$E_{ij}(I+\ell_{ij}e_ie_j^T)=I-\ell_{ij}^2(e_ie_j^T)^2=I$。
>
> **边界**：多个消元矩阵不可随意交换；“直接填倍数”的简洁规则依赖标准的逐列消元顺序。
>
> **结论**：$L$ 记录消元倍数，$U$ 记录消元后的上三角结果。
> <!-- bilingual-en:start -->
> **Goal**: Show why $L$ contains the elimination multipliers themselves, rather than their negatives.
> **Construction**: Let $e_i$ be the $i$th standard basis column vector. If $E_{ij}=I-\ell_{ij}e_ie_j^T$ performs $R_i\leftarrow R_i-\ell_{ij}R_j$, then $(e_ie_j^T)^2=0$ for $i\ne j$, so
> $E_{ij}^{-1}=I+\ell_{ij}e_ie_j^T$.
> Hence undoing the elimination places $+\ell_{ij}$ in the corresponding entry below the diagonal.
> **Verification**: $E_{ij}(I+\ell_{ij}e_ie_j^T)=I-\ell_{ij}^2(e_ie_j^T)^2=I$.
> **Boundary**: Elimination matrices cannot generally be reordered. The simple rule of inserting the multipliers directly into $L$ relies on the standard column-by-column elimination order.
> **Conclusion**: $L$ records the elimination multipliers, while $U$ records the upper triangular matrix produced by elimination.
> <!-- bilingual-en:end -->

### 2. 一个完整例子
<!-- bilingual-en:start -->
*2. A complete example*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}2&1&1\\4&-6&0\\-2&7&2\end{bmatrix}.
$$

第一列倍数为 $\ell_{21}=2$、$\ell_{31}=-1$：
<!-- bilingual-en:start -->
The first column is multiplied by $\ell_{21}=2$, $\ell_{31}=-1$:
<!-- bilingual-en:end -->

$$
R_2\leftarrow R_2-2R_1,\qquad R_3\leftarrow R_3+R_1,
$$

得到
<!-- bilingual-en:start -->
This gives
<!-- bilingual-en:end -->

$$
\begin{bmatrix}2&1&1\\0&-8&-2\\0&8&3\end{bmatrix}.
$$

第二列倍数 $\ell_{32}=-1$，做 $R_3\leftarrow R_3+R_2$：
<!-- bilingual-en:start -->
The second column is multiplied by $\ell_{32}=-1$ to do $R_3\leftarrow R_3+R_2$:
<!-- bilingual-en:end -->

$$
U=\begin{bmatrix}2&1&1\\0&-8&-2\\0&0&1\end{bmatrix},\qquad
L=\begin{bmatrix}1&0&0\\2&1&0\\-1&-1&1\end{bmatrix}.
$$

检查 $LU$ 的三行：第一行等于 $U$ 第一行；第二行是 $2U_{1:}+U_{2:}=(4,-6,0)$；第三行是 $-U_{1:}-U_{2:}+U_{3:}=(-2,7,2)$，确实恢复 $A$。
<!-- bilingual-en:start -->
Check the three rows of $LU$: the first equals the first row of $U$; the second is $2U_{1:}+U_{2:}=(4,-6,0)$; and the third is $-U_{1:}-U_{2:}+U_{3:}=(-2,7,2)$. These reproduce the rows of $A$.
<!-- bilingual-en:end -->

### 3. 为什么 LU 对多个右端特别有用
<!-- bilingual-en:start -->
*3. Why LUs are particularly useful for multiple right-hand ends*
<!-- bilingual-en:end -->

求 $Ax=b$ 时，若 $A=LU$，则先解
<!-- bilingual-en:start -->
When seeking $Ax=b$, if $A=LU$, the first solution
<!-- bilingual-en:end -->

$$
Lc=b
$$

（前代），再解
<!-- bilingual-en:start -->
(previous generation), then solve
<!-- bilingual-en:end -->

$$
Ux=c
$$

（回代）。$A$ 的消元只做一次；不同 $b$ 只需两次三角求解。对稠密 $n\times n$ 矩阵，分解约需 $O(n^3)$，每个新右端约需 $O(n^2)$。
<!-- bilingual-en:start -->
(Back).  The elimination of $A$ is only done once; the different $b$ only need to be triangulated twice.  For the dense $n\times n$ matrix, the decomposition takes about $O(n^3)$ and each new right-hand side takes about $O(n^2)$.
<!-- bilingual-en:end -->

### 4. 换行时的 $PA=LU$
<!-- bilingual-en:start -->
*4. $PA=LU$ when row exchanges are required*
<!-- bilingual-en:end -->

若某一步必须交换行，用 [[线性方程组与四个基本子空间#消元、主元与 LU|置换矩阵]] $P$ 记录交换。常用约定是
<!-- bilingual-en:start -->
If a step must exchange rows, record the exchange with [[线性方程组与四个基本子空间#消元、主元与 LU|permutation matrix]] $P$.  Common conventions are
<!-- bilingual-en:end -->

$$
PA=LU.
$$

例如 $A$ 第一主元为 $0$ 而下方非零，不能写无换行的标准 $A=LU$；先用 $P$ 把非零行换上来。实际数值计算还会主动选绝对值较大的主元以改善稳定性。
<!-- bilingual-en:start -->
For example, if the first pivot of $A$ is $0$ but a lower entry in the same column is nonzero, the standard factorization $A=LU$ cannot proceed without a row exchange. A permutation matrix $P$ records the required row swap. In numerical computation, pivoting deliberately chooses entries of larger magnitude to improve stability.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 4.1：从消元矩阵恢复 $L$
> **题目转述**：对
> $$
> A=\begin{bmatrix}1&3&0\\2&4&0\\2&0&1\end{bmatrix},
> $$
> 求 $E$ 使 $EA=U$，再用 $L=E^{-1}$ 写出 $A=LU$。
>
> **解答**：依次消去 $(2,1),(3,1),(3,2)$：
> $$
> R_2\leftarrow R_2-2R_1,\quad
> R_3\leftarrow R_3-2R_1,\quad
> R_3\leftarrow R_3-3R_2.
> $$
> 注意第三步中的 $R_2$ 是已消元后的 $R_2=(0,-2,0)$。结果
> $$
> U=\begin{bmatrix}1&3&0\\0&-2&0\\0&0&1\end{bmatrix}.
> $$
> 三个消元矩阵按执行顺序相乘得
> $$
> E=\begin{bmatrix}1&0&0\\-2&1&0\\4&-3&1\end{bmatrix}.
> $$
> 其逆为
> $$
> L=\begin{bmatrix}1&0&0\\2&1&0\\2&3&1\end{bmatrix}.
> $$
> 最终
> $$
> A=LU=
> \begin{bmatrix}1&0&0\\2&1&0\\2&3&1\end{bmatrix}
> \begin{bmatrix}1&3&0\\0&-2&0\\0&0&1\end{bmatrix}.
> $$
> 直接相乘恢复 $A$，尺寸均为 $3\times3$。
> <!-- bilingual-en:start -->
> **Restatement**: Find $E$ such that $EA=U$, then write $A=LU$ with $L=E^{-1}$.
> **Solution**: Eliminate entries $(2,1)$, $(3,1)$, and $(3,2)$ in that order. In the third step, remember that $R_2=(0,-2,0)$ is already the row produced by the preceding elimination.
> Multiplying the three elimination matrices in their order of application gives the displayed matrix $E$. Its inverse is the displayed matrix $L$, and therefore $A=LU$.
> Direct multiplication recovers $A$, and every factor is $3\times3$.
> <!-- bilingual-en:end -->

> [!question]- Problem 4.2：特殊对称矩阵的符号 LU
> **题目转述**：对
> $$
> A=\begin{bmatrix}
> a&a&a&a\\a&b&b&b\\a&b&c&c\\a&b&c&d
> \end{bmatrix},
> $$
> 求 $L,U$，并给出有四个主元的条件。
>
> **解答**：先从后面三行各减第一行，再从后面两行各减新的第二行，最后第四行减第三行：
> $$
> U=\begin{bmatrix}
> a&a&a&a\\
> 0&b-a&b-a&b-a\\
> 0&0&c-b&c-b\\
> 0&0&0&d-c
> \end{bmatrix}.
> $$
> 所有消元倍数为 $1$，故
> $$
> L=\begin{bmatrix}
> 1&0&0&0\\1&1&0&0\\1&1&1&0\\1&1&1&1
> \end{bmatrix}.
> $$
> 四个主元依次为 $a,b-a,c-b,d-c$，所以条件是
> $$
> a\ne0,\qquad b\ne a,\qquad c\ne b,\qquad d\ne c.
> $$
> 任一条件失败都会使对应主元为零；本题的特殊形状下没有另一个下方非零项可换上来。
> <!-- bilingual-en:start -->
> **Restatement**: Find $L$ and $U$ for the displayed matrix and state the conditions under which it has four pivots.
> **Solution**: Subtract row one from each of the last three rows, then subtract the new row two from each of the last two rows, and finally subtract row three from row four. This gives the displayed upper triangular matrix $U$.
> Every elimination multiplier is $1$, so $L$ is the displayed unit lower triangular matrix.
> The four pivots are $a,b-a,c-b,d-c$, yielding the conditions shown above. If any condition fails, the corresponding pivot is zero; because of this matrix's special structure, there is no nonzero entry below it available for a row exchange.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- $L$ 中保存的是消元倍数 $\ell_{ij}$，而消元矩阵中是 $-\ell_{ij}$。
- 若发生换行，不能只把 $P$ 忘掉后仍写 $A=LU$。
- $A=LU$ 不是逐元素乘法；必须用矩阵乘法验算。
- 三角矩阵可逆当且仅当所有对角元非零。
<!-- bilingual-en:start -->
- In $L$, the elimination multiple $\ell_{ij}$ is saved, and in the elimination matrix, $-\ell_{ij}$ is saved.
- If a row exchange occurs, do not omit $P$ and write only $A=LU$.
- $A=LU$ is not element-by-element multiplication; it must be checked with matrix multiplication.
- A triangular matrix is invertible if and only if every diagonal entry is nonzero.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. $A=LU$ 时，为什么先解 $Lc=b$ 再解 $Ux=c$？
> **答案**：令 $c=Ux$，则原式 $LUx=b$ 变为 $Lc=b$；两步合起来等价于原式。
> <!-- bilingual-en:start -->
> **Answer**: Let $c=Ux$, the original $LUx=b$ becomes $Lc=b$; two steps together are equivalent to the original.
> <!-- bilingual-en:end -->

> [!question]- 2. 消元倍数 $\ell_{31}=-2$ 在 $L$ 的哪里？
> **答案**：$L_{31}=-2$；消元矩阵对应位置则是 $2$。
> <!-- bilingual-en:start -->
> **Answer**:$L_{31}=-2$;The corresponding position of the elimination matrix is $2$.
> <!-- bilingual-en:end -->

> [!question]- 3. 若 $A=\begin{bmatrix}0&1\\2&3\end{bmatrix}$，为什么标准无换行 LU 失败？
> **答案**：第一主元为 $0$，无法用它消去下方的 $2$；需交换两行，写 $PA=LU$。
> <!-- bilingual-en:start -->
> **Answer:** The first pivot is $0$, so it cannot eliminate the $2$ below. Swap the two rows and write the factorization as $PA=LU$.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

消元矩阵 $E$ → $EA=U$ → 逆向恢复 $A=LU$ → 多个右端共享一次分解；接下来把矩阵本身放进更一般的向量空间语言。
<!-- bilingual-en:start -->
The elimination matrix $E$ → $EA=U$ → reverse recovery $A=LU$ → multiple right-hand sides share a decomposition; the matrix itself is then placed into a more general vector space language.
<!-- bilingual-en:end -->

## Session 1.6 Transposes, permutations, vector spaces

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：转置和置换怎样交换行列？一个集合满足什么条件才真的是向量空间？
<!-- bilingual-en:start -->
**Question**: How do you swap rows and columns by transposing and permuting?  What condition does a set meet to be really a vector space?
<!-- bilingual-en:end -->

**前置知识**：矩阵乘法、逆矩阵、线性组合。
<!-- bilingual-en:start -->
**Prerequisite knowledge**: Matrix multiplication, inverse matrix, linear combination.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.5sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S06_Lecture_Transposes_Permutations_Vector_Spaces.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S06_Recitation_Subspaces_of_Three_Dimensional_Space.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.5prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.5sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.5sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S06_Lecture_Transposes_Permutations_Vector_Spaces.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S06_Recitation_Subspaces_of_Three_Dimensional_Space.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.5prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.5sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 转置（transpose）
<!-- bilingual-en:start -->
*1. transpose*
<!-- bilingual-en:end -->

若 $A\in\mathbb R^{m\times n}$，其转置 $A^T\in\mathbb R^{n\times m}$ 定义为
<!-- bilingual-en:start -->
If $A\in\mathbb R^{m\times n}$, its transpose $A^T\in\mathbb R^{n\times m}$ is defined as
<!-- bilingual-en:end -->

$$
(A^T)_{ij}=A_{ji}.
$$

基本恒等式：
<!-- bilingual-en:start -->
Basic identity:
<!-- bilingual-en:end -->

$$
(A^T)^T=A,\qquad (A+B)^T=A^T+B^T,\qquad (cA)^T=cA^T,
$$

$$
(AB)^T=B^TA^T.
$$

> [!proof] 为什么乘积转置要反序
> **目标**：证明 $(AB)^T=B^TA^T$。
>
> **逐分量比较**：
> $$
> ((AB)^T)_{ij}=(AB)_{ji}=\sum_k a_{jk}b_{ki}.
> $$
> 另一方面，
> $$
> (B^TA^T)_{ij}=\sum_k(B^T)_{ik}(A^T)_{kj}
> =\sum_kb_{ki}a_{jk}.
> $$
> 标量相乘可交换，故两式相等。
>
> **尺寸检查**：$AB$ 为 $m\times p$，两边转置后均为 $p\times m$。
> <!-- bilingual-en:start -->
> **Goal**: Prove that $(AB)^T=B^TA^T$.
> **Component-by-Component Comparison**:
> On the other hand,
> Scalar multiplication is commutative, so the two forms are equal.
> **Size Check**: $m\times p$ for $AB$ and $p\times m$ for both sides.
> <!-- bilingual-en:end -->

若 $A^T=A$，称 $A$ 为 [[对称矩阵与正定二次型#对称矩阵与谱定理|对称矩阵]]；若 $A^T=-A$，称为斜对称矩阵，实数情形下其对角元必须为 $0$。
<!-- bilingual-en:start -->
If $A^T=A$, then $A$ is a [[对称矩阵与正定二次型#对称矩阵与谱定理|symmetric matrix]]. If $A^T=-A$, it is skew-symmetric; over the reals, every diagonal entry must then be zero.
<!-- bilingual-en:end -->

### 2. 置换矩阵
<!-- bilingual-en:start -->
*2. Permutation Matrix*
<!-- bilingual-en:end -->

置换矩阵 $P$ 是把 $I$ 的行重新排列得到的矩阵。左乘 $PA$ 重排 $A$ 的行，右乘 $AP$ 重排 $A$ 的列。每一行、每一列恰有一个 $1$，其余为 $0$，并且
<!-- bilingual-en:start -->
A permutation matrix $P$ is obtained by rearranging the rows of $I$. Left multiplication, $PA$, rearranges the rows of $A$; right multiplication, $AP$, rearranges its columns. Every row and every column of $P$ contains exactly one $1$, with all other entries zero, and
<!-- bilingual-en:end -->

$$
P^{-1}=P^T.
$$

证明：$P$ 的列是标准正交基的重排，所以 $P^TP=I$。
<!-- bilingual-en:start -->
The columns of $P$ are a permutation of the standard orthonormal basis, so $P^TP=I$.
<!-- bilingual-en:end -->

### 3. 向量空间与子空间
<!-- bilingual-en:start -->
*3. Vector Space and Subspace*
<!-- bilingual-en:end -->

一个 [[线性方程组与四个基本子空间#基、维数与秩|向量空间]] 必须对向量加法和标量乘法封闭，并满足通常的加法、数乘公理。若 $S\subseteq V$ 在继承 $V$ 的运算后仍为向量空间，称 $S$ 是 [[线性方程组与四个基本子空间#基、维数与秩|子空间]]。
<!-- bilingual-en:start -->
A [[线性方程组与四个基本子空间#基、维数与秩|vector space]] must be closed under vector addition and scalar multiplication and satisfy the usual vector-space axioms. If $S\subseteq V$ remains a vector space under the operations inherited from $V$, then $S$ is a [[线性方程组与四个基本子空间#基、维数与秩|subspace]].
<!-- bilingual-en:end -->

实用的子空间判别法：非空集合 $S$ 是子空间，当且仅当对任意 $u,v\in S$、任意标量 $\alpha,\beta$，有
<!-- bilingual-en:start -->
A practical subspace test says that a nonempty set $S$ is a subspace if and only if $\alpha u+\beta v\in S$ for all $u,v\in S$ and all scalars $\alpha,\beta$.
<!-- bilingual-en:end -->

$$
\alpha u+\beta v\in S.
$$

> [!proof] 为什么此判别已包含零向量和负向量
> 取 $\alpha=\beta=0$ 可得 $0\in S$；取 $\alpha=-1,\beta=0$ 可得 $-u\in S$；取 $\alpha=\beta=1$ 得加法封闭。因此线性组合封闭足够。
> <!-- bilingual-en:start -->
> Taking $\alpha=\beta=0$ gives $0\in S$; taking $\alpha=-1,\beta=0$ gives $-u\in S$; and taking $\alpha=\beta=1$ gives closure under addition. Thus closure under arbitrary linear combinations is sufficient.
> <!-- bilingual-en:end -->

Recitation 在 $\mathbb R^3$ 中展示：一个非零向量的 span 是过原点的直线；两个不共线向量的 span 是过原点的平面。两条不同直线的并集通常不是子空间，因为分别取一条线上的向量后，它们的和一般不在并集中；但两条线的 span 是它们的和空间。
<!-- bilingual-en:start -->
The recitation illustrates in $\mathbb R^3$ that the span of one nonzero vector is a line through the origin, while the span of two noncollinear vectors is a plane through the origin. The union of two distinct lines is generally not a subspace: adding one vector from each line usually leaves the union. Their joint span, by contrast, is the sum of the two line subspaces.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 5.1：置换的周期
> **题目转述**：（a）找 $3\times3$、$P\ne I$ 且 $P^3=I$ 的置换矩阵；（b）找 $4\times4$ 置换矩阵 $\widetilde P$ 使 $\widetilde P^4\ne I$。
>
> **解答**：取三循环
> $$
> P=\begin{bmatrix}0&0&1\\1&0&0\\0&1&0\end{bmatrix}.
> $$
> 它把 $e_1\to e_2\to e_3\to e_1$，所以三次后回到原处，$P^3=I$。
> 对（b），取 $\widetilde P=\operatorname{diag}(1,P)$。因 $P^3=I$，
> $$
> \widetilde P^4=\operatorname{diag}(1,P^4)=\operatorname{diag}(1,P)\ne I_4.
> $$
> <!-- bilingual-en:start -->
> **Restatement**: (a) Find a $3\times3$ permutation matrix $P\ne I$ with $P^3=I$; (b) find a $4\times4$ permutation matrix $\widetilde P$ with $\widetilde P^4\ne I$.
> **Answer**: Choose the displayed 3-cycle. It sends $e_1\to e_2\to e_3\to e_1$, so three applications return every basis vector to its starting point and $P^3=I$.
> For (b), take $\widetilde P=\operatorname{diag}(1,P)$.  Because $P^3=I$,
> <!-- bilingual-en:end -->

> [!question]- Problem 5.2：对称与斜对称矩阵的自由度
> **题目转述**：$4\times4$ 矩阵若为对称或斜对称，分别有多少个可独立选择的元素？
>
> **解答**：对称矩阵由对角线及上三角决定，共
> $$
> 4+\binom42=4+6=10.
> $$
> 斜对称矩阵对角元全为 $0$，下三角由上三角的负数决定，故有 $\binom42=6$ 个自由参数。
> <!-- bilingual-en:start -->
> **Question**: How many entries of a $4\times4$ symmetric or skew-symmetric matrix can be chosen independently?
> **Solution**: A symmetric matrix is determined by its diagonal and the entries above the diagonal, giving
> $$4+\binom42=4+6=10$$
> free parameters. A skew-symmetric matrix has zeros on the diagonal, and every entry below the diagonal is the negative of its counterpart above the diagonal, so it has $\binom42=6$ free parameters.
> <!-- bilingual-en:end -->

> [!question]- Problem 5.3：哪些矩阵集合是子空间
> **题目转述**：判断对称矩阵、斜对称矩阵、非对称矩阵集合是否构成矩阵空间 $M$ 的子空间。
>
> **解答**：若 $A^T=A,B^T=B$，则
> $$
> (\alpha A+\beta B)^T=\alpha A+\beta B,
> $$
> 所以对称矩阵构成子空间。若 $A^T=-A,B^T=-B$，同理
> $$
> (\alpha A+\beta B)^T=-(\alpha A+\beta B),
> $$
> 所以斜对称矩阵也构成子空间。非对称矩阵集合不含零矩阵，已违反必要条件；此外两个非对称矩阵也可能相加成对称矩阵，故不是子空间。
> <!-- bilingual-en:start -->
> **Title**: Judging whether the set of symmetric, skew symmetric and asymmetric matrices constitutes a subspace of the matrix space $M$.
> **Answer**: If $A^T=A,B^T=B$, then
> So the symmetric matrix constitutes a subspace.  The same holds true for $A^T=-A,B^T=-B$
> So the skew symmetric matrix also constitutes the subspace.  The set of asymmetric matrices does not contain zero matrices, which has violated the necessary conditions, and the two asymmetric matrices may add to a symmetric matrix, so they are not subspaces.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- 仿射平面 $ax+by+cz=1$ 不过原点，因此不是子空间；对应齐次平面 $ax+by+cz=0$ 才是。
- “集合里有很多向量”与“是子空间”无关；关键是所有线性组合是否仍在集合中。
- $(AB)^T$ 反序；只逐个转置但不反序是错误的。
- $P^T=P$ 并非所有置换矩阵都成立；正确恒等式是 $P^T=P^{-1}$。
<!-- bilingual-en:start -->
- The affine plane $ax+by+cz=1$ is not a subspace because it is not an origin; the corresponding homogeneous plane $ax+by+cz=0$ is.
- "There are many vectors in the set" is independent of "is a subspace"; the key is whether all linear combinations are still in the set.
- $(AB)^T$ reverse; it is wrong to invert only one by one, but not reverse.
- $P^T=P$ Not all permutation matrices hold; the correct identity is $P^T=P^{-1}$.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. $S=\{(x,y):x+y=1\}$ 是子空间吗？
> **答案**：不是；$(0,0)$ 不满足方程。
> <!-- bilingual-en:start -->
> **Answer**: No; $(0,0)$ does not satisfy the equation.
> <!-- bilingual-en:end -->

> [!question]- 2. 所有 $3\times3$ 上三角矩阵构成子空间吗？
> **答案**：构成。和与数乘仍保持下三角位置为零。
> <!-- bilingual-en:start -->
> **Answer**: Yes. Sums and scalar multiples of upper triangular matrices still have zeros below the diagonal.
> <!-- bilingual-en:end -->

> [!question]- 3. 若 $P$ 交换第一、第三行，写出 $P$。
> **答案**：$P=\begin{bmatrix}0&0&1\\0&1&0\\1&0&0\end{bmatrix}$，且 $P^{-1}=P^T=P$。
> <!-- bilingual-en:start -->
> **Answer**: $P=\begin{bmatrix}0&0&1\\0&1&0\\1&0&0\end{bmatrix}$, and $P^{-1}=P^T=P$.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

转置交换行列 → 置换矩阵实现可逆重排 → 子空间以线性组合封闭为核心 → 下一节证明列空间和零空间确实是子空间。
<!-- bilingual-en:start -->
permuting rows and columns → permuting matrices for invertible rearrangement → subspaces are linearly combinatorial closed → the next section proves that the column space and the null space are indeed subspaces.
<!-- bilingual-en:end -->

## Session 1.7 Column space and nullspace

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：哪些右端 $b$ 可达到？哪些输入方向被 $A$ 压成 $0$？
<!-- bilingual-en:start -->
**Q**: Which right-end $b$ are reachable?  Which input directions are pressed into $0$ by $A$?
<!-- bilingual-en:end -->

**前置知识**：span、子空间判别、矩阵乘向量。
<!-- bilingual-en:start -->
**Prerequisite knowledge**:span, subspace discrimination, matrix multiplication vector.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.6sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S07_Lecture_Column_Space_and_Nullspace.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S07_Recitation_Vector_Subspaces.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.6prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.6sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.6sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S07_Lecture_Column_Space_and_Nullspace.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S07_Recitation_Vector_Subspaces.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.6prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.6sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 定义与所在空间
<!-- bilingual-en:start -->
*1. Definition and space*
<!-- bilingual-en:end -->

对 $A\in\mathbb R^{m\times n}$：
<!-- bilingual-en:start -->
For $A\in\mathbb R^{m\times n}$:
<!-- bilingual-en:end -->

$$
C(A)=\{Ax:x\in\mathbb R^n\}
=\operatorname{span}\{a_1,\ldots,a_n\}\subseteq\mathbb R^m,
$$

$$
N(A)=\{x\in\mathbb R^n:Ax=0\}\subseteq\mathbb R^n.
$$

列空间回答 $Ax=b$ 的**存在性**：
<!-- bilingual-en:start -->
Column Space Answer**Existence of $Ax=b$**:
<!-- bilingual-en:end -->

$$
Ax=b\text{ 有解}\iff b\in C(A).
$$

零空间回答齐次自由度，并控制非齐次解的**唯一性**。
<!-- bilingual-en:start -->
The nullspace describes the homogeneous degrees of freedom and therefore controls **uniqueness** of solutions to the inhomogeneous system.
<!-- bilingual-en:end -->

> [!proof] 列空间和零空间为什么都是子空间
> **列空间**：若 $u=Ax$、$v=Ay$，则对任意 $\alpha,\beta$，
> $$
> \alpha u+\beta v=A(\alpha x+\beta y)\in C(A).
> $$
> **零空间**：若 $Ax=0$、$Ay=0$，则
> $$
> A(\alpha x+\beta y)=\alpha Ax+\beta Ay=0,
> $$
> 所以 $\alpha x+\beta y\in N(A)$。
>
> **尺寸**：前一组合发生在 $\mathbb R^m$，后一组合发生在 $\mathbb R^n$。
> <!-- bilingual-en:start -->
> **Row space**: If $u=Ax$, $v=Ay$, for any $\alpha,\beta$,
> **Nullspace**: If $Ax=0$, $Ay=0$, then
> So, $\alpha x+\beta y\in N(A)$.
> **Size**: The previous set occurs at $\mathbb R^m$ and the latter set at $\mathbb R^n$.
> <!-- bilingual-en:end -->

### 2. 一个同时读出两空间的例子
<!-- bilingual-en:start -->
*2. An example of simultaneous readout of two spaces*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}1&2&3\\2&4&6\end{bmatrix}.
$$

三列都是 $(1,2)^T$ 的倍数，所以
<!-- bilingual-en:start -->
All three columns are multiples of $(1,2)^T$, so
<!-- bilingual-en:end -->

$$
C(A)=\operatorname{span}\left\{\begin{bmatrix}1\\2\end{bmatrix}\right\}\subset\mathbb R^2.
$$

$b=(b_1,b_2)^T$ 可解当且仅当 $b_2=2b_1$。零空间方程只有一条独立约束：
<!-- bilingual-en:start -->
$b=(b_1,b_2)^T$ is solvable if and only if $b_2=2b_1$.  The null space equation has only one independent constraint:
<!-- bilingual-en:end -->

$$
x_1+2x_2+3x_3=0.
$$

令 $x_2=s,x_3=t$，则
<!-- bilingual-en:start -->
Let $x_2=s,x_3=t$, then
<!-- bilingual-en:end -->

$$
x=s\begin{bmatrix}-2\\1\\0\end{bmatrix}
+t\begin{bmatrix}-3\\0\\1\end{bmatrix}.
$$

所以 $N(A)$ 是 $\mathbb R^3$ 中的二维平面。输出仅剩一维、输入丢掉二维，预告 $1+2=3$ 的秩—零度关系。
<!-- bilingual-en:start -->
Thus $N(A)$ is a two-dimensional plane in $\mathbb R^3$. The output has one independent dimension and the input loses two dimensions, anticipating the rank–nullity count $1+2=3$.
<!-- bilingual-en:end -->

### 3. Recitation 的子空间快速判别
<!-- bilingual-en:start -->
*3. A quick subspace test from recitation*
<!-- bilingual-en:end -->

若集合由齐次线性条件 $b_1+b_2-b_3=0$ 描述，它正是矩阵 $[1\ 1\ -1]$ 的零空间，必为子空间。若条件改为 $b_3=b_1b_2$，则数乘不封闭，例如 $(1,1,1)$ 满足但 $(2,2,2)$ 不满足。若集合是固定向量加一个 span，必须检查固定向量是否已在该 span 中；否则它是仿射平移，不含 $0$。
<!-- bilingual-en:start -->
If a set is described by the homogeneous linear condition $b_1+b_2-b_3=0$, it is the nullspace of $[1\ 1\ -1]$ and is therefore a subspace. If the condition is changed to $b_3=b_1b_2$, closure under scalar multiplication fails: $(1,1,1)$ satisfies it but $(2,2,2)$ does not. If a set is a fixed vector plus a span, check whether the fixed vector already lies in that span; otherwise the set is an affine translate and does not contain $0$.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 6.1：子空间之和与并集
> **题目转述**：若 $S,T$ 是 $V$ 的子空间，证明 $S+T=\{s+t:s\in S,t\in T\}$ 是子空间；并解释两条直线时 $S+T$、$S\cup T$ 的区别以及 $\operatorname{span}(S\cup T)=S+T$。
>
> **解答**：零向量 $0=0_S+0_T\in S+T$。若 $u=s_1+t_1$、$v=s_2+t_2$，则
> $$
> \alpha u+\beta v=(\alpha s_1+\beta s_2)+(\alpha t_1+\beta t_2)\in S+T,
> $$
> 因为 $S,T$ 各自封闭，故 $S+T$ 是子空间。若 $S,T$ 是不同直线，$S\cup T$ 只含两条线，取 $s\in S\setminus T$、$t\in T\setminus S$ 后 $s+t$ 通常不在并集中；而 $S+T$ 是它们张成的平面。$S+T$ 是包含 $S\cup T$ 的子空间，所以包含其 span；反过来 span 包含所有 $s+t$，故二者相等。
> <!-- bilingual-en:start -->
> **Restatement**: If $S$ and $T$ are subspaces of $V$, prove that $S+T=\{s+t:s\in S,t\in T\}$ is a subspace. For two distinct lines, explain the difference between $S+T$ and $S\cup T$, and prove $\operatorname{span}(S\cup T)=S+T$.
> **Answer**: The zero vector satisfies $0=0_S+0_T\in S+T$. If $u=s_1+t_1$ and $v=s_2+t_2$, closure of $S$ and $T$ shows that every linear combination $\alpha u+\beta v$ belongs to $S+T$. Thus $S+T$ is a subspace. For distinct lines, $S\cup T$ contains only the two lines and is generally not closed under addition, whereas $S+T$ is the plane they span. Since $S+T$ is a subspace containing $S\cup T$, it contains $\operatorname{span}(S\cup T)$; conversely, that span contains every sum $s+t$, so the two sets are equal.
> <!-- bilingual-en:end -->

> [!question]- Problem 6.2：仿射平面的参数式
> **题目转述**：把平面 $x-3y-z=12$ 写成一个特解加两个齐次方向。
>
> **解答**：解出 $x=12+3y+z$，所以
> $$
> \begin{bmatrix}x\\y\\z\end{bmatrix}
> =\begin{bmatrix}12\\0\\0\end{bmatrix}
> +y\begin{bmatrix}3\\1\\0\end{bmatrix}
> +z\begin{bmatrix}1\\0\\1\end{bmatrix}.
> $$
> 后两向量都满足齐次式 $x-3y-z=0$；特解代入非齐次式得 $12$。原题英文第二次写平面方程时把最后的 $z$ 误排成 $x$，由上下文及官方解应读作 $x-3y-z=0$。
> <!-- bilingual-en:start -->
> **Restatement**: Write the plane $x-3y-z=12$ as one particular solution plus two homogeneous directions.
> **Answer**: Solve for $x=12+3y+z$ to obtain the displayed parametrisation. The final two vectors satisfy $x-3y-z=0$, while the particular solution gives $12$ in the nonhomogeneous equation. The original English problem statement misprints the final $z$ as $x$ on its second occurrence; the context and official solution both require $x-3y-z=0$.
> <!-- bilingual-en:end -->

> [!question]- Problem 6.3：纵向拼接矩阵的零空间
> **题目转述**：若 $C=\begin{bmatrix}A\\B\end{bmatrix}$，$N(C)$ 与 $N(A),N(B)$ 有什么关系？
>
> **解答**：$A,B$ 必须有相同列数 $n$，于是 $x\in\mathbb R^n$。有
> $$
> Cx=\begin{bmatrix}Ax\\Bx\end{bmatrix}=0
> \iff Ax=0\text{ 且 }Bx=0.
> $$
> 因而
> $$
> N(C)=N(A)\cap N(B).
> $$
> <!-- bilingual-en:start -->
> **Restatement:** If $C=\begin{bmatrix}A\\B\end{bmatrix}$, how is $N(C)$ related to $N(A)$ and $N(B)$?
> **Answer:** $A$ and $B$ must have the same number of columns, say $n$, so $x\in\mathbb R^n$. Then
> and therefore
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- 行操作通常改变列空间中的具体列，但保持行空间与零空间；找 $C(A)$ 的基必须回到原矩阵选 pivot columns。
- $C(A)$ 是列向量的 span，不是“列向量组成的有限集合”。
- 非齐次解集通常不是子空间，因为不含 $0$；它是零空间的仿射平移。
<!-- bilingual-en:start -->
- Row operations usually change specific columns in the column space, but maintain row space and nullspace; the basis for finding $C(A)$ must go back to the original matrix and select pivot columns.
- $C(A)$ is a span of column vectors, not a "limited set of column vectors".
- A nonhomogeneous solution set is usually not a subspace because it does not contain $0$; it is an affine translation of a null space.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. $C(A)$ 位于哪里？
> **答案**：若 $A$ 为 $m\times n$，则 $C(A)\subseteq\mathbb R^m$。
> <!-- bilingual-en:start -->
> **Answer**: If $A$ is $m\times n$, $C(A)\subseteq\mathbb R^m$.
> <!-- bilingual-en:end -->

> [!question]- 2. 为什么 $N(A)$ 一定含零向量？
> **答案**：线性性给 $A0=0$。
> <!-- bilingual-en:start -->
> **Answer**: Linearity for $A0=0$.
> <!-- bilingual-en:end -->

> [!question]- 3. 若两列相同，写出一个非零零空间向量。
> **答案**：若 $a_i=a_j$，则 $e_i-e_j\in N(A)$，因为 $A(e_i-e_j)=a_i-a_j=0$。
> <!-- bilingual-en:start -->
> **Answer**: If $a_i=a_j$, $e_i-e_j\in N(A)$, because $A(e_i-e_j)=a_i-a_j=0$.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

$C(A)$ = 可达输出 → $N(A)$ = 丢失输入 → 解的存在与唯一被拆开 → 下一节用 rref 为 $N(A)$ 构造可计算的基。
<!-- bilingual-en:start -->
$C(A)$ describes the reachable outputs → $N(A)$ describes input directions lost by the transformation → together they determine existence and uniqueness → the next section constructs a basis for $N(A)$ from the RREF.
<!-- bilingual-en:end -->

## Session 1.8 Solving Ax = 0: pivot variables and special solutions

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：怎样从消元后的矩阵系统地写出 $N(A)$ 的一组基？
<!-- bilingual-en:start -->
**Question:** How can a basis of $N(A)$ be constructed systematically from an eliminated matrix?
<!-- bilingual-en:end -->

**前置知识**：消元、主元、列空间与零空间。
<!-- bilingual-en:start -->
**Prerequisite knowledge**: elimination, pivot, column space and nullspace.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S08_Lecture_Solving_Ax_0_Pivot_Variables_Special_Solutions.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S08_Recitation_Solving_Ax_0.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.7prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.7sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S08_Lecture_Solving_Ax_0_Pivot_Variables_Special_Solutions.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S08_Recitation_Solving_Ax_0.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.7prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.7sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 行最简形与变量分工
<!-- bilingual-en:start -->
*1. Line simplest and variable division*
<!-- bilingual-en:end -->

把 $A$ 化为 [[线性方程组与四个基本子空间#消元、主元与 LU|行最简形]] $R$。由于行操作等价于左乘可逆矩阵 $E$，
<!-- bilingual-en:start -->
Turn $A$ into [[线性方程组与四个基本子空间#消元、主元与 LU|Line simplest]] $R$.  since that row operation is equivalent to the left multiply invertible matrix $E$,
<!-- bilingual-en:end -->

$$
R=EA,\qquad Rx=0\iff EAx=0\iff Ax=0,
$$

所以行操作保持零空间。
<!-- bilingual-en:start -->
So the row operations keep nullspace.
<!-- bilingual-en:end -->

- 主元列对应 **pivot variables（主元变量）**；
- 非主元列对应 **free variables（自由变量）**；
- 每个自由变量依次取 $1$、其余自由变量取 $0$，得到一个 **special solution（特殊解）**。
<!-- bilingual-en:start -->
- Pivot columns correspond to **pivot variables**;
- Non-primitive column mappings**free variables (free variables)**;
- Set one free variable to $1$ and all the others to $0$ in turn to obtain a **special solution**.
<!-- bilingual-en:end -->

若 $A$ 有 $n$ 列、rank 为 $r$，自由变量数是 $n-r$，也就是 $\dim N(A)$。
<!-- bilingual-en:start -->
If $A$ has $n$ rows, rank is $r$, and the number of free variables is $n-r$, that is, $\dim N(A)$.
<!-- bilingual-en:end -->

### 2. 标准块形式
<!-- bilingual-en:start -->
*2. Standard block form*
<!-- bilingual-en:end -->

适当重排列后，rref 可写成
<!-- bilingual-en:start -->
When properly rearranged, rref can be written as
<!-- bilingual-en:end -->

$$
R=\begin{bmatrix}I_r&F\\0&0\end{bmatrix},
$$

其中 $F$ 是主元方程里自由变量的系数块。把 $x$ 分成主元部分 $x_p$ 和自由部分 $x_f$：
<!-- bilingual-en:start -->
$F$ is the block of coefficients of the free variables in the pivot equations. Partition $x$ into its pivot-variable part $x_p$ and free-variable part $x_f$:
<!-- bilingual-en:end -->

$$
x_p+Fx_f=0\Longrightarrow x_p=-Fx_f.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
x=\begin{bmatrix}x_p\\x_f\end{bmatrix}
=\begin{bmatrix}-F\\I_{n-r}\end{bmatrix}x_f.
$$

矩阵 $\begin{bmatrix}-F\\I\end{bmatrix}$ 的列就是特殊解。
<!-- bilingual-en:start -->
The columns of $\begin{bmatrix}-F\\I\end{bmatrix}$ are the special solutions.
<!-- bilingual-en:end -->

> [!proof] 特殊解为什么构成零空间的基
> **张成**：任意 $x_f$ 都可按标准基展开，公式表明对应 $x$ 是特殊解的相同系数组合。
>
> **无关**：若特殊解的线性组合为 $0$，观察其自由变量分量；这些分量正好组成单位矩阵，所以所有系数都必须为 $0$。
>
> **结论**：它们既张成又线性无关，构成 $N(A)$ 的基；数量为 $n-r$。
> <!-- bilingual-en:start -->
> **Spanning:** Any $x_f$ can be expanded in the standard basis. The formula then shows that the corresponding $x$ is the same linear combination of the special solutions.
> **Independence:** If a linear combination of the special solutions equals $0$, inspect their free-variable components. Those components form an identity matrix, so every coefficient must be $0$.
> **Conclusion:** The special solutions are linearly independent and span $N(A)$, so they form a basis. Their number is $n-r$.
> <!-- bilingual-en:end -->

### 3. Recitation：齐次平面与仿射平面
<!-- bilingual-en:start -->
*3. Recitation: Homogeneous and Affine Planes*
<!-- bilingual-en:end -->

非齐次平面
<!-- bilingual-en:start -->
inhomogeneous plane
<!-- bilingual-en:end -->

$$
x-5y+2z=9

$$

与齐次平面 $x-5y+2z=0$ 平行。齐次式以 $x$ 为主元变量、$y,z$ 为自由变量：
<!-- bilingual-en:start -->
Parallel to the homogeneous plane $x-5y+2z=0$.  Homogeneity takes $x$ as the main variable and $y,z$ as the free variable:
<!-- bilingual-en:end -->

$$
x=5y-2z.
$$

特殊解为 $(5,1,0)^T$、$(-2,0,1)^T$，所以
<!-- bilingual-en:start -->
Special solutions are $(5,1,0)^T$, $(-2,0,1)^T$, so
<!-- bilingual-en:end -->

$$
N(A)=\operatorname{span}\left\{
\begin{bmatrix}5\\1\\0\end{bmatrix},
\begin{bmatrix}-2\\0\\1\end{bmatrix}
\right\}.
$$

非齐次平面再加特解 $(9,0,0)^T$。这是下一节完整解结构的几何原型。
<!-- bilingual-en:start -->
The inhomogeneous plane plus the extra solution $(9,0,0)^T$.  This is the geometric prototype of the complete solution structure in the next section.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 7.1：求 rref、rank 与特殊解
> **题目转述**：对
> $$
> A=\begin{bmatrix}1&5&7&9\\0&4&1&7\\2&-2&11&-3\end{bmatrix}
> $$
> 求 rref、rank 和 $Ax=0$ 的特殊解。
>
> **解答**：$R_3\leftarrow R_3-2R_1$ 得 $(0,-12,-3,-21)$，它是第二行的 $-3$ 倍，因此第三行归零。把第二行除以 $4$，再从第一行减 $5$ 倍第二行：
> $$
> R=\begin{bmatrix}
> 1&0&23/4&1/4\\
> 0&1&1/4&7/4\\
> 0&0&0&0
> \end{bmatrix}.
> $$
> 有两个主元，所以 rank $=2$；$x_3,x_4$ 自由。方程为
> $$
> x_1=-\frac{23}{4}x_3-\frac14x_4,\qquad
> x_2=-\frac14x_3-\frac74x_4.
> $$
> 依次令 $(x_3,x_4)=(1,0),(0,1)$，得到
> $$
> s_1=\begin{bmatrix}-23/4\\-1/4\\1\\0\end{bmatrix},\qquad
> s_2=\begin{bmatrix}-1/4\\-7/4\\0\\1\end{bmatrix}.
> $$
> 直接计算 $As_1=As_2=0$；两向量的自由分量分别是标准基，故线性无关。
> <!-- bilingual-en:start -->
> **Restatement**: Compute the RREF and rank of the displayed matrix and find the special solutions of $Ax=0$.
> **Solution**: The operation $R_3\leftarrow R_3-2R_1$ gives $(0,-12,-3,-21)$, which is $-3$ times row two, so row three reduces to zero. Divide row two by $4$, then subtract five times the new row two from row one.
> There are two pivots, so the rank is $2$, while $x_3$ and $x_4$ are free. The equations are shown above.
> Set $(x_3,x_4)=(1,0)$ and $(0,1)$ in turn to obtain the two special solutions.
> The free-variable components of the two vectors are distinct standard basis vectors, so the vectors are linearly independent. Direct calculation verifies $As_1=As_2=0$.
> <!-- bilingual-en:end -->

> [!question]- Problem 7.2：控制乘积的秩
> **题目转述**：令 $B=\begin{bmatrix}1&1\\1&1\end{bmatrix}$，找 $A_1,A_2$ 使 $\operatorname{rank}(A_1B)=1$、$\operatorname{rank}(A_2B)=0$。
>
> **解答**：取 $A_1=I_2$，则 $A_1B=B$，两列相同且非零，rank $=1$。取
> $$
> A_2=\begin{bmatrix}1&-1\\1&-1\end{bmatrix}.
> $$
> $B$ 的每一列都是 $(1,1)^T$，而 $A_2(1,1)^T=0$，所以 $A_2B=0$、rank $=0$。平凡选择 $A_2=0$ 也成立，但此选择更能显示 $C(B)\subseteq N(A_2)$ 才是乘积为零的结构原因。
> <!-- bilingual-en:start -->
> **Restatement**: For $B=\begin{bmatrix}1&1\\1&1\end{bmatrix}$, find $A_1,A_2$ such that $\operatorname{rank}(A_1B)=1$ and $\operatorname{rank}(A_2B)=0$.
> **Answer**: Take $A_1=I_2$, then $A_1B=B$, both columns are the same and non-zero, rank $=1$. Take
> Each column of $B$ is $(1,1)^T$, and $A_2(1,1)^T=0$, so $A_2B=0$ and its rank is $0$. The trivial choice $A_2=0$ also works, but the displayed choice better reveals the structural reason: $C(B)\subseteq N(A_2)$.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- 求 $N(A)$ 可使用 rref，但求 $C(A)$ 的基不能直接拿 rref 的 pivot columns；行操作改变列本身。
- “一个自由变量”对应“一条特殊解”，不是只对应一个解；其任意标量倍数都在零空间中。
- 零空间是子空间，参数式必须包含 $x=0$；若不包含，说明你混入了非齐次特解。
<!-- bilingual-en:start -->
- The basis for seeking $N(A)$ can use rref, but the basis for seeking $C(A)$ cannot directly take the pivot columns of the rref; row operations alter the column itself.
- One free variable corresponds to one special solution, not to only one nullspace vector; all scalar multiples of that special solution also lie in the nullspace.
- Nullspace is a subspace, and the argument must contain $x=0$; if not, you are mixing inhomogeneous solutions.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. $A$ 有 7 列、4 个主元，$N(A)$ 的维数是多少？
> **答案**：$7-4=3$。
> <!-- bilingual-en:start -->
> **Answer**: $7-4=3$.
> <!-- bilingual-en:end -->

> [!question]- 2. 为什么主元变量不能任意取值？
> **答案**：它们由 rref 中的主元方程表示为自由变量的线性组合；任意选择会违反方程。
> <!-- bilingual-en:start -->
> **Answer**: They are represented by the pivot equations in rref as linear combinations of free variables; any choice violates the equations.
> <!-- bilingual-en:end -->

> [!question]- 3. 若 $R=[I_2\ F]$ 且 $F=\begin{bmatrix}2\\-3\end{bmatrix}$，写出特殊解。
> **答案**：$x=(-F,1)^T=(-2,3,1)^T$。
> <!-- bilingual-en:start -->
> **Answer:** $x=(-F,1)^T=(-2,3,1)^T$.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

rref 保持 $N(A)$ → pivot/free 变量分工 → 特殊解给零空间基 → nullity $=n-r$ → 下一节把任意非齐次解写成“特解 + 零空间”。
<!-- bilingual-en:start -->
rref preserves $N(A)$ → separate pivot and free variables → construct special solutions as a nullspace basis → nullity $=n-r$ → the next section writes every solution of an inhomogeneous system as “particular solution + nullspace.”
<!-- bilingual-en:end -->

## Session 1.9 Solving Ax = b: row reduced form R

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：非齐次系统何时相容？相容时怎样一次写出全部解？
<!-- bilingual-en:start -->
**Q**: When are non-homogeneous systems compatible?  How do I write all the answers at once when I'm compatible?
<!-- bilingual-en:end -->

**前置知识**：rref、特殊解、列空间与零空间。
<!-- bilingual-en:start -->
**Prerequisites**: rref, Special Solutions, Column Space, and Nullspace.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.8sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S09_Lecture_Solving_Ax_b_Row_Reduced_Form_R.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S09_Recitation_Solving_Ax_b.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.8prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.8sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.8sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S09_Lecture_Solving_Ax_b_Row_Reduced_Form_R.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S09_Recitation_Solving_Ax_b.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.8prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.8sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 相容条件必须从增广矩阵读
<!-- bilingual-en:start -->
*1. Compatibility conditions must be read from the augmentation matrix*
<!-- bilingual-en:end -->

对 $[A\mid b]$ 做相同行操作。若出现
<!-- bilingual-en:start -->
Do the same row operation on $[A\mid b]$.  if
<!-- bilingual-en:end -->

$$
\begin{bmatrix}0&\cdots&0\mid c\end{bmatrix},\qquad c\ne0,
$$

即方程 $0=c$，则无解。否则系统相容。
<!-- bilingual-en:start -->
Namely the equation $0=c$, then there is no solution.  Otherwise, the system is compatible.
<!-- bilingual-en:end -->

等价的空间说法是
<!-- bilingual-en:start -->
The equivalent space claim is
<!-- bilingual-en:end -->

$$
Ax=b\text{ 相容}\iff b\in C(A).
$$

消元给可计算判据，列空间给结构解释。
<!-- bilingual-en:start -->
Elimination gives a computable criterion, while the column space gives the structural explanation.
<!-- bilingual-en:end -->

### 2. 完整解 = 特解 + 零空间
<!-- bilingual-en:start -->
*2. Complete solution = particular solution + nullspace*
<!-- bilingual-en:end -->

先找任意一个 particular solution（特解）$x_p$ 满足 $Ax_p=b$，再求 $N(A)$。所有解是
<!-- bilingual-en:start -->
First find any particular solution $x_p$ satisfying $Ax_p=b$, then compute $N(A)$. Every solution has the form
<!-- bilingual-en:end -->

$$
x=x_p+x_n,\qquad x_n\in N(A).
$$

> [!proof] 完整解公式的双向证明
> **目标**：证明上式不遗漏、也不加入假解。
>
> **正向构造**：若 $x_n\in N(A)$，则
> $$
> A(x_p+x_n)=Ax_p+Ax_n=b+0=b,
> $$
> 所以右侧每个向量都是解。
>
> **反向覆盖**：若 $x$ 是任意解，则
> $$
> A(x-x_p)=Ax-Ax_p=b-b=0,
> $$
> 因此 $x-x_p\in N(A)$，即 $x=x_p+x_n$。
>
> **边界**：若系统无特解，公式没有起点；若 $b=0$，可取 $x_p=0$，解集就是零空间本身。
>
> **结论**：相容系统的解集是 $N(A)$ 的一个仿射平移。
> <!-- bilingual-en:start -->
> **Goal:** show both that every true solution is included and that no false solution is introduced.
> **Forward construct**: If $x_n\in N(A)$, then
> So each vector on the right is a solution.
> **Backward Override**: If $x$ is an arbitrary solution, then
> Therefore, $x-x_p\in N(A)$, or $x=x_p+x_n$.
> **Boundary**: If the system has no particular solution, the formula has no starting point; if $b=0$, $x_p=0$ is preferred, and the solution set is the null space itself.
> **Conclusion**: The solution set of the compatible system is an affine translation of $N(A)$.
> <!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-affine-solution.png|760]]

### 3. Recitation：由 $b$ 决定的相容条件
<!-- bilingual-en:start -->
*3. Recitation: Compatibility condition determined by $b$*
<!-- bilingual-en:end -->

考虑
<!-- bilingual-en:start -->
consider
<!-- bilingual-en:end -->

$$
\begin{cases}
x-2y-2z=b_1,\\
2x-5y-4z=b_2,\\
4x-9y-8z=b_3.
\end{cases}
$$

对增广矩阵做 $R_2\leftarrow R_2-2R_1$、$R_3\leftarrow R_3-4R_1$，再消第二列，最后一行变为
<!-- bilingual-en:start -->
Do $R_2\leftarrow R_2-2R_1$, $R_3\leftarrow R_3-4R_1$ for the augmented matrix, then cancel the second column, the last row becomes
<!-- bilingual-en:end -->

$$
0=-2b_1-b_2+b_3.
$$

所以相容当且仅当
<!-- bilingual-en:start -->
So compatible if and only if
<!-- bilingual-en:end -->

$$
b_3=2b_1+b_2.
$$

相容时 $z$ 自由，取 $z=0$ 得
<!-- bilingual-en:start -->
Compatible $z$ free, take $z=0$
<!-- bilingual-en:end -->

$$
x_p=\begin{bmatrix}5b_1-2b_2\\2b_1-b_2\\0\end{bmatrix};
$$

齐次特殊解为 $(2,0,1)^T$，故
<!-- bilingual-en:start -->
The homogeneous special solution is $(2,0,1)^T$, so
<!-- bilingual-en:end -->

$$
x=x_p+t\begin{bmatrix}2\\0\\1\end{bmatrix}.
$$

验算策略：先验证 $Ax_p=b$，再验证 $A(2,0,1)^T=0$；线性性便保证整族都正确。
<!-- bilingual-en:start -->
Check strategy: Verify $Ax_p=b$ first, then verify $A(2,0,1)^T=0$; linearity guarantees that the whole family is correct.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 8.1：辨析完整解的三个错误说法
> **题目转述**：解释为什么以下说法都错：（a）完整解是 $x_p,x_n$ 的任意线性组合；（b）$Ax=b$ 至多有一个特解；（c）若 $A$ 可逆，零空间中没有解 $x_n$。
>
> **解答**：（a）$x_p$ 的系数必须是 $1$；若写 $\alpha x_p+x_n$，则其像为 $\alpha b$，只有 $\alpha=1$（或特殊的 $b=0$）才仍为 $b$。（b）若 $x_n\in N(A)$，则 $x_p+x_n$ 也是特解；只要零空间非平凡，就有无穷多个。（c）任何矩阵的零空间至少含 $x_n=0$；可逆只表示它不含非零向量。
> <!-- bilingual-en:start -->
> **Restatement**: Explain why each statement is false: (a) the complete solution is an arbitrary linear combination of $x_p$ and $x_n$; (b) $Ax=b$ has at most one particular solution; (c) if $A$ is invertible, its nullspace has no solution $x_n$.
> **Answer**: (a) The coefficient of $x_p$ must be $1$: $A(\alpha x_p+x_n)=\alpha b$, so only $\alpha=1$ gives $b$ unless $b=0$. (b) If $x_n\in N(A)$, then $x_p+x_n$ is another particular solution; a nontrivial nullspace therefore gives infinitely many. (c) Every nullspace contains $x_n=0$; invertibility means only that it contains no nonzero vector.
> <!-- bilingual-en:end -->

> [!question]- Problem 8.2：同时化简齐次与非齐次系统
> **题目转述**：令
> $$
> U=\begin{bmatrix}1&2&3\\0&0&4\end{bmatrix},\qquad
> c=\begin{bmatrix}5\\8\end{bmatrix}.
> $$
> 分别把 $[U\mid0]$、$[U\mid c]$ 化为 $[R\mid0]$、$[R\mid d]$，求解并代回。
>
> **解答**：第二行除以 $4$，第一行减 $3$ 倍第二行：
> $$
> R=\begin{bmatrix}1&2&0\\0&0&1\end{bmatrix}.
> $$
> 齐次式给 $x_1=-2x_2,x_3=0$，故
> $$
> x=t\begin{bmatrix}-2\\1\\0\end{bmatrix}.
> $$
> 官方解选择 $t=-1$，写成 $(2,-1,0)^T$，两者张成同一零空间。
>
> 对非齐次端，$8/4=2$，第一行右端变为 $5-3(2)=-1$：
> $$
> [R\mid d]=\left[\begin{array}{ccc|c}1&2&0&-1\\0&0&1&2\end{array}\right].
> $$
> 因而 $x_3=2$、$x_1=-1-2x_2$。取 $x_2=0$ 可得更简洁特解 $(-1,0,2)^T$；官方取 $x_2=1$ 得 $(-3,1,2)^T$。完整解为
> $$
> x=\begin{bmatrix}-1\\0\\2\end{bmatrix}
> +t\begin{bmatrix}-2\\1\\0\end{bmatrix}.
> $$
> 代入 $Ux$：第二分量恒为 $8$，第一分量为 $(-1-2t)+2t+6=5$。
> <!-- bilingual-en:start -->
> **Restatement**: Reduce $[U\mid0]$ and $[U\mid c]$ to $[R\mid0]$ and $[R\mid d]$, solve both systems, and verify the result by substitution.
> **Solution**: Divide row two by $4$, then subtract three times row two from row one.
> For the homogeneous system, $x_1=-2x_2$ and $x_3=0$, giving the nullspace family shown above. The official solution chooses $t=-1$ and writes $(2,-1,0)^T$; either vector spans the same nullspace.
> For the nonhomogeneous system, the second right-hand side becomes $8/4=2$, and the first becomes $5-3(2)=-1$.
> Thus $x_3=2$ and $x_1=-1-2x_2$. Choosing $x_2=0$ gives the simpler particular solution $(-1,0,2)^T$; choosing $x_2=1$ gives the official solution $(-3,1,2)^T$. Adding the nullspace gives the complete solution shown above.
> Substitution into $Ux$ gives second component $8$ and first component $(-1-2t)+2t+6=5$.
> <!-- bilingual-en:end -->

> [!question]- Problem 8.3：相同解算子是否意味着矩阵相同
> **题目转述**：若对每个 $b$，$Ax=b$ 与 $Cx=b$ 有完全相同的解集，是否必有 $A=C$？
>
> **解答**：是。任取尺寸合适的 $y$，令 $b=Ay$。于是 $y$ 是 $Ax=b$ 的解；按假设也是 $Cx=b$ 的解，所以 $Cy=b=Ay$。这对所有 $y$ 成立，特别对每个标准基 $e_j$ 成立，于是 $A$、$C$ 的第 $j$ 列分别为 $Ae_j,Ce_j$ 且相等。因此 $A=C$。
> <!-- bilingual-en:start -->
> **Question**: If for each $b$, $Ax=b$ has exactly the same solution set as $Cx=b$, is $A=C$ required?
> **Answer**: Yes.  Choose the right size of $y$ to make $b=Ay$.  So $y$ is the solution of $Ax=b$; it is also the solution of $Cx=b$, so $Cy=b=Ay$.  This holds for all $y$, especially for each standard basis $e_j$, so that the $j$ columns of $A$ and $C$ are $Ae_j,Ce_j$ and equal, respectively.  So $A=C$.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- 特解不唯一；“particular”只是任选一个方便代表。
- 完整解不是 $\operatorname{span}\{x_p,N(A)\}$，因为那会允许改变 $x_p$ 的系数。
- 相容条件来自左端行之间的依赖关系；消元时必须带着 $b$。
- 若 $N(A)=\{0\}$，相容系统才唯一；它并不自动保证相容。
<!-- bilingual-en:start -->
- The Special Rapporteur is not unique; the "particular" is simply an optional facilitative representative.
- The complete solution is not $\operatorname{span}\{x_p,N(A)\}$ because it allows the coefficient of $x_p$ to be changed.
- Compatibility conditions are derived from dependencies between left-hand rows; they must be eliminated with $b$.
- For $N(A)=\{0\}$, a compatible system is unique; it does not automatically guarantee compatibility.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若 $x_1,x_2$ 都满足 $Ax=b$，$x_1-x_2$ 在哪里？
> **答案**：在 $N(A)$，因为 $A(x_1-x_2)=b-b=0$。
> <!-- bilingual-en:start -->
> **Answer**: In $N(A)$, because of $A(x_1-x_2)=b-b=0$.
> <!-- bilingual-en:end -->

> [!question]- 2. 解集 $x=x_p+su+tv$ 的几何维数是多少？
> **答案**：若 $u,v$ 线性无关，则是二维仿射平面；其方向空间是 $\operatorname{span}\{u,v\}=N(A)$。
> <!-- bilingual-en:start -->
> **Answer:** If $u$ and $v$ are linearly independent, the set is a two-dimensional affine plane with direction space $\operatorname{span}\{u,v\}=N(A)$.
> <!-- bilingual-en:end -->

> [!question]- 3. 若 $Ax=b$ 有两个不同解，能否有恰好两个解？
> **答案**：不能。在 $\mathbb R$ 上，它们之差给非零零空间方向，$x_1+t(x_2-x_1)$ 对每个实数 $t$ 都是解，因此有无穷多个。
> <!-- bilingual-en:start -->
> **Answer**: No.  On the $\mathbb R$, their difference gives the non-zero null space direction, and $x_1+t(x_2-x_1)$ is the solution for each real $t$, so there are infinitely many.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

增广消元判相容 → 特解定位仿射平移 → 零空间给全部方向 → 完整解 = $x_p+N(A)$ → 下一节用线性无关、基和维数准确计数这些方向。
<!-- bilingual-en:start -->
Augmented elimination criterion compatibility→particular solution location affine translation→null space to all directions→complete solution = $x_p+N(A)$→the next section counts these directions exactly with linear independence, basis and dimension.
<!-- bilingual-en:end -->

## Session 1.10 Independence, basis, and dimension

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：一组生成向量中哪些方向真正不可替代？怎样用最少且不冗余的向量描述空间？
<!-- bilingual-en:start -->
**Question**: Which directions in a set of generated vectors are truly irreplaceable?  How to describe the space with the least and non-redundant vectors?
<!-- bilingual-en:end -->

**前置知识**：span、零空间、主元与自由变量。
<!-- bilingual-en:start -->
**Prerequisites**:span, Nullspace, Pivot and Free Variable.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S10_Lecture_Independence_Basis_and_Dimension.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S10_Recitation_Basis_and_Dimension.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.9prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.9sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S10_Lecture_Independence_Basis_and_Dimension.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S10_Recitation_Basis_and_Dimension.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.9prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.9sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 线性无关
<!-- bilingual-en:start -->
*1. Linear independence*
<!-- bilingual-en:end -->

向量 $v_1,\ldots,v_k$ 称为 [[线性方程组与四个基本子空间#基、维数与秩|线性无关]]，若
<!-- bilingual-en:start -->
The vectors $v_1,\ldots,v_k$ are [[线性方程组与四个基本子空间#基、维数与秩|linearly independent]] if
<!-- bilingual-en:end -->

$$
c_1v_1+\cdots+c_kv_k=0

$$

只允许平凡系数 $c_1=\cdots=c_k=0$。把这些向量作为矩阵 $V$ 的列，则
<!-- bilingual-en:start -->
the only possible coefficients are the trivial ones, $c_1=\cdots=c_k=0$. Placing the vectors in the columns of a matrix $V$ gives
<!-- bilingual-en:end -->

$$
\{v_j\}\text{ 无关}\iff N(V)=\{0\}\iff V\text{ 的每一列都是主元列}.
$$

一组向量若包含零向量、重复向量，或某个向量是其余向量的组合，必然相关。
<!-- bilingual-en:start -->
A set of vectors that contains zero vectors, repeating vectors, or a vector that is a combination of the rest of the vectors is necessarily relevant.
<!-- bilingual-en:end -->

### 2. 基与维数
<!-- bilingual-en:start -->
*2. Basis and dimension*
<!-- bilingual-en:end -->

空间 $S$ 的一组 [[线性方程组与四个基本子空间#基、维数与秩|基]] 同时满足：
<!-- bilingual-en:start -->
A set of [[线性方程组与四个基本子空间#基、维数与秩|basis]] for the space $S$ simultaneously satisfies:
<!-- bilingual-en:end -->

1. 张成 $S$；
2. 线性无关。
<!-- bilingual-en:start -->
1. Chang Cheng $S$;
2. Linear independence.
<!-- bilingual-en:end -->

基实现“存在且唯一的坐标表示”。若 $v_1,\ldots,v_k$ 是基，每个 $s\in S$ 可唯一写成 $s=\sum c_iv_i$。
<!-- bilingual-en:start -->
The basis implements an "existing and unique coordinate representation".  If $v_1,\ldots,v_k$ is basis, each $s\in S$ can be uniquely written as a $s=\sum c_iv_i$.
<!-- bilingual-en:end -->

> [!proof] 为什么基坐标唯一
> 若 $s=\sum c_iv_i=\sum d_iv_i$，两式相减：
> $$
> \sum_i(c_i-d_i)v_i=0.
> $$
> 基向量线性无关，所以每个 $c_i-d_i=0$，即 $c_i=d_i$。
> <!-- bilingual-en:start -->
> If $s=\sum c_iv_i=\sum d_iv_i$, two-fold subtraction:
> The basis vectors are linear, so each $c_i-d_i=0$ is $c_i=d_i$.
> <!-- bilingual-en:end -->

有限维空间任意两组基含有相同数量的向量，这个数量称为 [[线性方程组与四个基本子空间#基、维数与秩|维数]]。零空间的基由特殊解给出；列空间的基由原矩阵的主元列给出。
<!-- bilingual-en:start -->
Any two bases of a finite-dimensional vector space contain the same number of vectors; this number is the [[线性方程组与四个基本子空间#基、维数与秩|dimension]]. A basis of the nullspace is obtained from the special solutions, while a basis of the column space is formed by the pivot columns of the original matrix.
<!-- bilingual-en:end -->

### 3. [[线性方程组与四个基本子空间#基、维数与秩|秩—零度定理]]
<!-- bilingual-en:start -->
*3. [[线性方程组与四个基本子空间#基、维数与秩|The rank–nullity theorem]]*
<!-- bilingual-en:end -->

对 $A\in\mathbb R^{m\times n}$，设 rank $=r$：
<!-- bilingual-en:start -->
For $A\in\mathbb R^{m\times n}$, set rank $=r$:
<!-- bilingual-en:end -->

$$
\dim C(A)=r,\qquad \dim N(A)=n-r.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{\operatorname{rank}(A)+\operatorname{nullity}(A)=n}.
$$

> [!proof] [[线性方程组与四个基本子空间#基、维数与秩|目标—构造—计数证明]]
> **目标**：证明输入空间的维数 $n$ 被行空间有效方向与零空间丢失方向分成 $r+(n-r)$。
>
> **构造**：把 $A$ 化为 rref。恰有 $r$ 个主元列，因此有 $r$ 个主元变量；余下 $n-r$ 个变量自由。
>
> **逐步依据**：每个自由变量产生一条特殊解；上一节已证明这些特殊解构成 $N(A)$ 的基，所以 $\dim N(A)=n-r$。原矩阵的 $r$ 个主元列构成 $C(A)$ 的基，所以 $\dim C(A)=r$。
>
> **边界**：$r$ 可能为 $0$ 或 $\min(m,n)$，公式仍成立。
>
> **结论**：$r+(n-r)=n$。
> <!-- bilingual-en:start -->
> **Goal**: Show that the $n$ input dimensions split into $r$ effective row-space directions and $n-r$ directions lost in the nullspace.
> **Construct**: Convert $A$ to rref.  There are exactly $r$ pivot columns, so there are $r$ pivot variables; the remaining $n-r$ variables are free.
> **Step by step:** Each free variable produces one special solution. The previous section showed that these special solutions form a basis of $N(A)$, so $\dim N(A)=n-r$. The $r$ pivot columns of the original matrix form a basis of $C(A)$, so $\dim C(A)=r$.
> **Boundary**: $r$ may be $0$ or $\min(m,n)$, and the formula still holds.
> **Conclusion**: $r+(n-r)=n$.
> <!-- bilingual-en:end -->

### 4. 如何从生成集抽取基
<!-- bilingual-en:start -->
*4. How to Extract a Basis from a Generated Set*
<!-- bilingual-en:end -->

若 $v_1,\ldots,v_k$ 是列向量，把它们组成 $A=[v_1\cdots v_k]$ 并消元。rref 的主元**位置**告诉你应从**原矩阵**选哪些列。行操作保持列之间的线性依赖关系，但不保持原列空间中的具体列，因此不能拿 rref 的主元列替代原列。
<!-- bilingual-en:start -->
Place the vectors $v_1,\ldots,v_k$ as the columns of $A=[v_1\cdots v_k]$ and row-reduce the matrix. The **pivot positions in rref** tell you which columns to select from **the original matrix**. Row operations preserve linear dependence relations among columns but not the actual column vectors or the original column space, so pivot columns of rref cannot replace the corresponding original columns.
<!-- bilingual-en:end -->

Recitation 也讨论把向量作为行消元：非零行可构成行空间的基；但若原问题问原列向量中的一个子集，必须按列放置并回到原列选择。
<!-- bilingual-en:start -->
The recitation also treats vectors as rows: the nonzero rows after row reduction can form a basis for the row space. But if the problem asks for a subset of the original column vectors, place those vectors in columns and select the pivot columns from the original matrix.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 9.1：六个向量中最多多少个独立
> **题目转述**：给定 $\mathbb R^4$ 中
> $$
> v_1=(1,-1,0,0)^T, v_2=(1,0,-1,0)^T, v_3=(1,0,0,-1)^T,
> $$
> $$
> v_4=(0,1,-1,0)^T, v_5=(0,1,0,-1)^T, v_6=(0,0,1,-1)^T,
> $$
> 求最多可选多少个线性无关向量。
>
> **解答**：先看依赖关系：
> $$
> v_4=v_2-v_1,\qquad v_5=v_3-v_1,\qquad v_6=v_3-v_2.
> $$
> 所以后三个不增加 span。把前三个作列，消元可得三个主元；也可直接令
> $$
> c_1v_1+c_2v_2+c_3v_3=0,
> $$
> 查看第 2、3、4 分量依次得 $c_1=c_2=c_3=0$。因此 $v_1,v_2,v_3$ 无关，最大数量为 $3$，整个 $4\times6$ 列矩阵 rank $=3$。
> <!-- bilingual-en:start -->
> **Title Restatement**: In the given $\mathbb R^4$
> Find the maximum number of linearly independent vectors that can be selected.
> **Answer**: See dependencies first:
> So the last three do not add span.  By taking the first three as a row, the elimination may obtain three pivots; it may also be directly ordered that
> $c_1=c_2=c_3=0$ for components 2, 3, and 4.  Therefore, $v_1,v_2,v_3$ is independent, the maximum number is $3$, the entire $4\times6$ column matrix rank $=3$.
> <!-- bilingual-en:end -->

> [!question]- Problem 9.2：平面、交线与正交方向的基
> **题目转述**：求平面 $x-2y+3z=0$ 的基；求它与 $xy$ 平面的交的基；再求所有垂直于该平面的向量空间的基。
>
> **解答**：令 $y=s,z=t$，则 $x=2s-3t$，所以
> $$
> \begin{bmatrix}x\\y\\z\end{bmatrix}
> =s\begin{bmatrix}2\\1\\0\end{bmatrix}
> +t\begin{bmatrix}-3\\0\\1\end{bmatrix}.
> $$
> 两特殊解无关，构成平面基。与 $xy$ 平面相交要求 $z=0$，即 $t=0$，故交线基为 $(2,1,0)^T$。原平面的法向量是系数向量 $(1,-2,3)^T$；所有垂直于平面的向量构成其 span，基为
> $$
> \left\{\begin{bmatrix}1\\-2\\3\end{bmatrix}\right\}.
> $$
> 验证点积：$(1,-2,3)\cdot(2,1,0)=0$，与另一基向量点积也为 $0$。
> <!-- bilingual-en:start -->
> **Title**: Find the basis of the plane $x-2y+3z=0$; Find the basis of the intersection of it and the plane $xy$; Find the basis of all vector spaces perpendicular to the plane.
> **Answer**: $y=s,z=t$, then $x=2s-3t$, so
> The two special solutions are linearly independent and form a basis of the plane. Intersecting with the $xy$-plane requires $z=0$, hence $t=0$, so a basis of the line of intersection is $(2,1,0)^T$. The normal vector of the original plane is the coefficient vector $(1,-2,3)^T$; the vectors perpendicular to the plane form its span, with basis
> As a check, $(1,-2,3)\cdot(2,1,0)=0$, and its dot product with the other basis vector is also $0$.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- “张成”不保证无关；“无关”也不保证张成目标空间。
- $k>n$ 个 $\mathbb R^n$ 向量必相关；$k<n$ 个向量不可能张成整个 $\mathbb R^n$。
- 不同基可以长得完全不同，但基向量数量相同。
- 行操作后选择列空间基，必须使用原矩阵对应列。
<!-- bilingual-en:start -->
- "Jang Sung" does not guarantee irrelevance; "irrelevance" does not guarantee that Jang Sung's target space.
- $k>n$ $\mathbb R^n$ vectors are correlated; $k<n$ vectors cannot be spanned into an entire $\mathbb R^n$.
- Different bases can be completely different in length, but have the same number of basis vectors.
- Columns must be mapped using the original matrix when the column space basis is selected after the row operation.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. $\mathbb R^5$ 中能否有 6 个线性无关向量？
> **答案**：不能；把它们作列得到 $5\times6$ 矩阵，rank 至多为 $5$。
> <!-- bilingual-en:start -->
> **Answer**: No; list them as $5\times6$ matrices, rank at most $5$.
> <!-- bilingual-en:end -->

> [!question]- 2. 两个向量张成一个平面需要什么条件？
> **答案**：二者都非零且不互为倍数，即线性无关。
> <!-- bilingual-en:start -->
> **Answer**: Both are non-zero and are not multiples of each other, i.e. linear.
> <!-- bilingual-en:end -->

> [!question]- 3. $A$ 有 8 列且 $N(A)$ 的基有 3 个向量，rank 是多少？
> **答案**：$8-3=5$。
> <!-- bilingual-en:start -->
> **Answer**: $8-3=5$.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

无关 = 无冗余 → 基 = 无冗余且完整 → 维数 = 基的长度 → rank-nullity 计数输入自由度 → 下一节把同样的计数推广到四个基本子空间。
<!-- bilingual-en:start -->
Independent = non-redundant → basis = non-redundant and complete → dimension = length of basis → rank-nullity count input degrees of freedom → The next section extends the same count to four basic subspaces.
<!-- bilingual-en:end -->

## Session 1.11 The four fundamental subspaces

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：一个 $m\times n$ 矩阵在输入端和输出端分别决定哪四个空间？它们的维数和正交关系是什么？
<!-- bilingual-en:start -->
**Problem**: Which four spaces are determined on the input side and the output side of a $m\times n$ matrix?  What are their dimensions and orthogonality?
<!-- bilingual-en:end -->

**前置知识**：列空间、零空间、基、维数、转置。
<!-- bilingual-en:start -->
**Prerequisites**: Column Space, Nullspace, Basis, Dimension, Transpose.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S11_Lecture_The_Four_Fundamental_Subspaces.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S11_Recitation_Computing_the_Four_Fundamental_Subspaces.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.10prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.10sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S11_Lecture_The_Four_Fundamental_Subspaces.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S11_Recitation_Computing_the_Four_Fundamental_Subspaces.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.10prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.10sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 四空间总表
<!-- bilingual-en:start -->
*1. Four-space matrix*
<!-- bilingual-en:end -->

对 $A\in\mathbb R^{m\times n}$、rank $=r$：
<!-- bilingual-en:start -->
For $A\in\mathbb R^{m\times n}$, rank $=r$:
<!-- bilingual-en:end -->

| 空间 | 定义 | 所在环境 | 维数 |
|---|---|---:|---:|
| $C(A)$ | $A$ 的列的 span | $\mathbb R^m$ | $r$ |
| $N(A)$ | $Ax=0$ 的全部解 | $\mathbb R^n$ | $n-r$ |
| [[线性方程组与四个基本子空间#四个基本子空间|行空间]] $C(A^T)$ | $A$ 的行的 span | $\mathbb R^n$ | $r$ |
| [[线性方程组与四个基本子空间#四个基本子空间|左零空间]] $N(A^T)$ | $A^Ty=0$ 的全部解 | $\mathbb R^m$ | $m-r$ |
<!-- bilingual-en:start -->
| Space | Definition | Ambient space | Dimension |
| --- | --- | ---: | ---: |
| [[线性方程组与四个基本子空间#四个基本子空间|Column space]] $C(A)$ | span of the columns of $A$ | $\mathbb R^m$ | $r$ |
| [[线性方程组与四个基本子空间#四个基本子空间|Nullspace]] $N(A)$ | solution set of $Ax=0$ | $\mathbb R^n$ | $n-r$ |
| [[线性方程组与四个基本子空间#四个基本子空间|Row space]] $C(A^T)$ | span of the rows of $A$ | $\mathbb R^n$ | $r$ |
| [[线性方程组与四个基本子空间#四个基本子空间|Left nullspace]] $N(A^T)$ | solution set of $A^Ty=0$ | $\mathbb R^m$ | $m-r$ |
<!-- bilingual-en:end -->

行空间与列空间维数相同，都是 rank；这就是“行秩 = 列秩”。实践中，rref 的非零行给行空间基，原矩阵的主元列给列空间基。
<!-- bilingual-en:start -->
The row space and column space have the same dimension, namely the rank: this is the theorem that row rank equals column rank. In practice, the nonzero rows of rref form a row-space basis, while the pivot columns of the original matrix form a column-space basis.
<!-- bilingual-en:end -->

### 2. 正交关系
<!-- bilingual-en:start -->
*2. Orthogonal relations*
<!-- bilingual-en:end -->

$$
C(A^T)=N(A)^\perp\subseteq\mathbb R^n,
$$

$$
C(A)=N(A^T)^\perp\subseteq\mathbb R^m.
$$

> [!proof] 行空间为何与零空间正交互补
> **正交性**：若 $x\in N(A)$，则 $Ax=0$。$Ax$ 的第 $i$ 个分量是第 $i$ 行 $r_i^T$ 与 $x$ 的点积，所以每行都与 $x$ 正交；行的任意线性组合也与 $x$ 正交。因此 $C(A^T)\subseteq N(A)^\perp$。
>
> **维数闭合**：$\dim C(A^T)=r$；而
> $$
> \dim N(A)^\perp=n-\dim N(A)=n-(n-r)=r.
> $$
> 一个子空间包含于另一个且维数相同，故二者相等。
>
> **另一对**：对 $A^T$ 应用同一论证，得到 $C(A)=N(A^T)^\perp$。
> <!-- bilingual-en:start -->
> **Orthogonality**: If $x\in N(A)$, then $Ax=0$.  The $i$ component of $Ax$ is the dot product of the $i$ row $r_i^T$ and $x$, so each row is orthogonal to $x$; any linear combination of rows is also orthogonal to $x$.  So $C(A^T)\subseteq N(A)^\perp$.
> **Closed**:$\dim C(A^T)=r$;and
> One subspace is contained in the other and has the same dimension, so the two subspaces are equal.
> **Another pair of**: Apply the same argument to $A^T$, resulting in $C(A)=N(A^T)^\perp$.
> <!-- bilingual-en:end -->

这说明
<!-- bilingual-en:start -->
This means
<!-- bilingual-en:end -->

$$
\mathbb R^n=C(A^T)\oplus N(A),\qquad
\mathbb R^m=C(A)\oplus N(A^T),
$$

其中 $\oplus$ 表示每个向量都有唯一的“两部分相加”表示；这里两直和还是正交直和。
<!-- bilingual-en:start -->
where $\oplus$ denotes that each vector has a unique "two-part addition"; here two straight sums or orthogonal straight sums.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-four-subspaces.png|760]]

### 3. $A$ 如何在四空间图中作用
<!-- bilingual-en:start -->
*3. How $A$ Works in Four-Space Diagram*
<!-- bilingual-en:end -->

- $A$ 把 $N(A)$ 中所有向量送到 $0$；
- $A$ 把行空间中的 $r$ 维有效输入一一映到列空间；
- $A^T$ 把左零空间送到 $0$；
- $A^T$ 把列空间中的有效输出映回行空间。
<!-- bilingual-en:start -->
- $A$ sends all vectors in $N(A)$ to $0$;
- $A$ maps the $r$ dimension in row space to column space one by one;
- $A^T$ sends left nullspace to $0$;
- $A^T$ maps valid output in column space back to row space.
<!-- bilingual-en:end -->

严格地说，限制映射
<!-- bilingual-en:start -->
Strictly speaking, limit mapping
<!-- bilingual-en:end -->

$$
A\big|_{C(A^T)}:C(A^T)\to C(A)
$$

是双射。它是满射，因为每个 $Ax$ 只依赖 $x$ 的行空间分量；它是单射，因为行空间与零空间交只有 $0$。
<!-- bilingual-en:start -->
This restricted map is a bijection. It is surjective because every $Ax$ depends only on the row-space component of $x$, and injective because the row space intersects the nullspace only at $0$.
<!-- bilingual-en:end -->

### 4. 从消元读四空间
<!-- bilingual-en:start -->
*4. Reading the Four Spaces from Elimination*
<!-- bilingual-en:end -->

见 [[线性方程组与四个基本子空间#四个基本子空间|从 RREF 读四个基本子空间]]：
<!-- bilingual-en:start -->
See [[线性方程组与四个基本子空间#四个基本子空间|Reading Four Basic Subspaces from RREF]]:
<!-- bilingual-en:end -->

1. $C(A)$：取**原矩阵**的主元列；
2. $N(A)$：从 rref 的自由变量构造特殊解；
3. $C(A^T)$：取 rref 的非零行；
4. $N(A^T)$：解 $A^Ty=0$，或在完整消元矩阵 $E$ 中读取把 $A$ 消成零行的行组合。
<!-- bilingual-en:start -->
1. $C(A)$: take the pivot columns of the **original matrix**;
2. $N(A)$: Construct the special solutions from the free variables in rref;
3. $C(A^T)$: take the non-zero row of rref;
4. $N(A^T)$: Solve $A^Ty=0$, or read the combination of rows that eliminate $A$ to zero in the complete elimination matrix $E$.
<!-- bilingual-en:end -->

Recitation 用 $B=LU$ 的 rank-2 例子说明：取 $L$ 中与 $U$ 的两个非零 pivot positions 对应的两列，可给出 $C(B)$ 的基；$U$ 给 $N(B)$ 与行空间，$E=L^{-1}$ 中对应 $U$ 零行的那一行给左零空间向量。
<!-- bilingual-en:start -->
In the recitation's rank-two example $B=LU$, the two columns of $L$ corresponding to the nonzero pivot rows of $U$ form a basis of $C(B)$. The matrix $U$ reveals $N(B)$ and the row space, while the row of $E=L^{-1}$ corresponding to the zero row of $U$ gives a vector in the left nullspace.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 10.1：由“某些 $b$ 无解”推出维数关系
> **题目转述**：$A$ 是 $m\times n$、rank $r$。已知存在某些 $b$ 使 $Ax=b$ 无解。（a）写出 $m,n,r$ 的全部必然不等式；（b）解释为什么 $A^Ty=0$ 有非零解。
>
> **解答**：总有 $r\le n$、$r\le m$。存在不可达 $b\in\mathbb R^m$ 说明 $C(A)\ne\mathbb R^m$，所以
> $$
> r<m.
> $$
> $m,n$ 之间没有由题设强制的大小关系；完整陈述是 $r\le n$ 且 $r<m$。左零空间维数
> $$
> \dim N(A^T)=m-r>0,
> $$
> 因而存在 $y\ne0$ 满足 $A^Ty=0$。原官方解一处把列空间误写成 $\mathbb R^n$，按尺寸应为 $\mathbb R^m$。
> <!-- bilingual-en:start -->
> **Restatement**: Let $A$ be $m\times n$ with rank $r$, and suppose that $Ax=b$ is unsolvable for some $b$. (a) State every necessary inequality among $m,n,r$. (b) Explain why $A^Ty=0$ has a nonzero solution.
> **Answer**: Always $r\le n$ and $r\le m$. The existence of an unreachable $b\in\mathbb R^m$ means $C(A)\ne\mathbb R^m$, so $r<m$. The problem imposes no ordering between $m$ and $n$; the complete conclusion is $r\le n$ and $r<m$. Since $\dim N(A^T)=m-r>0$, some $y\ne0$ satisfies $A^Ty=0$. The official solution once writes the column space as $\mathbb R^n$; dimensional consistency requires $\mathbb R^m$.
> <!-- bilingual-en:end -->

> [!question]- Problem 10.2：转置系统的存在与唯一
> **题目转述**：$A^Ty=d$ 在 $d$ 属于哪个基本子空间时可解？在什么空间只含零向量时解唯一？
>
> **解答**：$A^T$ 的列空间是 $C(A^T)$，也就是 $A$ 的行空间，所以
> $$
> A^Ty=d\text{ 可解}\iff d\in C(A^T).
> $$
> 两个解之差位于 $N(A^T)$，因此解唯一当且仅当左零空间 $N(A^T)=\{0\}$。
> <!-- bilingual-en:start -->
> **Question**: $A^Ty=d$ is solvable when $d$ belongs to which basic subspace?  When there is only zero vector in what space?
> **Answer**: $A^T$'s column space is $C(A^T)$, which is $A$'s row space, so
> The difference between the two solutions lies in $N(A^T)$, so the solutions are unique if and only if the left null space $N(A^T)=\{0\}$.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- $C(A)$ 与 $N(A^T)$ 都在 $\mathbb R^m$；$C(A^T)$ 与 $N(A)$ 都在 $\mathbb R^n$。正交只能在同一个环境空间内谈。
- 列空间的基来自原矩阵主元列，行空间基可以来自 rref 非零行。
- “四个空间互相正交”是错的；只有两对互为正交补。
- $N(A)$ 的维数是 $n-r$，左零空间的维数是 $m-r$，不要把 $m,n$ 对调。
<!-- bilingual-en:start -->
- $C(A)$ and $N(A^T)$ in $\mathbb R^m$; $C(A^T)$ and $N(A)$ in $\mathbb R^n$.  Orthogonality can only be discussed in the same environmental space.
- A basis of the column space comes from the pivot columns of the original matrix; a basis of the row space can be taken from the nonzero rows of rref.
- "Four spaces are orthogonal to each other" is wrong; only two pairs are orthogonal to each other.
- The dimension of $N(A)$ is $n-r$, the dimension of left nullspace is $m-r$, do not adjust $m,n$.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. $A$ 为 $7\times5$、rank $3$，四空间维数各是多少？
> **答案**：$\dim C(A)=3$、$\dim N(A)=2$、$\dim C(A^T)=3$、$\dim N(A^T)=4$。
> <!-- bilingual-en:start -->
> **Answer**: $\dim C(A)=3$, $\dim N(A)=2$, $\dim C(A^T)=3$, $\dim N(A^T)=4$.
> <!-- bilingual-en:end -->

> [!question]- 2. 若 $y\in N(A^T)$，证明它与每列正交。
> **答案**：$A^Ty=0$ 的第 $j$ 个分量是 $a_j^Ty=0$。
> <!-- bilingual-en:start -->
> **Answer**: The $j$ component of $A^Ty=0$ is $a_j^Ty=0$.
> <!-- bilingual-en:end -->

> [!question]- 3. 为什么 $A$ 在整个 $\mathbb R^n$ 上不一定一一对应，但在行空间上是一一对应？
> **答案**：整个空间可能含非零 $N(A)$；行空间与 $N(A)$ 正交，交集仅有 $0$，故限制到行空间后核为零。
> <!-- bilingual-en:start -->
> **Answer**: The full space may contain nonzero vectors in $N(A)$. The row space is orthogonal to $N(A)$, so their intersection is only $\{0\}$; therefore the restriction of $A$ to the row space has trivial kernel and is one-to-one.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

列空间/零空间 → 转置产生行空间/左零空间 → 两对正交补 → rank 同时给两个有效维数 → 四空间图统一 $A$ 的输入、输出结构。
<!-- bilingual-en:start -->
column space/nullspace→transpose produces row space/left nullspace→two pairs of orthogonal complements→rank gives two effective dimensions→four-space diagrams unify the input and output structures of $A$.
<!-- bilingual-en:end -->

## Session 1.12 Matrix spaces, rank 1, and small world graphs

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：向量空间的“向量”能否本身就是矩阵？为什么秩一矩阵是一般矩阵的基本构件？
<!-- bilingual-en:start -->
**Problem**: Can a vector in vector space be a matrix in itself?  Why is a rank-one matrix a fundamental component of a general matrix?
<!-- bilingual-en:end -->

**前置知识**：子空间、基、维数、外积、rank。
<!-- bilingual-en:start -->
**Prerequisites**: Subspace, Basis, Dimension, Product, rank.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.11sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S12_Lecture_Matrix_Spaces_Rank_1_Small_World_Graphs.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S12_Recitation_Matrix_Spaces.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.11prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.11sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.11sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S12_Lecture_Matrix_Spaces_Rank_1_Small_World_Graphs.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S12_Recitation_Matrix_Spaces.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.11prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.11sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 矩阵空间
<!-- bilingual-en:start -->
*1. Matrix space*
<!-- bilingual-en:end -->

所有 $m\times n$ 实矩阵组成向量空间 $M_{m\times n}$。标准基为
<!-- bilingual-en:start -->
All real $m\times n$ matrices form the vector space $M_{m\times n}$. A standard basis is
<!-- bilingual-en:end -->

$$
E_{ij}\quad(1\le i\le m,\ 1\le j\le n),
$$

其中只有 $(i,j)$ 元为 $1$。任意矩阵唯一写成
<!-- bilingual-en:start -->
where $E_{ij}$ has a $1$ in position $(i,j)$ and zeros elsewhere. Every matrix has the unique expansion
<!-- bilingual-en:end -->

$$
A=\sum_{i=1}^m\sum_{j=1}^n a_{ij}E_{ij},
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\dim M_{m\times n}=mn.
$$

对称矩阵、斜对称矩阵、上三角矩阵、trace 为零的方阵都可构成子空间；可逆矩阵集合不是子空间，因为不含零矩阵，且两个可逆矩阵的和可能奇异。
<!-- bilingual-en:start -->
Symmetric matrices, skew-symmetric matrices, upper triangular matrices, and square matrices with trace zero each form a subspace. Invertible matrices do not form a subspace: the zero matrix is absent, and the sum of two invertible matrices may be singular.
<!-- bilingual-en:end -->

### 2. [[线性方程组与四个基本子空间#矩阵乘法的结构读法|秩一矩阵]]与外积
<!-- bilingual-en:start -->
*2. [[线性方程组与四个基本子空间#矩阵乘法的结构读法|Rank-one matrices]] and outer products*
<!-- bilingual-en:end -->

若 $u\in\mathbb R^m$、$v\in\mathbb R^n$ 均非零，则
<!-- bilingual-en:start -->
If both $u\in\mathbb R^m$ and $v\in\mathbb R^n$ are non-zero,
<!-- bilingual-en:end -->

$$
A=uv^T
$$

是 $m\times n$ 矩阵，其第 $j$ 列为 $v_j u$，所有列都在同一直线上，所以 rank $=1$。
<!-- bilingual-en:start -->
is an $m\times n$ matrix whose $j$th column is $v_ju$. All columns lie on the same one-dimensional subspace, so its rank is $1$.
<!-- bilingual-en:end -->

反过来，任一非零 rank-1 矩阵的所有列都是某个非零列 $u$ 的倍数；把倍数收进 $v$，便得 $A=uv^T$。
<!-- bilingual-en:start -->
Conversely, every column of a nonzero rank-one matrix is a scalar multiple of some nonzero column $u$. Collecting those scalars into $v$ gives $A=uv^T$.
<!-- bilingual-en:end -->

> [!proof] rank-$r$ 矩阵可分成 $r$ 个秩一矩阵
> 取 $A$ 的 $r$ 个主元列组成 $C\in\mathbb R^{m\times r}$。每一列都由这些主元列组合，所以存在 $R\in\mathbb R^{r\times n}$ 使 $A=CR$。按内维展开：
> $$
> A=\sum_{k=1}^r C_{:k}R_{k:}.
> $$
> 每项是列乘行，rank 至多 $1$。这给出 $r$ 个秩一构件；不能少于 $r$ 个，否则秩的次可加性会使总秩小于 $r$。
> <!-- bilingual-en:start -->
> Form $C\in\mathbb R^{m\times r}$ from the $r$ pivot columns of $A$. Every column of $A$ is a linear combination of these pivot columns, so some $R\in\mathbb R^{r\times n}$ satisfies $A=CR$. Expanding through the inner dimension gives
> Each term is an outer product of a column and a row, so it has rank at most $1$. This produces $r$ rank-one components. Fewer than $r$ cannot suffice, because subadditivity of rank would then make the total rank smaller than $r$.
> <!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-rank-one.png|760]]

### 3. Recitation：带固定零空间向量的矩阵子空间
<!-- bilingual-en:start -->
*3. Recitation: Matrix Subspace with Fixed Nullspace Vector*
<!-- bilingual-en:end -->

考虑所有满足
<!-- bilingual-en:start -->
Consider all matrices satisfying
<!-- bilingual-en:end -->

$$
A\begin{bmatrix}2\\1\\1\end{bmatrix}=0,\qquad A\in M_{2\times3}
$$

的矩阵。若 $A,B$ 满足条件，则 $(\alpha A+\beta B)v=0$，故它是子空间。每一行 $(a,b,c)$ 满足 $2a+b+c=0$，即
<!-- bilingual-en:start -->
If $A$ and $B$ satisfy the condition, then $(\alpha A+\beta B)v=0$, so the set is a subspace. Each row $(a,b,c)$ must satisfy $2a+b+c=0$, and can therefore be written as
<!-- bilingual-en:end -->

$$
(a,b,c)=a(1,0,-2)+b(0,1,-1).
$$

两行可独立选择，因此基可取
<!-- bilingual-en:start -->
The two rows can be chosen independently, so one possible basis is
<!-- bilingual-en:end -->

$$
\begin{bmatrix}1&0&-2\\0&0&0\end{bmatrix},
\begin{bmatrix}0&1&-1\\0&0&0\end{bmatrix},
\begin{bmatrix}0&0&0\\1&0&-2\end{bmatrix},
\begin{bmatrix}0&0&0\\0&1&-1\end{bmatrix},
$$

维数为 $4$。相比之下，“列空间包含固定非零向量 $(2,1)^T$”的矩阵集合不含零矩阵，不是子空间。
<!-- bilingual-en:start -->
The dimension is $4$. By contrast, the set of matrices whose column space contains the fixed nonzero vector $(2,1)^T$ excludes the zero matrix and is therefore not a subspace.
<!-- bilingual-en:end -->

### 4. Small-world graph 的矩阵视角
<!-- bilingual-en:start -->
*4. A matrix view of small-world graphs*
<!-- bilingual-en:end -->

图的邻接矩阵把连接关系变为矩阵。矩阵幂 $(A^k)_{ij}$ 可计数从节点 $i$ 到 $j$ 的长度 $k$ walk；局部边加上少量远程边可能显著缩短平均路径。这里的重点不是图论细节，而是同一套矩阵乘法能编码网络中的传播与连接。
<!-- bilingual-en:start -->
A graph's adjacency matrix encodes its connections algebraically. The entry $(A^k)_{ij}$ counts walks of length $k$ from vertex $i$ to vertex $j$; adding a small number of long-range edges to local connections can sharply reduce average path length. The point here is not the graph-theoretic detail, but that ordinary matrix multiplication encodes propagation and connectivity in a network.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 11.1（Optional）：五个置换矩阵的基
> **题目转述**：把 $3\times3$ 单位矩阵写成其余五个置换矩阵的组合，并证明这五个矩阵线性无关；它们构成“所有行和、列和均相等”的矩阵子空间的一组基。
>
> **解答**：记三个换位矩阵为 $P_{21},P_{31},P_{32}$，两个三循环为 $P_{32}P_{21},P_{21}P_{32}$。前三者相加得到全一矩阵；后两者相加得到对角为 $0$、非对角为 $1$ 的矩阵，所以
> $$
> I=P_{21}+P_{31}+P_{32}-P_{32}P_{21}-P_{21}P_{32}.
> $$
> 若五者线性组合为零，查看三个对角位置可依次迫使三个换位矩阵的系数为零；再看剩余非对角位置迫使两个三循环系数为零，故五者无关。每个置换矩阵的各行和、列和都是 $1$，其组合具有共同的行和、列和。目标空间原有 $9$ 个参数；“三行和相等”给两个独立约束，“三列和相等”再给两个独立约束，而共同的行和与列和因总元素和相同而自动相等，所以维数是 $9-4=5$。已有五个无关矩阵，故它们确实是一组基。
> <!-- bilingual-en:start -->
> **Restatement:** Express the $3\times3$ identity matrix as a combination of the other five permutation matrices, and prove that those five matrices form a basis of the matrix subspace whose row sums are all equal and whose column sums are all equal.
> **Answer:** Denote the three transposition matrices by $P_{21},P_{31},P_{32}$ and the two three-cycles by $P_{32}P_{21},P_{21}P_{32}$. The sum of the first three is the all-ones matrix, while the sum of the last two has zeros on the diagonal and ones off the diagonal.
> If a linear combination of these five matrices is zero, inspecting the three diagonal entries forces the coefficients of the three transposition matrices to vanish one by one; the remaining off-diagonal entries then force the coefficients of the two three-cycles to vanish. Thus the five matrices are linearly independent. Every permutation matrix has all row sums and column sums equal to $1$, so any linear combination has a common row sum and a common column sum. The ambient matrix space has $9$ parameters. Equality of the three row sums imposes two independent constraints, and equality of the three column sums imposes two more. The common row sum automatically equals the common column sum because both equal the sum of all entries. Hence the target subspace has dimension $9-4=5$. Since it contains five independent matrices, those matrices form a basis.
> <!-- bilingual-en:end -->

> [!question]- Problem 11.2：线性算子 $T(X)=AX$ 的核与像
> **题目转述**：在 $M_{3\times3}$ 上令
> $$
> A=\begin{bmatrix}1&0&-1\\-1&1&0\\0&-1&1\end{bmatrix}.
> $$
> （a）哪些 $X$ 满足 $AX=0$？（b）哪些矩阵可写成 $AX$？（c）求核、像维数并解释为何和为 $9$。
>
> **解答**：$A(1,1,1)^T=0$ 且 rank $A=2$，故 $N(A)=\operatorname{span}\{(1,1,1)^T\}$。（a）$AX=0$ 当且仅当 $X$ 的每一列都在 $N(A)$，所以
> $$
> X=\begin{bmatrix}a&b&c\\a&b&c\\a&b&c\end{bmatrix},
> $$
> 有三个自由参数，$\dim\ker T=3$。
>
> （b）$AX$ 的每列均在 $C(A)$。因为 $A$ 每列分量和为 $0$ 且 $C(A)$ 为二维平面，
> $$
> C(A)=\{(p,q,-p-q)^T:p,q\in\mathbb R\}.
> $$
> 因而像中的矩阵恰为
> $$
> \begin{bmatrix}
> a&b&c\\d&e&f\\-a-d&-b-e&-c-f
> \end{bmatrix},
> $$
> 有六个自由参数，$\dim\operatorname{im}T=6$。
>
> （c）输入空间 $M_{3\times3}$ 维数为 $9$；线性算子的 rank-nullity 给 $3+6=9$。这是矩阵作为“向量”时同一维数定理的直接应用。
> <!-- bilingual-en:start -->
> **Restatement**: On $M_{3\times3}$, (a) characterize the matrices $X$ satisfying $AX=0$; (b) characterize the matrices that can be written as $AX$; and (c) find the dimensions of the kernel and image and explain why they add to $9$.
> **Solution**: Since $A(1,1,1)^T=0$ and $\operatorname{rank}(A)=2$, $N(A)=\operatorname{span}\{(1,1,1)^T\}$.
> (a) The equation $AX=0$ holds exactly when every column of $X$ belongs to $N(A)$, giving three free parameters and $\dim\ker T=3$.
> (b) Every column of $AX$ lies in $C(A)$. Each column of $A$ has entries summing to zero, and $C(A)$ is the two-dimensional plane displayed above. Hence the matrices in the image have the displayed form, with six free parameters, so $\dim\operatorname{im}T=6$.
> (c) The domain $M_{3\times3}$ has dimension $9$, and rank–nullity for the linear operator gives $3+6=9$. This is the usual dimension theorem applied with matrices themselves treated as vectors.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- “rank 恰为 1 的矩阵集合”不是子空间：两个 rank-1 矩阵相加可能 rank 2，且不含零矩阵。
- $uv^T$ 的尺寸由 $u$ 的长度给行数、$v$ 的长度给列数。
- 矩阵空间中的线性算子也有核、像、rank-nullity；不要把输入维数误写成矩阵的行数或列数，$M_{m\times n}$ 的维数是 $mn$。
<!-- bilingual-en:start -->
- The set of matrices of rank exactly $1$ is not a subspace: it excludes the zero matrix, and the sum of two rank-one matrices may have rank $2$.
- The length of $u$ determines the number of rows of $uv^T$, while the length of $v$ determines the number of columns.
- Linear operators on matrix spaces also have kernels, images, and a rank–nullity theorem. Do not confuse the dimension of the domain with the number of rows or columns of one matrix: $\dim M_{m\times n}=mn$.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. $M_{2\times4}$ 的维数是多少？
> **答案**：$2\cdot4=8$。
> <!-- bilingual-en:start -->
> **Answer**: $2\cdot4=8$.
> <!-- bilingual-en:end -->

> [!question]- 2. 若 $u\ne0,v\ne0$，为什么 $uv^T$ 不可能 rank 0？
> **答案**：取 $v_j\ne0$，第 $j$ 列 $v_ju\ne0$，矩阵非零；所有列共线，故 rank 恰为 1。
> <!-- bilingual-en:start -->
> **Answer**: Choose $j$ with $v_j\ne0$. Then column $j$ equals $v_ju\ne0$, so the matrix is nonzero; since all columns are collinear, its rank is exactly $1$.
> <!-- bilingual-en:end -->

> [!question]- 3. 所有 $3\times3$ 可逆矩阵构成子空间吗？
> **答案**：不构成；零矩阵不可逆，且 $I+(-I)=0$。
> <!-- bilingual-en:start -->
> **Answer:** No. The zero matrix is singular, and $I+(-I)=0$ shows that the set is not closed under addition.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

向量空间可由矩阵充当元素 → rank-1 外积是矩阵的原子构件 → rank-$r$ 是 $r$ 个外积之和 → 矩阵算子仍满足核—像维数定理 → 下一节用关联矩阵编码图与网络。
<!-- bilingual-en:start -->
Matrices can themselves be elements of a vector space → rank-one outer products are the atomic building blocks of matrices → a rank-$r$ matrix is a sum of $r$ outer products → linear operators on matrix spaces still satisfy rank–nullity → the next section encodes graphs and networks with incidence matrices.
<!-- bilingual-en:end -->

## Session 1.13 Graphs, networks, and incidence matrices

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：怎样用一个矩阵同时编码节点、边、势差、流量守恒与网络能量？
<!-- bilingual-en:start -->
**Question**: How can one matrix simultaneously encode vertices, edges, potential differences, flow conservation, and network energy?
<!-- bilingual-en:end -->

**前置知识**：四个基本子空间、转置、矩阵乘法。
<!-- bilingual-en:start -->
**Prerequisites**: the four fundamental subspaces, transposes, and matrix multiplication.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.12sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S13_Lecture_Graphs_Networks_Incidence_Matrices.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S13_Recitation_Graphs_and_Networks.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.12prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.12sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.12sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S13_Lecture_Graphs_Networks_Incidence_Matrices.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S13_Recitation_Graphs_and_Networks.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.12prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.12sol.pdf#page=1|official solution p.1]]
<!-- bilingual-en:end -->

### 1. 关联矩阵的定义
<!-- bilingual-en:start -->
*1. Definition of Incidence Matrix*
<!-- bilingual-en:end -->

给有向图任意指定每条边的方向。若图有 $n$ 个节点、$m$ 条边，其 [[图的基本结构、路径与遍历|关联矩阵]] $A\in\mathbb R^{m\times n}$ 每行对应一条边：边从节点 $i$ 指向节点 $j$，则该行在第 $i$ 列写 $-1$、第 $j$ 列写 $+1$，其余写 $0$。
<!-- bilingual-en:start -->
Choose an arbitrary orientation for each edge of the graph. For a graph with $n$ vertices and $m$ edges, its [[图的基本结构、路径与遍历|incidence matrix]] $A\in\mathbb R^{m\times n}$ has one row per edge. If an edge is oriented from vertex $i$ to vertex $j$, its row contains $-1$ in column $i$, $+1$ in column $j$, and zeros elsewhere.
<!-- bilingual-en:end -->

方向只是记号选择；反转一条边只会把对应行乘 $-1$，不改变图的连通结构或 $A^TA$。
<!-- bilingual-en:start -->
The orientation is only a sign convention. Reversing one edge multiplies the corresponding row by $-1$ but does not change the graph's connectivity or the matrix $A^TA$.
<!-- bilingual-en:end -->

### 2. 四空间在网络中的含义
<!-- bilingual-en:start -->
*2. The four fundamental subspaces in a network*
<!-- bilingual-en:end -->

令节点势（potential）为 $x\in\mathbb R^n$，则
<!-- bilingual-en:start -->
Let the vector of vertex potentials be $x\in\mathbb R^n$. Then
<!-- bilingual-en:end -->

$$
e=Ax\in\mathbb R^m
$$

给出每条有向边的终点势减起点势。
<!-- bilingual-en:start -->
each component gives the terminal potential minus the initial potential along one oriented edge.
<!-- bilingual-en:end -->

- $N(A)$：所有边势差都为零的节点势。连通图中所有节点势相同，所以 $N(A)=\operatorname{span}\{\mathbf1\}$。
- $C(A)$：可由节点势产生的边势差。
- $N(A^T)$：满足每个节点净流量为零的边流，称为 cycle space；环流属于这里。
- $C(A^T)$：由边量累积到节点的净注入向量；其分量总和为零。
<!-- bilingual-en:start -->
- $N(A)$: vertex-potential assignments that produce zero difference across every edge. In a connected graph, every vertex must have the same potential, so $N(A)=\operatorname{span}\{\mathbf1\}$.
- $C(A)$: edge-potential differences that can be generated by vertex potentials.
- $N(A^T)$: edge flows with zero net flow at every vertex, called the cycle space; circulating flows lie here.
- $C(A^T)$: net-injection vectors obtained by accumulating edge quantities at the vertices; their components sum to zero.
<!-- bilingual-en:end -->

若图有 $c$ 个连通分量，则每个分量可有一个独立常势：
<!-- bilingual-en:start -->
If the graph has $c$ connected components, each component can have an independent constant:
<!-- bilingual-en:end -->

$$
\dim N(A)=c,\qquad \operatorname{rank}(A)=n-c,\qquad
\dim N(A^T)=m-n+c.
$$

> [!proof] 连通图的关联矩阵为什么 rank 为 $n-1$
> **目标**：证明 $\dim N(A)=1$，再用 rank-nullity。
>
> **逐步依据**：$Ax=0$ 表示每条边两端势相等。若图连通，任意两节点之间存在路径；沿路径逐边使用“端点相等”，得到所有节点势相等。因此 $x=c\mathbf1$。
>
> **反向**：$A\mathbf1=0$，因为每行含一个 $-1$ 和一个 $+1$。
>
> **结论**：$N(A)=\operatorname{span}\{\mathbf1\}$，nullity $=1$，所以 rank $=n-1$。
>
> **边界**：不连通时每个连通分量各有一个常数，nullity 变为连通分量数 $c$。
> <!-- bilingual-en:start -->
> **Goal**: Prove $\dim N(A)=1$, then apply rank–nullity.
> **Forward direction**: The equation $Ax=0$ says that the potentials at the two endpoints of every edge are equal. In a connected graph, any two vertices are joined by a path, so propagating this equality along the path shows that all vertex potentials are equal. Hence $x=c\mathbf1$.
> **Reverse direction**: $A\mathbf1=0$ because each row contains one $-1$ and one $+1$.
> **Conclusion**: $N(A)=\operatorname{span}\{\mathbf1\}$, so the nullity is $1$ and the rank is $n-1$.
> **Boundary**: In a disconnected graph, the potential may take an independent constant value on each connected component, so the nullity equals the number $c$ of connected components.
> <!-- bilingual-en:end -->

### 3. 传导、守恒与图 Laplacian
<!-- bilingual-en:start -->
*3. Conductance, conservation, and the graph Laplacian*
<!-- bilingual-en:end -->

令 $C\in\mathbb R^{m\times m}$ 是对角 conductance（电导）矩阵，边流可按符号约定写成
<!-- bilingual-en:start -->
Let $C\in\mathbb R^{m\times m}$ be the diagonal matrix of edge conductances. Then the edge-current vector can be written as
<!-- bilingual-en:end -->

$$
y=-CAx.
$$

若 $f\in\mathbb R^n$ 表示节点的**外部注入**，取“注入为正、网络净流出为正”的约定，则节点守恒写成
<!-- bilingual-en:start -->
Let $f\in\mathbb R^n$ denote **external injection at the vertices**. Under the convention that both injection and net outflow are positive, conservation at the vertices is written as
<!-- bilingual-en:end -->

$$
-A^Ty=f.
$$

与 $y=-CAx$ 合并得
<!-- bilingual-en:start -->
Combining this equation with $y=-CAx$ gives
<!-- bilingual-en:end -->

$$
A^TCAx=f.
$$

若把 $f$ 定义成外部净流出，或改用 $y=CAx$，守恒式中的符号会相应改变。重要的是从同一方向约定一致推导，不是死记正负号。矩阵
<!-- bilingual-en:start -->
If $f$ is instead defined as external net outflow, or if the convention $y=CAx$ is used, the sign in the conservation equation changes accordingly. The important point is to derive all signs consistently from one orientation convention rather than memorize them. The matrix
<!-- bilingual-en:end -->

$$
L_G=A^TCA
$$

称加权[[图的基本结构、路径与遍历|图 Laplacian（graph Laplacian）]]。它对称，并且
<!-- bilingual-en:start -->
is called the weighted [[图的基本结构、路径与遍历|graph Laplacian]]. It is symmetric, and
<!-- bilingual-en:end -->

$$
x^TL_Gx=(Ax)^TC(Ax)=\sum_{e=1}^m c_e(\Delta x_e)^2\ge0.
$$

连通图中 $L_G\mathbf1=0$；势只确定到加一个常数，通常把一个节点接地来选定唯一代表。可解的注入必须满足 $\mathbf1^Tf=0$，即总流入等于总流出。
<!-- bilingual-en:start -->
For a connected graph, $L_G\mathbf1=0$. Potentials are therefore determined only up to an additive constant; grounding one vertex selects a unique representative. A feasible injection must satisfy $\mathbf1^Tf=0$, meaning that total inflow equals total outflow.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-incidence-network.png|760]]

### 4. Recitation：不用消元读出核
<!-- bilingual-en:start -->
*4. Recitation: Read out the kernel without elimination*
<!-- bilingual-en:end -->

五节点六边的连通图，其关联矩阵为 $6\times5$。由连通性立刻得
<!-- bilingual-en:start -->
For a connected graph with five vertices and six edges, the incidence matrix is $6\times5$. Connectivity immediately gives
<!-- bilingual-en:end -->

$$
N(A)=\operatorname{span}\{(1,1,1,1,1)^T\},\qquad \operatorname{rank}(A)=4.
$$

故 $\dim N(A^T)=6-4=2$：两个独立基本环流生成全部平衡流。大环流可以写成两个小环流之和，所以不是新独立方向。
<!-- bilingual-en:start -->
Thus, $\dim N(A^T)=6-4=2$: Two separate fundamental circulations generate all the equilibrium flows.  The large circulation can be written as the sum of two small circulation currents, so it is not a new independent direction.
<!-- bilingual-en:end -->

此外，$\operatorname{tr}(M)$ 表示方阵 $M$ 的对角元之和。于是
<!-- bilingual-en:start -->
For any square matrix $M$, $\operatorname{tr}(M)$ is the sum of its diagonal entries. Therefore,
<!-- bilingual-en:end -->

$$
\operatorname{tr}(A^TA)=\sum_{j=1}^n\|A_{:j}\|^2.
$$

关联矩阵第 $j$ 列每条相邻边贡献一个 $\pm1$，平方和等于节点度数。因此 trace 等于所有节点度数之和，也就是 $2m$；例中为 $12$。
<!-- bilingual-en:start -->
Column $j$ of the incidence matrix contains one $\pm1$ for each edge incident to vertex $j$, so its squared norm equals the degree of that vertex. Thus the trace equals the sum of all vertex degrees, which is $2m$; in this example it is $12$.
<!-- bilingual-en:end -->

### Homework：全部题目与逐步解答
<!-- bilingual-en:start -->
*Homework: All Questions and Step-by-Step Answers*
<!-- bilingual-en:end -->

> [!question]- Problem 12.1：正方形图的关联矩阵与零空间
> **题目转述**：按图中方向写正方形四节点、四边的关联矩阵，求 $N(A)$，并说明为什么 $(1,0,0,0)$ 不在行空间。
>
> **解答**：按官方方向可写
> $$
> A=\begin{bmatrix}
> -1&1&0&0\\
> 0&-1&1&0\\
> 0&0&1&-1\\
> -1&0&0&1
> \end{bmatrix}.
> $$
> $Ax=0$ 给
> $$
> x_2=x_1,\quad x_3=x_2,\quad x_3=x_4,\quad x_4=x_1,
> $$
> 所以
> $$
> N(A)=\operatorname{span}\{(1,1,1,1)^T\}.
> $$
> 行空间等于 $N(A)^\perp$；而
> $$
> (1,0,0,0)\cdot(1,1,1,1)=1\ne0,
> $$
> 故该向量不在行空间。
> <!-- bilingual-en:start -->
> **Restatement**: Using the orientations shown in the figure, write the incidence matrix of the four-vertex, four-edge cycle, find $N(A)$, and explain why $(1,0,0,0)$ is not in the row space.
> **Solution**: With the stated orientations, use the matrix displayed above. Solving $Ax=0$ forces all four coordinates to be equal, so $N(A)=\operatorname{span}\{(1,1,1,1)^T\}$. The row space is $N(A)^\perp$, but $(1,0,0,0)\cdot(1,1,1,1)=1\ne0$; therefore the vector is not in the row space.
> <!-- bilingual-en:end -->

> [!question]- Problem 12.2：电导网络
> **题目转述**：沿用上一题，令
> $$
> C=\operatorname{diag}(1,2,2,1).
> $$
> 求 $A^TCA$；对 $f=(1,0,-1,0)^T$，求 $A^TCAx=f$ 的一个解，并求 $y=-CAx$。
>
> **解答**：逐项相乘得到
> $$
> A^TCA=\begin{bmatrix}
> 2&-1&0&-1\\
> -1&3&-2&0\\
> 0&-2&4&-2\\
> -1&0&-2&3
> \end{bmatrix}.
> $$
> 该矩阵每行和为 $0$，反映常势在核中。选择节点 3 接地，即 $x_3=0$，解得
> $$
> x=\begin{bmatrix}3/4\\1/4\\0\\1/4\end{bmatrix}.
> $$
> 验算：第一行 $2(3/4)-1/4-1/4=1$；第二行 $-3/4+3/4=0$；第三行 $-2(1/4)-2(1/4)=-1$；第四行 $-3/4+3(1/4)=0$。
>
> 再算
> $$
> Ax=\begin{bmatrix}-1/2\\-1/4\\-1/4\\-1/2\end{bmatrix},
> \qquad
> y=-CAx=\begin{bmatrix}1/2\\1/2\\1/2\\1/2\end{bmatrix}.
> $$
> 此时 $-A^Ty=f$，与上面的外部注入约定一致。
> 因 $A\mathbf1=0$，$x+c\mathbf1$ 都是势的等价解；接地只是选定一个代表。
> <!-- bilingual-en:start -->
> **Restatement**: Continuing from the previous problem, compute $A^TCA$. For $f=(1,0,-1,0)^T$, find one solution of $A^TCAx=f$ and then compute $y=-CAx$.
> **Solution**: Direct matrix multiplication gives the displayed Laplacian. Every row sums to zero, reflecting that constant potentials lie in its nullspace. Ground vertex 3 by setting $x_3=0$; solving gives the displayed vector $x$.
> Check each row: $2(3/4)-1/4-1/4=1$, $-3/4+3/4=0$, $-2(1/4)-2(1/4)=-1$, and $-3/4+3(1/4)=0$.
> Computing $Ax$ and then $y=-CAx$ gives the displayed uniform edge flow. The identity $-A^Ty=f$ agrees with the external-injection convention above.
> Since $A\mathbf1=0$, every $x+c\mathbf1$ represents the same potential differences; grounding selects just one representative.
> <!-- bilingual-en:end -->

### 易错点、边界与反例
<!-- bilingual-en:start -->
*Errors, Boundaries and Counterexamples*
<!-- bilingual-en:end -->

- 关联矩阵的形状是“边数 $\times$ 节点数”，因为每一行对应边。
- 边方向可任取，但一旦选定，$A$、流量和势差的符号约定必须一致。
- 连通图的 $A^TA$ 不是可逆矩阵：常数向量总在核中。
- 环的数量不能凭肉眼数所有闭合路径；独立环维数是 $m-n+c$。
<!-- bilingual-en:start -->
- The shape of the incidence matrix is "number of edges $\times$ nodes" because each row corresponds to an edge.
- The edge direction is optional, but once selected, the sign conventions for $A$, flow, and potential differences must be consistent.
- For a connected graph, $A^TA$ is not invertible because the constant vectors always lie in its nullspace.
- Do not count every visibly closed path as an independent cycle; the dimension of the cycle space is $m-n+c$.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 一棵有 $n$ 个节点的树有多少条边，关联矩阵 rank 多少？
> **答案**：边数 $n-1$；树连通，所以 rank $n-1$，且 $N(A^T)$ 维数为 $0$，没有独立环流。
> <!-- bilingual-en:start -->
> **Answer:** It has $n-1$ edges. Because it is connected, the incidence matrix has rank $n-1$; consequently $\dim N(A^T)=0$, so there is no independent circulation.
> <!-- bilingual-en:end -->

> [!question]- 2. 为什么可解的节点注入必须总和为零？
> **答案**：$f\in C(A^T)=N(A)^\perp$，而连通图 $N(A)=\operatorname{span}\{\mathbf1\}$，故 $\mathbf1^Tf=0$。
> <!-- bilingual-en:start -->
> **Answer**: $f\in C(A^T)=N(A)^\perp$, and for a connected graph $N(A)=\operatorname{span}\{\mathbf1\}$, so $\mathbf1^Tf=0$.
> <!-- bilingual-en:end -->

> [!question]- 3. 反转一条边会怎样改变 $A^TA$？
> **答案**：只把 $A$ 对应行乘 $-1$；写成 $DA$，其中 $D^TD=I$，故 $(DA)^T(DA)=A^TA$，不变。
> <!-- bilingual-en:start -->
> **Answer**: Only multiply the corresponding rows of $A$ by $-1$; write as $DA$, where $D^TD=I$, and therefore $(DA)^T(DA)=A^TA$, unchanged.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

关联矩阵把节点势映为边差 → $N(A)$ 表示分量常势 → $N(A^T)$ 表示环流 → $A^TCA$ 汇总守恒与能量 → 四空间获得具体网络意义。
<!-- bilingual-en:start -->
The incidence matrix maps vertex potentials to edge differences → $N(A)$ consists of potentials constant on each connected component → $N(A^T)$ consists of circulations → $A^TCA$ encodes conservation and energy. This gives the network interpretation of the four-subspace structure.
<!-- bilingual-en:end -->

## Session 1.14 Exam 1 review

### 本节问题与前置知识
<!-- bilingual-en:start -->
*Questions and Prerequisites for This Section*
<!-- bilingual-en:end -->

**问题**：如何把 Unit I 的算法线和结构线压缩为一条稳定的解题流程？
<!-- bilingual-en:start -->
**Problem**: How to compress the algorithm line and structure line of Unit I into a stable problem solving flow?
<!-- bilingual-en:end -->

**前置知识**：Sessions 1.1–1.13 全部内容。
<!-- bilingual-en:start -->
**Prerequisites**: Sessions 1.1-1.13.
<!-- bilingual-en:end -->

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.14sum.pdf#page=1|review summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S14_Lecture_Exam_1_Review.pdf#page=1|review lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S14_Recitation_Exam_1_Problem_Solving.pdf#page=1|problem-solving transcript p.1]]
<!-- bilingual-en:start -->
**Local**: [[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.14sum.pdf#page=1|review summary p.1]] [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S14_Lecture_Exam_1_Review.pdf#page=1|review lecture transcript p.1]] [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S14_Recitation_Exam_1_Problem_Solving.pdf#page=1|problem-solving transcript p.1]]
<!-- bilingual-en:end -->

### 1. Unit I 解题总流程
<!-- bilingual-en:start -->
*1. A general problem-solving workflow for Unit I*
<!-- bilingual-en:end -->

拿到 $Ax=b$ 后依次问：
<!-- bilingual-en:start -->
Given $Ax=b$, ask the following questions in order:
<!-- bilingual-en:end -->

1. **尺寸**：$A$ 是 $m\times n$ 吗？$x\in\mathbb R^n$、$b\in\mathbb R^m$ 吗？
2. **消元**：$[A\mid b]$ 的主元在哪里？是否出现矛盾行？
3. **存在性**：$b\in C(A)$ 吗？
4. **唯一性**：$N(A)=\{0\}$ 吗？
5. **完整解**：先取特解，再加所有特殊解。
6. **空间基**：列空间回原矩阵取 pivot columns；行空间取 rref 非零行；两个零空间解齐次系统。
7. **维数检查**：$r+(n-r)=n$、$r+(m-r)=m$。
8. **验算**：代回原矩阵，而不是只代回 rref。
<!-- bilingual-en:start -->
1. **Dimensions**: Is $A$ an $m\times n$ matrix, with $x\in\mathbb R^n$ and $b\in\mathbb R^m$?
2. **Elimination:** Where are the pivots of $[A\mid b]$? Does a contradictory row appear?
3. **Existence**: Is $b\in C(A)$?
4. **Uniqueness**: Is $N(A)=\{0\}$?
5. **Complete solution:** First find one particular solution, then add every homogeneous solution.
6. **Bases for the subspaces**: For the column space, return to the original matrix and select its pivot columns; for the row space, use the nonzero rows of the RREF; obtain the two nullspaces by solving the corresponding homogeneous systems.
7. **Dimension check**: Verify $r+(n-r)=n$ and $r+(m-r)=m$.
8. **Verification**: Substitute into the original matrix, not only into its RREF.
<!-- bilingual-en:end -->

### 2. 必会证明链
<!-- bilingual-en:start -->
*2. Essential proofs*
<!-- bilingual-en:end -->

考前应能不查笔记完成：
<!-- bilingual-en:start -->
Before the exam, you should be able to prove the following without consulting your notes:
<!-- bilingual-en:end -->

- 行操作保持解集；
- 可逆矩阵的逆唯一；
- $Ax=b$ 完整解是 $x_p+N(A)$；
- 特殊解构成零空间基；
- rank-nullity；
- 行空间与零空间正交，列空间与左零空间正交；
- 连通图关联矩阵的零空间由常数向量张成。
<!-- bilingual-en:start -->
- row operations keep the solution set;
- Inverse uniqueness of invertible matrices;
- The complete $Ax=b$ solution is $x_p+N(A)$;
- The special solutions form a basis of the nullspace;
- rank-nullity;
- row space is orthogonal to nullspace, column space is orthogonal to left nullspace;
- The nullspace of the incidence matrix of a connected graph is spanned by the constant vector.
<!-- bilingual-en:end -->

### 3. Recitation 参数题完整闭环
<!-- bilingual-en:start -->
*3. A complete workflow for the recitation's parameter problem*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}1&1&1\\1&2&3\\3&4&k\end{bmatrix},
\qquad b=\begin{bmatrix}2\\3\\7\end{bmatrix}.
$$

对增广矩阵依次做
<!-- bilingual-en:start -->
Apply the following row operations to the augmented matrix:
<!-- bilingual-en:end -->

$$
R_2\leftarrow R_2-R_1,\quad
R_3\leftarrow R_3-3R_1,\quad
R_3\leftarrow R_3-R_2,
$$

得到
<!-- bilingual-en:start -->
This gives
<!-- bilingual-en:end -->

$$
\left[\begin{array}{ccc|c}
1&1&1&2\\
0&1&2&1\\
0&0&k-5&0
\end{array}\right].
$$

- 若 $k\ne5$，三个主元，$x_3=0,x_2=1,x_1=1$，唯一解 $(1,1,0)^T$。
- 若 $k=5$，第三行是 $0=0$，令 $x_3=t$：
  $$
  x_2=1-2t,\qquad x_1=1+t,
  $$
  $$
  x=\begin{bmatrix}1\\1\\0\end{bmatrix}
  +t\begin{bmatrix}1\\-2\\1\end{bmatrix}.
  $$
<!-- bilingual-en:start -->
- If $k\ne5$, there are three pivots and $x_3=0,x_2=1,x_1=1$, so the unique solution is $(1,1,0)^T$.
- If $k=5$, the third row becomes $0=0$. Letting $x_3=t$ gives the parameterized family displayed above.
<!-- bilingual-en:end -->

消元倍数为 $1,3,1$，所以
<!-- bilingual-en:start -->
The elimination multiplier is $1,3,1$, so
<!-- bilingual-en:end -->

$$
L=\begin{bmatrix}1&0&0\\1&1&0\\3&1&1\end{bmatrix},\qquad
U=\begin{bmatrix}1&1&1\\0&1&2\\0&0&k-5\end{bmatrix}.
$$

即使 $k=5$，分解仍成立，只是 $U$ 奇异。
<!-- bilingual-en:start -->
The factorization remains valid when $k=5$; in that case $U$ is singular.
<!-- bilingual-en:end -->

### 4. 考试中的错误诊断
<!-- bilingual-en:start -->
*4. Troubleshooting in the exam*
<!-- bilingual-en:end -->

- **只报 rank 不报尺寸**：rank 必须同时满足 $r\le m,n$。
- **把 rref 主元列当列空间基**：应回原矩阵取同编号列。
- **参数题不分特殊值**：任何可能成为 $0$ 的主元都必须单独讨论。
- **只给一组解却声称完整**：有自由变量时必须写参数族。
- **把左零空间写进 $\mathbb R^n$**：它属于 $\mathbb R^m$。
- **用 $A^TA$ 消元替代 $A$**：会改变条件数且不是本单元必要步骤；直接消元更清楚。
<!-- bilingual-en:start -->
- **Reporting rank without dimensions**: rank must also satisfy $r\le m$ and $r\le n$.
- **Using pivot columns of the RREF as a basis for the column space**: return to the original matrix and select columns with the same indices.
- **Ignoring special parameter values**: discuss separately every value that can make a pivot vanish.
- **Giving one solution and calling it complete**: when free variables exist, write the full parameterized family.
- **Placing the left nullspace in $\mathbb R^n$**: it belongs to $\mathbb R^m$.
- **Eliminating $A^TA$ instead of $A$**: this squares the condition number and is unnecessary here; direct elimination of $A$ is clearer.
<!-- bilingual-en:end -->

### 三道自检题
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若 $A$ 是 $4\times6$、rank $4$，对每个 $b\in\mathbb R^4$ 解的情况如何？
> **答案**：列空间是 $\mathbb R^4$，所以每个 $b$ 都可解；nullity $=2$，所以每个相容系统都有无穷多解。
> <!-- bilingual-en:start -->
> **Answer**: The column space is $\mathbb R^4$, so every $b$ is attainable. The nullity is $2$, so every consistent system has infinitely many solutions.
> <!-- bilingual-en:end -->

> [!question]- 2. 若 $A$ 是 $6\times4$、rank $4$，解的情况如何？
> **答案**：$N(A)=\{0\}$，所以至多一个解；列空间是 $\mathbb R^6$ 中四维子空间，并非每个 $b$ 可解。
> <!-- bilingual-en:start -->
> **Answer**: $N(A)=\{0\}$, so a solution, when it exists, is unique. The column space is only a four-dimensional subspace of $\mathbb R^6$, so not every $b$ is attainable.
> <!-- bilingual-en:end -->

> [!question]- 3. 一个 $n\times n$ 方阵有 $n$ 个主元时，列出三条立即可得的结论。
> **答案**：可逆；$N(A)=\{0\}$；$C(A)=\mathbb R^n$；等价地每个 $b$ 有唯一解，任选三条即可。
> <!-- bilingual-en:start -->
> **Answer**: It is invertible; $N(A)=\{0\}$; $C(A)=\mathbb R^n$; equivalently, every $b$ gives a unique solution. Any three of these statements suffice.
> <!-- bilingual-en:end -->

### 知识链小结
<!-- bilingual-en:start -->
*summary of knowledge chain*
<!-- bilingual-en:end -->

尺寸 → 消元 → rank → 相容性 → 特解与零空间 → 四空间基与维数 → 代回验算；下面用 Exam 1 的四道题把这条链完整实践。
<!-- bilingual-en:start -->
Dimensions → elimination → rank → consistency → particular solutions and nullspace → bases and dimensions of the four fundamental subspaces → verification by substitution. The following Exam 1 problem exercises this entire chain.
<!-- bilingual-en:end -->

## Exam 1 完整题解
<!-- bilingual-en:start -->
*Complete Exam 1 solutions*
<!-- bilingual-en:end -->

**本地试卷**：[[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex1.pdf#page=1|Unit 1 Exam p.1]]
<!-- bilingual-en:start -->
**Local Quiz Paper**:[[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex1.pdf#page=1|Unit 1 Exam p.1]]
<!-- bilingual-en:end -->

**官方答案**：[[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex1s.pdf#page=1|Unit 1 Exam Solutions p.1]]
<!-- bilingual-en:start -->
**Official answer**: [[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex1s.pdf#page=1|Unit 1 Exam Solutions p.1]]
<!-- bilingual-en:end -->

> [!warning] PDF 文本层说明
> 官方答案 PDF 的文本层编码损坏，但页面公式可正常阅读。本节按题目页面、答案页面与直接代数验算交叉核对；所有矩阵均额外做尺寸和乘积检查。
> <!-- bilingual-en:start -->
> The official answer PDF's text-layer encoding is corrupted, but the page formula reads correctly.  This section cross-checks by topic page, answer page, and direct algebraic checking; all matrices are additionally checked for size and product.
> <!-- bilingual-en:end -->

### Exam Problem 1：由存在性与唯一性反推尺寸和秩
<!-- bilingual-en:start -->
*Exam Problem 1: Size and rank deduced from existence and uniqueness*
<!-- bilingual-en:end -->

题设：$A$ 为 $m\times n$，
<!-- bilingual-en:start -->
Question: $A$ is $m\times n$.
<!-- bilingual-en:end -->

$$
Ax=\begin{bmatrix}1\\1\\1\end{bmatrix}\text{ 无解},
\qquad
Ax=\begin{bmatrix}0\\1\\0\end{bmatrix}\text{ 恰有一解}.
$$

#### (a) 求 $m,n,r$ 的全部可能信息
<!-- bilingual-en:start -->
*(a) Determine all possible information about $m,n,r$*
<!-- bilingual-en:end -->

**已知与目标**：两个右端都有三个分量，所以输出空间是 $\mathbb R^3$；要从一个无解和一个唯一解推断 rank。
<!-- bilingual-en:start -->
**Given and goal**: Both right-hand sides have three components, so the output space is $\mathbb R^3$. Use one inconsistent system and one uniquely solvable system to infer the rank and dimensions.
<!-- bilingual-en:end -->

**逐步推导**：
<!-- bilingual-en:start -->
**Derive step by step**:
<!-- bilingual-en:end -->

1. $Ax$ 有三个分量，故 $m=3$。
2. 第二个系统存在且唯一。若 $N(A)$ 含非零 $z$，则由一个解 $x_p$ 可产生 $x_p+tz$ 的无穷多个解，矛盾。因此 $N(A)=\{0\}$。
3. rank-nullity 给 $r+n-r=n$ 且 nullity $=0$，所以 $r=n$。
4. 第一个系统无解，说明 $C(A)\ne\mathbb R^3$，所以 $r<3$。
5. rank 为非负整数，而第二个非零右端可达，所以 $r\ge1$。
<!-- bilingual-en:start -->
1. Since $Ax$ has three components, $m=3$.
2. The second system has exactly one solution. If $N(A)$ contained a nonzero vector $z$, one solution $x_p$ would generate infinitely many solutions $x_p+tz$, a contradiction. Hence $N(A)=\{0\}$.
3. Rank–nullity gives $r+(n-r)=n$. Since the nullity is zero, $r=n$.
4. The first system is inconsistent, so $C(A)\ne\mathbb R^3$ and therefore $r<3$.
5. The rank is an integer, and the nonzero second right-hand side is attainable, so $r\ge1$.
<!-- bilingual-en:end -->

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{m=3,\qquad r=n\in\{1,2\}.}
$$

#### (b) 求 $Ax=0$ 的全部解
<!-- bilingual-en:start -->
*(b) Finding all solutions of $Ax=0$*
<!-- bilingual-en:end -->

由上一步 $N(A)=\{0\}$：
<!-- bilingual-en:start -->
From the previous $N(A)=\{0\}$:
<!-- bilingual-en:end -->

$$
\boxed{x=0\in\mathbb R^n.}
$$

#### (c) 给出一个例子
<!-- bilingual-en:start -->
*(c) Give an example*
<!-- bilingual-en:end -->

取 $n=r=1$：
<!-- bilingual-en:start -->
Take $n=r=1$:
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}0\\1\\0\end{bmatrix}.
$$

则 $Ax=(0,x,0)^T$。右端 $(0,1,0)^T$ 唯一对应 $x=1$；$(1,1,1)^T$ 不在 $A$ 的一维列空间中。也可取 $n=r=2$ 的例子
<!-- bilingual-en:start -->
$Ax=(0,x,0)^T$.  The right-hand $(0,1,0)^T$ uniquely corresponds to $x=1$; $(1,1,1)^T$ is not in the one-dimensional column space of $A$.  An example of $n=r=2$ may also be used
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}.
$$

#### 错误诊断
<!-- bilingual-en:start -->
*Troubleshooting*
<!-- bilingual-en:end -->

- 仅从“一个 $b$ 唯一可解”不能推出 $C(A)=\mathbb R^3$；它只推出核为零。
- $r=n$ 不等于 $r=m$；本题恰因 $r<m$ 才有不可达右端。
<!-- bilingual-en:start -->
- The fact that one particular $b$ has a unique solution does not imply $C(A)=\mathbb R^3$; it implies only that the nullspace is zero.
- The equality $r=n$ does not imply $r=m$. Here $r<m$, which is precisely why some right-hand sides are unattainable.
<!-- bilingual-en:end -->

### Exam Problem 2：初等矩阵、逆矩阵与 LU
<!-- bilingual-en:start -->
*Exam Problem 2: Elementary Matrix, Inverse Matrix and LU*
<!-- bilingual-en:end -->

题设：$A$ 经以下顺序化为 $I$：
<!-- bilingual-en:start -->
Title: $A$ is converted to $I$ in the following order:
<!-- bilingual-en:end -->

1. $E_{21}$：$R_2\leftarrow R_2-4R_1$；
2. $E_{31}$：$R_3\leftarrow R_3-3R_1$；
3. $E_{23}$：$R_2\leftarrow R_2-R_3$。

对应矩阵为
<!-- bilingual-en:start -->
The mapping matrix is
<!-- bilingual-en:end -->

$$
E_{21}=\begin{bmatrix}1&0&0\\-4&1&0\\0&0&1\end{bmatrix},
$$

$$
E_{31}=\begin{bmatrix}1&0&0\\0&1&0\\-3&0&1\end{bmatrix},
\qquad
E_{23}=\begin{bmatrix}1&0&0\\0&1&-1\\0&0&1\end{bmatrix}.
$$

#### (a) 求 $A^{-1}$
<!-- bilingual-en:start -->
*(a) Find $A^{-1}$*
<!-- bilingual-en:end -->

因
<!-- bilingual-en:start -->
because
<!-- bilingual-en:end -->

$$
E_{23}E_{31}E_{21}A=I,
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
A^{-1}=E_{23}E_{31}E_{21}.
$$

先乘后两项，再左乘 $E_{23}$（即把第二行减第三行）：
<!-- bilingual-en:start -->
First multiply the latter two matrices, then multiply on the left by $E_{23}$, which subtracts row three from row two:
<!-- bilingual-en:end -->

$$
A^{-1}
=\begin{bmatrix}
1&0&0\\
-1&1&-1\\
-3&0&1
\end{bmatrix}.
$$

#### (b) 求原矩阵 $A$
<!-- bilingual-en:start -->
*(b) Find the original matrix $A$*
<!-- bilingual-en:end -->

撤销操作时顺序反转且符号改变：
<!-- bilingual-en:start -->
Undoing the operations reverses their order and changes the signs of the elimination multipliers:
<!-- bilingual-en:end -->

$$
A=E_{21}^{-1}E_{31}^{-1}E_{23}^{-1}
=\begin{bmatrix}
1&0&0\\
4&1&1\\
3&0&1
\end{bmatrix}.
$$

验算：
<!-- bilingual-en:start -->
Check:
<!-- bilingual-en:end -->

$$
AA^{-1}
=\begin{bmatrix}1&0&0\\4&1&1\\3&0&1\end{bmatrix}
\begin{bmatrix}1&0&0\\-1&1&-1\\-3&0&1\end{bmatrix}
=I_3.
$$

#### (c) 求 $A=LU$ 中的 $L$
<!-- bilingual-en:start -->
*(c) Find $L$ in $A=LU$*
<!-- bilingual-en:end -->

LU 只消到上三角，不必把上方的 $1$ 再消掉。对 $A$：
<!-- bilingual-en:start -->
LU elimination stops once the matrix is upper triangular; it does not continue by clearing the entries above the pivots. For $A$:
<!-- bilingual-en:end -->

$$
R_2\leftarrow R_2-4R_1,\qquad
R_3\leftarrow R_3-3R_1,
$$

得到
<!-- bilingual-en:start -->
This gives
<!-- bilingual-en:end -->

$$
U=\begin{bmatrix}1&0&0\\0&1&1\\0&0&1\end{bmatrix}.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{L=\begin{bmatrix}1&0&0\\4&1&0\\3&0&1\end{bmatrix}.}
$$

直接计算 $LU=A$。第三个 Gauss–Jordan 操作 $E_{23}$ 是把 $U$ 进一步化为 $I$，不属于标准 LU 的下三角消元，因此其倍数不放进 $L$。
<!-- bilingual-en:start -->
Direct multiplication verifies that $LU=A$. The third Gauss–Jordan operation, $E_{23}$, reduces $U$ further to $I$; it is not part of the forward elimination used in a standard LU factorization, so its multiplier does not belong in $L$.
<!-- bilingual-en:end -->

### Exam Problem 3：参数矩阵的列空间、零空间与完整解
<!-- bilingual-en:start -->
*Exam Problem 3: Column Space, Nullspace and Complete Solution of Parameter Matrix*
<!-- bilingual-en:end -->

题设
<!-- bilingual-en:start -->
question creation
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}
1&1&2&4\\
3&c&2&8\\
0&0&2&2
\end{bmatrix}.
$$

做
<!-- bilingual-en:start -->
do
<!-- bilingual-en:end -->

$$
R_2\leftarrow R_2-3R_1
$$

得到第二行 $(0,c-3,-4,-4)$；第三行是 $(0,0,2,2)$。特殊值只有 $c=3$。
<!-- bilingual-en:start -->
This produces row two $(0,c-3,-4,-4)$ and row three $(0,0,2,2)$. The only special parameter value is $c=3$.
<!-- bilingual-en:end -->

#### (a) 对每个 $c$ 求列空间基
<!-- bilingual-en:start -->
*(a) Column space basis for each $c$*
<!-- bilingual-en:end -->

若 $c\ne3$，第二列产生第二个主元，第三列产生第三个主元，rank $=3$。取原矩阵前三个主元列：
<!-- bilingual-en:start -->
If $c\ne3$, column two supplies the second pivot and column three supplies the third, so the rank is $3$. Take the first three pivot columns from the original matrix:
<!-- bilingual-en:end -->

$$
\boxed{
\left\{
\begin{bmatrix}1\\3\\0\end{bmatrix},
\begin{bmatrix}1\\c\\0\end{bmatrix},
\begin{bmatrix}2\\2\\2\end{bmatrix}
\right\}.}
$$

若 $c=3$，第二列与第一列相同，且消元后的第二、第三非零行互为倍数，rank $=2$。主元列为原第 1、3 列：
<!-- bilingual-en:start -->
If $c=3$, the second column equals the first, and the second and third nonzero rows after elimination are scalar multiples of one another, so the rank is $2$. The pivot columns are columns 1 and 3 of the original matrix:
<!-- bilingual-en:end -->

$$
\boxed{
\left\{
\begin{bmatrix}1\\3\\0\end{bmatrix},
\begin{bmatrix}2\\2\\2\end{bmatrix}
\right\}.}
$$

#### (b) 对每个 $c$ 求零空间基
<!-- bilingual-en:start -->
*(b) Basis of nullspace for each $c$*
<!-- bilingual-en:end -->

齐次系统第三行给 $x_3=-x_4$。第二行化简为
<!-- bilingual-en:start -->
In the homogeneous system, the third row gives $x_3=-x_4$. The second row then reduces to
<!-- bilingual-en:end -->

$$
(c-3)x_2-4x_3-4x_4=(c-3)x_2=0.
$$

若 $c\ne3$，$x_2=0$；第一行给 $x_1=-2x_4$，所以
<!-- bilingual-en:start -->
If $c\ne3$, then $x_2=0$, and the first row gives $x_1=-2x_4$. Hence
<!-- bilingual-en:end -->

$$
\boxed{N(A)=\operatorname{span}\left\{
\begin{bmatrix}-2\\0\\-1\\1\end{bmatrix}
\right\}.}
$$

若 $c=3$，$x_2,x_4$ 都自由，且
<!-- bilingual-en:start -->
If $c=3$, both $x_2$ and $x_4$ are free, and
<!-- bilingual-en:end -->

$$
x_1=-x_2-2x_4,\qquad x_3=-x_4.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\boxed{N(A)=\operatorname{span}\left\{
\begin{bmatrix}-1\\1\\0\\0\end{bmatrix},
\begin{bmatrix}-2\\0\\-1\\1\end{bmatrix}
\right\}.}
$$

维数检查：$c\ne3$ 时 $r+\text{nullity}=3+1=4$；$c=3$ 时 $2+2=4$。
<!-- bilingual-en:start -->
Dimension checking: $r+\text{nullity}=3+1=4$ for $c\ne3$; $2+2=4$ for $c=3$.
<!-- bilingual-en:end -->

#### (c) 求 $Ax=(1,c,0)^T$ 的完整解
<!-- bilingual-en:start -->
*(c) Finding the complete solution of $Ax=(1,c,0)^T$*
<!-- bilingual-en:end -->

容易验证
<!-- bilingual-en:start -->
Easy to verify
<!-- bilingual-en:end -->

$$
x_p=\begin{bmatrix}0\\1\\0\\0\end{bmatrix}
$$

对所有 $c$ 都满足 $Ax_p=(1,c,0)^T$。因此把对应零空间加上即可。
<!-- bilingual-en:start -->
$Ax_p=(1,c,0)^T$ is satisfied for all $c$.  So we can add the corresponding nullspace.
<!-- bilingual-en:end -->

若 $c\ne3$：
<!-- bilingual-en:start -->
If $c\ne3$:
<!-- bilingual-en:end -->

$$
\boxed{x=\begin{bmatrix}0\\1\\0\\0\end{bmatrix}
+t\begin{bmatrix}-2\\0\\-1\\1\end{bmatrix}.}
$$

若 $c=3$：
<!-- bilingual-en:start -->
If $c=3$:
<!-- bilingual-en:end -->

$$
\boxed{x=\begin{bmatrix}0\\1\\0\\0\end{bmatrix}
+s\begin{bmatrix}-1\\1\\0\\0\end{bmatrix}
+t\begin{bmatrix}-2\\0\\-1\\1\end{bmatrix}.}
$$

#### 错误诊断
<!-- bilingual-en:start -->
*Troubleshooting*
<!-- bilingual-en:end -->

- 参数矩阵必须在 $c=3$ 分情况；否则会非法除以 $c-3$。
- 列空间基必须取原矩阵列，不取消元结果的列。
- 特解对全部 $c$ 都成立，是本题最省计算的入口。
<!-- bilingual-en:start -->
- The parameterized matrix must be split into the cases $c=3$ and $c\ne3$; otherwise one may divide illegally by $c-3$.
- A basis for the column space must use columns of the original matrix, not columns of the row-reduced matrix.
- The particular solution is valid for every $c$ and provides the most efficient starting point for the problem.
<!-- bilingual-en:end -->

### Exam Problem 4：矩形矩阵、列关系与 RREF 空间
<!-- bilingual-en:start -->
*Exam Problem 4: Rectangular Matrices, Column Relationships and RREF Spaces*
<!-- bilingual-en:end -->

#### (a) $3\times5$ 矩阵的零空间信息
<!-- bilingual-en:start -->
*(a) Zero-space information of $3\times5$ matrices*
<!-- bilingual-en:end -->

$A$ 有 $5$ 列而 rank $r\le3$，所以
<!-- bilingual-en:start -->
$A$ has $5$ columns and rank $r\le3$, so
<!-- bilingual-en:end -->

$$
\dim N(A)=5-r\ge2.
$$

因此 $N(A)$ 是 $\mathbb R^5$ 的子空间，至少有两个线性无关的非零方向；$Ax=0$ 必有无穷多个解。
<!-- bilingual-en:start -->
Therefore $N(A)$ is a subspace of $\mathbb R^5$ with at least two linearly independent nonzero directions, and $Ax=0$ has infinitely many solutions.
<!-- bilingual-en:end -->

#### (b) 由给定 rref 推断原列关系
<!-- bilingual-en:start -->
*(b) Inference of original column relationship from given rref*
<!-- bilingual-en:end -->

给定
<!-- bilingual-en:start -->
given
<!-- bilingual-en:end -->

$$
R=\operatorname{rref}(A)=
\begin{bmatrix}
1&4&0&0&0\\
0&0&0&1&0\\
0&0&0&0&1
\end{bmatrix}.
$$

主元列是 $1,4,5$，所以原矩阵的 $a_1,a_4,a_5$ 线性无关，并构成 $C(A)$ 的基。rank $=3=m$，故
<!-- bilingual-en:start -->
The pivot columns are $1,4,5$, so $a_1,a_4,a_5$ in the original matrix are linearly independent and form a basis for $C(A)$. Since $\operatorname{rank}(A)=3=m$,
<!-- bilingual-en:end -->

$$
C(A)=\mathbb R^3.
$$

消元保持列之间的线性关系。由 $R$ 的列可见
<!-- bilingual-en:start -->
Row operations preserve linear relations among columns. From the columns of $R$ we read
<!-- bilingual-en:end -->

$$
R_{:2}=4R_{:1},\qquad R_{:3}=0,
$$

因此原列满足
<!-- bilingual-en:start -->
So the original column satisfies
<!-- bilingual-en:end -->

$$
\boxed{a_2=4a_1,\qquad a_3=0.}
$$

完整信息可概括为：三条 pivot columns $a_1,a_4,a_5$ 是 $\mathbb R^3$ 的一组基；其余两列分别是 $4a_1$ 与零列。
<!-- bilingual-en:start -->
The complete information can be summarized as follows: the three pivot columns $a_1,a_4,a_5$ form a basis of $\mathbb R^3$; the remaining two columns are $4a_1$ and zero.
<!-- bilingual-en:end -->

#### (c) 所有 $3\times3$ RREF 张成什么子空间
<!-- bilingual-en:start -->
*(c) What subspace is spanned by all $3\times3$ RREF matrices?*
<!-- bilingual-en:end -->

任何 rref 的第 $i$ 个主元位置至少在第 $i$ 列，且主元左侧为零、零行在底部。因此每个 $3\times3$ rref 都是上三角矩阵，故其 span 包含于上三角矩阵空间
<!-- bilingual-en:start -->
In any RREF matrix, the $i$th pivot can occur no earlier than column $i$; entries to the left of each pivot are zero, and all zero rows lie at the bottom. Thus every $3\times3$ RREF matrix is upper triangular, so their span is contained in the space of upper triangular matrices
<!-- bilingual-en:end -->

$$
S=\left\{
\begin{bmatrix}a&b&c\\0&d&e\\0&0&f\end{bmatrix}:a,b,c,d,e,f\in\mathbb R
\right\}.
$$

反过来，六个上三角标准基矩阵均可由 rref 的差得到：
<!-- bilingual-en:start -->
Conversely, the six upper triangular standard basis matrices can be obtained by the difference of rref:
<!-- bilingual-en:end -->

- $E_{11}$ 本身是 rref；
- $E_{12}=\begin{bmatrix}1&1&0\\0&0&0\\0&0&0\end{bmatrix}-E_{11}$，$E_{13}$ 同理；
- $E_{22}=\operatorname{diag}(1,1,0)-E_{11}$；
- $E_{23}$ 可由 $\begin{bmatrix}1&0&0\\0&1&1\\0&0&0\end{bmatrix}-\operatorname{diag}(1,1,0)$ 得到；
- $E_{33}=I-\operatorname{diag}(1,1,0)$。
<!-- bilingual-en:start -->
- $E_{11}$ itself is rref;
- $E_{12}=\begin{bmatrix}1&1&0\\0&0&0\\0&0&0\end{bmatrix}-E_{11}$, $E_{13}$ likewise;
- $E_{22}=\operatorname{diag}(1,1,0)-E_{11}$;
- $E_{23}$ can be obtained by $\begin{bmatrix}1&0&0\\0&1&1\\0&0&0\end{bmatrix}-\operatorname{diag}(1,1,0)$;
- $E_{33}=I-\operatorname{diag}(1,1,0)$.
<!-- bilingual-en:end -->

所以所有上三角矩阵都在该 span 中，最终
<!-- bilingual-en:start -->
Thus every upper triangular matrix lies in the span, and hence
<!-- bilingual-en:end -->

$$
\boxed{S=\{3\times3\text{ 上三角矩阵}\},\qquad \dim S=6.}
$$

### Exam 1 题后复盘
<!-- bilingual-en:start -->
*Review after Exam 1*
<!-- bilingual-en:end -->

四题分别检查了 Unit I 的四个层次：
<!-- bilingual-en:start -->
The four questions examine the four levels of Unit I:
<!-- bilingual-en:end -->

1. 从存在性/唯一性反推维数结构；
2. 把行操作、逆与 LU 串起来；
3. 对参数值做 rank、四空间和完整解分流；
4. 把矩形矩阵及矩阵空间纳入统一的基与维数语言。
<!-- bilingual-en:start -->
1. Infer dimensions and rank from existence and uniqueness;
2. connect row operations, inverses, and LU factorization;
3. separate parameter cases and determine rank, fundamental subspaces, and complete solutions;
4. describe rectangular matrices and matrix spaces in one language of bases and dimensions.
<!-- bilingual-en:end -->

若某题计算正确却无法解释“为什么要分情况、这个向量位于哪个空间、维数是否闭合”，说明还停留在算法层，尚未完成 Unit I 的结构化理解。
<!-- bilingual-en:start -->
If you can compute an answer but cannot explain why cases must be separated, which space a vector belongs to, or whether the dimensions are consistent, then you are still operating only at the algorithmic level and have not yet formed the structural understanding required by Unit I.
<!-- bilingual-en:end -->

## 本单元最终检查表
<!-- bilingual-en:start -->
*Final Checklist in this module*
<!-- bilingual-en:end -->

### 概念与尺寸
<!-- bilingual-en:start -->
*Concepts and Dimensions*
<!-- bilingual-en:end -->

- [ ] 我能在任何 $m\times n$ 矩阵旁立刻写出四个基本子空间所在的 $\mathbb R^m$ 或 $\mathbb R^n$。
- [ ] 我能区分行图像、列图像与线性映射图像。
- [ ] 我能解释 rank 是有效方向数，nullity 是丢失的输入自由度。
<!-- bilingual-en:start -->
- [ ] For any $m\times n$ matrix, I can immediately state whether each of the four fundamental subspaces lies in $\mathbb R^m$ or $\mathbb R^n$.
- [ ] I can distinguish the row space, column space, and image of a linear map.
- [ ] I can explain rank as the number of effective output directions and nullity as the number of lost input degrees of freedom.
<!-- bilingual-en:end -->

### 算法
<!-- bilingual-en:start -->
*algorithm*
<!-- bilingual-en:end -->

- [ ] 我能用增广矩阵消元，并在需要时换行。
- [ ] 我能从消元倍数构造 $L$，从结果读出 $U$，并处理 $PA=LU$。
- [ ] 我能由 rref 构造零空间特殊解，并从原矩阵挑列空间基。
- [ ] 我能把相容系统写成 $x_p+N(A)$，再代回原式验算。
<!-- bilingual-en:start -->
- [ ] I can perform elimination on an augmented matrix and swap rows when needed.
- [ ] I can construct $L$ from the elimination multipliers, read $U$ from the result, and handle the factorization $PA=LU$.
- [ ] I can construct special nullspace solutions from rref and choose bases for the relevant subspaces from the original matrix.
- [ ] I can write a consistent system's complete solution as $x_p+N(A)$ and verify it in the original equation.
<!-- bilingual-en:end -->

### 证明
<!-- bilingual-en:start -->
*prove*
<!-- bilingual-en:end -->

- [ ] 我能证明行操作保持解集、逆矩阵唯一、完整解公式和 rank-nullity。
- [ ] 我能证明两对基本子空间互为正交补。
- [ ] 我能证明连通图的关联矩阵零空间由常数向量张成。
<!-- bilingual-en:start -->
- [ ] I can prove that row operations preserve the solution set, inverse matrix uniqueness, the complete solution formula, and rank-nullity.
- [ ] I can prove that two pairs of elementary subspaces complement each other orthogonally.
- [ ] I can prove that the nullspace of the incidence matrix of a connected graph is spanned by a constant vector.
<!-- bilingual-en:end -->

### 下一单元接口
<!-- bilingual-en:start -->
*Next Unit Interface*
<!-- bilingual-en:end -->

Unit II 将从
<!-- bilingual-en:start -->
The Unit II will be launched from the
<!-- bilingual-en:end -->

$$
C(A^T)\perp N(A),\qquad C(A)\perp N(A^T)
$$

出发，研究正交投影、最小二乘和正交基；也就是说，当 $b\notin C(A)$、精确方程无解时，我们将寻找 $C(A)$ 中离 $b$ 最近的向量。
<!-- bilingual-en:start -->
In other words, when $b\notin C(A)$, the exact equation has no solution, we will look for the vector in $C(A)$ which is closest to $b$.
<!-- bilingual-en:end -->
