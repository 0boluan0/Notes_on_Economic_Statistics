---
aliases:
  - MIT 18.06SC Unit II
  - 最小二乘、行列式与特征值
tags:
  - 线性代数
  - mit-ocw
  - course-note
---

# Least Squares, Determinants and Eigenvalues

> [!abstract] 本单元要解决什么
> Unit II 有三条彼此衔接的主线。第一条用**正交性**把无解的 $Ax=b$ 改造成唯一的最佳逼近；第二条用**行列式**判断可逆性并度量有向体积；第三条寻找矩阵不改变方向的**特征向量**，从而快速计算矩阵幂、矩阵指数和长期稳态。
>
> 本文按 MIT 18.06SC Fall 2011 官方顺序组织：Session 2.1–2.11 → Session 2.12 Exam Review → Exam 2。正文自足；PDF 用于核对原材料，而不是理解正文的前提。
> <!-- bilingual-en:start -->
> Unit II develops three connected themes. First, **orthogonality** turns an inconsistent system $Ax=b$ into a uniquely defined best approximation. Second, the **determinant** tests invertibility and measures oriented volume. Third, **eigenvectors** identify directions that a matrix does not rotate, making matrix powers, matrix exponentials, and long-run behavior easier to compute.
> This article is organized in the official order MIT 18.06SC Fall 2011: Session 2.1-2.11 → Session 2.12 Exam Review → Exam 2.  Body self-sufficiency; PDFs are used to check raw materials, not to understand the body.
> <!-- bilingual-en:end -->

## 记号、空间与尺寸约定
<!-- bilingual-en:start -->
*Notation, spaces, and dimension conventions*
<!-- bilingual-en:end -->

- 除非特别说明，向量属于实数域，$A\in\mathbb R^{m\times n}$，$x\in\mathbb R^n$，$b\in\mathbb R^m$。
- $C(A)\subseteq\mathbb R^m$ 是列空间，$C(A^T)\subseteq\mathbb R^n$ 是行空间；$N(A)\subseteq\mathbb R^n$，$N(A^T)\subseteq\mathbb R^m$。
- $x^Ty$ 是欧氏内积，$\|x\|=\sqrt{x^Tx}$。本单元后半出现复向量时，内积必须改为 $x^*y$。
- $I_m$ 表示 $m\times m$ 单位矩阵；尺寸不含混时简写为 $I$。
- $\hat x$ 表示最小二乘解，$p=A\hat x$ 表示投影，$e=b-p$ 表示残差。
<!-- bilingual-en:start -->
- Unless stated otherwise, vectors and matrices are real: $A\in\mathbb R^{m\times n}$, $x\in\mathbb R^n$, and $b\in\mathbb R^m$.
- $C(A)\subseteq\mathbb R^m$ is column space and $C(A^T)\subseteq\mathbb R^n$ is row space; $N(A)\subseteq\mathbb R^n$, $N(A^T)\subseteq\mathbb R^m$.
- $x^Ty$ is the Euclidean inner product and $\|x\|=\sqrt{x^Tx}$. When complex vectors appear later in the unit, the inner product must be written as $x^*y$.
- $I_m$ denotes the $m\times m$ identity matrix; write simply $I$ when its size is unambiguous.
- $\hat x$ for the least squares solution, $p=A\hat x$ for the projection, and $e=b-p$ for the residual.
<!-- bilingual-en:end -->

## Session 导航

1. [[#Session 2.1 Orthogonal vectors and subspaces|正交向量、正交子空间与正交补]]
2. [[#Session 2.2 Projections onto subspaces|投影到子空间]]
3. [[#Session 2.3 Projection matrices and least squares|投影矩阵与最小二乘]]
4. [[#Session 2.4 Orthogonal matrices and Gram–Schmidt|正交矩阵、Gram–Schmidt 与 QR]]
5. [[#Session 2.5 Properties of determinants|行列式的定义性质]]
6. [[#Session 2.6 Determinant formulas and cofactors|大公式、余子式展开]]
7. [[#Session 2.7 Cramer's rule, inverse matrix and volume|Cramer 法则、逆矩阵与体积]]
8. [[#Session 2.8 Eigenvalues and eigenvectors|特征值与特征向量]]
9. [[#Session 2.9 Diagonalization and powers of A|对角化与矩阵幂]]
10. [[#Session 2.10 Differential equations and $e^{At}$|微分方程与矩阵指数]]
11. [[#Session 2.11 Markov matrices and Fourier series|Markov 矩阵与 Fourier 级数]]
12. [[#Session 2.12 Exam 2 review|Exam 2 复习]]
13. [[#Exam 2|Exam 2 完整题解]]

---

## Session 2.1 Orthogonal vectors and subspaces

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节回答三个问题：怎样把“垂直”推广到任意维；四个基本子空间为什么成对正交；为什么 $A^TA$ 可逆恰好等价于 $A$ 的列向量线性无关。
<!-- bilingual-en:start -->
This section answers three questions: how to generalize “perpendicular” to arbitrary dimensions; why the four fundamental subspaces occur in orthogonal pairs; and why invertibility of $A^TA$ is equivalent to linear independence of the columns of $A$.
<!-- bilingual-en:end -->

前置知识是四个基本子空间、秩—零度定理以及矩阵乘法尺寸。若 $A\in\mathbb R^{m\times n}$，则 $A^TA\in\mathbb R^{n\times n}$，所以 $N(A)$ 与 $N(A^TA)$ 都是 $\mathbb R^n$ 的子空间，二者才可以比较。
<!-- bilingual-en:start -->
The prerequisites are the four fundamental subspaces, the rank–nullity theorem, and matrix-dimension rules. If $A\in\mathbb R^{m\times n}$, then $A^TA\in\mathbb R^{n\times n}$, so both $N(A)$ and $N(A^TA)$ are subspaces of $\mathbb R^n$ and can be compared directly.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S01_Lecture_Orthogonal_Vectors_and_Subspaces.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S01_Recitation_Problem_Solving_Orthogonal_Vectors_and_Subspaces.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.1prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.1sol.pdf#page=1)

### Lecture：从正交向量到四个基本子空间
<!-- bilingual-en:start -->
*Lecture: From Orthogonal Vector to Four Basic Subspaces*
<!-- bilingual-en:end -->

两个向量 $x,y\in\mathbb R^n$ 的[[正交投影与最小二乘#正交补与最近点|正交性（orthogonality）]]定义为
<!-- bilingual-en:start -->
The [[正交投影与最小二乘#正交补与最近点|orthogonality]] of two vectors $x,y\in\mathbb R^n$ is defined as
<!-- bilingual-en:end -->

$$
x\perp y\quad\Longleftrightarrow\quad x^Ty=0.
$$

它不是二维图形的偶然规则。由余弦公式 $x^Ty=\|x\|\|y\|\cos\theta$，非零向量的内积为零恰好对应 $\theta=90^\circ$。零向量与所有向量正交，但零向量没有可定义的方向或夹角。
<!-- bilingual-en:start -->
This is not an accidental rule confined to two-dimensional diagrams. For nonzero vectors, the cosine formula $x^Ty=\|x\|\|y\|\cos\theta$ shows that a zero inner product is equivalent to $\theta=90^\circ$. The zero vector is orthogonal to every vector, although it has no defined direction or angle.
<!-- bilingual-en:end -->

若 $x\perp y$，则
<!-- bilingual-en:start -->
If $x\perp y$,
<!-- bilingual-en:end -->

$$
\begin{aligned}
\|x+y\|^2
&=(x+y)^T(x+y)\\
&=x^Tx+x^Ty+y^Tx+y^Ty\\
&=\|x\|^2+\|y\|^2.
\end{aligned}
$$

这就是高维勾股定理。后面“投影是最近点”的证明，只是在这个等式中把 $x,y$ 换成两个互相正交的误差分量。
<!-- bilingual-en:start -->
This is the Pythagorean theorem in higher dimensions. The later proof that an orthogonal projection is the nearest point uses the same identity with $x$ and $y$ replaced by two orthogonal error components.
<!-- bilingual-en:end -->

两个子空间 $S,T\subseteq\mathbb R^n$ 正交，是指
<!-- bilingual-en:start -->
Two subspaces $S,T\subseteq\mathbb R^n$ are orthogonal if
<!-- bilingual-en:end -->

$$
s^Tt=0\qquad\text{对所有 }s\in S, t\in T.
$$

要求是“任意一对”，不是只找到一对正交向量。例如三维空间中两个相交平面并不是正交子空间：它们的交线中有非零向量同时属于两者，该向量不可能与自身正交。
<!-- bilingual-en:start -->
The condition must hold for every pair, not merely for one chosen pair of orthogonal vectors. For example, two planes that intersect in a line in three-dimensional space cannot be orthogonal subspaces: a nonzero vector on their intersection belongs to both planes and cannot be orthogonal to itself.
<!-- bilingual-en:end -->

子空间 $S$ 的[[正交投影与最小二乘#正交补与最近点|正交补（orthogonal complement）]]定义为
<!-- bilingual-en:start -->
The [[正交投影与最小二乘#正交补与最近点|orthogonal complement]] of the subspace $S$ is defined as
<!-- bilingual-en:end -->

$$
S^\perp=\{x\in\mathbb R^n:x^Ts=0\text{ 对所有 }s\in S\}.
$$

$S^\perp$ 本身是子空间：若 $x,y\in S^\perp$ 且 $\alpha,\beta\in\mathbb R$，则对任意 $s\in S$，
<!-- bilingual-en:start -->
The $S^\perp$ itself is a subspace: if $x,y\in S^\perp$ and $\alpha,\beta\in\mathbb R$, for any $s\in S$,
<!-- bilingual-en:end -->

$$
(\alpha x+\beta y)^Ts=\alpha x^Ts+\beta y^Ts=0.
$$

#### 四个基本子空间的两对正交关系
<!-- bilingual-en:start -->
*Two-pair Orthogonal Relations of Four Basic Subspaces*
<!-- bilingual-en:end -->

设 $A$ 的行向量为 $r_1^T,\dots,r_m^T$。若 $x\in N(A)$，则
<!-- bilingual-en:start -->
Let $A$'s row vector be $r_1^T,\dots,r_m^T$.  If $x\in N(A)$,
<!-- bilingual-en:end -->

$$
Ax=0\quad\Longrightarrow\quad r_i^Tx=0\quad(i=1,\dots,m).
$$

所以 $x$ 与每一行正交，也与行的任意线性组合正交：
<!-- bilingual-en:start -->
So $x$ is orthogonal to each row and any linear combination of rows:
<!-- bilingual-en:end -->

$$
C(A^T)\perp N(A)\qquad(\text{都位于 }\mathbb R^n).
$$

把同一论证用于 $A^T$ 得
<!-- bilingual-en:start -->
Use the same argument with $A^T$
<!-- bilingual-en:end -->

$$
C(A)\perp N(A^T)\qquad(\text{都位于 }\mathbb R^m).
$$

若 $\operatorname{rank}(A)=r$，则
<!-- bilingual-en:start -->
If $\operatorname{rank}(A)=r$,
<!-- bilingual-en:end -->

$$
\dim C(A^T)=r,\quad \dim N(A)=n-r,
$$

两者维数相加为 $n$。因此它们不仅正交，而且互为正交补：
<!-- bilingual-en:start -->
The sum of the two dimensions is $n$.  So they are not only orthogonal, but also complementary to each other:
<!-- bilingual-en:end -->

$$
C(A^T)^\perp=N(A),\qquad N(A)^\perp=C(A^T).
$$

把上面的逐行论证应用于 $A^T$，并使用 $\dim C(A)+\dim N(A^T)=r+(m-r)=m$，可得 $C(A)$ 与 $N(A^T)$ 在 $\mathbb R^m$ 中互为正交补。由此每个 $v\in\mathbb R^n$ 都能唯一写为
<!-- bilingual-en:start -->
Applying the row-by-row argument above to $A^T$ and using $\dim C(A)+\dim N(A^T)=r+(m-r)=m$ shows that $C(A)$ and $N(A^T)$ are orthogonal complements in $\mathbb R^m$. Correspondingly, on the input side every $v\in\mathbb R^n$ has the unique decomposition
<!-- bilingual-en:end -->

$$
v=r+n,\qquad r\in C(A^T),\quad n\in N(A).
$$

唯一性证明：若 $v=r_1+n_1=r_2+n_2$，则 $r_1-r_2=n_2-n_1$ 同时属于两个正交补；它与自身正交，只能是零向量，故 $r_1=r_2,n_1=n_2$。
<!-- bilingual-en:start -->
The uniqueness proof is that if $v=r_1+n_1=r_2+n_2$, $r_1-r_2=n_2-n_1$ belongs to two orthogonal complements at the same time; it is orthogonal to itself and can only be a zero vector, so $r_1=r_2,n_1=n_2$.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-orthogonal-complements.png|760]]

#### 为什么 $N(A^TA)=N(A)$
<!-- bilingual-en:start -->
*Why, $N(A^TA)=N(A)$*
<!-- bilingual-en:end -->

> [!note] 定理
> 对任意实矩阵 $A\in\mathbb R^{m\times n}$，
> $$N(A^TA)=N(A).$$
> 因而 $\operatorname{rank}(A^TA)=\operatorname{rank}(A)$；$A^TA$ 可逆当且仅当 $A$ 满列秩。
> <!-- bilingual-en:start -->
> For any real matrix $A\in\mathbb R^{m\times n}$,
> $$N(A^TA)=N(A).$$
> Thus $\operatorname{rank}(A^TA)=\operatorname{rank}(A)$, and $A^TA$ is invertible if and only if $A$ has full column rank.
> <!-- bilingual-en:end -->

**第一方向。** 若 $Ax=0$，左乘 $A^T$ 得 $A^TAx=0$，所以 $N(A)\subseteq N(A^TA)$。
<!-- bilingual-en:start -->
**First direction.**If $Ax=0$, take $A^T$ on the left to get $A^TAx=0$, so $N(A)\subseteq N(A^TA)$.
<!-- bilingual-en:end -->

**反方向。** 若 $A^TAx=0$，左乘 $x^T$：
<!-- bilingual-en:start -->
**Reverse direction.** If $A^TAx=0$, multiply on the left by $x^T$:
<!-- bilingual-en:end -->

$$
0=x^TA^TAx=(Ax)^T(Ax)=\|Ax\|^2.
$$

平方范数只有在向量为零时才为零，故 $Ax=0$，于是 $N(A^TA)\subseteq N(A)$。两边合并即得结论。
<!-- bilingual-en:start -->
The squared norm is zero only for the zero vector, so $Ax=0$ and therefore $N(A^TA)\subseteq N(A)$. Combining the two inclusions proves the equality.
<!-- bilingual-en:end -->

因为 $A^TA$ 是 $n\times n$ 方阵，它可逆等价于 $N(A^TA)=\{0\}$；由上式又等价于 $N(A)=\{0\}$，即 $A$ 的 $n$ 个列向量线性无关。这里不要求 $A$ 是方阵，只要求列数不超过可独立的维数，即必有 $n\le m$。
<!-- bilingual-en:start -->
Because $A^TA$ is an $n\times n$ square matrix, it is invertible exactly when $N(A^TA)=\{0\}$. By the equality above, this is equivalent to $N(A)=\{0\}$, meaning that the $n$ columns of $A$ are linearly independent. The matrix $A$ need not be square, but full column rank requires $n\le m$.
<!-- bilingual-en:end -->

### Recitation：求 $S^\perp$ 并证明正交分解唯一
<!-- bilingual-en:start -->
*Recitation: Finding $S^\perp$ and Proving the Uniqueness of Orthogonal Decomposition*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
S=\operatorname{span}\left\{
\begin{bmatrix}1\\2\\2\\3\end{bmatrix},
\begin{bmatrix}1\\3\\3\\2\end{bmatrix}
\right\}\subseteq\mathbb R^4.
$$

$x\in S^\perp$ 当且仅当它同时与两个生成向量正交，所以
<!-- bilingual-en:start -->
$x\in S^\perp$ if and only if it is orthogonal to both generating vectors, so
<!-- bilingual-en:end -->

$$
\begin{bmatrix}
1&2&2&3\\
1&3&3&2
\end{bmatrix}x=0.
$$

第二行减第一行，取 $x_3=a,x_4=b$：
<!-- bilingual-en:start -->
Subtract row one from row two, then let $x_3=a$ and $x_4=b$:
<!-- bilingual-en:end -->

$$
x_2=-a+b,\qquad x_1=-5b.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
x=a\begin{bmatrix}0\\-1\\1\\0\end{bmatrix}
+b\begin{bmatrix}-5\\1\\0\\1\end{bmatrix},
$$

故括号中的两个向量构成 $S^\perp$ 的一组基。原来两个生成向量线性无关，$\dim S=2$；现在 $\dim S^\perp=2$，两组基合起来给出 $\mathbb R^4$ 的四个线性无关向量，所以每个 $v\in\mathbb R^4$ 都有唯一的 $v=s+s_\perp$ 分解。
<!-- bilingual-en:start -->
The two vectors in parentheses therefore form a basis for $S^\perp$. The original two generators are linearly independent, so $\dim S=2$; likewise $\dim S^\perp=2$. Combining the two bases gives four linearly independent vectors in $\mathbb R^4$, so every $v\in\mathbb R^4$ has a unique decomposition $v=s+s_\perp$.
<!-- bilingual-en:end -->

### Homework

> [!question]- Problem 16.1：用方程的线性组合得到矛盾 $0=1$
> 方程为 $x_1-x_2=1$、$x_2-x_3=1$、$x_1-x_3=1$。求 $y_1,y_2,y_3$，使三式加权后左边为 $0$、右边为 $1$。
>
> **解。** 取 $y=(1,1,-1)^T$。左边为
> $$
> (x_1-x_2)+(x_2-x_3)-(x_1-x_3)=0,
> $$
> 右边为 $1+1-1=1$。矩阵语言中，这说明 $A^Ty=0$ 而 $y^Tb=1$。若 $Ax=b$ 有解，则应有 $y^TAx=(A^Ty)^Tx=0$，但 $y^Tb=1$，矛盾；因此原系统无解。这正是 Fredholm alternative 在有限维情形的证书。
> <!-- bilingual-en:start -->
> The equations are $x_1-x_2=1$, $x_2-x_3=1$, and $x_1-x_3=1$. Find weights $y_1,y_2,y_3$ such that the weighted left-hand sides sum to $0$ while the weighted right-hand sides sum to $1$.
> **Solution.** Take $y=(1,1,-1)^T$. The weighted left-hand side is zero, while the right-hand side is $1+1-1=1$. In matrix language, $A^Ty=0$ but $y^Tb=1$. If $Ax=b$ had a solution, then $y^TAx=(A^Ty)^Tx=0$ would have to equal $y^Tb=1$, a contradiction. Thus the system is inconsistent. This is the finite-dimensional Fredholm alternative expressed as an inconsistency certificate.
> <!-- bilingual-en:end -->

> [!question]- Problem 16.2：给定四条一维子空间，构造秩一矩阵
> 给定非零 $r,n,c,\ell\in\mathbb R^2$，希望它们分别成为 $C(A^T),N(A),C(A),N(A^T)$ 的基。条件是什么？给出一个 $A$。
>
> **解。** 必要条件是
> $$r^Tn=0,\qquad c^T\ell=0.$$
> 四个向量都非零，故四个空间都为一维，维数条件自动满足。取
> $$A=cr^T.$$
> $A$ 的每一列都是 $c$ 的倍数，所以 $C(A)=\operatorname{span}(c)$；每一行都是 $r^T$ 的倍数，所以 $C(A^T)=\operatorname{span}(r)$。又因零空间是对应行/列空间的正交补，便得到指定的 $n$ 与 $\ell$。任意非零标量倍 $\alpha cr^T$ 也可行。
> <!-- bilingual-en:start -->
> Given nonzero vectors $r,n,c,\ell\in\mathbb R^2$, under what conditions can they serve as bases for $C(A^T),N(A),C(A),N(A^T)$, respectively? Construct such a matrix $A$ when the conditions hold.
> **Solution.** The necessary conditions are
> $$r^Tn=0,\qquad c^T\ell=0.$$
> The four vectors are all non-zero, so the four spaces are all one-dimensional, and the dimensionality condition is automatically satisfied.
> $$A=cr^T.$$
> Every column of $A$ is a multiple of $c$, so $C(A)=\operatorname{span}(c)$; every row is a multiple of $r^T$, so $C(A^T)=\operatorname{span}(r)$. Since each nullspace is the orthogonal complement of the corresponding row or column space, the prescribed vectors $n$ and $\ell$ span them. Any nonzero scalar multiple $\alpha cr^T$ also works.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- $S\perp T$ 强于 $S\cap T=\{0\}$；两条不垂直但不同的直线交集也是 $\{0\}$。
- 不可把 $C(A)\subseteq\mathbb R^m$ 与 $N(A)\subseteq\mathbb R^n$ 直接称为正交，除非 $m=n$ 且另有说明。
- $A^TA$ 总是对称，但未必可逆；公式 $(A^TA)^{-1}$ 只有在 $A$ **满列秩**时才存在。
- 从 $x^TA^TAx=0$ 推出 $Ax=0$ 用到了实数欧氏内积的正定性；复数情形应写 $x^*A^*Ax=\|Ax\|^2$。
<!-- bilingual-en:start -->
- $S\perp T$ is stronger than $S\cap T=\{0\}$; two distinct, nonorthogonal lines through the origin can also intersect only at $\{0\}$.
- $C(A)\subseteq\mathbb R^m$ and $N(A)\subseteq\mathbb R^n$ cannot be called orthogonal to one another unless they lie in the same ambient space, which in particular requires $m=n$.
- $A^TA$ is always symmetric, but it need not be invertible; the formula $(A^TA)^{-1}$ exists only when $A$ has full column rank.
- The positive definiteness of the real Euclidean inner product is used to infer $Ax=0$ from $x^TA^TAx=0$; over $\mathbb C$, write $x^*A^*Ax=\|Ax\|^2$.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若 $A\in\mathbb R^{7\times4}$ 且 $\operatorname{rank}(A)=3$，四个基本子空间的维数分别是多少？
> $\dim C(A)=\dim C(A^T)=3$，$\dim N(A)=4-3=1$，$\dim N(A^T)=7-3=4$。
> <!-- bilingual-en:start -->
> $\dim C(A)=\dim C(A^T)=3$,$\dim N(A)=4-3=1$,$\dim N(A^T)=7-3=4$.
> <!-- bilingual-en:end -->

> [!question]- 2. 证明 $C(A^T)\cap N(A)=\{0\}$。
> 设 $z$ 同时属于两空间。因为二者正交，$z^Tz=0$，所以 $z=0$。
> <!-- bilingual-en:start -->
> Let $z$ belong to both spaces.  Because they're orthogonal, $z^Tz=0$, $z=0$.
> <!-- bilingual-en:end -->

> [!question]- 3. 若 $A\in\mathbb R^{3\times5}$，$A^TA$ 能否可逆？
> 不能。五个列向量位于 $\mathbb R^3$，不可能线性无关，所以 $N(A)\ne\{0\}$；由 $N(A^TA)=N(A)$，$A^TA$ 奇异。
> <!-- bilingual-en:start -->
> No.  The five column vectors are located at $\mathbb R^3$ and cannot be linearly independent, so $N(A)\ne\{0\}$; by $N(A^TA)=N(A)$, $A^TA$ is singular.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

[[线性方程组与四个基本子空间#四个基本子空间|列空间]]与[[线性方程组与四个基本子空间#四个基本子空间|零空间]] → 正交补 → $N(A^TA)=N(A)$ → 下一节的[[正交投影与最小二乘#投影矩阵|正交投影]]。
<!-- bilingual-en:start -->
[[线性方程组与四个基本子空间#四个基本子空间|column space]] and [[线性方程组与四个基本子空间#四个基本子空间|null space]]→orthogonal complement→$N(A^TA)=N(A)$→[[正交投影与最小二乘#投影矩阵|orthogonal projection]] in the next section.
<!-- bilingual-en:end -->

---

## Session 2.2 Projections onto subspaces

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

给定 $b\in\mathbb R^m$ 和子空间 $S\subseteq\mathbb R^m$，怎样严格找出 $S$ 中离 $b$ 最近的向量？若 $S=C(A)$ 且 $A\in\mathbb R^{m\times n}$ 满列秩，投影系数 $\hat x\in\mathbb R^n$，投影 $p=A\hat x\in\mathbb R^m$，残差 $e=b-p\in\mathbb R^m$。
<!-- bilingual-en:start -->
Given $b\in\mathbb R^m$ and a subspace $S\subseteq\mathbb R^m$, how can we find the vector in $S$ closest to $b$ exactly? If $S=C(A)$ and $A\in\mathbb R^{m\times n}$ has full column rank, then the projection coefficients satisfy $\hat x\in\mathbb R^n$, the projection is $p=A\hat x\in\mathbb R^m$, and the residual is $e=b-p\in\mathbb R^m$.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S02_Lecture_Lecture_15_Projections_onto_Subspaces.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S02_Recitation_Problem_Solving_Projection_onto_Subspaces.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.2prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.2sol.pdf#page=1)

### Lecture：最近点由“残差正交”唯一决定
<!-- bilingual-en:start -->
*Lecture: Orthogonality of the residual uniquely determines the nearest point*
<!-- bilingual-en:end -->

先投影到直线 $S=\operatorname{span}(a)$，其中 $a\ne0$。因为 $p\in S$，写成 $p=\hat xa$。正交投影要求误差 $e=b-p$ 与 $a$ 垂直：
<!-- bilingual-en:start -->
First project onto the line $S=\operatorname{span}(a)$, where $a\ne0$. Since $p\in S$, write $p=\hat xa$. Orthogonal projection requires the residual $e=b-p$ to be perpendicular to $a$:
<!-- bilingual-en:end -->

$$
a^T(b-\hat xa)=0.
$$

解得
<!-- bilingual-en:start -->
Solving gives
<!-- bilingual-en:end -->

$$
\hat x=\frac{a^Tb}{a^Ta},\qquad
p=a\frac{a^Tb}{a^Ta}.
$$

于是投影矩阵为
<!-- bilingual-en:start -->
So the projection matrix is
<!-- bilingual-en:end -->

$$
P=\frac{aa^T}{a^Ta}\in\mathbb R^{m\times m},\qquad p=Pb.
$$

分母 $a^Ta$ 是标量，分子 $aa^T$ 是 $m\times m$ 矩阵；把两者误写成 $a^Ta/(aa^T)$ 是尺寸错误。
<!-- bilingual-en:start -->
The denominator $a^Ta$ is a scalar, whereas the numerator $aa^T$ is an $m\times m$ matrix. Reversing them to write $a^Ta/(aa^T)$ is dimensionally invalid.
<!-- bilingual-en:end -->

#### 为什么正交条件保证最近
<!-- bilingual-en:start -->
*Why Orthogonal Conditions Guarantee Nearest*
<!-- bilingual-en:end -->

设 $p$ 满足 $p\in S$ 且 $e=b-p\perp S$。对任意 $s\in S$，
<!-- bilingual-en:start -->
Let $p$ satisfy $p\in S$ and $e=b-p\perp S$.  For any $s\in S$,
<!-- bilingual-en:end -->

$$
b-s=(b-p)+(p-s)=e+(p-s).
$$

因为 $p-s\in S$ 而 $e\perp S$，勾股定理给出
<!-- bilingual-en:start -->
Because $p-s\in S$ and $e\perp S$, the Pythagorean theorem
<!-- bilingual-en:end -->

$$
\|b-s\|^2=\|e\|^2+\|p-s\|^2\ge\|e\|^2=\|b-p\|^2.
$$

等号要求 $p-s=0$，即 $s=p$。因此 $p$ 不只是局部极小，而是唯一的全局最近点。
<!-- bilingual-en:start -->
The equal sign requires $p-s=0$, or $s=p$.  Therefore, $p$ is not only a local minimum, but also a unique global closest point.
<!-- bilingual-en:end -->

#### 投影到列空间
<!-- bilingual-en:start -->
*Projection into Column Space*
<!-- bilingual-en:end -->

令 $S=C(A)$，所以 $p=A\hat x$。要求残差与 $A$ 的每一列正交：
<!-- bilingual-en:start -->
$S=C(A)$, so $p=A\hat x$.  Require the residual to be orthogonal to each column of $A$:
<!-- bilingual-en:end -->

$$
A^T(b-A\hat x)=0.
$$

整理得**正规方程（normal equations）**
<!-- bilingual-en:start -->
Rearranging gives the **normal equations**
<!-- bilingual-en:end -->

$$
A^TA\hat x=A^Tb.
$$

若 $A$ 满列秩，$A^TA$ 可逆，于是
<!-- bilingual-en:start -->
If $A$ has full column rank, then $A^TA$ is invertible, and
<!-- bilingual-en:end -->

$$
\hat x=(A^TA)^{-1}A^Tb,\qquad
p=A(A^TA)^{-1}A^Tb.
$$

因此投影到 $C(A)$ 的[[正交投影与最小二乘#投影矩阵|投影矩阵（projection matrix）]]是
<!-- bilingual-en:start -->
So the [[正交投影与最小二乘#投影矩阵|projection matrix]] projected onto the $C(A)$ is
<!-- bilingual-en:end -->

$$
P=A(A^TA)^{-1}A^T.
$$

逐项检查尺寸：$A^T b\in\mathbb R^n$，$(A^TA)^{-1}\in\mathbb R^{n\times n}$，最终 $Pb\in\mathbb R^m$。
<!-- bilingual-en:start -->
Check the dimensions one by one: $A^T b\in\mathbb R^n$, $(A^TA)^{-1}\in\mathbb R^{n\times n}$, and finally $Pb\in\mathbb R^m$.
<!-- bilingual-en:end -->

#### 投影矩阵的结构
<!-- bilingual-en:start -->
*The Structure of Projection Matrix*
<!-- bilingual-en:end -->

在满列秩条件下，
<!-- bilingual-en:start -->
Under the full-column-rank assumption,
<!-- bilingual-en:end -->

$$
P^T=P,
$$

因为 $A^TA$ 对称，其逆也对称；并且
<!-- bilingual-en:start -->
Because $A^TA$ is symmetric, its inverse is also symmetric; and
<!-- bilingual-en:end -->

$$
\begin{aligned}
P^2
&=A(A^TA)^{-1}A^TA(A^TA)^{-1}A^T\\
&=A(A^TA)^{-1}A^T=P.
\end{aligned}
$$

所以正交投影矩阵同时满足**对称**与**幂等**。还有
<!-- bilingual-en:start -->
So the orthogonal projection matrices satisfy**symmetry**and**idempotence**.  and
<!-- bilingual-en:end -->

$$
C(P)=C(A),\qquad N(P)=N(A^T).
$$

第一式因为 $Pb$ 总在 $C(A)$ 中，且对 $p\in C(A)$ 有 $Pp=p$；第二式因为 $Pb=0$ 正好表示 $b$ 完全位于 $C(A)^\perp=N(A^T)$。
<!-- bilingual-en:start -->
The first is because $Pb$ is always in $C(A)$ and has $Pp=p$ for $p\in C(A)$; the second is because $Pb=0$ just means $b$ is entirely in $C(A)^\perp=N(A^T)$.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-projection.png|760]]

### Recitation：投影到平面 $x+y-z=0$
<!-- bilingual-en:start -->
*Recitation:projecting to a plane $x+y-z=0$*
<!-- bilingual-en:end -->

平面的法向量是 $n=(1,1,-1)^T$。可以选平面基
<!-- bilingual-en:start -->
The normal vector of the plane is $n=(1,1,-1)^T$.  You can select a planar basis
<!-- bilingual-en:end -->

$$
a_1=(1,-1,0)^T,\qquad a_2=(1,0,1)^T,\qquad A=[a_1\ a_2],
$$

再代入 $A(A^TA)^{-1}A^T$。更短的办法是先投影到法线，再用互补投影：
<!-- bilingual-en:start -->
and then repopulate the $A(A^TA)^{-1}A^T$.  A shorter approach is to project to the normal first, and then project with complementary projections:
<!-- bilingual-en:end -->

$$
P_n=\frac{nn^T}{n^Tn}
=\frac13
\begin{bmatrix}
1&1&-1\\
1&1&-1\\
-1&-1&1
\end{bmatrix},
$$

$$
P_{\text{plane}}=I-P_n
=\frac13
\begin{bmatrix}
2&-1&1\\
-1&2&1\\
1&1&2
\end{bmatrix}.
$$

验算：$P_{\text{plane}}n=0$；$P_{\text{plane}}a_i=a_i$；矩阵对称且平方等于自身。无论选择平面的哪一组基，最后的投影矩阵都相同，因为最近点是唯一的。
<!-- bilingual-en:start -->
Check that $P_{\text{plane}}n=0$ and $P_{\text{plane}}a_i=a_i$, and that the matrix is symmetric and idempotent. The final projection matrix is independent of the basis chosen for the plane because the closest point is unique.
<!-- bilingual-en:end -->

### Homework

> [!question]- Problem 15.1：投影到 $\mathbb R^4$ 的前三个坐标方向
> $A$ 是 $4\times4$ 单位矩阵删去最后一列所得的 $4\times3$ 矩阵。把 $b=(1,2,3,4)^T$ 投影到 $C(A)$，并写出 $P$ 的尺寸与数值。
>
> **解。** $C(A)=\{(x_1,x_2,x_3,0)^T\}$。输入与输出都在 $\mathbb R^4$，所以 $P\in\mathbb R^{4\times4}$：
> $$
> P=\begin{bmatrix}
> 1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&0
> \end{bmatrix},\qquad
> p=Pb=\begin{bmatrix}1\\2\\3\\0\end{bmatrix}.
> $$
> 残差 $(0,0,0,4)^T$ 与前三个坐标方向正交。
> <!-- bilingual-en:start -->
> $A$ is the $4\times3$ matrix obtained by deleting the last column from the $4\times4$ identity matrix.  Projection $b=(1,2,3,4)^T$ to $C(A)$, and write out the size and value of $P$.
> **Solution.**$C(A)=\{(x_1,x_2,x_3,0)^T\}$.  Input and output are both at $\mathbb R^4$, so $P\in\mathbb R^{4\times4}$:
> The residual $(0,0,0,4)^T$ is orthogonal to the first three coordinate directions.
> <!-- bilingual-en:end -->

> [!question]- Problem 15.2：证明互补投影仍然幂等
> 已知 $P^2=P$，证明 $(I-P)^2=I-P$；说明上一题的 $I-P$ 投影到哪里。
>
> **解。** 因 $IP=PI=P$，
> $$
> (I-P)^2=I-IP-PI+P^2=I-P-P+P=I-P.
> $$
> 上题中
> $$
> I-P=\operatorname{diag}(0,0,0,1),
> $$
> 它投影到 $C(A)^\perp=N(A^T)=\operatorname{span}(e_4)$，即 $A$ 的左零空间。
> <!-- bilingual-en:start -->
> Known $P^2=P$, certifies $(I-P)^2=I-P$; explains where the $I-P$ from the previous question is projected.
> **Solution.**Because of $IP=PI=P$,
> above question
> It projects to $C(A)^\perp=N(A^T)=\operatorname{span}(e_4)$, the left nullspace of $A$.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- $(A^TA)^{-1}$ 只在 $A$ 满列秩时存在；列相关时应删去冗余列、换一组基，或在 Unit III 使用伪逆。
- $P^2=P$ 只说明“投影”意义上的幂等；若还要是**正交**投影，必须有 $P^T=P$。斜投影可以幂等但不对称。
- 一般不能把 $(A^TA)^{-1}$ 拆为 $A^{-1}(A^T)^{-1}$，因为长方矩阵 $A$ 没有双侧逆。
- 投影到同一个子空间的矩阵与所选基无关；系数 $\hat x$ 会随基改变，几何向量 $p$ 不变。
<!-- bilingual-en:start -->
- $(A^TA)^{-1}$ exists only when $A$ has full column rank. If the columns are dependent, remove redundant columns, choose a basis for the same column space, or use the pseudoinverse introduced in Unit III.
- $P^2=P$ only states the idempotence in the sense of "projection"; in addition to**orthogonal**projection, $P^T=P$ is required.  Oblique projections can be idempotent but asymmetric.
- In general, $(A^TA)^{-1}$ cannot be rewritten as $A^{-1}(A^T)^{-1}$ because a rectangular matrix $A$ has no two-sided inverse.
- The matrix projected onto the same subspace is independent of the basis selected; the coefficient $\hat x$ changes with the basis and the geometric vector $p$ remains unchanged.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 把 $b=(3,1)^T$ 投影到 $a=(1,2)^T$ 张成的直线。
> $a^Tb=5,a^Ta=5$，故 $\hat x=1$，$p=(1,2)^T$，$e=(2,-1)^T$，并且 $a^Te=0$。
> <!-- bilingual-en:start -->
> $a^Tb=5,a^Ta=5$, hence $\hat x=1$, $p=(1,2)^T$, $e=(2,-1)^T$, and $a^Te=0$.
> <!-- bilingual-en:end -->

> [!question]- 2. 若 $P=P^T=P^2$，证明 $b=Pb+(I-P)b$ 是正交分解。
> 两项之和是 $b$；内积为
> $$
> (Pb)^T(I-P)b=b^TP(I-P)b=b^T(P-P^2)b=0.
> $$
> <!-- bilingual-en:start -->
> The sum of the two terms is $b$; the inner product is
> <!-- bilingual-en:end -->

> [!question]- 3. $A=[a\ 2a]$ 能否直接使用 $A(A^TA)^{-1}A^T$？怎样修复？
> 不能，两列相关使 $A^TA$ 奇异。删去第二列，用非零列 $a$ 作为同一列空间的一组基，再用 $aa^T/(a^Ta)$。
> <!-- bilingual-en:start -->
> No. The two columns are dependent, so $A^TA$ is singular. Delete the second column, use the nonzero column $a$ as a basis for the same column space, and then use $aa^T/(a^Ta)$.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

正交补 → [[正交投影与最小二乘#投影矩阵|正交投影]] → 正规方程 → 下一节的[[正交投影与最小二乘#最小二乘与正规方程|最小二乘]]与残差分析。
<!-- bilingual-en:start -->
Orthogonal complement→[[正交投影与最小二乘#投影矩阵|orthogonal projection]]→Normal equation→[[正交投影与最小二乘#最小二乘与正规方程|least squares]] and residual analysis in the next section.
<!-- bilingual-en:end -->

---

## Session 2.3 Projection matrices and least squares

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

当 $Ax=b$ 因 $b\notin C(A)$ 而无解时，我们不伪造精确解，而是在所有 $Ax$ 中寻找离 $b$ 最近的一个。本节把这个几何问题写成[[正交投影与最小二乘#最小二乘与正规方程|最小二乘（least squares）]]，并解释正规方程的来源、解的唯一性条件和残差所属的子空间。
<!-- bilingual-en:start -->
When $Ax=b$ is inconsistent because $b\notin C(A)$, we do not invent an exact solution. Instead, among all vectors $Ax$, we find the one closest to $b$. This section formulates that geometric problem as [[正交投影与最小二乘#最小二乘与正规方程|least squares]] and explains where the normal equations come from, when the coefficient vector is unique, and which subspace contains the residual.
<!-- bilingual-en:end -->

仍设 $A\in\mathbb R^{m\times n}$、$b\in\mathbb R^m$。若 $A$ 满列秩，$\hat x\in\mathbb R^n$ 唯一；无论坐标是否唯一，最佳拟合向量 $p=A\hat x\in C(A)$ 都是唯一的正交投影。
<!-- bilingual-en:start -->
Let $A\in\mathbb R^{m\times n}$ and $b\in\mathbb R^m$. If $A$ has full column rank, then $\hat x\in\mathbb R^n$ is unique. Whether or not the coefficient vector is unique, however, the best-fit vector $p=A\hat x\in C(A)$ is always the unique orthogonal projection.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.3sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S03_Lecture_Lecture_16_Projection_Matrices_and_Least_Squares.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S03_Recitation_Problem_Solving_Least_Squares_Approximation.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.3prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.3sol.pdf#page=1)

### Lecture：把“拟合”翻译成列空间投影
<!-- bilingual-en:start -->
*Lecture: Translating Fit into Column Space Projection*
<!-- bilingual-en:end -->

最小二乘问题是
<!-- bilingual-en:start -->
least squares problem
<!-- bilingual-en:end -->

$$
\min_{x\in\mathbb R^n}\|Ax-b\|^2.
$$

$Ax$ 只能在 $C(A)$ 中移动，因此问题等价于：求 $b$ 在 $C(A)$ 上的投影 $p$，再解 $A\hat x=p$。最佳残差
<!-- bilingual-en:start -->
The vector $Ax$ can vary only within $C(A)$, so the problem is equivalent to projecting $b$ onto $C(A)$ to obtain $p$, then solving $A\hat x=p$. The optimal residual
<!-- bilingual-en:end -->

$$
e=b-A\hat x
$$

必须属于 $C(A)^\perp=N(A^T)$，所以
<!-- bilingual-en:start -->
must belong to $C(A)^\perp=N(A^T)$, so
<!-- bilingual-en:end -->

$$
A^Te=0
\quad\Longleftrightarrow\quad
A^T(b-A\hat x)=0
\quad\Longleftrightarrow\quad
A^TA\hat x=A^Tb.
$$

这就是[[正交投影与最小二乘#最小二乘与正规方程|用正规方程求最小二乘解]]。它不是对 $Ax=b$ 随意左乘 $A^T$，而是把“残差与全部列正交”同时写成一组方程。
<!-- bilingual-en:start -->
These are the [[正交投影与最小二乘#最小二乘与正规方程|normal equations for the least-squares solution]]. They do not arise from arbitrarily multiplying $Ax=b$ by $A^T$; they encode in one system the condition that the residual be orthogonal to every column of $A$.
<!-- bilingual-en:end -->

若希望用微积分核对，令
<!-- bilingual-en:start -->
If you want to use calculus to check,
<!-- bilingual-en:end -->

$$
F(x)=\|Ax-b\|^2=(Ax-b)^T(Ax-b).
$$

展开为
<!-- bilingual-en:start -->
Expand As
<!-- bilingual-en:end -->

$$
F(x)=x^TA^TAx-2x^TA^Tb+b^Tb.
$$

因为 $A^TA$ 对称，
<!-- bilingual-en:start -->
because $A^TA$ is symmetric,
<!-- bilingual-en:end -->

$$
\nabla F(x)=2A^TAx-2A^Tb.
$$

令梯度为零同样得到正规方程。几何证明更强地说明这是全局最小：对任意 $x$，写
<!-- bilingual-en:start -->
Setting the gradient to zero gives the same normal equations. The geometric proof additionally shows that the solution is a global minimum: for any $x$, write
<!-- bilingual-en:end -->

$$
b-Ax=(b-p)+(p-Ax)=e+A(\hat x-x),
$$

两项正交，因而
<!-- bilingual-en:start -->
Two Orthogonal
<!-- bilingual-en:end -->

$$
\|b-Ax\|^2=\|e\|^2+\|A(\hat x-x)\|^2\ge\|e\|^2.
$$

#### 唯一性究竟在哪里
<!-- bilingual-en:start -->
*Where the Uniqueness Is*
<!-- bilingual-en:end -->

- 投影 $p$ 总是唯一，因为子空间中的最近点唯一。
- 若 $A$ 满列秩，则 $A^TA$ 正定且可逆，$\hat x$ 唯一：
  $$
  \hat x=(A^TA)^{-1}A^Tb.
  $$
- 若 $A$ 列相关，则可能有多个 $x$ 给出同一 $p$；任何两个最小二乘解之差都在 $N(A)$ 中。此时不能写 $(A^TA)^{-1}$。
<!-- bilingual-en:start -->
- The projection $p$ is always unique because the closest point in the subspace is unique.
- If $A$ has full column rank, then $A^TA$ is positive definite and invertible, so $\hat x$ is unique.
- If the columns of $A$ are linearly dependent, multiple vectors $x$ may produce the same projection $p$. The difference between any two least-squares solutions lies in $N(A)$, and $(A^TA)^{-1}$ does not exist.
<!-- bilingual-en:end -->

投影矩阵与互补投影为
<!-- bilingual-en:start -->
Projection Matrix and Complementary Projection as
<!-- bilingual-en:end -->

$$
P=A(A^TA)^{-1}A^T,\qquad I-P,
$$

并且
<!-- bilingual-en:start -->
and
<!-- bilingual-en:end -->

$$
p=Pb\in C(A),\qquad e=(I-P)b\in N(A^T),\qquad p^Te=0.
$$

### 课堂例题：三点的最佳拟合直线
<!-- bilingual-en:start -->
*Classroom Example: Best Fit Line of Three Points*
<!-- bilingual-en:end -->

拟合 $(1,1),(2,2),(3,2)$，设直线 $y=C+Dt$。则
<!-- bilingual-en:start -->
Fit $(1,1),(2,2),(3,2)$ and set the line $y=C+Dt$. Then
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\quad
x=\begin{bmatrix}C\\D\end{bmatrix},\quad
b=\begin{bmatrix}1\\2\\2\end{bmatrix}.
$$

原系统三式两未知，一般不相容。正规方程是
<!-- bilingual-en:start -->
The original system is unknown and generally incompatible.  The normal equation is
<!-- bilingual-en:end -->

$$
\underbrace{\begin{bmatrix}3&6\\6&14\end{bmatrix}}_{A^TA}
\begin{bmatrix}\hat C\\\hat D\end{bmatrix}
=
\underbrace{\begin{bmatrix}5\\11\end{bmatrix}}_{A^Tb}.
$$

第一式乘 $2$ 后从第二式相减：$2\hat D=1$，所以 $\hat D=1/2$；代回得 $\hat C=2/3$。于是
<!-- bilingual-en:start -->
The first formula is multiplied by $2$ and subtracted from the second formula: $2\hat D=1$, so $\hat D=1/2$; $\hat C=2/3$ is substituted.  therefore
<!-- bilingual-en:end -->

$$
p=A\hat x=
\begin{bmatrix}7/6\\5/3\\13/6\end{bmatrix},\qquad
e=b-p=
\begin{bmatrix}-1/6\\1/3\\-1/6\end{bmatrix}.
$$

直接验算
<!-- bilingual-en:start -->
direct checking
<!-- bilingual-en:end -->

$$
A^Te=
\begin{bmatrix}
-1/6+1/3-1/6\\
-1/6+2/3-3/6
\end{bmatrix}=0.
$$

第一行说明残差总和为零，第二行说明“时间加权残差”也为零；这两个条件分别来自常数列和时间列。
<!-- bilingual-en:start -->
The first line indicates that the sum of the residuals is zero, and the second line indicates that the time-weighted residuals are also zero; the two conditions are from the constant column and the time column, respectively.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-least-squares-fit.png|780]]

### Recitation：过原点的最佳二次曲线，并修正转录算术
<!-- bilingual-en:start -->
*Recitation: Optimal conic over origin and modified transcription arithmetic*
<!-- bilingual-en:end -->

对点 $(1,1),(2,5),(-1,-2)$ 拟合 $y=ct+dt^2$。设计矩阵为
<!-- bilingual-en:start -->
Fit $y=ct+dt^2$ to the points $(1,1),(2,5),(-1,-2)$. The design matrix is
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}1&1\\2&4\\-1&1\end{bmatrix},\qquad
b=\begin{bmatrix}1\\5\\-2\end{bmatrix}.
$$

按定义逐项计算：
<!-- bilingual-en:start -->
Calculate by definition:
<!-- bilingual-en:end -->

$$
A^TA=\begin{bmatrix}6&8\\8&18\end{bmatrix},\qquad
A^Tb=\begin{bmatrix}13\\19\end{bmatrix}.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\begin{bmatrix}6&8\\8&18\end{bmatrix}
\begin{bmatrix}\hat c\\\hat d\end{bmatrix}
=\begin{bmatrix}13\\19\end{bmatrix}.
$$

消元得 $22\hat d=5$，故 $\hat d=5/22$；再由 $6\hat c+8\hat d=13$ 得 $\hat c=41/22$。最佳拟合为
<!-- bilingual-en:start -->
Elimination gives $22\hat d=5$, so $\hat d=5/22$. Substituting into $6\hat c+8\hat d=13$ gives $\hat c=41/22$. The best fit is
<!-- bilingual-en:end -->

$$
y=\frac{41}{22}t+\frac5{22}t^2.
$$

> [!warning] 原 transcript 的算术错误
> 转录稿在这里把 $(A^TA)_{22}=1^2+4^2+1^2$ 记成了 $10$，并给出不满足正规方程的 $c=11/2,d=-5/2$。按题面数据正确值必须是 $18$，上面的解可由 $A^T(b-A\hat x)=0$ 直接验算。笔记保留课程思路，但不沿用这一算术错误。
> <!-- bilingual-en:start -->
> The transcript records $(A^TA)_{22}=1^2+4^2+1^2$ as $10$ and gives $c=11/2,d=-5/2$, which does not satisfy the normal equations. For the stated data, the correct entry is $18$. The solution above is verified directly by $A^T(b-A\hat x)=0$. These notes preserve the course's method but correct this arithmetic error.
> <!-- bilingual-en:end -->

### Homework

以下六题共用
<!-- bilingual-en:start -->
The following six problems use
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}1&-1\\1&1\\1&2\end{bmatrix}
$$

（最后一题另建五行设计矩阵）。
<!-- bilingual-en:start -->
(In the last question, a five-line design matrix will be created.)
<!-- bilingual-en:end -->

> [!question]- Problem 16.1：拟合 $(-1,7),(1,7),(2,21)$
> 写出三式并求 $\hat x=(C,D)^T$。
>
> **解。** 三式为 $C-D=7,C+D=7,C+2D=21$。有
> $$
> A^TA=\begin{bmatrix}3&2\\2&6\end{bmatrix},\qquad
> A^Tb=\begin{bmatrix}35\\42\end{bmatrix}.
> $$
> 解 $3C+2D=35,2C+6D=42$，得
> $$\boxed{C=9,\quad D=4},$$
> 即最佳直线 $b=9+4t$。
> <!-- bilingual-en:start -->
> Write the three equations and solve for $\hat x=(C,D)^T$.
> **Solution.** The equations are $C-D=7$, $C+D=7$, and $C+2D=21$. Solving the normal equations $3C+2D=35$ and $2C+6D=42$ gives
> $$\boxed{C=9,\quad D=4},$$
> That's the best line, $b=9+4t$.
> <!-- bilingual-en:end -->

> [!question]- Problem 16.2：求投影与残差，并解释 $Pe=0$
> **解。**
> $$
> p=A\hat x=\begin{bmatrix}5\\13\\17\end{bmatrix},\qquad
> e=b-p=\begin{bmatrix}2\\-6\\4\end{bmatrix}.
> $$
> 验算 $A^Te=(2-6+4,-2-6+8)^T=0$，故 $e\in N(A^T)=C(A)^\perp$。投影到 $C(A)$ 后为零，即 $Pe=0$；也可写成 $Pe=P(b-p)=Pb-Pp=p-p=0$。
> <!-- bilingual-en:start -->
> **Solution.**
> $A^Te=(2-6+4,-2-6+8)^T=0$, so $e\in N(A^T)=C(A)^\perp$.  It is zero after projection into $C(A)$, which is $Pe=0$; it can be written as $Pe=P(b-p)=Pb-Pp=p-p=0$.
> <!-- bilingual-en:end -->

> [!question]- Problem 16.3：把上一题的误差本身当成新数据
> **解。** 新 $b=e=(2,-6,4)^T$ 满足 $A^Tb=0$，正规方程右端为零。因 $A$ 满列秩，唯一解为 $\hat x=0$，最近直线是零函数，$p=0$。原因是 $b$ 已经垂直于整个 $C(A)$。
> <!-- bilingual-en:start -->
> **Solution.** The new vector $b=e=(2,-6,4)^T$ satisfies $A^Tb=0$, so the right-hand side of the normal equations is zero. Since $A$ has full column rank, the unique solution is $\hat x=0$, and the closest line is the zero function, $p=0$. This occurs because $b$ is already orthogonal to all of $C(A)$.
> <!-- bilingual-en:end -->

> [!question]- Problem 16.4：把上一题的投影本身当成数据
> **解。** $b=(5,13,17)^T=A(9,4)^T\in C(A)$，所以精确可解：$\hat x=(9,4)^T$，$p=b$，$e=0$。
> <!-- bilingual-en:start -->
> **Solution.** Since $b=(5,13,17)^T=A(9,4)^T\in C(A)$, the system is exactly solvable: $\hat x=(9,4)^T$, $p=b$, and $e=0$.
> <!-- bilingual-en:end -->

> [!question]- Problem 16.5：$e,p,\hat x$ 分别属于哪个基本子空间？
> **解。** $e\in N(A^T)\subseteq\mathbb R^3$；$p\in C(A)\subseteq\mathbb R^3$；$\hat x\in\mathbb R^2=C(A^T)$，因为 $A$ 的秩为 $2$、行空间填满 $\mathbb R^2$。两列独立，故 $N(A)=\{0\}$。
> <!-- bilingual-en:start -->
> **Solution.** We have $e\in N(A^T)\subseteq\mathbb R^3$ and $p\in C(A)\subseteq\mathbb R^3$. Also $\hat x\in\mathbb R^2=C(A^T)$ because $A$ has rank $2$, so its row space fills $\mathbb R^2$. The two columns are linearly independent, hence $N(A)=\{0\}$.
> <!-- bilingual-en:end -->

> [!question]- Problem 16.6：拟合五个对称时刻的数据
> 在 $t=-2,-1,0,1,2$，数据为 $4,2,-1,0,0$。求最佳直线 $C+Dt$。
>
> **解。**
> $$
> A=\begin{bmatrix}1&-2\\1&-1\\1&0\\1&1\\1&2\end{bmatrix},\quad
> A^TA=\begin{bmatrix}5&0\\0&10\end{bmatrix},\quad
> A^Tb=\begin{bmatrix}5\\-10\end{bmatrix}.
> $$
> 对称时刻使 $\sum t_i=0$，所以正规方程解耦。得到 $C=1,D=-1$，最佳直线为
> $$\boxed{b=1-t}.$$
> 预测值为 $(3,2,1,0,-1)^T$，残差 $(1,0,-2,0,1)^T$；其总和和时间加权和都为零。
> <!-- bilingual-en:start -->
> In $t=-2,-1,0,1,2$, the data is $4,2,-1,0,0$.  Find the best line, $C+Dt$.
> **Solution.**
> The symmetric sampling times give $\sum t_i=0$, so the normal equations decouple. Solving gives $C=1,D=-1$, and the best-fit line is
> $$\boxed{b=1-t}.$$
> The predicted vector is $(3,2,1,0,-1)^T$, and the residual is $(1,0,-2,0,1)^T$; both its ordinary sum and its time-weighted sum are zero.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- “least squares” 最小化的是残差平方和 $\sum e_i^2$，不是 $\sum e_i$；后者可由正负抵消。
- 拟合直线的几何图在 $(t,b)$ 平面，而投影 $b\mapsto p$ 发生在数据空间 $\mathbb R^m$；两种图景不能混成同一个空间。
- 离群点会因平方而获得更大权重，普通最小二乘并不稳健。
- 正规方程会把条件数近似平方；数值计算常优先用 QR，而不是显式形成 $A^TA$。
<!-- bilingual-en:start -->
- “Least squares” minimizes the residual sum of squares $\sum e_i^2$, not $\sum e_i$; positive and negative residuals can cancel in the latter sum.
- The fitted line is drawn in the $(t,b)$ plane, whereas the projection $b\mapsto p$ takes place in the data space $\mathbb R^m$; these are two different geometric pictures.
- Squaring gives outliers greater weight, so ordinary least squares is not robust to them.
- Forming the normal equations roughly squares the condition number; numerical algorithms therefore often prefer QR to forming $A^TA$ explicitly.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若设计矩阵含常数列，为什么最小二乘残差的分量和为零？
> 常数列是 $\mathbf1$，正规方程中的正交条件给出 $\mathbf1^Te=\sum_i e_i=0$。
> <!-- bilingual-en:start -->
> The constant column is $\mathbf1$, and the orthogonality condition in the normal equation gives $\mathbf1^Te=\sum_i e_i=0$.
> <!-- bilingual-en:end -->

> [!question]- 2. 若 $b\in C(A)$ 且 $A$ 满列秩，最小二乘解是什么性质？
> 原方程精确可解且解唯一；$p=b,e=0$，最小残差为零。
> <!-- bilingual-en:start -->
> The original equation is exactly solvable and its solution is unique. Thus $p=b$, $e=0$, and the minimum residual norm is zero.
> <!-- bilingual-en:end -->

> [!question]- 3. 两个最小二乘解 $x_1,x_2$ 为什么可能不同却有相同预测？
> 因最佳投影唯一，所以 $Ax_1=Ax_2=p$，从而 $x_1-x_2\in N(A)$。只有 $N(A)=\{0\}$ 时系数才唯一。
> <!-- bilingual-en:start -->
> The best projection $p$ is unique, so $Ax_1=Ax_2=p$ and hence $x_1-x_2\in N(A)$. The coefficient vector is unique exactly when $N(A)=\{0\}$.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

投影 → 正规方程 → 残差 $N(A^T)$ → 下一节用[[正交投影与最小二乘#Gram–Schmidt 与 QR|Gram–Schmidt]]和 QR 更稳定地求投影系数。
<!-- bilingual-en:start -->
Projection → normal equations → residual in $N(A^T)$ → more stable computation of projection coefficients by [[正交投影与最小二乘#Gram–Schmidt 与 QR|Gram–Schmidt]] and QR in the next section.
<!-- bilingual-en:end -->

---

## Session 2.4 Orthogonal matrices and Gram–Schmidt

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

怎样把任意一组线性无关向量变成张成同一子空间的标准正交基？为什么在标准正交坐标中，投影、最小二乘和坐标恢复都会变简单？
<!-- bilingual-en:start -->
How can any linearly independent set be converted into an orthonormal basis for the same subspace? Why do projection, least squares, and coordinate recovery become simpler in orthonormal coordinates?
<!-- bilingual-en:end -->

设 $A=[a_1\ \cdots\ a_n]\in\mathbb R^{m\times n}$ 且列独立（$m\ge n$）。Gram–Schmidt 产生 $Q=[q_1\ \cdots\ q_n]\in\mathbb R^{m\times n}$，满足 $Q^TQ=I_n$，以及上三角 $R\in\mathbb R^{n\times n}$，使 $A=QR$。
<!-- bilingual-en:start -->
Let $A=[a_1\ \cdots\ a_n]\in\mathbb R^{m\times n}$ have linearly independent columns, so $m\ge n$. Gram–Schmidt produces $Q=[q_1\ \cdots\ q_n]\in\mathbb R^{m\times n}$ with $Q^TQ=I_n$ and an upper-triangular matrix $R\in\mathbb R^{n\times n}$ such that $A=QR$.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S04_Lecture_Lecture_17_Orthogonal_Matrices_and_Gram_Schmidt.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S04_Recitation_Problem_Solving_Gram_Schmidt_Orthogonalization.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.4prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.4sol.pdf#page=1)

### Lecture：标准正交列为何特别方便
<!-- bilingual-en:start -->
*Lecture: Why orthonormal columns are so convenient*
<!-- bilingual-en:end -->

向量 $q_1,\dots,q_n$ **标准正交（orthonormal）**，是指
<!-- bilingual-en:start -->
Vector $q_1,\dots,q_n$**orthonormal**means
<!-- bilingual-en:end -->

$$
q_i^Tq_j=\delta_{ij}=
\begin{cases}1,&i=j,\\0,&i\ne j.\end{cases}
$$

把它们作为列组成 $Q$，上式一次写成
<!-- bilingual-en:start -->
Placing these vectors as the columns of $Q$ expresses all of the orthonormality conditions at once:
<!-- bilingual-en:end -->

$$
Q^TQ=I_n.
$$

若 $Q$ 是方阵，则称为[[对称矩阵与正定二次型#对称矩阵与谱定理|正交矩阵（orthogonal matrix）]]，且
<!-- bilingual-en:start -->
If $Q$ is a square matrix, it is called [[对称矩阵与正定二次型#对称矩阵与谱定理|orthogonal matrix]], and
<!-- bilingual-en:end -->

$$
Q^{-1}=Q^T,\qquad QQ^T=Q^TQ=I.
$$

若 $Q$ 是高矩阵（$m>n$），只有 $Q^TQ=I_n$；此时
<!-- bilingual-en:start -->
If $Q$ is tall ($m>n$), only $Q^TQ=I_n$ holds; in that case,
<!-- bilingual-en:end -->

$$
QQ^T=P_{C(Q)}\ne I_m
$$

一般是秩为 $n$ 的投影矩阵。
<!-- bilingual-en:start -->
Generally, it is a projection matrix with rank $n$.
<!-- bilingual-en:end -->

正交矩阵保持内积和长度：
<!-- bilingual-en:start -->
An orthogonal matrix preserves inner products and lengths:
<!-- bilingual-en:end -->

$$
(Qx)^T(Qy)=x^TQ^TQy=x^Ty,\qquad \|Qx\|=\|x\|.
$$

因此它只执行旋转、反射或这些变换的组合，不拉伸长度。
<!-- bilingual-en:start -->
So it performs only rotation, reflection, or a combination of these transformations, without stretching the length.
<!-- bilingual-en:end -->

#### 标准正交向量必线性无关
<!-- bilingual-en:start -->
*Orthonormal vectors are linearly independent*
<!-- bilingual-en:end -->

若 $Qx=0$，左乘 $Q^T$：
<!-- bilingual-en:start -->
If $Qx=0$, multiply on the left by $Q^T$:
<!-- bilingual-en:end -->

$$
Q^TQx=Q^T0\quad\Longrightarrow\quad I_nx=0\quad\Longrightarrow\quad x=0.
$$

所以 $N(Q)=\{0\}$，列向量线性无关。注意 $Q$ 可以是长方矩阵，不能在第一步写 $Q^{-1}$。
<!-- bilingual-en:start -->
Thus $N(Q)=\{0\}$ and the columns are linearly independent. Note that $Q$ may be rectangular, so the first step cannot assume that $Q^{-1}$ exists.
<!-- bilingual-en:end -->

#### Gram–Schmidt 的逐步构造
<!-- bilingual-en:start -->
*Stepwise Construction of Gram-Schmidt*
<!-- bilingual-en:end -->

先取
<!-- bilingual-en:start -->
Begin with
<!-- bilingual-en:end -->

$$
u_1=a_1,\qquad q_1=\frac{u_1}{\|u_1\|}.
$$

第二个向量减去在 $q_1$ 上的投影：
<!-- bilingual-en:start -->
The second vector subtracts the projection on the $q_1$:
<!-- bilingual-en:end -->

$$
u_2=a_2-(q_1^Ta_2)q_1,\qquad q_2=\frac{u_2}{\|u_2\|}.
$$

一般地，
<!-- bilingual-en:start -->
In general,
<!-- bilingual-en:end -->

$$
u_k=a_k-\sum_{j=1}^{k-1}(q_j^Ta_k)q_j,\qquad
q_k=\frac{u_k}{\|u_k\|}.
$$

验证 $u_k\perp q_i$（$i<k$）：
<!-- bilingual-en:start -->
Verify $u_k\perp q_i$ ($i<k$):
<!-- bilingual-en:end -->

$$
q_i^Tu_k=q_i^Ta_k-\sum_{j<k}(q_j^Ta_k)q_i^Tq_j
=q_i^Ta_k-q_i^Ta_k=0.
$$

每一步只从 $a_k$ 中减去先前向量的线性组合，所以
<!-- bilingual-en:start -->
Each step only subtracts the linear combination of the previous vectors from the $a_k$, so
<!-- bilingual-en:end -->

$$
\operatorname{span}(q_1,\dots,q_k)
=\operatorname{span}(a_1,\dots,a_k).
$$

列独立保证 $u_k\ne0$；若某一步 $u_k=0$，正说明 $a_k$ 已落入前面列的张成空间。
<!-- bilingual-en:start -->
Column independence guarantee $u_k\ne0$; if a step $u_k=0$, the $a_k$ is already in the row row row's cast space.
<!-- bilingual-en:end -->

#### 从 Gram–Schmidt 到 [[正交投影与最小二乘#Gram–Schmidt 与 QR|QR 分解]]
<!-- bilingual-en:start -->
*From Gram-Schmidt to [[正交投影与最小二乘#Gram–Schmidt 与 QR|QR decomposition]]*
<!-- bilingual-en:end -->

每个 $a_j$ 都能写成 $q_1,\dots,q_j$ 的组合：
<!-- bilingual-en:start -->
Each $a_j$ can be written as a combination of $q_1,\dots,q_j$:
<!-- bilingual-en:end -->

$$
a_j=\sum_{i=1}^j r_{ij}q_i,\qquad r_{ij}=q_i^Ta_j.
$$

把这些系数排成上三角矩阵 $R$，便有
<!-- bilingual-en:start -->
Put these coefficients in the upper triangular matrix $R$, and there are
<!-- bilingual-en:end -->

$$
A=QR,\qquad R=Q^TA.
$$

若约定 $r_{jj}=\|u_j\|>0$，满列秩矩阵的薄 QR 分解唯一。投影矩阵立即简化为
<!-- bilingual-en:start -->
With the convention $r_{jj}=\|u_j\|>0$, the thin QR factorization of a full-column-rank matrix is unique. The projection matrix simplifies immediately to
<!-- bilingual-en:end -->

$$
P=QQ^T,
$$

最小二乘正规方程也可化为
<!-- bilingual-en:start -->
The least-squares normal equations can also be written as
<!-- bilingual-en:end -->

$$
QR\hat x\approx b
\quad\Longrightarrow\quad
R\hat x=Q^Tb,
$$

只需回代上三角系统，避免显式形成 $A^TA$。
<!-- bilingual-en:start -->
Only the upper-triangular system needs to be solved, avoiding explicit formation of $A^TA$.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-gram-schmidt.png|760]]

### Recitation：完整 QR 计算
<!-- bilingual-en:start -->
*Recitation: Full QR Calculation*
<!-- bilingual-en:end -->

给定
<!-- bilingual-en:start -->
Given
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}
1&2&4\\
0&0&5\\
0&3&6
\end{bmatrix}
=[a\ b\ c].
$$

第一列 $a=(1,0,0)^T$ 已是单位向量，所以 $q_1=(1,0,0)^T$。接着
<!-- bilingual-en:start -->
The first column, $a=(1,0,0)^T$, is already a unit vector, so $q_1=(1,0,0)^T$. Next,
<!-- bilingual-en:end -->

$$
u_2=b-(q_1^Tb)q_1=(2,0,3)^T-2(1,0,0)^T=(0,0,3)^T,
$$

故 $q_2=(0,0,1)^T$。再算
<!-- bilingual-en:start -->
So, $q_2=(0,0,1)^T$.  then count
<!-- bilingual-en:end -->

$$
\begin{aligned}
u_3
&=c-(q_1^Tc)q_1-(q_2^Tc)q_2\\
&=(4,5,6)^T-4(1,0,0)^T-6(0,0,1)^T\\
&=(0,5,0)^T,
\end{aligned}
$$

所以 $q_3=(0,1,0)^T$。于是
<!-- bilingual-en:start -->
So, $q_3=(0,1,0)^T$.  therefore
<!-- bilingual-en:end -->

$$
Q=\begin{bmatrix}1&0&0\\0&0&1\\0&1&0\end{bmatrix},\qquad
R=Q^TA=\begin{bmatrix}1&2&4\\0&3&6\\0&0&5\end{bmatrix},\qquad A=QR.
$$

$R$ 的第 $j$ 列正是 $a_j$ 在标准正交基 $q_i$ 下的坐标。
<!-- bilingual-en:start -->
The $j$ column of $R$ is exactly the coordinate of $a_j$ in the orthonormal basis $q_i$.
<!-- bilingual-en:end -->

### Homework

> [!question]- Problem 17.1：用矩阵证明标准正交列自动独立
> **解。** 设 $Q$ 的列标准正交，所以 $Q^TQ=I$。若 $Qx=0$，则
> $$x=Ix=Q^TQx=Q^T0=0.$$
> 因齐次方程只有零解，$Q$ 的列线性无关。$Q$ 未必为方阵，因此不可假设 $Q^{-1}$ 存在。
> <!-- bilingual-en:start -->
> **Solution.** Suppose the columns of $Q$ are orthonormal, so $Q^TQ=I$. If $Qx=0$, then
> $$x=Ix=Q^TQx=Q^T0=0.$$
> Since the homogeneous equation has only the zero solution, the columns of $Q$ are linearly independent. The matrix $Q$ need not be square, so we cannot assume that $Q^{-1}$ exists.
> <!-- bilingual-en:end -->

> [!question]- Problem 17.2：对三个差分向量做 Gram–Schmidt
> 给定
> $$
> a=(1,-1,0,0),\quad b=(0,1,-1,0),\quad c=(0,0,1,-1).
> $$
> 求张成同一空间的正交向量，并证明它们与原向量都是 $d=(1,1,1,1)$ 的正交补的一组基。
>
> **解。** 先取
> $$A=a=(1,-1,0,0).$$
> 因 $A^Tb=-1,A^TA=2$，
> $$
> B=b-\frac{A^Tb}{A^TA}A
> =b+\frac12A
> =\left(\frac12,\frac12,-1,0\right).
> $$
> $A^Tc=0$，而 $B^Tc=-1,B^TB=3/2$，所以
> $$
> C=c-\frac{A^Tc}{A^TA}A-\frac{B^Tc}{B^TB}B
> =c+\frac23B
> =\left(\frac13,\frac13,\frac13,-1\right).
> $$
> 直接检查 $A^TB=A^TC=B^TC=0$。此外 $a,b,c$ 各分量之和为零，因此都垂直于 $d$；$A,B,C$ 亦然。$d^\perp\subseteq\mathbb R^4$ 是一个三维子空间，三条非零正交向量自动独立，故 $\{A,B,C\}$ 是其一组基。Gram–Schmidt 不改变张成空间，所以 $\{a,b,c\}$ 也是同一空间的一组基。
> <!-- bilingual-en:start -->
> **Given**
> Find orthogonal vectors spanning the same subspace, and prove that both the new vectors and the original vectors form bases of the orthogonal complement of $d=(1,1,1,1)$.
> **Solution.** Start with
> $$A=a=(1,-1,0,0).$$
> Because $A^Tb=-1,A^TA=2$,
> $A^Tc=0$, and $B^Tc=-1,B^TB=3/2$, so
> Check directly that $A^TB=A^TC=B^TC=0$. The components of each of $a,b,c$ sum to zero, so they are orthogonal to $d$; the same is true of $A,B,C$. Since $d^\perp\subseteq\mathbb R^4$ is three-dimensional, three nonzero mutually orthogonal vectors form a basis. Thus $\{A,B,C\}$ is a basis, and because Gram–Schmidt preserves the span, $\{a,b,c\}$ is also a basis of the same subspace.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- “orthonormal matrix” 常泛指列标准正交的长方矩阵；只有方阵才有 $Q^{-1}=Q^T$。
- Gram–Schmidt 必须减去在**所有已得到的 $q_j$** 上的投影，不能直接用尚未正交的 $a_j$ 代替。
- 原始 Gram–Schmidt 在列几乎相关时有数值误差；实际计算常用 modified Gram–Schmidt 或 Householder QR。
- $QR$ 中 $R$ 上三角来自“第 $j$ 列只需要前 $j$ 个 $q_i$”，不是事后巧合。
<!-- bilingual-en:start -->
- “Matrix with orthonormal columns” may refer to a rectangular matrix; only when $Q$ is square does $Q^{-1}=Q^T$ hold.
- Gram–Schmidt must subtract the projections onto **all previously constructed $q_j$**; a vector $a_j$ that has not yet been orthogonalized cannot be substituted for them.
- Classical Gram–Schmidt can suffer numerical error when the columns are nearly linearly dependent; numerical work often uses modified Gram–Schmidt or Householder QR instead.
- The upper triangular structure of $R$ in $A=QR$ follows from the fact that column $j$ uses only $q_1,\ldots,q_j$.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. $Q\in\mathbb R^{5\times3}$ 且 $Q^TQ=I_3$。$QQ^T$ 的尺寸、秩与行列式分别是什么？
> $QQ^T$ 是 $5\times5$，秩为 $3$，是投影到 $C(Q)$ 的矩阵；因秩小于 $5$，行列式为 $0$。
> <!-- bilingual-en:start -->
> $QQ^T$ is $5\times5$, the rank is $3$, is the projection to $C(Q)$ matrix; because the rank is less than $5$, the determinant is $0$.
> <!-- bilingual-en:end -->

> [!question]- 2. 对 $a_1=(1,0)^T,a_2=(1,1)^T$ 做 Gram–Schmidt。
> $q_1=(1,0)^T$；$u_2=a_2-(q_1^Ta_2)q_1=(0,1)^T$，故 $q_2=(0,1)^T$。
> <!-- bilingual-en:start -->
> $q_1=(1,0)^T$; $u_2=a_2-(q_1^Ta_2)q_1=(0,1)^T$, so $q_2=(0,1)^T$.
> <!-- bilingual-en:end -->

> [!question]- 3. 若 $A=QR$ 是薄 QR 且 $A$ 满列秩，为什么 $R$ 可逆？
> $R=Q^TA$ 是 $n\times n$ 上三角；其对角元是每一步非零残量的长度 $\|u_j\|>0$，所以所有主元非零，$R$ 可逆。
> <!-- bilingual-en:start -->
> $R=Q^TA$ is an $n\times n$ upper triangular matrix. Its diagonal entries are the norms $\|u_j\|>0$ of the nonzero residual vectors produced at each step, so every pivot is nonzero and $R$ is invertible.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

正交投影 → 标准正交基 → [[正交投影与最小二乘#Gram–Schmidt 与 QR|Gram–Schmidt 正交化]] → QR → 更稳定的最小二乘。
<!-- bilingual-en:start -->
Orthogonal projection→orthonormal basis→[[正交投影与最小二乘#Gram–Schmidt 与 QR|Gram-Schmidt orthogonalization]]→QR→more stable least squares.
<!-- bilingual-en:end -->

---

## Session 2.5 Properties of determinants

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节不把行列式当成待背的展开式，而是从三条基本性质推出所有计算规则。[[行列式#行列式的结构含义|行列式（determinant）]]只对方阵 $A\in\mathbb R^{n\times n}$ 定义，输出一个标量 $\det A$。
<!-- bilingual-en:start -->
This section does not treat determinants as expansions to be followed, but rather derives all the rules of calculation from three basic properties.  The [[行列式#行列式的结构含义|determinant]] is defined only by the matrix $A\in\mathbb R^{n\times n}$ and outputs a scalar $\det A$.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.5sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S05_Lecture_Lecture_18_Properties_of_Determinants.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S05_Recitation_Problem_Solving_Properties_of_Determinants.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.5prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.5sol.pdf#page=1)

### Lecture：三条公理推出整套规则
<!-- bilingual-en:start -->
*Lecture: Three axioms that lay out the rules*
<!-- bilingual-en:end -->

行列式由以下三条性质唯一确定：
<!-- bilingual-en:start -->
The determinant is uniquely determined by the following three properties:
<!-- bilingual-en:end -->

1. $\det I=1$。
2. 交换两行，行列式变号。
3. 固定其他行时，行列式对某一行是线性的。例如
   $$
   \det\begin{bmatrix}\alpha u+\beta v\\\text{其余各行}\end{bmatrix}
   =\alpha\det\begin{bmatrix}u\\\text{其余各行}\end{bmatrix}
   +\beta\det\begin{bmatrix}v\\\text{其余各行}\end{bmatrix}.
   $$
<!-- bilingual-en:start -->
1. $\det I=1$.
2. Swapping two rows changes the sign of the determinant.
3. With all other rows fixed, the determinant is linear in any one row. For example,
<!-- bilingual-en:end -->

#### 推论 1：有两行相同则行列式为零
<!-- bilingual-en:start -->
*Consequence 1: Equal rows force the determinant to be zero*
<!-- bilingual-en:end -->

若 $D$ 有两行相同，交换这两行后矩阵不变；但性质 2 说行列式变成 $-D$。因此 $D=-D$，在实数域中 $D=0$。
<!-- bilingual-en:start -->
If $D$ has two identical rows, swapping those rows leaves the matrix unchanged, but property 2 says that its determinant changes sign. Thus $\det D=-\det D$, so $\det D=0$ over the real numbers.
<!-- bilingual-en:end -->

#### 推论 2：一行加上另一行的倍数不改变行列式
<!-- bilingual-en:start -->
*Consequence 2: Adding a multiple of one row to another preserves the determinant*
<!-- bilingual-en:end -->

把第 $i$ 行 $r_i$ 换成 $r_i+cr_j$。按线性性，新行列式等于原行列式，加上 $c$ 乘以一个第 $i,j$ 行相同的行列式；后者为零，所以值不变。
<!-- bilingual-en:start -->
Replace row $i$, $r_i$, by $r_i+cr_j$. By linearity, the new determinant equals the original determinant plus $c$ times a determinant whose $i$th and $j$th rows are equal. The latter determinant is zero, so the value is unchanged.
<!-- bilingual-en:end -->

#### 推论 3：零行、相关行与奇异矩阵
<!-- bilingual-en:start -->
*Consequence 3: Zero rows, dependent rows, and singular matrices*
<!-- bilingual-en:end -->

有零行时，从该行提出标量 $0$，得行列式为零。若行向量线性相关，可通过不改变行列式的消元产生零行，所以 $\det A=0$。反过来，若消元有 $n$ 个非零主元，便可还原 $A$，所以 $A$ 可逆且行列式非零。于是
<!-- bilingual-en:start -->
A zero row contributes a factor of $0$, so the determinant vanishes. If the rows are linearly dependent, determinant-preserving elimination produces a zero row, again giving $\det A=0$. Conversely, if elimination produces $n$ nonzero pivots, the operations can be reversed, so $A$ is invertible and its determinant is nonzero. Therefore,
<!-- bilingual-en:end -->

$$
\det A\ne0
\Longleftrightarrow A\text{ 可逆}
\Longleftrightarrow \operatorname{rank}(A)=n
\Longleftrightarrow N(A)=\{0\}.
$$

#### 推论 4：三角矩阵的行列式是对角线乘积
<!-- bilingual-en:start -->
*Consequence 4: the determinant of a triangular matrix is the product of its diagonal entries*
<!-- bilingual-en:end -->

对上三角矩阵，从左上角开始利用行线性和零元素，或沿用消元规则，得到
<!-- bilingual-en:start -->
For upper triangular matrices, row linearities and zero elements are used from the upper left corner, or the elimination rules are used to get the final result
<!-- bilingual-en:end -->

$$
\det U=u_{11}u_{22}\cdots u_{nn}.
$$

若 $A$ 通过不换行的消元得到 $U$，则 $\det A=\det U$；每换一次行额外乘 $-1$。若某行放大 $c$ 倍，行列式也放大 $c$ 倍。
<!-- bilingual-en:start -->
If elimination transforms $A$ into $U$ using no row exchanges, then $\det A=\det U$. Each row exchange contributes an additional factor of $-1$, while multiplying a row by $c$ multiplies the determinant by $c$.
<!-- bilingual-en:end -->

#### 乘法与转置
<!-- bilingual-en:start -->
*Multiplication and Transpose*
<!-- bilingual-en:end -->

重要结论为
<!-- bilingual-en:start -->
The important conclusion is
<!-- bilingual-en:end -->

$$
\det(AB)=\det A\det B,\qquad \det(A^T)=\det A.
$$

第一式可从“左乘初等矩阵对应一次行操作”证明：每类初等矩阵对行列式的影响与乘法相容，任意可逆 $A$ 都是初等矩阵的乘积；奇异情形两边同为零。由 $AA^{-1}=I$ 还得
<!-- bilingual-en:start -->
The first formula can be proved from the following proof: the influence of each kind of elementary matrix on the determinant is compatible with multiplication, any invertible $A$ is the product of the elementary matrix, and both sides of the singular case are zero.  By $AA^{-1}=I$.
<!-- bilingual-en:end -->

$$
\det(A^{-1})=\frac1{\det A}.
$$

对标量 $c$，$cA$ 是把 **每一行** 放大 $c$，所以
<!-- bilingual-en:start -->
For the scalar $c$, $cA$ magnifies**every row**$c$, so
<!-- bilingual-en:end -->

$$
\det(cA)=c^n\det A,
$$

不是一般的 $c\det A$。
<!-- bilingual-en:start -->
Not just any $c\det A$.
<!-- bilingual-en:end -->

### Recitation：先识别结构，再决定算法
<!-- bilingual-en:start -->
*Recitation: Identify the structure before deciding on the algorithm*
<!-- bilingual-en:end -->

1. 若相邻两行相减后出现两条相同行，行列式立即为零，不必展开大数。
2. Vandermonde 矩阵
   $$
   V=\begin{bmatrix}1&a&a^2\\1&b&b^2\\1&c&c^2\end{bmatrix}
   $$
   经 $R_2-R_1,R_3-R_1$ 后，可分别提出 $(b-a),(c-a)$，再做一次消元得到
   $$
   \det V=(b-a)(c-a)(c-b).
   $$
3. 外积 $uv^T\in\mathbb R^{3\times3}$ 的所有行互为倍数，故 $\operatorname{rank}(uv^T)\le1<3$，行列式为零；若 $u,v$ 都非零，秩才恰好等于 $1$。
4. 若 $D^T=-D$ 且阶数 $n$ 为奇数，则
   $$
   \det D=\det D^T=\det(-D)=(-1)^n\det D=-\det D,
   $$
   所以 $\det D=0$。偶数阶斜对称矩阵未必奇异，例如
   $$\begin{bmatrix}0&1\\-1&0\end{bmatrix}$$
   的行列式为 $1$。
<!-- bilingual-en:start -->
1. Subtracting two adjacent rows makes two rows identical, so the determinant is immediately zero; no large cofactor expansion is needed.
2. For the Vandermonde matrix, perform $R_2-R_1$ and $R_3-R_1$, factor out $(b-a)$ and $(c-a)$, and eliminate once more to obtain $\det V=(b-a)(c-a)(c-b)$.
3. Every row of the outer product $uv^T\in\mathbb R^{3\times3}$ is a scalar multiple of every other row, so $\operatorname{rank}(uv^T)\le1<3$ and its determinant is zero. If both $u$ and $v$ are nonzero, its rank is exactly $1$.
4. If $D^T=-D$ and $n$ is odd, then $\det D=(-1)^n\det D=-\det D$, so $\det D=0$. An even-order skew-symmetric matrix need not be singular; for example,
   $$\begin{bmatrix}0&1\\-1&0\end{bmatrix}$$
   has determinant $1$.
<!-- bilingual-en:end -->

### Homework

> [!question]- Problem 18.1：行和揭示特征值与奇异性
> 若方阵 $A$ 每一行元素和为 $0$，证明 $\det A=0$。若每行和为 $1$，证明 $\det(A-I)=0$；这是否推出 $\det A=1$？
>
> **解。** 令 $\mathbf1=(1,\dots,1)^T\ne0$。行和为零意味着 $A\mathbf1=0$，所以 $A$ 有非零零空间，$\det A=0$。行和为一意味着 $A\mathbf1=\mathbf1$，故 $(A-I)\mathbf1=0$，所以 $\det(A-I)=0$；这只说明 $1$ 是 $A$ 的特征值，不说明其他特征值乘积为 $1$。反例
> $$
> A=\begin{bmatrix}0&1\\1&0\end{bmatrix}
> $$
> 每行和为 $1$，但 $\det A=-1$。
> <!-- bilingual-en:start -->
> If every row of the square matrix $A$ sums to $0$, prove that $\det A=0$. If every row sums to $1$, prove that $\det(A-I)=0$. Does this imply $\det A=1$?
> **Solution.** Let $\mathbf1=(1,\dots,1)^T\ne0$. Zero row sums give $A\mathbf1=0$, so $A$ has a nontrivial nullspace and $\det A=0$. Row sums equal to one give $A\mathbf1=\mathbf1$, hence $(A-I)\mathbf1=0$ and $\det(A-I)=0$. This shows only that $1$ is an eigenvalue of $A$; it does not determine the product of the remaining eigenvalues. As a counterexample,
> each row sums to $1$, but $\det A=-1$.
> <!-- bilingual-en:end -->

> [!question]- Problem 18.2：完整推导三阶 Vandermonde 行列式
> **解。**
> $$
> \begin{aligned}
> \det V
> &=\det\begin{bmatrix}1&a&a^2\\0&b-a&b^2-a^2\\0&c-a&c^2-a^2\end{bmatrix}\\
> &=(b-a)(c-a)
> \det\begin{bmatrix}1&a&a^2\\0&1&b+a\\0&1&c+a\end{bmatrix}\\
> &=(b-a)(c-a)
> \det\begin{bmatrix}1&a&a^2\\0&1&b+a\\0&0&c-b\end{bmatrix}\\
> &=(b-a)(c-a)(c-b).
> \end{aligned}
> $$
> 当任意两点相等时有两行相同，公式也相应出现一个零因子。
> <!-- bilingual-en:start -->
> **Solution.**
> When any two points are equal, two rows are identical, and a zero factor appears accordingly.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- 行列式不是逐元素线性，也不是对整个矩阵线性：通常 $\det(A+B)\ne\det A+\det B$。
- 行加法不改变行列式；行交换变号；行缩放会同比缩放。三种操作不要混记。
- $\det(AB)=\det A\det B$，但一般 $\det(A+B)$ 没有类似公式。
- $\det A=0$ 只告诉你至少一个方向被压扁；它不告诉你秩具体缺几维。
<!-- bilingual-en:start -->
- The determinant is not element-by-element linear, nor is it linear to the entire matrix: typically $\det(A+B)\ne\det A+\det B$.
- Row addition does not change the determinant; the rows swap the numbers; the rows scale year-over-year.  Do three things without remembering.
- $\det(AB)=\det A\det B$, but there is no similar formula for a typical $\det(A+B)$.
- $\det A=0$ tells you only that at least one direction is flattened; it does not tell you how many dimensions the rank is missing.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若 $A$ 是 $4\times4$ 且 $\det A=3$，求 $\det(2A)$。
> $\det(2A)=2^4\det A=48$。
> <!-- bilingual-en:start -->
> $\det(2A)=2^4\det A=48$.
> <!-- bilingual-en:end -->

> [!question]- 2. 消元中交换两次行并把一行乘以 $5$，最后上三角对角为 $2,3,4$。原矩阵行列式是多少？
> 两次交换符号抵消；缩放后的矩阵行列式为 $2\cdot3\cdot4=24$，它是原值的 $5$ 倍，所以原值为 $24/5$。
> <!-- bilingual-en:start -->
> Two swaps are canceled; the scaled matrix determinant is $2\cdot3\cdot4=24$, which is $5$ times the original value, so the original value is $24/5$.
> <!-- bilingual-en:end -->

> [!question]- 3. 奇数阶斜对称矩阵为什么一定不可逆？
> 因 $\det A=\det A^T=\det(-A)=(-1)^n\det A=-\det A$，故行列式为零。
> <!-- bilingual-en:start -->
> The determinant is zero because of $\det A=\det A^T=\det(-A)=(-1)^n\det A=-\det A$.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

消元与可逆性 → [[行列式#行列式的结构含义|行列式]]三公理 → 大公式与余子式 → 下一节的可计算公式。
<!-- bilingual-en:start -->
Elimination and invertibility → the three axioms of the [[行列式#行列式的结构含义|determinant]] → the Leibniz and cofactor-expansion formulas → the inverse formula in the next section.
<!-- bilingual-en:end -->

---

## Session 2.6 Determinant formulas and cofactors

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

三条性质说明“行列式应怎样变化”，本节把它们变成两个通用计算公式：含 $n!$ 项的排列公式，以及沿任意一行或一列展开的余子式公式。
<!-- bilingual-en:start -->
The three axioms describe how a determinant must behave. This section turns them into two general computational formulas: the $n!$-term Leibniz formula and cofactor expansion along any row or column.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S06_Lecture_Lecture_19_Determinant_Formulas_and_Cofactors.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S06_Recitation_Problem_Solving_Determinants.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.6prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.6sol.pdf#page=1)

### Lecture：排列大公式
<!-- bilingual-en:start -->
*Lecture: The Leibniz permutation formula*
<!-- bilingual-en:end -->

从每一行取一个元素，并要求所取元素来自不同列。列指标 $(\sigma(1),\dots,\sigma(n))$ 必须是 $1,\dots,n$ 的一个排列。于是
<!-- bilingual-en:start -->
Select one entry from each row, with the selected entries coming from distinct columns. The column indices $(\sigma(1),\dots,\sigma(n))$ must therefore form a permutation of $1,\dots,n$, giving
<!-- bilingual-en:end -->

$$
\det A=
\sum_{\sigma\in S_n}\operatorname{sgn}(\sigma)
\prod_{i=1}^n a_{i,\sigma(i)}.
$$

$S_n$ 是全部 $n!$ 个排列；$\operatorname{sgn}(\sigma)=+1$ 表示偶排列，$-1$ 表示奇排列。符号可由“恢复自然顺序所需交换次数”的奇偶性判断。
<!-- bilingual-en:start -->
$S_n$ is the set of all $n!$ permutations. The sign $\operatorname{sgn}(\sigma)$ is $+1$ for an even permutation and $-1$ for an odd one, determined by the parity of the number of swaps needed to restore the natural order.
<!-- bilingual-en:end -->

为什么每项不能在两行选同一列？若同一列重复，就会遗漏另一列；这样的项不能满足交换两行变号的交替性。也可以从行线性展开看出，只有每列恰好选一次的项能保留下来。
<!-- bilingual-en:start -->
Why can a term not select the same column in two different rows? Repeating one column necessarily omits another, and such a term cannot satisfy the alternating sign rule under row swaps. Equivalently, expanding by row linearity shows that only terms selecting every column exactly once survive.
<!-- bilingual-en:end -->

大公式适合稀疏且非零项很少的矩阵，但一般计算量为 $n!$，远逊于消元的约 $O(n^3)$。
<!-- bilingual-en:start -->
The Leibniz formula can be convenient for very sparse matrices with few nonzero terms, but in general it requires $n!$ terms and is vastly slower than elimination, which costs roughly $O(n^3)$ operations.
<!-- bilingual-en:end -->

### [[行列式#行列式的结构含义|余子式展开]]与代数余子式
<!-- bilingual-en:start -->
*[[行列式#行列式的结构含义|Cofactor expansion]] and cofactors*
<!-- bilingual-en:end -->

删去第 $i$ 行、第 $j$ 列所得的 $(n-1)\times(n-1)$ 矩阵记为 $M_{ij}$。对应的**代数余子式（cofactor）**是
<!-- bilingual-en:start -->
The $(n-1)\times(n-1)$ matrix obtained by deleting row $i$ and column $j$ is denoted by $M_{ij}$. The corresponding **cofactor** is
<!-- bilingual-en:end -->

$$
C_{ij}=(-1)^{i+j}\det M_{ij}.
$$

符号棋盘为
<!-- bilingual-en:start -->
symbol checkerboard as
<!-- bilingual-en:end -->

$$
\begin{bmatrix}
+&-&+&\cdots\\
-&+&-&\cdots\\
+&-&+&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{bmatrix}.
$$

把排列大公式按第一行所选列分组，就得到第一行展开：
<!-- bilingual-en:start -->
Grouping large equations into the first row of selected columns gives you the first row to expand:
<!-- bilingual-en:end -->

$$
\det A=a_{11}C_{11}+a_{12}C_{12}+\cdots+a_{1n}C_{1n}.
$$

若改为按“第 $i$ 行选中了哪一列”给排列项分组，就得到第 $i$ 行展开；按“第 $j$ 列选中了哪一行”分组，则得到第 $j$ 列展开：
<!-- bilingual-en:start -->
If you instead group the permutations by "which column is selected in row $i$", you get row $i$ expanded; if you group by "which row is selected in column $j$", you get column $j$ expanded:
<!-- bilingual-en:end -->

$$
\det A=\sum_{k=1}^n a_{ik}C_{ik}
=\sum_{k=1}^n a_{kj}C_{kj}.
$$

实际选择零最多的一行或一列，可显著减少子式数量。
<!-- bilingual-en:start -->
You can significantly reduce the number of subformulas by actually selecting a row or column with the most zeros.
<!-- bilingual-en:end -->

### Recitation：按矩阵结构混合三种方法
<!-- bilingual-en:start -->
*Recitation: Mix three methods by matrix structure*
<!-- bilingual-en:end -->

第一类 $5\times5$ 稀疏循环矩阵在对角线上为 $x$，循环邻位为 $y$。沿第一列展开，只剩两个三角子式，得到
<!-- bilingual-en:start -->
The first kind of $5\times5$ sparse cyclic matrix is $x$ on the diagonal and $y$ on the adjacent cyclic bits.  Expanding along the first column, there are only two trigonometrics left, giving
<!-- bilingual-en:end -->

$$
\det A=x^5+y^5.
$$

第二类矩阵 $B$ 的对角线全为 $x$、非对角线全为 $y$。用相邻行相减，再累加列，可化为对角线包含 $x+4y$ 与四个 $x-y$ 的三角矩阵：
<!-- bilingual-en:start -->
The diagonal of the second kind of matrix $B$ is all $x$, and the off-diagonal is all $y$.  Subtract adjacent rows and then add columns to reduce to a diagonal triangular matrix containing $x+4y$ and four $x-y$:
<!-- bilingual-en:end -->

$$
\det B=(x+4y)(x-y)^4.
$$

这也可从特征方向预见：$\mathbf1$ 的特征值是 $x+4y$，其正交补上的特征值都是 $x-y$。此处特征值将在 Session 2.8 正式建立。
<!-- bilingual-en:start -->
The same result can be anticipated from the eigendirections: $\mathbf1$ has eigenvalue $x+4y$, while every vector in its orthogonal complement has eigenvalue $x-y$. Eigenvalues will be developed formally in Session 2.8.
<!-- bilingual-en:end -->

### Homework

> [!question]- Problem 19.1：计算循环置换矩阵的行列式
> $$
> A=\begin{bmatrix}
> 0&0&0&1\\1&0&0&0\\0&1&0&0\\0&0&1&0
> \end{bmatrix}.
> $$
>
> **解。** 沿第一行展开，只有 $a_{14}=1$：
> $$
> \det A=(-1)^{1+4}\det I_3=-1.
> $$
> 也可理解为列排列 $(4,1,2,3)$ 是一个四循环，等价于三次交换，符号为 $(-1)^3=-1$。稀疏时余子式法最短。
> <!-- bilingual-en:start -->
> **Solution.** Expanding along the first row leaves only $a_{14}=1$.
> Equivalently, the column permutation $(4,1,2,3)$ is a four-cycle, which has the same parity as three swaps and therefore sign $(-1)^3=-1$. For this sparse matrix, cofactor expansion is the shortest method.
> <!-- bilingual-en:end -->

> [!question]- Problem 19.2：Pascal 矩阵末项减一为何使行列式从 $1$ 变成 $0$
> 已知 $n\times n$ 对称 Pascal 矩阵行列式为 $1$，其左上 $(n-1)\times(n-1)$ Pascal 子矩阵行列式也为 $1$。
>
> **解。** 只改变 $a_{nn}$，且行列式对该元素线性。$a_{nn}$ 的 cofactor 为
> $$
> C_{nn}=(-1)^{2n}\det M_{nn}=1\cdot1=1.
> $$
> 将 $a_{nn}$ 减 $1$ 会令整体行列式减少 $1\cdot C_{nn}=1$，故从 $1$ 变成 $0$。
> <!-- bilingual-en:start -->
> The $n\times n$ symmetric Pascal matrix has determinant $1$, and so does its upper-left $(n-1)\times(n-1)$ Pascal submatrix.
> **Solution.** Only $a_{nn}$ changes, and the determinant is linear in that entry. Its cofactor is $C_{nn}=1$, as shown above. Decreasing $a_{nn}$ by $1$ therefore decreases the determinant by $1\cdot C_{nn}=1$, changing it from $1$ to $0$.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- minor $\det M_{ij}$ 与 cofactor $C_{ij}$ 相差符号 $(-1)^{i+j}$。
- 沿某行展开时，每一项都使用**同一行元素**与其 cofactor；不可混用另一行。
- 大公式每项恰取每行、每列各一个元素；只检查行而忘记列会产生错误项。
- 余子式展开理论通用，但对稠密大矩阵递归计算极慢；数值计算应使用消元分解。
<!-- bilingual-en:start -->
- The minor $\det M_{ij}$ and the cofactor $C_{ij}$ differ by the sign factor $(-1)^{i+j}$.
- In an expansion along one row, every term uses an entry from **that same row** multiplied by its cofactor; entries from another row cannot be mixed in.
- In the Leibniz formula, each term selects exactly one entry from every row and every column. Checking the row condition but forgetting the column condition creates invalid terms.
Cofactor expansion is fully general, but recursive expansion is very slow for large dense matrices; numerical computation should use elimination-based factorizations instead.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 求 $C_{23}$ 的符号。
> $(-1)^{2+3}=-1$，所以 $C_{23}=-\det M_{23}$。
> <!-- bilingual-en:start -->
> $(-1)^{2+3}=-1$, $C_{23}=-\det M_{23}$.
> <!-- bilingual-en:end -->

> [!question]- 2. 为什么置换矩阵的行列式只能是 $\pm1$？
> 大公式中只有与该置换匹配的一项非零，所有选中元素均为 $1$，只剩排列符号 $\pm1$。
> <!-- bilingual-en:start -->
> In the Leibniz formula, only the term matching that permutation is nonzero. Every selected entry equals $1$, leaving only the permutation sign $\pm1$.
> <!-- bilingual-en:end -->

> [!question]- 3. $4\times4$ 行列式的大公式有多少项？每项含多少个矩阵元素？
> 有 $4!=24$ 项；每项从四行和四列各取一次，共含四个元素的乘积。
> <!-- bilingual-en:start -->
> There are $4!=24$ terms. Each term selects one entry from each of the four rows and each of the four columns, so it is a product of four entries.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

行列式性质 → 排列大公式 → cofactor 展开 → 下一节的 adjugate、[[行列式#乘法性与可逆性|Cramer 法则]]与体积。
<!-- bilingual-en:start -->
Determinant properties → permutation formula → cofactor expansion → adjugate, [[行列式#乘法性与可逆性|Cramer's rule]], and volume in the next section.
<!-- bilingual-en:end -->

---

## Session 2.7 Cramer's rule, inverse matrix and volume

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节把 cofactor 矩阵用于三个方向：构造逆矩阵、推导 Cramer 法则、解释坐标变换的面积或体积缩放。设 $A\in\mathbb R^{n\times n}$；逆矩阵与 Cramer 法则都要求 $\det A\ne0$。
<!-- bilingual-en:start -->
This section uses cofactors in three ways: to construct the inverse, derive Cramer's rule, and interpret the area or volume scaling of a linear transformation. Let $A\in\mathbb R^{n\times n}$; both the inverse formula and Cramer's rule require $\det A\ne0$.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S07_Lecture_Lecture_20_Cramer_s_Rule_Inverse_Matrix_and_Volume.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S07_Recitation_Problem_Solving_Determinants_and_Volume.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.7prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.7sol.pdf#page=1)

### Lecture：为什么 cofactor 的转置给出逆矩阵
<!-- bilingual-en:start -->
*Lecture: Why the transpose of the cofactor matrix gives the inverse*
<!-- bilingual-en:end -->

令 $C=(C_{ij})$ 为 cofactor 矩阵。考察 $AC^T$ 的 $(i,j)$ 元素：
<!-- bilingual-en:start -->
Let $C=(C_{ij})$ be the cofactor matrix. Consider the $(i,j)$ entry of $AC^T$:
<!-- bilingual-en:end -->

$$
(AC^T)_{ij}=\sum_{k=1}^n a_{ik}C_{jk}.
$$

- 若 $i=j$，这正是沿第 $i$ 行的 cofactor 展开，等于 $\det A$。
- 若 $i\ne j$，该和等于“把 $A$ 的第 $j$ 行替换为第 $i$ 行”所得矩阵沿第 $j$ 行展开。新矩阵有两行相同，行列式为零。
<!-- bilingual-en:start -->
- If $i=j$, this is exactly the cofactor expansion along row $i$, equal to $\det A$.
- If $i\ne j$, the sum is the cofactor expansion along row $j$ of the matrix obtained by replacing row $j$ of $A$ with row $i$. That matrix has two identical rows, so its determinant is zero.
<!-- bilingual-en:end -->

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
AC^T=(\det A)I.
$$

$C^T$ 称为伴随矩阵 $\operatorname{adj}(A)$。当 $\det A\ne0$ 时，两边除以 $\det A$，得到[[线性方程组与四个基本子空间#可解性与完整解|逆矩阵]]公式：
<!-- bilingual-en:start -->
$C^T$ is the adjugate matrix $\operatorname{adj}(A)$. When $\det A\ne0$, dividing both sides by $\det A$ gives the [[线性方程组与四个基本子空间#可解性与完整解|inverse-matrix]] formula:
<!-- bilingual-en:end -->

$$
A^{-1}
=\frac{1}{\det A}C^T.
$$

这个公式适合理论推导和小矩阵，不适合大型数值求逆。
<!-- bilingual-en:start -->
This formula is suitable for theoretical derivation and small matrix, and is not suitable for large-scale numerical inversion.
<!-- bilingual-en:end -->

#### Cramer 法则
<!-- bilingual-en:start -->
*Cramer's rule*
<!-- bilingual-en:end -->

对 $Ax=b$，第 $j$ 个分量
<!-- bilingual-en:start -->
For $Ax=b$, the $j$ component
<!-- bilingual-en:end -->

$$
x_j=e_j^TA^{-1}b
=\frac1{\det A}\sum_{i=1}^n C_{ij}b_i.
$$

右侧分子正是把 $A$ 的第 $j$ 列替换为 $b$ 所得矩阵 $B_j$ 沿该列展开。因此
<!-- bilingual-en:start -->
The numerator on the right is the cofactor expansion, along column $j$, of the matrix $B_j$ obtained by replacing column $j$ of $A$ with $b$. Therefore,
<!-- bilingual-en:end -->

$$
x_j=\frac{\det B_j}{\det A}.
$$

这就是[[行列式#乘法性与可逆性|Cramer 法则（Cramer's rule）]]。它清楚揭示解对数据的依赖，但求全部分量需要许多行列式，计算上通常不如消元。
<!-- bilingual-en:start -->
That's [[行列式#乘法性与可逆性|Cramer's rule]].  It clearly reveals the dependence of the solution on the data, but it requires many determinants to obtain all the quantities, which are usually less computationally efficient than elimination.
<!-- bilingual-en:end -->

#### 行列式的几何意义
<!-- bilingual-en:start -->
*The Geometric Meaning of Determinants*
<!-- bilingual-en:end -->

矩阵 $A=[a_1\ \cdots\ a_n]$ 把单位立方体映成由列向量张成的平行多面体。其 $n$ 维体积为
<!-- bilingual-en:start -->
The matrix $A=[a_1\ \cdots\ a_n]$ maps the unit cube to a parallel polyhedron spanned by row vectors.  Its $n$ dimension volume is
<!-- bilingual-en:end -->

$$
\operatorname{Vol}=|\det A|.
$$

绝对值给普通体积；符号记录定向是否翻转。这个结论满足行列式三公理：单位立方体体积为 $1$；交换两条边翻转定向；一条边变化时有向体积对该边线性。乘法公式则表示先做 $B$、再做 $A$ 时，体积缩放因子相乘。
<!-- bilingual-en:start -->
The absolute value gives ordinary volume, while the sign records whether orientation is preserved or reversed. This interpretation satisfies the three determinant axioms: the unit cube has volume $1$; exchanging two edge vectors reverses orientation; and oriented volume is linear in each edge vector when the others are fixed. The product rule says that applying $B$ and then $A$ multiplies their volume-scaling factors.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-determinant-volume.png|780]]

### Recitation：四面体体积与“不改变高度”的行操作
<!-- bilingual-en:start -->
*Recitation: tetrahedral volume and No Height Change row operations*
<!-- bilingual-en:end -->

四面体顶点为
<!-- bilingual-en:start -->
The tetrahedral vertices are
<!-- bilingual-en:end -->

$$
O=(0,0,0),\quad A_1=(2,2,-1),\quad A_2=(1,3,0),\quad A_3=(-1,1,4).
$$

三条边张成的平行六面体体积为
<!-- bilingual-en:start -->
The parallel hexahedron volume spanned by the three edges is
<!-- bilingual-en:end -->

$$
\left|\det\begin{bmatrix}2&2&-1\\1&3&0\\-1&1&4\end{bmatrix}\right|=12.
$$

同底同高关系给出四面体体积是平行六面体的 $1/6$：
<!-- bilingual-en:start -->
The tetrahedron volume is the $1/6$ of the parallelohedron.
<!-- bilingual-en:end -->

$$
V_T=\frac16\cdot12=2.
$$

若把 $A_3$ 移到 $A_3'=(-201,-199,104)$，则
<!-- bilingual-en:start -->
If you move $A_3$ to $A_3'=(-201,-199,104)$,
<!-- bilingual-en:end -->

$$
A_3'=A_3-100A_1.
$$

这对应第三行减第一行的 $100$ 倍，不改变行列式；几何上沿底面方向移动顶点，不改变到底面的高度。因此新四面体体积仍为 $2$。
<!-- bilingual-en:start -->
This corresponds to the third row minus the first row by a factor of $100$, without changing the determinant; moving the vertex geometrically along the bottom without changing the height of the bottom.  The volume of the new tetrahedron is thus still $2$.
<!-- bilingual-en:end -->

### Homework

> [!question]- Problem 20.1：cofactor 矩阵与 $AC^T=(\det A)I$
> $$
> A=\begin{bmatrix}1&1&4\\1&2&2\\1&2&5\end{bmatrix}.
> $$
> 求 cofactor 矩阵 $C$、$AC^T$ 与 $\det A$；解释为何把 $a_{13}=4$ 改为 $100$ 不改变行列式。
>
> **解。** 逐个删行删列并带符号，得
> $$
> C=\begin{bmatrix}
> 6&-3&0\\
> 3&1&-1\\
> -6&2&1
> \end{bmatrix}.
> $$
> 因而
> $$
> AC^T=
> \begin{bmatrix}3&0&0\\0&3&0\\0&0&3\end{bmatrix}=3I,
> $$
> 所以 $\det A=3$。$a_{13}$ 的 cofactor $C_{13}=0$；行列式对该元素的线性系数就是 $C_{13}$，故把 $4$ 改成任意数都不改变 $\det A$。
> <!-- bilingual-en:start -->
> Find the cofactor matrix $C$, $AC^T$ and $\det A$; explain why changing $a_{13}=4$ to $100$ does not change the determinant.
> **Solution.** Delete each row and column in turn and apply the appropriate cofactor sign to obtain the displayed matrix $C$.
> thus
> So, $\det A=3$.  cofactor $C_{13}=0$ of $a_{13}$; The linear coefficient of determinant to this element is $C_{13}$, so changing $4$ to any number does not change $\det A$.
> <!-- bilingual-en:end -->

> [!question]- Problem 20.2：球坐标 Jacobian
> 给定
> $$x=\rho\sin\phi\cos\theta,\quad y=\rho\sin\phi\sin\theta,\quad z=\rho\cos\phi,$$
> 求偏导矩阵的行列式。
>
> **解。** Jacobian 矩阵为
> $$
> J_m=\begin{bmatrix}
> \sin\phi\cos\theta&\rho\cos\phi\cos\theta&-\rho\sin\phi\sin\theta\\
> \sin\phi\sin\theta&\rho\cos\phi\sin\theta&\rho\sin\phi\cos\theta\\
> \cos\phi&-\rho\sin\phi&0
> \end{bmatrix}.
> $$
> 沿第三行展开：
> $$
> \begin{aligned}
> \det J_m
> &=\cos\phi\bigl(\rho^2\cos\phi\sin\phi(\cos^2\theta+\sin^2\theta)\bigr)\\
> &\quad+\rho\sin\phi\bigl(\rho\sin^2\phi(\cos^2\theta+\sin^2\theta)\bigr)\\
> &=\rho^2\sin\phi(\cos^2\phi+\sin^2\phi)\\
> &=\boxed{\rho^2\sin\phi}.
> \end{aligned}
> $$
> 在通常范围 $\rho\ge0,0\le\phi\le\pi$，该值非负，所以体积元为 $dV=\rho^2\sin\phi\,d\rho\,d\phi\,d\theta$。
> <!-- bilingual-en:start -->
> given
> $$x=\rho\sin\phi\cos\theta,\quad y=\rho\sin\phi\sin\theta,\quad z=\rho\cos\phi,$$
> Find the determinant of the partial derivative matrix.
> **Solution.** The Jacobian matrix is shown above. Expanding its determinant along the third row yields the displayed calculation.
> On the standard range $\rho\ge0$ and $0\le\phi\le\pi$, the result is nonnegative, so the volume element is $dV=\rho^2\sin\phi\,d\rho\,d\phi\,d\theta$.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- 逆矩阵公式使用 $C^T$，不是 $C$；转置来自 $(AC^T)_{ij}$ 中固定 cofactor 的“行”。
- Cramer 法则要求 $\det A\ne0$；奇异系统不能用分母为零的比值判断有解性。
- $\det A$ 是有向体积，普通体积必须取 $|\det A|$。
- Jacobian 的行列顺序改变会改变符号；体积元取绝对值，但若已限定坐标范围可判断符号。
<!-- bilingual-en:start -->
- The inverse formula uses $C^T$, not $C$; the transpose aligns a fixed row of cofactors with the corresponding matrix product $(AC^T)_{ij}$.
- Cramer's rule requires $\det A\ne0$; when the denominator is zero, its ratios cannot decide whether a singular system is consistent.
- $\det A$ gives oriented volume; ordinary volume is $|\det A|$.
- Changing the order of the Jacobian's rows or columns changes its sign. A volume element uses the absolute value, although a restricted coordinate range may make the sign known in advance.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若 $\det A=-4$，$A$ 把单位立方体的体积与定向怎样改变？
> 体积放大为 $4$ 倍；负号表示定向翻转。
> <!-- bilingual-en:start -->
> The volume is $4$ times larger; a negative sign indicates directional flip.
> <!-- bilingual-en:end -->

> [!question]- 2. 为什么 $AC^T$ 的非对角元为零？
> $(i,j)$ 非对角元等于把第 $j$ 行替换成第 $i$ 行后沿第 $j$ 行展开；所得矩阵有两行相同，行列式为零。
> <!-- bilingual-en:start -->
> The $(i,j)$ off-diagonal element is expanded along the $j$ after replacing the $j$ row with the $i$ row; the resulting matrix has two identical rows and the determinant is zero.
> <!-- bilingual-en:end -->

> [!question]- 3. 三维四面体由同一点出发的边向量组成列矩阵 $A$，体积公式是什么？
> 平行六面体体积为 $|\det A|$，四面体体积为 $|\det A|/6$。
> <!-- bilingual-en:start -->
> The parallel hexahedron volume is $|\det A|$, and the tetrahedron volume is $|\det A|/6$.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

cofactor → adjugate 与[[线性方程组与四个基本子空间#可解性与完整解|逆矩阵]] → [[行列式#乘法性与可逆性|Cramer 法则]] → 体积缩放 → 下一节由 $\det(A-\lambda I)=0$ 寻找特征方向。
<!-- bilingual-en:start -->
cofactor → adjugate and [[线性方程组与四个基本子空间#可解性与完整解|inverse matrix]] → [[行列式#乘法性与可逆性|Cramer's rule]] → volume scaling → the next section finds eigenvector directions from $\det(A-\lambda I)=0$.
<!-- bilingual-en:end -->

---

## Session 2.8 Eigenvalues and eigenvectors

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

一般向量被矩阵作用后会改变方向。哪些特殊方向只被缩放而不转向？设 $A\in\mathbb R^{n\times n}$；只有方阵才能在同一空间中比较 $x$ 与 $Ax$ 的方向。
<!-- bilingual-en:start -->
A general vector changes direction when a matrix acts on it. Which special directions are merely scaled, with no change of direction? Let $A\in\mathbb R^{n\times n}$; a square matrix is required so that $x$ and $Ax$ lie in the same space and their directions can be compared.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.8sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S08_Lecture_Lecture_21_Eigenvalues_and_Eigenvectors.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S08_Recitation_Problem_Solving_Eigenvalues_and_Eigenvectors.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.8prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.8sol.pdf#page=1)

### Lecture：从 $Ax=\lambda x$ 到特征方程
<!-- bilingual-en:start -->
*Lecture: From $Ax=\lambda x$ to Characteristic Equations*
<!-- bilingual-en:end -->

非零向量 $x\ne0$ 若满足
<!-- bilingual-en:start -->
If a nonzero vector $x$ satisfies
<!-- bilingual-en:end -->

$$
Ax=\lambda x,
$$

则 $x$ 是 $A$ 的[[特征值、对角化与线性动力系统#特征值与特征向量|特征向量（eigenvector）]]，$\lambda$ 是对应的[[特征值、对角化与线性动力系统#特征值与特征向量|特征值（eigenvalue）]]。必须排除 $x=0$，因为零向量会对任意 $\lambda$ 满足等式，却不代表任何方向。
<!-- bilingual-en:start -->
Then $x$ is the [[特征值、对角化与线性动力系统#特征值与特征向量|eigenvector]] of $A$, and $\lambda$ is the corresponding [[特征值、对角化与线性动力系统#特征值与特征向量|eigenvalue]].  $x=0$ must be excluded because the zero vector satisfies the equation for any $\lambda$ but does not represent any direction.
<!-- bilingual-en:end -->

移项得
<!-- bilingual-en:start -->
Rearranging gives
<!-- bilingual-en:end -->

$$
(A-\lambda I)x=0.
$$

要有非零解，$A-\lambda I$ 必须奇异，所以
<!-- bilingual-en:start -->
To have a non-zero solution, $A-\lambda I$ must be singular, so
<!-- bilingual-en:end -->

$$
\det(A-\lambda I)=0.
$$

这称为[[特征值、对角化与线性动力系统#特征值与特征向量|特征多项式（characteristic polynomial）]]对应的**特征方程**。具体计算可按[[特征值、对角化与线性动力系统#特征值与特征向量|特征对计算流程]]执行：
<!-- bilingual-en:start -->
This is the **characteristic equation** associated with the [[特征值、对角化与线性动力系统#特征值与特征向量|characteristic polynomial]]. An [[特征值、对角化与线性动力系统#特征值与特征向量|eigenvalue–eigenvector pair]] can be computed as follows:
<!-- bilingual-en:end -->

1. 解标量多项式 $\det(A-\lambda I)=0$ 得特征值；
2. 对每个 $\lambda$ 解零空间 $N(A-\lambda I)$ 得特征向量。
<!-- bilingual-en:start -->
1. Solve the scalar polynomial equation $\det(A-\lambda I)=0$ for the eigenvalues.
2. For each eigenvalue $\lambda$, compute the eigenspace $N(A-\lambda I)$; its nonzero vectors are eigenvectors.
<!-- bilingual-en:end -->

> [!example] 一个 $2\times2$ 例子
> 对
> $$A=\begin{bmatrix}3&1\\1&3\end{bmatrix},$$
> $$
> \det(A-\lambda I)=(3-\lambda)^2-1=(\lambda-4)(\lambda-2).
> $$
> 当 $\lambda=4$，可取 $x_1=(1,1)^T$；当 $\lambda=2$，可取 $x_2=(1,-1)^T$。几何上，矩阵沿两条互相垂直的对角方向分别放大 $4$ 倍与 $2$ 倍。
> <!-- bilingual-en:start -->
> For
> $$A=\begin{bmatrix}3&1\\1&3\end{bmatrix},$$
> When $\lambda=4$, one may take $x_1=(1,1)^T$; when $\lambda=2$, one may take $x_2=(1,-1)^T$. Geometrically, the matrix scales these two perpendicular diagonal directions by factors of $4$ and $2$.
> <!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-eigenvectors.png|760]]

#### 行列式、迹与特征值
<!-- bilingual-en:start -->
*determinant, trace and eigenvalue*
<!-- bilingual-en:end -->

把特征多项式写成
<!-- bilingual-en:start -->
write characteristic polynomial as
<!-- bilingual-en:end -->

$$
\det(\lambda I-A)=\lambda^n-(\operatorname{tr}A)\lambda^{n-1}+\cdots+(-1)^n\det A.
$$

若在复数域计入[[特征值、对角化与线性动力系统#特征值与特征向量|代数重数]]，根为 $\lambda_1,\dots,\lambda_n$，比较系数得
<!-- bilingual-en:start -->
Over a field containing all roots, list the roots of the characteristic polynomial with [[特征值、对角化与线性动力系统#特征值与特征向量|algebraic multiplicity]] as $\lambda_1,\dots,\lambda_n$. Comparing coefficients gives
<!-- bilingual-en:end -->

$$
\operatorname{tr}A=\sum_{i=1}^n\lambda_i,\qquad
\det A=\prod_{i=1}^n\lambda_i.
$$

因此 $A$ 可逆当且仅当没有零特征值；若 $A$ 可逆，$A^{-1}$ 在同一特征向量上的特征值是 $1/\lambda$：
<!-- bilingual-en:start -->
Therefore, $A$ is invertible if and only if there is no zero eigenvalue, and if $A$ is invertible, the eigenvalue of $A^{-1}$ on the same eigenvector is $1/\lambda$:
<!-- bilingual-en:end -->

$$
Ax=\lambda x
\Longrightarrow
x=\lambda A^{-1}x
\Longrightarrow
A^{-1}x=\lambda^{-1}x.
$$

更一般地，对多项式 $p$，
<!-- bilingual-en:start -->
more generally, for polynomial $p$,
<!-- bilingual-en:end -->

$$
p(A)x=p(\lambda)x,
$$

因为 $A^kx=\lambda^kx$，再按线性组合相加即可。
<!-- bilingual-en:start -->
This follows because $A^kx=\lambda^kx$ for every power, after which the terms are combined linearly.
<!-- bilingual-en:end -->

#### 不同特征值的特征向量线性无关
<!-- bilingual-en:start -->
*Eigenvectors corresponding to distinct eigenvalues are linearly independent*
<!-- bilingual-en:end -->

> [!note] 定理
> 对应于两两不同特征值 $\lambda_1,\dots,\lambda_k$ 的特征向量 $x_1,\dots,x_k$ 线性无关。
> <!-- bilingual-en:start -->
> Eigenvectors $x_1,\dots,x_k$ corresponding to distinct eigenvalues $\lambda_1,\dots,\lambda_k$ are linearly independent.
> <!-- bilingual-en:end -->

**证明。** 对 $k$ 归纳。$k=1$ 时 $x_1\ne0$，结论成立。假设前 $k-1$ 个独立，并设
<!-- bilingual-en:start -->
**Proof.** Use induction on $k$. For $k=1$, $x_1\ne0$, so the claim holds. Assume the first $k-1$ vectors are independent and suppose
<!-- bilingual-en:end -->

$$
c_1x_1+\cdots+c_kx_k=0.
$$

左乘 $A$：
<!-- bilingual-en:start -->
Multiply on the left by $A$:
<!-- bilingual-en:end -->

$$
c_1\lambda_1x_1+\cdots+c_k\lambda_kx_k=0.
$$

用第二式减去 $\lambda_k$ 倍第一式：
<!-- bilingual-en:start -->
Subtract $\lambda_k$ times the first formula with the second formula:
<!-- bilingual-en:end -->

$$
c_1(\lambda_1-\lambda_k)x_1+\cdots+c_{k-1}(\lambda_{k-1}-\lambda_k)x_{k-1}=0.
$$

由归纳假设和 $\lambda_i-\lambda_k\ne0$，得 $c_1=\cdots=c_{k-1}=0$。代回原式，$c_kx_k=0$，因 $x_k\ne0$，故 $c_k=0$。所以全部系数为零，证毕。
<!-- bilingual-en:start -->
From the inductive hypothesis and $\lambda_i-\lambda_k\ne0$, we get $c_1=\cdots=c_{k-1}=0$.  $c_kx_k=0$, $c_k=0$ because of $x_k\ne0$.  So all the coefficients are zero.
<!-- bilingual-en:end -->

### Recitation：不直接平方或求逆
<!-- bilingual-en:start -->
*Recitation: Do not square or invert directly*
<!-- bilingual-en:end -->

取
<!-- bilingual-en:start -->
take
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}
1&2&3\\
0&1&-2\\
0&1&4
\end{bmatrix}.
$$

沿第一列展开：
<!-- bilingual-en:start -->
Expand along the first column:
<!-- bilingual-en:end -->

$$
\det(A-\lambda I)
=(1-\lambda)\bigl((1-\lambda)(4-\lambda)+2\bigr)
=(1-\lambda)(\lambda-2)(\lambda-3).
$$

所以特征值为 $1,2,3$。相应可取
<!-- bilingual-en:start -->
Therefore the eigenvalues are $1,2,3$. One possible choice is
<!-- bilingual-en:end -->

$$
x_1=\begin{bmatrix}1\\0\\0\end{bmatrix},\quad
x_2=\begin{bmatrix}1\\2\\-1\end{bmatrix},\quad
x_3=\begin{bmatrix}1\\-2\\2\end{bmatrix}.
$$

不用计算 $A^2$ 或 $A^{-1}$：
<!-- bilingual-en:start -->
Do not calculate $A^2$ or $A^{-1}$:
<!-- bilingual-en:end -->

- $A^2$ 的特征值是 $1,4,9$，特征向量仍为 $x_i$；
- $A^{-1}-I$ 的特征值是 $1/\lambda_i-1$，即 $0,-1/2,-2/3$，特征向量仍为 $x_i$。
<!-- bilingual-en:start -->
- $A^2$'s eigenvalue is $1,4,9$, and the eigenvector is still $x_i$;
- The eigenvalue of $A^{-1}-I$ is $1/\lambda_i-1$, i.e. $0,-1/2,-2/3$, and the eigenvector is still $x_i$.
<!-- bilingual-en:end -->

这展示了特征坐标的优势：矩阵函数在特征方向上退化为标量函数。
<!-- bilingual-en:start -->
This shows the advantage of the characteristic coordinates: the matrix function reduces to the scalar function in the characteristic direction.
<!-- bilingual-en:end -->

### Homework

> [!question]- Problem 21.1：只知道 $B$ 的特征值 $0,1,2$，能确定什么？
> **(a) 秩。** 零特征值说明 $B$ 奇异，秩至多 $2$；两个不同的非零特征值有两个独立特征向量，它们都在 $C(B)$ 中，故秩至少 $2$。所以 $\operatorname{rank}(B)=2$。
>
> **(b) $\det(B^TB)$。**
> $$\det(B^TB)=\det(B^T)\det B=(\det B)^2=0.$$
>
> **(c) $B^TB$ 的特征值。** 不能由 $B$ 的特征值确定。$B^TB$ 的特征值是奇异值平方，除非 $B$ 具有额外的正规性，通常不等于 $|\lambda_i(B)|^2$。相似但非正规矩阵可有相同特征值而有不同的 $B^TB$。
>
> **(d) $(B^2+I)^{-1}$ 的特征值。** 对每个 $B$ 的特征向量，标量变为 $(\lambda^2+1)^{-1}$，所以为
> $$\boxed{1,\frac12,\frac15}.$$
> <!-- bilingual-en:start -->
> **(a) Rank.** The zero eigenvalue makes $B$ singular, so its rank is at most $2$. The two distinct nonzero eigenvalues have linearly independent eigenvectors, both lying in $C(B)$, so its rank is at least $2$. Hence $\operatorname{rank}(B)=2$.
> **(b) $\det(B^TB)$.**
> $$\det(B^TB)=\det(B^T)\det B=(\det B)^2=0.$$
> **(c) Eigenvalues of $B^TB$.** They cannot be determined from the eigenvalues of $B$ alone. The eigenvalues of $B^TB$ are the squared singular values of $B$ and equal $|\lambda_i(B)|^2$ only under additional conditions such as normality. Similar nonnormal matrices can have the same eigenvalues but different matrices $B^TB$.
> **(d) Eigenvalues of $(B^2+I)^{-1}$.** On an eigenvector of $B$ with eigenvalue $\lambda$, this matrix acts by $(\lambda^2+1)^{-1}$, so its eigenvalues are
> $$\boxed{1,\frac12,\frac15}.$$
> <!-- bilingual-en:end -->

> [!question]- Problem 21.2：三个结构不同的矩阵
> $$
> A=\begin{bmatrix}1&2&3\\0&4&5\\0&0&6\end{bmatrix},\quad
> B=\begin{bmatrix}0&0&1\\0&2&0\\3&0&0\end{bmatrix},\quad
> C=\begin{bmatrix}2&2&2\\2&2&2\\2&2&2\end{bmatrix}.
> $$
>
> **解。** $A$ 为三角矩阵，特征值就是对角元 $1,4,6$。
>
> 对 $B$，
> $$
> \det(B-\lambda I)=(2-\lambda)(\lambda^2-3),
> $$
> 所以特征值为 $2,\sqrt3,-\sqrt3$。
>
> $C=2\mathbf1\mathbf1^T$。对 $\mathbf1=(1,1,1)^T$，$C\mathbf1=6\mathbf1$；对任意 $x\perp\mathbf1$，$Cx=2\mathbf1(\mathbf1^Tx)=0$。故特征值为 $6,0,0$。迹与行列式分别核对为 $6$ 与 $0$。
> <!-- bilingual-en:start -->
> **Solution.** $A$ is triangular, so its eigenvalues are its diagonal entries $1,4,6$.
> For $B$,
> Thus the eigenvalues are $2,\sqrt3,-\sqrt3$.
> For $C=2\mathbf1\mathbf1^T$, the vector $\mathbf1=(1,1,1)^T$ satisfies $C\mathbf1=6\mathbf1$, while every $x\perp\mathbf1$ satisfies $Cx=2\mathbf1(\mathbf1^Tx)=0$. Hence the eigenvalues are $6,0,0$, consistent with trace $6$ and determinant $0$.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- 特征向量不能是零向量；同一特征空间中的任意非零倍数都是特征向量。
- $B$ 的特征值通常不能推出 $B^TB$ 的特征值；后者属于奇异值问题。
- 实矩阵可能没有足够的实特征值，例如二维旋转 $90^\circ$ 的特征值为 $\pm i$。
- 重复特征值未必提供足够多独立特征向量；代数重数与特征空间维数要分开。
<!-- bilingual-en:start -->
- The eigenvector cannot be zero; any non-zero multiple in the same eigenspace is an eigenvector.
The eigenvalues of $B$ cannot usually be deduced from the eigenvalues of $B^TB$; the latter is a singular value problem.
- The real matrix may not have enough real eigenvalues, such as $\pm i$ for the two-dimensional rotation $90^\circ$.
- Repeated eigenvalues may fail to provide enough linearly independent eigenvectors; algebraic multiplicity must be distinguished from eigenspace dimension.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若 $Ax=3x$，求 $(A^2-4A+I)x$。
> 等于 $(3^2-4\cdot3+1)x=-2x$。
> <!-- bilingual-en:start -->
> Equals $(3^2-4\cdot3+1)x=-2x$.
> <!-- bilingual-en:end -->

> [!question]- 2. 为什么 $0$ 是 $A$ 的特征值等价于 $A$ 奇异？
> $Ax=0x$ 有非零解恰好表示 $N(A)$ 非平凡，也恰好表示 $A$ 奇异。
> <!-- bilingual-en:start -->
> The equation $Ax=0x$ has a nonzero solution exactly when $N(A)$ is nontrivial, which is equivalent to $A$ being singular.
> <!-- bilingual-en:end -->

> [!question]- 3. 一个 $3\times3$ 矩阵有三个互不相同的特征值，能否保证有一组特征向量基？
> 能。不同特征值对应的三个特征向量线性无关，在三维空间中构成一组基。
> <!-- bilingual-en:start -->
> Yes. Eigenvectors associated with distinct eigenvalues are linearly independent, so three such eigenvectors form a basis of the three-dimensional space.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

$\det(A-\lambda I)=0$ → [[特征值、对角化与线性动力系统#特征值与特征向量|特征值]]与[[特征值、对角化与线性动力系统#特征值与特征向量|特征向量]] → 独立特征方向 → 下一节的[[特征值、对角化与线性动力系统#对角化与矩阵幂|对角化]]。
<!-- bilingual-en:start -->
$\det(A-\lambda I)=0$ → [[特征值、对角化与线性动力系统#特征值与特征向量|eigenvalues]] and [[特征值、对角化与线性动力系统#特征值与特征向量|eigenvectors]] → independent eigendirections → [[特征值、对角化与线性动力系统#对角化与矩阵幂|diagonalization]] in the next section.
<!-- bilingual-en:end -->

---

## Session 2.9 Diagonalization and powers of A

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

什么时候能选一组特征向量作为整个空间的基？一旦做到，为什么 $A^k$ 会变成只对标量取幂？本节始终假设 $A\in\mathbb F^{n\times n}$，其中 $\mathbb F$ 至少包含所需的特征值。
<!-- bilingual-en:start -->
When can a set of eigenvectors be chosen as a basis for the whole space? Once this is possible, why does computing $A^k$ reduce to taking scalar powers? Throughout this section, $A\in\mathbb F^{n\times n}$ and the field $\mathbb F$ contains the required eigenvalues.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.9sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S09_Lecture_Lecture_22_Diagonalization_and_Powers_of_A.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S09_Recitation_Problem_Solving_Powers_of_a_Matrix.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.9prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.9sol.pdf#page=1)

### Lecture：把全部特征方程并排放置
<!-- bilingual-en:start -->
*Lecture: Putting All Characteristic Equations Parallel*
<!-- bilingual-en:end -->

若 $A$ 有 $n$ 个线性无关特征向量 $x_1,\dots,x_n$，令
<!-- bilingual-en:start -->
If $A$ has $n$ linearly independent eigenvectors $x_1,\dots,x_n$, then
<!-- bilingual-en:end -->

$$
S=[x_1\ \cdots\ x_n],\qquad
\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n).
$$

把 $Ax_i=\lambda_i x_i$ 并排写：
<!-- bilingual-en:start -->
Tile $Ax_i=\lambda_i x_i$:
<!-- bilingual-en:end -->

$$
AS=A[x_1\ \cdots\ x_n]
=[\lambda_1x_1\ \cdots\ \lambda_nx_n]
=S\Lambda.
$$

$S$ 的列独立，所以可逆。右乘 $S^{-1}$ 得
<!-- bilingual-en:start -->
The columns of $S$ are independent, so they are invertible.  Right by $S^{-1}$
<!-- bilingual-en:end -->

$$
A=S\Lambda S^{-1},\qquad S^{-1}AS=\Lambda.
$$

这称为[[特征值、对角化与线性动力系统#对角化与矩阵幂|对角化（diagonalization）]]。它的含义不是把 $A$ 通过行操作变成对角矩阵，而是**换到特征向量基底**：$S^{-1}$ 把标准坐标换成特征坐标，$\Lambda$ 在各坐标独立缩放，$S$ 再换回标准坐标。
<!-- bilingual-en:start -->
This is [[特征值、对角化与线性动力系统#对角化与矩阵幂|diagonalization]]. It is not obtained by row-reducing $A$. Rather, $S$ changes eigencoordinates into standard coordinates, $S^{-1}$ changes standard coordinates into eigencoordinates, and $\Lambda$ describes the action of $A$ in the eigenvector basis.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-diagonalization.png|780]]

#### 可对角化的条件
<!-- bilingual-en:start -->
*Conditions for diagonalization*
<!-- bilingual-en:end -->

$$
A\text{ 可对角化}
\Longleftrightarrow
A\text{ 有 }n\text{ 个线性无关特征向量}.
$$

若有 $n$ 个互不相同的特征值，则上一节定理保证可对角化；反之不成立，例如 $I$ 只有一个不同特征值 $1$，却有任意基作为特征向量基。
<!-- bilingual-en:start -->
If $A$ has $n$ distinct eigenvalues, the preceding theorem guarantees diagonalizability. The converse is false: for example, $I$ has only one distinct eigenvalue, $1$, yet every basis is an eigenbasis.
<!-- bilingual-en:end -->

重复特征值的关键是几何重数。矩阵
<!-- bilingual-en:start -->
Geometric multiplicity is the key issue for repeated eigenvalues. Consider
<!-- bilingual-en:end -->

$$
\begin{bmatrix}1&1\\0&1\end{bmatrix}
$$

的特征值 $1$ 代数重数为 $2$，但 $N(A-I)$ 只有一维，因而不可对角化。
<!-- bilingual-en:start -->
The eigenvalue $1$ has algebraic multiplicity $2$, but its eigenspace $N(A-I)$ is only one-dimensional, so the matrix is not diagonalizable.
<!-- bilingual-en:end -->

#### 矩阵幂
<!-- bilingual-en:start -->
*matrix power*
<!-- bilingual-en:end -->

利用中间的 $S^{-1}S=I$：
<!-- bilingual-en:start -->
Using the intermediate cancellations $S^{-1}S=I$ gives
<!-- bilingual-en:end -->

$$
\begin{aligned}
A^k
&=(S\Lambda S^{-1})^k\\
&=S\Lambda(S^{-1}S)\Lambda\cdots(S^{-1}S)\Lambda S^{-1}\\
&=S\Lambda^kS^{-1}.
\end{aligned}
$$

其中
<!-- bilingual-en:start -->
where
<!-- bilingual-en:end -->

$$
\Lambda^k=\operatorname{diag}(\lambda_1^k,\dots,\lambda_n^k).
$$

对差分方程 $u_{k+1}=Au_k$，若
<!-- bilingual-en:start -->
For the difference equation $u_{k+1}=Au_k$, if
<!-- bilingual-en:end -->

$$
u_0=c_1x_1+\cdots+c_nx_n,
$$

则
<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->

$$
u_k=A^ku_0=c_1\lambda_1^kx_1+\cdots+c_n\lambda_n^kx_n.
$$

[[特征值、对角化与线性动力系统#线性动力系统|谱半径（spectral radius）]] $\rho(A)=\max_i|\lambda_i|$ 给出长期增长的基本尺度：$|\lambda|<1$ 衰减，$|\lambda|>1$ 增长，$\lambda<0$ 伴随交替翻转，复特征值带来旋转振荡。若存在 Jordan 块，还要额外考虑多项式因子。
<!-- bilingual-en:start -->
The [[特征值、对角化与线性动力系统#线性动力系统|spectral radius]] $\rho(A)=\max_i|\lambda_i|$ sets the basic long-run growth scale: modes with $|\lambda|<1$ decay, those with $|\lambda|>1$ grow, negative eigenvalues alternate in sign, and complex eigenvalues generate rotation and oscillation. Nontrivial Jordan blocks also contribute polynomial factors.
<!-- bilingual-en:end -->

### Recitation：参数矩阵的 $k$ 次幂
<!-- bilingual-en:start -->
*Recitation: The $k$th power of a parameterized matrix*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
C=\begin{bmatrix}
2b-a&a-b\\
2b-2a&2a-b
\end{bmatrix}.
$$

它的特征值为 $a,b$，可取对应特征向量 $(1,2)^T,(1,1)^T$。因此
<!-- bilingual-en:start -->
Its eigenvalues are $a$ and $b$, with corresponding eigenvectors $(1,2)^T$ and $(1,1)^T$. Therefore,
<!-- bilingual-en:end -->

$$
S=\begin{bmatrix}1&1\\2&1\end{bmatrix},\quad
\Lambda=\begin{bmatrix}a&0\\0&b\end{bmatrix},\quad
S^{-1}=\begin{bmatrix}-1&1\\2&-1\end{bmatrix}.
$$

相乘得到
<!-- bilingual-en:start -->
multiply
<!-- bilingual-en:end -->

$$
C^k=S\Lambda^kS^{-1}
=\begin{bmatrix}
2b^k-a^k&a^k-b^k\\
2b^k-2a^k&2a^k-b^k
\end{bmatrix}.
$$

令 $k=1$ 可恢复原矩阵，这是必要的代数检查。若 $a=b=-1$，则 $C=-I$，所以 $C^{100}=I$。
<!-- bilingual-en:start -->
Setting $k=1$ recovers the original matrix, an essential algebraic check. If $a=b=-1$, then $C=-I$, so $C^{100}=I$.
<!-- bilingual-en:end -->

### Homework

> [!question]- Problem 22.1：描述所有可对角化 $A$ 的矩阵 $S$
> $$
> A=\begin{bmatrix}4&0\\1&2\end{bmatrix}.
> $$
>
> **解。** 特征值为 $4,2$。对 $\lambda=4$，$(A-4I)x=0$ 给出 $x_1=2x_2$，可取 $(2,1)^T$；对 $\lambda=2$，给出 $x_1=0$，可取 $(0,1)^T$。因此所有使 $S^{-1}AS$ 对角化的 $S$，其两列必须分别是这两个特征方向的非零倍数，次序可以交换：
> $$
> S=\begin{bmatrix}2\alpha&0\\\alpha&\beta\end{bmatrix}
> \quad\text{或}\quad
> S=\begin{bmatrix}0&2\alpha\\\beta&\alpha\end{bmatrix},\qquad \alpha\beta\ne0.
> $$
> 因 $A^{-1}=S\Lambda^{-1}S^{-1}$，同样这些 $S$ 也对角化 $A^{-1}$，特征值变成 $1/4,1/2$。
> <!-- bilingual-en:start -->
> describing all diagonalizable $A$
> **Solution.** The eigenvalues are $4$ and $2$. For $\lambda=4$, solving $(A-4I)x=0$ gives $x_1=2x_2$, so use $(2,1)^T$. For $\lambda=2$, solving gives $x_1=0$, so use $(0,1)^T$. Therefore, in every matrix $S$ that diagonalizes $A$, the two columns must be nonzero multiples of these two eigendirections, possibly in the opposite order:
> Due to $A^{-1}=S\Lambda^{-1}S^{-1}$, these same $S$ also diagonalize $A^{-1}$, and the eigenvalues become $1/4,1/2$.
> <!-- bilingual-en:end -->

> [!question]- Problem 22.2：Markov 型矩阵的幂极限
> $$
> A=\begin{bmatrix}0.6&0.9\\0.4&0.1\end{bmatrix}.
> $$
>
> **解。** 每列和为 $1$，所以有特征值 $1$；迹为 $0.7$，另一特征值为 $-0.3$。可取
> $$
> x_1=\begin{bmatrix}9\\4\end{bmatrix},\qquad
> x_2=\begin{bmatrix}1\\-1\end{bmatrix}.
> $$
> 因而
> $$
> S=\begin{bmatrix}9&1\\4&-1\end{bmatrix},\qquad
> \Lambda=\begin{bmatrix}1&0\\0&-0.3\end{bmatrix}.
> $$
> 当 $k\to\infty$，$(-0.3)^k\to0$，所以
> $$
> \Lambda^k\to\begin{bmatrix}1&0\\0&0\end{bmatrix},\qquad
> A^k\to\frac1{13}\begin{bmatrix}9&9\\4&4\end{bmatrix}.
> $$
> 两列都是归一化稳态向量 $(9/13,4/13)^T$，说明长期状态忘记初始列。
> <!-- bilingual-en:start -->
> **Solution.** Each column sums to $1$, so $1$ is an eigenvalue. Since the trace is $0.7$, the other eigenvalue is $-0.3$. One may take the two eigenvectors displayed above, giving the stated matrices $S$ and $\Lambda$.
> As $k\to\infty$, $(-0.3)^k\to0$, so $A^k$ converges to the displayed rank-one matrix.
> Both columns of the limit are the normalized stationary vector $(9/13,4/13)^T$, showing that the long-run state becomes independent of the initial basis state.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- 行化简一般改变特征值；对角化是相似变换 $S^{-1}AS$，不是消元。
- $A=S\Lambda S^{-1}$ 的 $S$ 列顺序必须和 $\Lambda$ 对角元顺序一致。
- 有重复特征值不等于不可对角化；要检查对应特征空间总维数。
- $A^k=S\Lambda^kS^{-1}$ 要求可对角化；不可对角化矩阵需 Unit III 的 Jordan 方法。
<!-- bilingual-en:start -->
- Row reduction generally changes eigenvalues; diagonalization is a similarity transformation $S^{-1}AS$, not elimination.
- In $A=S\Lambda S^{-1}$, the order of the columns of $S$ must match the order of the diagonal entries of $\Lambda$.
- A repeated eigenvalue does not imply that the matrix is non-diagonalizable; check the total dimension of its eigenspaces.
- The formula $A^k=S\Lambda^kS^{-1}$ requires $A$ to be diagonalizable; non-diagonalizable matrices require the Jordan-form methods of Unit III.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若 $A=S\operatorname{diag}(2,-1)S^{-1}$，$A^{10}$ 的特征值是什么？
> $2^{10}=1024$ 与 $(-1)^{10}=1$，特征向量不变。
> <!-- bilingual-en:start -->
> $2^{10}=1024$ and $(-1)^{10}=1$, the eigenvectors are invariant.
> <!-- bilingual-en:end -->

> [!question]- 2. 为什么 $n$ 个不同特征值足以保证对角化？
> 对应的 $n$ 个特征向量线性无关，在 $n$ 维空间中形成基；以它们为 $S$ 的列即可。
> <!-- bilingual-en:start -->
> Their corresponding eigenvectors are linearly independent and therefore form a basis of the $n$-dimensional space. Using them as the columns of $S$ diagonalizes $A$.
> <!-- bilingual-en:end -->

> [!question]- 3. $A=I$ 只有一个不同特征值，为什么仍可对角化？
> 每个非零向量都是特征值 $1$ 的特征向量，可任取一组基；$A=SIS^{-1}$。
> <!-- bilingual-en:start -->
> Every nonzero vector is an eigenvector with eigenvalue $1$, so any basis can serve as an eigenbasis and $A=SIS^{-1}$.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

独立特征向量 → [[特征值、对角化与线性动力系统#对角化与矩阵幂|对角化]] → $A^k$ 与差分方程 → 下一节把标量 $e^{\lambda t}$ 提升为[[特征值、对角化与线性动力系统#对角化与矩阵幂|矩阵指数]]。
<!-- bilingual-en:start -->
Independent eigenvectors → [[特征值、对角化与线性动力系统#对角化与矩阵幂|diagonalization]] → $A^k$ and difference equations → the next section lifts the scalar exponential $e^{\lambda t}$ to the [[特征值、对角化与线性动力系统#对角化与矩阵幂|matrix exponential]].
<!-- bilingual-en:end -->

---

## Session 2.10 Differential equations and $e^{At}$

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

如何求解耦合常系数系统
<!-- bilingual-en:start -->
How to Solve the Coupling Constant Coefficient System
<!-- bilingual-en:end -->

$$
\frac{du}{dt}=Au,\qquad u(0)=u_0,\qquad A\in\mathbb R^{n\times n}?
$$

核心思想是：特征向量方向把向量微分方程化为标量指数增长；把全部方向合起来得到[[特征值、对角化与线性动力系统#对角化与矩阵幂|矩阵指数（matrix exponential）]] $e^{At}\in\mathbb R^{n\times n}$。
<!-- bilingual-en:start -->
The main idea is that the eigenvector direction converts the vector differential equation into a scalar exponential growth, and all the eigenvector directions are combined to form a [[特征值、对角化与线性动力系统#对角化与矩阵幂|matrix exponential]] $e^{At}\in\mathbb R^{n\times n}$.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.10sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S10_Lecture_Lecture_23_Differential_Equations_and_expAt.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S10_Recitation_Problem_Solving_Differential_Equations_and_expAt.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.10prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.10sol.pdf#page=1)

### Lecture：每个特征方向是一种指数模式
<!-- bilingual-en:start -->
*Lecture: Each eigendirection evolves as an exponential mode*
<!-- bilingual-en:end -->

若 $Ax=\lambda x$，尝试
<!-- bilingual-en:start -->
If $Ax=\lambda x$, try
<!-- bilingual-en:end -->

$$
u(t)=e^{\lambda t}x.
$$

则
<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->

$$
u'(t)=\lambda e^{\lambda t}x=e^{\lambda t}Ax=A(e^{\lambda t}x)=Au(t).
$$

所以每一对特征值—特征向量都给出一个解。若 $A$ 有特征向量基 $x_1,\dots,x_n$，把初值分解为
<!-- bilingual-en:start -->
Thus every eigenvalue–eigenvector pair gives a solution mode. If $A$ has an eigenbasis $x_1,\dots,x_n$, decompose the initial condition as
<!-- bilingual-en:end -->

$$
u_0=c_1x_1+\cdots+c_nx_n,
$$

由线性叠加得唯一解
<!-- bilingual-en:start -->
A unique solution is obtained by linear superposition
<!-- bilingual-en:end -->

$$
u(t)=c_1e^{\lambda_1t}x_1+\cdots+c_ne^{\lambda_nt}x_n.
$$

这与离散系统 $u_k=A^ku_0$ 完全平行：$\lambda_i^k$ 被 $e^{\lambda_i t}$ 取代。
<!-- bilingual-en:start -->
This is completely parallel to the discrete system $u_k=A^ku_0$: $\lambda_i^k$ was replaced by $e^{\lambda_i t}$.
<!-- bilingual-en:end -->

#### 矩阵指数的定义与推导
<!-- bilingual-en:start -->
*Definition and derivation of the matrix exponential*
<!-- bilingual-en:end -->

标量指数的幂级数提示定义
<!-- bilingual-en:start -->
The scalar exponential's power series motivates the definition
<!-- bilingual-en:end -->

$$
e^{At}=I+At+\frac{A^2t^2}{2!}+\frac{A^3t^3}{3!}+\cdots.
$$

逐项求导：
<!-- bilingual-en:start -->
Differentiate term by term:
<!-- bilingual-en:end -->

$$
\frac{d}{dt}e^{At}
=A+A^2t+\frac{A^3t^2}{2!}+\cdots
=Ae^{At}.
$$

并且 $e^{A\cdot0}=I$，所以
<!-- bilingual-en:start -->
And $e^{A\cdot0}=I$, so
<!-- bilingual-en:end -->

$$
u(t)=e^{At}u_0
$$

满足微分方程和初值。若 $A=S\Lambda S^{-1}$，则 $A^k=S\Lambda^kS^{-1}$，代入级数：
<!-- bilingual-en:start -->
satisfies both the differential equation and the initial condition. If $A=S\Lambda S^{-1}$, then $A^k=S\Lambda^kS^{-1}$; substituting this into the series gives
<!-- bilingual-en:end -->

$$
e^{At}
=S\left(I+\Lambda t+\frac{\Lambda^2t^2}{2!}+\cdots\right)S^{-1}
=Se^{\Lambda t}S^{-1},
$$

其中
<!-- bilingual-en:start -->
where
<!-- bilingual-en:end -->

$$
e^{\Lambda t}=\operatorname{diag}(e^{\lambda_1t},\dots,e^{\lambda_nt}).
$$

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-matrix-exponential.png|780]]

必须区分：一般 $e^{(A+B)t}\ne e^{At}e^{Bt}$；只有 $AB=BA$ 时该标量指数法则才成立。另一方面，同一个 $A$ 的指数总满足
<!-- bilingual-en:start -->
In general, $e^{(A+B)t}\ne e^{At}e^{Bt}$; the familiar scalar rule holds only when $AB=BA$. Exponentials of the same matrix, however, always satisfy
<!-- bilingual-en:end -->

$$
e^{At}e^{As}=e^{A(t+s)},\qquad (e^{At})^{-1}=e^{-At}.
$$

### Recitation：三阶 ODE 化为一阶矩阵系统
<!-- bilingual-en:start -->
*Recitation: Third-order ODE into First-order Matrix System*
<!-- bilingual-en:end -->

考虑
<!-- bilingual-en:start -->
consider
<!-- bilingual-en:end -->

$$
y'''+2y''-y'-2y=0.
$$

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
u=\begin{bmatrix}y''\\y'\\y\end{bmatrix},\qquad
u'=\begin{bmatrix}y'''\\y''\\y'\end{bmatrix}
=
\underbrace{\begin{bmatrix}-2&1&2\\1&0&0\\0&1&0\end{bmatrix}}_{A}
u.
$$

特征多项式为
<!-- bilingual-en:start -->
The characteristic polynomial is
<!-- bilingual-en:end -->

$$
\det(A-\lambda I)=(1-\lambda)(1+\lambda)(2+\lambda),
$$

故特征值为 $1,-1,-2$，可取
<!-- bilingual-en:start -->
Therefore the eigenvalues are $1,-1,-2$. One possible choice is
<!-- bilingual-en:end -->

$$
x_1=\begin{bmatrix}1\\1\\1\end{bmatrix},\quad
x_2=\begin{bmatrix}1\\-1\\1\end{bmatrix},\quad
x_3=\begin{bmatrix}4\\-2\\1\end{bmatrix}.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
u(t)=c_1e^tx_1+c_2e^{-t}x_2+c_3e^{-2t}x_3,
$$

读第三个分量即得
<!-- bilingual-en:start -->
Reading the third component gives
<!-- bilingual-en:end -->

$$
y(t)=c_1e^t+c_2e^{-t}+c_3e^{-2t}.
$$

令 $S=[x_1\ x_2\ x_3]$。$S^{-1}$ 第一列为 $(1/6,-1/2,1/3)^T$，因此 $e^{At}=Se^{\Lambda t}S^{-1}$ 的第一列为
<!-- bilingual-en:start -->
Get $S=[x_1\ x_2\ x_3]$.  The first column of $S^{-1}$ is $(1/6,-1/2,1/3)^T$, so the first column of $e^{At}=Se^{\Lambda t}S^{-1}$ is
<!-- bilingual-en:end -->

$$
\frac16e^tx_1-\frac12e^{-t}x_2+\frac13e^{-2t}x_3.
$$

这里只计算所需的一列，体现了“先判断输出需求，再做最少矩阵乘法”的策略。
<!-- bilingual-en:start -->
Here, only one column is calculated, which reflects the strategy of "first determine the output requirements, then do the least matrix multiplication".
<!-- bilingual-en:end -->

### Homework

> [!question]- Problem 23.1：斜对称系统保持长度
> 设
> $$
> u'=Au,\qquad
> A=\begin{bmatrix}0&c&-b\\-c&0&a\\b&-a&0\end{bmatrix},\qquad A^T=-A.
> $$
> 求 $\frac d{dt}\|u(t)\|^2$。
>
> **解法一：分量。**
> $$
> \begin{aligned}
> \frac d{dt}(u_1^2+u_2^2+u_3^2)
> &=2u_1(cu_2-bu_3)+2u_2(au_3-cu_1)\\
> &\quad+2u_3(bu_1-au_2)=0.
> \end{aligned}
> $$
> **解法二：矩阵。**
> $$
> \frac d{dt}(u^Tu)=u'^Tu+u^Tu'=u^T(A^T+A)u=0.
> $$
> 所以 $\|u(t)\|=\|u(0)\|$。再令 $w=(a,b,c)^T$；直接计算 $w^TA=0$，故
> $$
> \frac d{dt}(w^Tu)=w^TAu=0.
> $$
> 当 $w\ne0$ 时，轨迹同时位于球面 $\|u\|=\|u(0)\|$ 与固定平面 $w^Tu=w^Tu(0)$ 上，因此是两者的交圆，或在初值平行于 $w$ 时退化为一点。圆半径为
> $$
> \sqrt{\|u(0)\|^2-\frac{(w^Tu(0))^2}{\|w\|^2}}.
> $$
> 当 $w=0$ 时 $A=0$，轨迹同样退化为初始点。
> <!-- bilingual-en:start -->
> if
> Find $\frac d{dt}\|u(t)\|^2$.
> **Solution 1: Component.** **Solution 2: Matrix.**
> So, $\|u(t)\|=\|u(0)\|$.  $w=(a,b,c)^T$; directly calculate $w^TA=0$, therefore
> When $w\ne0$, the trajectory lies on both the sphere $\|u\|=\|u(0)\|$ and the fixed plane $w^Tu=w^Tu(0)$, so it is an intersection circle between them, or it is reduced to a point when the initial value is parallel to $w$.  Circle radius is
> When $w=0$, $A=0$, the trajectory also reduces to the initial point.
> <!-- bilingual-en:end -->

> [!question]- Problem 23.2：由对角化计算 $e^{At}$
> $$
> A=\begin{bmatrix}1&1\\0&3\end{bmatrix}.
> $$
>
> **解。** 特征值为 $1,3$，可取 $x_1=(1,0)^T,x_2=(1,2)^T$：
> $$
> S=\begin{bmatrix}1&1\\0&2\end{bmatrix},\quad
> \Lambda=\begin{bmatrix}1&0\\0&3\end{bmatrix},\quad
> S^{-1}=\begin{bmatrix}1&-1/2\\0&1/2\end{bmatrix}.
> $$
> 因而
> $$
> e^{At}=Se^{\Lambda t}S^{-1}
> =\begin{bmatrix}
> e^t&\frac12(e^{3t}-e^t)\\
> 0&e^{3t}
> \end{bmatrix}.
> $$
> 在 $t=0$ 得 $I$；求导后在 $t=0$ 得
> $$
> \left.\frac d{dt}e^{At}\right|_{t=0}
> =\begin{bmatrix}1&1\\0&3\end{bmatrix}=A,
> $$
> 两项检查均通过。
> <!-- bilingual-en:start -->
> **Solution.** The eigenvalues are $1$ and $3$, with possible eigenvectors $x_1=(1,0)^T$ and $x_2=(1,2)^T$. This gives the displayed matrices $S$, $\Lambda$, and $S^{-1}$, and hence the stated expression for $e^{At}$.
> At $t=0$ the result equals $I$, and differentiating then evaluating at $t=0$ gives $A$. Both checks pass.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- $e^{At}$ 不是把 $A$ 的每个元素分别取指数；定义来自矩阵幂级数。
- 只有可对角化时才能直接写 $Se^{\Lambda t}S^{-1}$；级数定义本身对所有方阵都有效。
- $\lambda<0$ 表示该特征模式随 $t\to\infty$ 衰减；不是说解向量“为负”。
- 复共轭特征值产生实数形式的正弦—余弦振荡；不可简单丢弃虚部。
<!-- bilingual-en:start -->
- $e^{At}$ is not obtained by exponentiating the entries of $A$ separately; it is defined by the matrix power series.
- The formula $Se^{\Lambda t}S^{-1}$ can be used directly only when $A$ is diagonalizable; the power-series definition is valid for every square matrix.
- $\lambda<0$ indicates that the eigenmode decays with $t\to\infty$; it does not mean that the solution vector is "negative."
- The complex conjugate eigenvalues produce a real form of sine—cosine oscillation; the imaginary part cannot be simply discarded.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 若 $Ax=-2x$，初值 $u_0=3x$，求 $u(t)$。
> $u(t)=3e^{-2t}x$。
> <!-- bilingual-en:start -->
> $u(t)=3e^{-2t}x$.
> <!-- bilingual-en:end -->

> [!question]- 2. 验证 $e^{At}u_0$ 满足初值。
> $e^{A0}u_0=Iu_0=u_0$；且 $\frac d{dt}(e^{At}u_0)=Ae^{At}u_0$。
> <!-- bilingual-en:start -->
> $e^{A0}u_0=Iu_0=u_0$; and $\frac d{dt}(e^{At}u_0)=Ae^{At}u_0$.
> <!-- bilingual-en:end -->

> [!question]- 3. $e^{At}e^{Bt}=e^{(A+B)t}$ 总成立吗？
> 不成立。矩阵乘法不交换会使幂级数交叉项次序不同；当 $AB=BA$ 时才成立。
> <!-- bilingual-en:start -->
> Not true.  The order of the cross-terms of the power series is different if the matrix multiplication is not commutative; it is valid only when $AB=BA$.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

对角化 → 特征模式 $e^{\lambda t}x$ → [[特征值、对角化与线性动力系统#对角化与矩阵幂|矩阵指数]] → 连续动力系统 → 下一节的概率稳态与正交函数展开。
<!-- bilingual-en:start -->
diagonalization → eigenmodes $e^{\lambda t}x$ → [[特征值、对角化与线性动力系统#对角化与矩阵幂|matrix exponential]] → continuous-time dynamical systems → the next section's stationary probability distributions and orthogonal function expansions.
<!-- bilingual-en:end -->

---

## Session 2.11 Markov matrices and Fourier series

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节把此前工具用于两个看似不同的问题：概率在状态之间转移时为什么会趋于稳态；函数怎样像向量一样投影到正交“坐标轴”上。二者共同核心是：选择能让变换解耦的基。
<!-- bilingual-en:start -->
This section applies the previous tool to two seemingly different questions: why probability tends to stabilize as it transitions between states; and how functions are projected as vectors onto orthogonal "axes."  The common core of the two methods is to choose the basis which can decouple the transformation.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S11_Recitation_Problem_Solving_Markov_Matrices.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.11prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.11sol.pdf#page=1)

### Lecture A：Markov 矩阵与稳态
<!-- bilingual-en:start -->
*Lecture A:Markov Matrix and Steady State*
<!-- bilingual-en:end -->

课程采用“列随机”约定：[[特征值、对角化与线性动力系统#Markov 矩阵与稳态|Markov 矩阵（Markov matrix）]] $A=(a_{ij})$ 满足
<!-- bilingual-en:start -->
Course uses the "column random" convention: [[特征值、对角化与线性动力系统#Markov 矩阵与稳态|Markov matrix]] $A=(a_{ij})$ meets
<!-- bilingual-en:end -->

$$
a_{ij}\ge0,\qquad \sum_{i=1}^n a_{ij}=1\quad\text{对每一列 }j.
$$

$a_{ij}$ 表示“当前在状态 $j$，下一步到状态 $i$”的概率。若 $p_k$ 是分量非负、和为 $1$ 的概率列向量，则
<!-- bilingual-en:start -->
$a_{ij}$ represents the probability of "currently in state $j$, next to state $i$".  If $p_k$ is a probability column vector whose components are non-negative and $1$, then
<!-- bilingual-en:end -->

$$
p_{k+1}=Ap_k.
$$

概率总和保持：令 $\mathbf1=(1,\dots,1)^T$，列和为 $1$ 等价于
<!-- bilingual-en:start -->
To see why total probability is preserved, let $\mathbf1=(1,\dots,1)^T$. Having every column sum to $1$ is equivalent to
<!-- bilingual-en:end -->

$$
\mathbf1^TA=\mathbf1^T.
$$

故
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
\mathbf1^Tp_{k+1}=\mathbf1^TAp_k=\mathbf1^Tp_k=1.
$$

$\mathbf1^T$ 是左特征向量；由于 $A$ 与 $A^T$ 特征值相同，$1$ 是 $A$ 的特征值。满足
<!-- bilingual-en:start -->
$\mathbf1^T$ is a left eigenvector. Since $A$ and $A^T$ have the same eigenvalues, $1$ is also an eigenvalue of $A$. A probability vector satisfying
<!-- bilingual-en:end -->

$$
Ap_*=p_*
$$

的概率向量称为**稳态（steady state）**。
<!-- bilingual-en:start -->
is called a **steady state**.
<!-- bilingual-en:end -->

若 $A$ 可对角化，特征值 $1$ 是单特征值，并且其他特征值均满足 $|\lambda_i|<1$，则
<!-- bilingual-en:start -->
If $A$ is diagonalizable, the eigenvalue $1$ is a single eigenvalue, and all other eigenvalues satisfy $|\lambda_i|<1$, then
<!-- bilingual-en:end -->

$$
p_k=A^kp_0=c_1p_*+\sum_{i\ge2}c_i\lambda_i^kx_i
\longrightarrow c_1p_*.
$$

归一化使极限分量和为 $1$。若 $1$ 的重数大于 $1$，稳态可能不唯一；若还存在 $\lambda=-1$ 或其他单位圆上的特征值，则可能周期振荡而不收敛。因此“Markov”本身并不自动保证收敛到唯一稳态。有限状态的正矩阵满足更强的 Perron--Frobenius 条件，本节图中的具体矩阵正是这一情形。
<!-- bilingual-en:start -->
The limiting component is normalized to be $1$.  If the multiplicity of $1$ is greater than $1$, the steady state may not be unique, and if there are eigenvalues on $\lambda=-1$ or other unit circles, it may oscillate periodically without convergence.  Therefore, "Markov" itself does not automatically guarantee convergence to a unique steady state.  Positive matrices in finite states satisfy the stronger Perron—Frobenius condition, which is the case for the specific matrices in the figure in this section.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-markov-steady-state.png|760]]

### Recitation：两状态粒子的长期分布
<!-- bilingual-en:start -->
*Recitation: Long-Term Distribution of Two-State Particles*
<!-- bilingual-en:end -->

粒子在 $A,B$ 两状态之间跳转：从 $A$ 留在 $A$ 的概率 $0.6$、到 $B$ 的概率 $0.4$；从 $B$ 到 $A$ 的概率 $0.2$、留在 $B$ 的概率 $0.8$。所以
<!-- bilingual-en:start -->
A particle moves between states $A$ and $B$: from $A$, it remains at $A$ with probability $0.6$ and moves to $B$ with probability $0.4$; from $B$, it moves to $A$ with probability $0.2$ and remains at $B$ with probability $0.8$. Therefore,
<!-- bilingual-en:end -->

$$
M=\begin{bmatrix}0.6&0.2\\0.4&0.8\end{bmatrix},\qquad p_0=\begin{bmatrix}1\\0\end{bmatrix}.
$$

一步后 $p_1=Mp_0=(0.6,0.4)^T$。特征值为 $1,0.4$，可取
<!-- bilingual-en:start -->
$p_1=Mp_0=(0.6,0.4)^T$. The eigenvalues are $1$ and $0.4$, with possible eigenvectors
<!-- bilingual-en:end -->

$$
x_1=\begin{bmatrix}1\\2\end{bmatrix},\qquad x_2=\begin{bmatrix}1\\-1\end{bmatrix}.
$$

由
<!-- bilingual-en:start -->
by
<!-- bilingual-en:end -->

$$
p_0=\frac13x_1+\frac23x_2
$$

得到
<!-- bilingual-en:start -->
This gives
<!-- bilingual-en:end -->

$$
p_n=\frac13x_1+\frac23(0.4)^nx_2
=\frac13\begin{bmatrix}1+2(0.4)^n\\2-2(0.4)^n\end{bmatrix}.
$$

因此
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
p_n\longrightarrow\begin{bmatrix}1/3\\2/3\end{bmatrix}.
$$

衰减因子 $(0.4)^n$ 精确描述“忘记初始状态”的速度。
<!-- bilingual-en:start -->
The decay factor $(0.4)^n$ gives the exact rate at which the chain forgets its initial state.
<!-- bilingual-en:end -->

### Lecture B：Fourier 级数是无限维投影
<!-- bilingual-en:start -->
*Lecture B: A Fourier series is an infinite-dimensional projection*
<!-- bilingual-en:end -->

有限维中，若 $q_1,\dots,q_n$ 是标准正交基，
<!-- bilingual-en:start -->
In finite dimensions, if $q_1,\dots,q_n$ is an orthonormal basis,
<!-- bilingual-en:end -->

$$
v=\sum_i(q_i^Tv)q_i.
$$

函数空间中把内积定义为
<!-- bilingual-en:start -->
The inner product is defined as
<!-- bilingual-en:end -->

$$
\langle f,g\rangle=\int_0^{2\pi}f(x)g(x)\,dx.
$$

函数 $1,\cos nx,\sin nx$ 两两正交，例如
<!-- bilingual-en:start -->
The function $1,\cos nx,\sin nx$ is biorthogonal, for example
<!-- bilingual-en:end -->

$$
\int_0^{2\pi}\sin x\cos x\,dx=0,
$$

但它们尚未全部归一化：$\|1\|^2=2\pi$，$\|\cos nx\|^2=\|\sin nx\|^2=\pi$。因此[[特征值、对角化与线性动力系统#对角化与矩阵幂|Fourier 级数（Fourier series）]]写成
<!-- bilingual-en:start -->
But they have not all been normalized: $\|1\|^2=2\pi$, $\|\cos nx\|^2=\|\sin nx\|^2=\pi$.  So [[特征值、对角化与线性动力系统#对角化与矩阵幂|Fourier series]] wrote
<!-- bilingual-en:end -->

$$
f(x)\sim a_0+\sum_{n=1}^{\infty}\bigl(a_n\cos nx+b_n\sin nx\bigr),
$$

其中投影系数为
<!-- bilingual-en:start -->
where the projection coefficient is
<!-- bilingual-en:end -->

$$
a_0=\frac1{2\pi}\int_0^{2\pi}f(x)\,dx,
$$

$$
a_n=\frac1\pi\int_0^{2\pi}f(x)\cos(nx)\,dx,\qquad
b_n=\frac1\pi\int_0^{2\pi}f(x)\sin(nx)\,dx.
$$

注意这里采用课程的“常数项写 $a_0$”约定；另一些教材写 $a_0/2$，相应地把 $a_0$ 定义为 $\frac1\pi\int f$。
<!-- bilingual-en:start -->
Note that the "constant term writes $a_0$" convention is used here; some textbooks write $a_0/2$, and define $a_0$ as $\frac1\pi\int f$ accordingly.
<!-- bilingual-en:end -->

截断到有限个三角函数时，Fourier 部分和就是在所张成函数子空间中的最小二乘投影；残差与每个保留的基函数正交。
<!-- bilingual-en:start -->
When truncated to a finite number of trigonometric functions, the Fourier partial sum is the least squares projection in the subspace of the expanded function; the residual is orthogonal to each preserved basis function.
<!-- bilingual-en:end -->

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-fourier-basis.png|780]]

### Homework

> [!question]- Problem 24.1：对称 $2\times2$ 矩阵何时有负特征值
> $$
> A=\begin{bmatrix}1&b\\b&1\end{bmatrix}.
> $$
>
> **解。** 特征值为 $1+b$ 与 $1-b$。当 $|b|>1$ 时恰有一个负特征值，例如 $b=2$ 时为 $3,-1$。消元主元为
> $$
> 1,\qquad 1-b^2,
> $$
> 第二个为负；对称消元的惯性与特征值正负个数一致。它不可能有两个负特征值，因为两特征值之和等于迹 $2>0$；也可直接看 $b>1$ 与 $b<-1$ 时总是一正一负。
> <!-- bilingual-en:start -->
> **Solution.** The eigenvalues are $1+b$ and $1-b$. When $|b|>1$, exactly one is negative; for example, if $b=2$, they are $3$ and $-1$. The elimination pivots are
> $1$ and $1-b^2$,
> so the second pivot is negative. Under symmetric elimination, the inertia agrees with the numbers of positive and negative eigenvalues. The matrix cannot have two negative eigenvalues because their sum is the trace, $2>0$. Equivalently, for either $b>1$ or $b<-1$, one eigenvalue is positive and the other negative.
> <!-- bilingual-en:end -->

> [!question]- Problem 24.2：矩阵分类与分解
> $$
> A=\begin{bmatrix}0&0&1\\0&1&0\\1&0&0\end{bmatrix},\qquad
> B=\frac13\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}.
> $$
>
> **矩阵 $A$。** $\det A=-1$，所以可逆；$A^TA=I$，所以正交；$A^2=I\ne A$，不是投影；每行每列恰有一个 $1$，所以是置换矩阵；$A=A^T$，所以可正交对角化；每列和为 $1$，所以也是 Markov。无换行的普通 $LU$ 因首主元为零而失败；QR、$S\Lambda S^{-1}$ 与 $Q\Lambda Q^T$ 均可行。
>
> **矩阵 $B$。** 秩为 $1$，不可逆、非正交；$B^2=B$，是投影；不是置换矩阵；$B=B^T$，故可正交对角化；每列和为 $1$，也是 Markov。可写退化的 $LU$；标准“满列秩薄 QR”不适用；$S\Lambda S^{-1}$ 与 $Q\Lambda Q^T$ 可行，特征值为 $1,0,0$。
> <!-- bilingual-en:start -->
> **Matrix $A$.** Since $\det A=-1$, it is invertible. Since $A^TA=I$, it is orthogonal. Because $A^2=I\ne A$, it is not a projection. It has exactly one $1$ in each row and column, so it is a permutation matrix. Since $A=A^T$, it is orthogonally diagonalizable; its columns sum to $1$, so it is also a Markov matrix. Standard LU without row exchanges fails because the first pivot is zero, but QR, $S\Lambda S^{-1}$, and $Q\Lambda Q^T$ are all available.
> **Matrix $B$.** Its rank is $1$, so it is singular and not orthogonal. Since $B^2=B$, it is a projection; it is not a permutation matrix. Because $B=B^T$, it is orthogonally diagonalizable, and because each column sums to $1$, it is also a Markov matrix. A rank-deficient $LU$ factorization can be written, but the standard thin QR factorization for full-column-rank matrices does not apply. Both spectral decompositions $S\Lambda S^{-1}$ and $Q\Lambda Q^T$ are available, with eigenvalues $1,0,0$.
> <!-- bilingual-en:end -->

> [!question]- Problem 24.3：补全对称 Markov 矩阵并求稳态
> 已知前两行
> $$
> \begin{bmatrix}.7&.1&.2\\.1&.6&.3\end{bmatrix}.
> $$
>
> **解。** 每列补到和为 $1$，第三行为 $(.2,.3,.5)$：
> $$
> A=\begin{bmatrix}.7&.1&.2\\.1&.6&.3\\.2&.3&.5\end{bmatrix}.
> $$
> 它又是对称矩阵，所以行和也为 $1$，从而 $A\mathbf1=\mathbf1$。稳态方向为 $(1,1,1)^T$；作为概率向量应归一化为 $(1/3,1/3,1/3)^T$。
> <!-- bilingual-en:start -->
> The first two rows are given above.
> **Solution.** Completing each column so that it sums to $1$ gives third row $(.2,.3,.5)$. The matrix is also symmetric, so its rows sum to $1$ as well and $A\mathbf1=\mathbf1$. The stationary direction is $(1,1,1)^T$; normalized as a probability vector, it is $(1/3,1/3,1/3)^T$.
> <!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- 本课用“列和为 $1$、概率列向量左乘”的约定；采用行向量的教材常定义行和为 $1$，两者互为转置。
- $\mathbf1^TA=\mathbf1^T$ 给的是左特征向量；稳态 $Ap_*=p_*$ 是右特征向量，除非矩阵对称，不可混同。
- Markov 矩阵可有周期而不收敛，例如交换矩阵的特征值含 $-1$。
- Fourier 基函数是正交但未按上述写法归一；系数分母来自各基函数的平方范数。
<!-- bilingual-en:start -->
- This section uses the convention that each column sums to $1$ and probability column vectors are multiplied on the left by the matrix. Texts that use row probability vectors instead use row-stochastic matrices; the two conventions are transposes of one another.
- The identity $\mathbf1^TA=\mathbf1^T$ identifies a left eigenvector. A steady state satisfies $Ap_*=p_*$ and is a right eigenvector; the two should not be confused unless symmetry makes them coincide.
- A Markov matrix can be periodic and fail to converge; for example, the eigenvalue $-1$ may occur in a permutation matrix.
- The Fourier basis functions used above are orthogonal but not normalized; the denominators in the coefficients are their squared norms.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. 为什么列随机矩阵保持概率总和？
> 列和为 $1$ 等价于 $\mathbf1^TA=\mathbf1^T$，所以 $\mathbf1^TAp=\mathbf1^Tp$。
> <!-- bilingual-en:start -->
> Column and are $1$ equivalent to $\mathbf1^TA=\mathbf1^T$, so $\mathbf1^TAp=\mathbf1^Tp$.
> <!-- bilingual-en:end -->

> [!question]- 2. 若 Markov 矩阵特征值为 $1,0.8,-0.2$，长期收敛速度由哪个数控制？
> 除稳态外最大模为 $0.8$，误差通常按 $0.8^k$ 的量级衰减。
> <!-- bilingual-en:start -->
> Except for steady state, the maximum mode is $0.8$, and the error usually attenuates by the order of $0.8^k$.
> <!-- bilingual-en:end -->

> [!question]- 3. 在 $[0,2\pi]$ 上，为什么 $\cos x$ 的投影系数分母是 $\pi$？
> 因 $\langle\cos x,\cos x\rangle=\int_0^{2\pi}\cos^2x\,dx=\pi$。
> <!-- bilingual-en:start -->
> Because of $\langle\cos x,\cos x\rangle=\int_0^{2\pi}\cos^2x\,dx=\pi$.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

矩阵幂 → [[特征值、对角化与线性动力系统#Markov 矩阵与稳态|Markov 稳态]]；标准正交投影 → [[特征值、对角化与线性动力系统#对角化与矩阵幂|Fourier 展开]]。两条线都体现“在合适的基中解耦”。
<!-- bilingual-en:start -->
Matrix Power → [[特征值、对角化与线性动力系统#Markov 矩阵与稳态|Markov steady state]]; Orthonormal Projection → [[特征值、对角化与线性动力系统#对角化与矩阵幂|Fourier Expansion]].  Both lines represent "decoupling in the proper basis".
<!-- bilingual-en:end -->

---

## Session 2.12 Exam 2 review

### 本节问题、前置知识与尺寸
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Dimensions for this section*
<!-- bilingual-en:end -->

本节不增加新定理，而是训练在有限时间内识别题型、先做结构判断、再选择计算工具。Exam 2 Review 串联三块内容：正交/投影/最小二乘/QR；行列式/cofactor/逆矩阵；特征值/对角化/矩阵幂。
<!-- bilingual-en:start -->
Rather than adding new theorems, this section trains you to identify the problem type under time pressure: first make a structural diagnosis, then choose the computational tool. The Exam 2 review connects three clusters: orthogonality, projection, least squares, and QR; determinants, cofactors, and inverses; and eigenvalues, diagonalization, and matrix powers.
<!-- bilingual-en:end -->

> [!info] 本地材料
> - [Review summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.12sum.pdf#page=1)
> - [Review lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S12_Lecture_Exam_2_Review.pdf#page=1)
> - [Exam problem-solving recitation](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S12_Recitation_Exam_2_Problem_Solving.pdf#page=1)
> - [Exam 2 problems](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=1)

> [!note] Review 与所附样卷的范围不同
> Review lecture 会复习 eigenvalues、diagonalization 与 $A^k$，但下面完整解答的官方 Exam 2 样卷只覆盖正交/投影、最小二乘、Gram--Schmidt 与行列式。官网也明确说明 eigenvalue questions 会在随后出现；因此不能把 Review 的全部内容误称为这份样卷的题目范围。
> <!-- bilingual-en:start -->
> The review lecture includes eigenvalues, diagonalization, and $A^k$, whereas the official Exam 2 sample solved below covers only orthogonality and projection, least squares, Gram–Schmidt, and determinants. The official site also states that eigenvalue questions appear later, so the review lecture's full content should not be mistaken for the scope of this particular sample exam.
> <!-- bilingual-en:end -->

### 题型分流表
<!-- bilingual-en:start -->
*Problem-type routing table*
<!-- bilingual-en:end -->

| 题面信号 | 第一反应 | 必查条件 |
|---|---|---|
| closest / best fit / inconsistent | $A^TA\hat x=A^Tb$ 或 QR | $A$ 是否满列秩；残差是否满足 $A^Te=0$ |
| project onto $C(A)$ | $P=A(A^TA)^{-1}A^T$；若 $Q^TQ=I$ 则 $P=QQ^T$ | $P^T=P,P^2=P$；尺寸为 ambient space 的维数 |
| orthonormal / Gram–Schmidt | 逐项减投影并归一化；$A=QR$ | 长方 $Q$ 只有 $Q^TQ=I$ |
| determinant | 先看三角、零、相同行/列、稀疏 | 行交换、缩放、行加法影响不同 |
| cofactor / inverse column | $C_{ij}=(-1)^{i+j}\det M_{ij}$；$A^{-1}=C^T/\det A$ | 转置与符号 |
| $A^k$ / long run | 先找 eigen，再 $S\Lambda^kS^{-1}$ | 是否有足够独立特征向量 |
| matrix class | 分别查定义，不凭外观 | projection: $P^2=P$；orthogonal: $Q^TQ=I$；Markov: 非负且列和 1 |
<!-- bilingual-en:start -->
| Topic Signal | First Response | Required |
|---|---|---|
| nearest point / best fit / inconsistent system | $A^TA\hat x=A^Tb$ or QR | whether $A$ has full column rank; whether the residual satisfies $A^Te=0$ |
| project onto $C(A)$ | $P=A(A^TA)^{-1}A^T$; if $Q^TQ=I$, $P=QQ^T$ | $P^T=P,P^2=P$; dimension of ambient space |
| orthonormal / Gram–Schmidt | Subtract all previous projections, then normalize; $A=QR$ | For rectangular $Q$, only $Q^TQ=I$ is guaranteed |
| determinant | First check for triangular form, zero rows, repeated rows or columns, and sparsity | Row swaps, row scaling, and row addition affect the determinant differently |
| cofactor / inverse column | $C_{ij}=(-1)^{i+j}\det M_{ij}$; $A^{-1}=C^T/\det A$ | Transposition and signs |
| $A^k$ / long-run behavior | Find eigenvalues and eigenvectors, then use $S\Lambda^kS^{-1}$ | Whether there are enough linearly independent eigenvectors |
| matrix class | Check each definition rather than relying on appearance | projection: $P^2=P$; orthogonal: $Q^TQ=I$; Markov: nonnegative entries and column sums equal to $1$ |
<!-- bilingual-en:end -->

### Review lecture：一题串起投影、秩、特征值和动力学
<!-- bilingual-en:start -->
*Review lecture: A Series of Projections, Rank, Eigenvalues and Dynamics*
<!-- bilingual-en:end -->

令 $a=(2,1,2)^T$，投影到 $\operatorname{span}(a)$：
<!-- bilingual-en:start -->
Let $a=(2,1,2)^T$ and project onto $\operatorname{span}(a)$:
<!-- bilingual-en:end -->

$$
P=\frac{aa^T}{a^Ta}
=\frac19\begin{bmatrix}4&2&4\\2&1&2\\4&2&4\end{bmatrix}.
$$

- $\operatorname{rank}(P)=1$，$C(P)=\operatorname{span}(a)$。
- $a$ 是特征值 $1$ 的特征向量；$a^\perp$ 中的两个独立方向对应特征值 $0$。
- 因 $P^2=P$，对 $k\ge1$ 有 $P^k=P$。
- 若 $u_{k+1}=Pu_k$，只做一次投影后便稳定：$u_k=Pu_0$（$k\ge1$）。
<!-- bilingual-en:start -->
- $\operatorname{rank}(P)=1$,$C(P)=\operatorname{span}(a)$.
- $a$ is the eigenvector of the eigenvalue $1$; two independent directions in $a^\perp$ correspond to the eigenvalue $0$.
- $P^k=P$ for $k\ge1$ due to $P^2=P$.
- For $u_{k+1}=Pu_k$, stabilization occurs after one drop: $u_k=Pu_0$ ($k\ge1$).
<!-- bilingual-en:end -->

这类综合题应先利用矩阵类别，而不是重新展开特征多项式或反复相乘。
<!-- bilingual-en:start -->
Such synthesis questions should first use the matrix category rather than reexpanding the characteristic polynomial or multiplying it repeatedly.
<!-- bilingual-en:end -->

另一个 review 例子是拟合过原点直线 $y=Dt$ 到 $(1,4),(2,5),(3,8)$：
<!-- bilingual-en:start -->
Another example of review is the fitting of the origin line $y=Dt$ to $(1,4),(2,5),(3,8)$:
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}1\\2\\3\end{bmatrix},\quad
b=\begin{bmatrix}4\\5\\8\end{bmatrix},\quad
\hat D=\frac{A^Tb}{A^TA}=\frac{38}{14}=\frac{19}{7}.
$$

这里只有一个未知数，不应人为添加截距列。
<!-- bilingual-en:start -->
There is only one unknown and you should not add intercept columns artificially.
<!-- bilingual-en:end -->

### Recitation：15 分钟行列式综合题
<!-- bilingual-en:start -->
*Recitation:15 Minute Determinant Synthesis*
<!-- bilingual-en:end -->

考虑
<!-- bilingual-en:start -->
consider
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}
1&2&3&4\\
5&6&7&8\\
0&0&9&10\\
0&0&11&12
\end{bmatrix}.
$$

大公式中，第三、四行只能从第三、四列取非零元素，前两行便只能从前两列取。因此仅有 $2\times2=4$ 个非零排列项，而不是盲目写满 $4!=24$ 项。更快地把它视为分块上三角：
<!-- bilingual-en:start -->
In the Leibniz formula, rows three and four can select nonzero entries only from columns three and four, leaving columns one and two for the first two rows. Thus only $2\times2=4$ permutation terms can be nonzero, rather than all $4!=24$ terms. More quickly, recognize the matrix as block upper triangular:
<!-- bilingual-en:end -->

$$
\det A
=\det\begin{bmatrix}1&2\\5&6\end{bmatrix}
\det\begin{bmatrix}9&10\\11&12\end{bmatrix}
=(-4)(-2)=8.
$$

第一行 cofactors 为
<!-- bilingual-en:start -->
The first line of cofactors is
<!-- bilingual-en:end -->

$$
C_{11}=-12,\qquad C_{12}=10,\qquad C_{13}=C_{14}=0.
$$

核对：$1(-12)+2(10)=8=\det A$。由逆矩阵公式，$A^{-1}$ 第一列是 $C^T$ 第一列，即 cofactor 矩阵第一行的转置除以行列式：
<!-- bilingual-en:start -->
Check: $1(-12)+2(10)=8=\det A$. By the inverse formula, the first column of $A^{-1}$ is the first column of $C^T$—equivalently, the transpose of the first row of the cofactor matrix divided by the determinant:
<!-- bilingual-en:end -->

$$
(A^{-1})_{:,1}=\frac18\begin{bmatrix}-12\\10\\0\\0\end{bmatrix}.
$$

考试策略是让前一问的结果直接服务后一问，并在每阶段做一次低成本核对。
<!-- bilingual-en:start -->
The test strategy is to have the results of the previous question serve the latter directly, with a low-cost check done at each stage.
<!-- bilingual-en:end -->

### 边界、反例与易错点
<!-- bilingual-en:start -->
*Boundaries, Counterexamples and Errors*
<!-- bilingual-en:end -->

- 投影矩阵的行列式：若不是恒等投影，至少有一个零特征值，所以方阵投影的行列式一定为零。
- $P_A=P_Q$ 的原因是投影到同一子空间，不是因为 $A=Q$。
- 行列式大公式中的“最多次数”由每项每行每列各选一个元素限制。
- Exam 中写出尺寸和一行验算，常能及时发现把 $Q^TQ$ 与 $QQ^T$ 交换等错误。
<!-- bilingual-en:start -->
- Determinant of the projection matrix: If it is not an identity projection, there is at least one zero eigenvalue, so the determinant of the square projection must be zero.
- $P_A=P_Q$ is due to projection into the same subspace, not $A=Q$.
- The Maximum Number of Times in a large determinant formula is limited by one element per row and column.
- Exam writes out size and a line of checking, often finding errors such as swapping $Q^TQ$ with $QQ^T$ in time.
<!-- bilingual-en:end -->

### 三道自检
<!-- bilingual-en:start -->
*Three self-check questions*
<!-- bilingual-en:end -->

> [!question]- 1. $Q\in\mathbb R^{8\times3}$ 列标准正交，投影矩阵是什么？其秩和尺寸是什么？
> $P=QQ^T\in\mathbb R^{8\times8}$，秩为 $3$。
> <!-- bilingual-en:start -->
> $P=QQ^T\in\mathbb R^{8\times8}$ with rank $3$.
> <!-- bilingual-en:end -->

> [!question]- 2. $P^2=P$ 的特征值只能是多少？
> 若 $Px=\lambda x$，则 $P^2x=\lambda^2x$，又等于 $Px=\lambda x$，故 $\lambda(\lambda-1)=0$，只能是 $0$ 或 $1$。
> <!-- bilingual-en:start -->
> If $Px=\lambda x$, then $P^2x=\lambda^2x$ is equal to $Px=\lambda x$, so $\lambda(\lambda-1)=0$ can only be $0$ or $1$.
> <!-- bilingual-en:end -->

> [!question]- 3. 若 $A$ 有特征值 $2,1/2,-1$，$A^k$ 长期一定收敛吗？
> 不一定且通常发散：特征值 $2$ 的分量指数增长；即使初值无该分量，$-1$ 分量也会交替而不收敛。
> <!-- bilingual-en:start -->
> Not necessarily and usually divergent: the component of the eigenvalue $2$ grows exponentially; even if the initial value does not have the component, the $-1$ component alternates without convergence.
> <!-- bilingual-en:end -->

### 知识链
<!-- bilingual-en:start -->
*knowledge chain*
<!-- bilingual-en:end -->

投影/QR、行列式/cofactor、eigen/diagonalization 三条线在 Unit II 与 Exam 2 Review 中汇合；下一单元将用对称性把它们进一步统一为正交对角化、正定性与 SVD。
<!-- bilingual-en:start -->
Projection/QR, determinant/cofactor and eigen/diagonalization are combined in Unit II and Exam 2 Review; the next unit will further unify them into orthogonal diagonalization, positive definiteness and SVD by symmetry.
<!-- bilingual-en:end -->

---

# Exam 2

> [!info] 试卷与官方答案
> - [Exam 2 原题，第 1 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=1)
> - [Exam 2 原题，第 2 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=2)
> - [Exam 2 原题，第 3 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=3)
> - [Exam 2 原题，第 4 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=4)
> - [官方答案，第 1 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=1) · [第 2 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=3) · [第 3 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=4) · [第 4 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=5)
> <!-- bilingual-en:start -->
> - [Exam 2 Original Question 1](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=1)
> - [Exam 2 Original Question 2](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=2)
> - [Exam 2 Original Question 3](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=3)
> - [Exam 2 Original Question 4](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=4)
> - [Official Answer, Question 1](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=1) · [Question 2](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=3) · [Question 3](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=4) · [Question 4](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=5)
> <!-- bilingual-en:end -->

## Problem 1：标准正交基构成的三个行列式
<!-- bilingual-en:start -->
*Problem 1: Three Determinants of Orthonormal Bases*
<!-- bilingual-en:end -->

设 $q_1,q_2,q_3$ 是 $\mathbb R^3$ 中的标准正交向量。求下列行列式的所有可能值，并说明理由。
<!-- bilingual-en:start -->
Let $q_1,q_2,q_3$ be the orthonormal vector in $\mathbb R^3$.  Find all possible values for the following determinants and explain the reasons.
<!-- bilingual-en:end -->

### (a) $\det[q_1\ q_2\ q_3]$

> [!success]- 完整解答
> 令 $Q=[q_1\ q_2\ q_3]$。标准正交条件给出
> $$Q^TQ=I.$$
> 两边取行列式：
> $$
> \det(Q^TQ)=\det(Q^T)\det Q=(\det Q)^2=1.
> $$
> 因为实数平方为 $1$，
> $$\boxed{\det Q=\pm1}.$$
> 正号对应保持定向的标准正交基，负号对应翻转定向的标准正交基。
> <!-- bilingual-en:start -->
> Get $Q=[q_1\ q_2\ q_3]$.  The orthonormality condition is given
> $$Q^TQ=I.$$
> Both sides take the determinant:
> because the real square is $1$,
> $$\boxed{\det Q=\pm1}.$$
> A positive sign corresponds to an orientation-preserving orthonormal basis, while a negative sign corresponds to an orientation-reversing orthonormal basis.
> <!-- bilingual-en:end -->

### (b) $\det[q_1+q_2\quad q_2+q_3\quad q_3+q_1]$

> [!success]- 完整解答
> 把新列写成 $Q$ 与系数矩阵的乘积：
> $$
> [q_1+q_2\quad q_2+q_3\quad q_3+q_1]
> =Q\begin{bmatrix}1&0&1\\1&1&0\\0&1&1\end{bmatrix}.
> $$
> 系数矩阵的行列式为
> $$
> \det\begin{bmatrix}1&0&1\\1&1&0\\0&1&1\end{bmatrix}
> =1+1=2.
> $$
> 所以新行列式为 $2\det Q$，所有可能值是
> $$\boxed{\pm2}.$$
> 这也可由行列式对各列的线性性展开；含重复列的项全部为零，只留下两个同号的循环排列项。
> <!-- bilingual-en:start -->
> The new column is written as the product of $Q$ and coefficient matrix:
> The determinant of the coefficient matrix is
> So the new determinant is $2\det Q$, and all possible values are
> $$\boxed{\pm2}.$$
> This can also be expanded by the linearity of the columns by the determinant; items with duplicate columns are all zeroes, leaving only two circularly arranged items of the same sign.
> <!-- bilingual-en:end -->

### (c) $\det[q_1\ q_2\ q_3]\det[q_2\ q_3\ q_1]$

> [!success]- 完整解答
> 第二个列次序由 $(1,2,3)$ 变为 $(2,3,1)$，是三循环，可由两次交换完成，所以是偶排列，行列式不变：
> $$
> \det[q_2\ q_3\ q_1]=\det[q_1\ q_2\ q_3].
> $$
> 乘积为 $(\det Q)^2=1$，故
> $$\boxed{1}.$$
> <!-- bilingual-en:start -->
> The second column order is changed from $(1,2,3)$ to $(2,3,1)$, which is a three-cycle, and can be completed by two exchanges, so it is even arrangement, the determinant is unchanged:
> Product is $(\det Q)^2=1$, so
> $$\boxed{1}.$$
> <!-- bilingual-en:end -->

> [!warning] 常见错误
> 不能因为列向量都是单位向量就直接断言行列式为 $1$；反射型标准正交基的行列式为 $-1$。只有乘积平方才固定为 $1$。
> <!-- bilingual-en:start -->
> Unit-length columns alone do not imply determinant $1$. Even for an orthonormal basis, an orientation-reversing basis has determinant $-1$; only the square of the determinant is forced to equal $1$.
> <!-- bilingual-en:end -->

## Problem 2：21 个数据点的最小二乘直线
<!-- bilingual-en:start -->
*Problem 2:Least Squares Line for 21 Data Points*
<!-- bilingual-en:end -->

在 $t=-10,-9,\dots,9,10$ 共 21 个时刻测量。除 $t=0$ 的测量为 $1$ 外，其余测量全为 $0$。
<!-- bilingual-en:start -->
It was measured at $t=-10,-9,\dots,9,10$ for 21 times.  All measurements are $0$ except for the $t=0$ measurement of $1$.
<!-- bilingual-en:end -->

### (a) 求最佳直线 $C+Dt$
<!-- bilingual-en:start -->
*(a) For the best straight line $C+Dt$*
<!-- bilingual-en:end -->

> [!success]- 完整解答
> 设计矩阵与数据向量为
> $$
> A=\begin{bmatrix}
> 1&-10\\1&-9\\\vdots&\vdots\\1&0\\\vdots&\vdots\\1&10
> \end{bmatrix}\in\mathbb R^{21\times2},\qquad
> b=(0,\dots,0,1,0,\dots,0)^T.
> $$
> 利用时刻关于零对称，
> $$
> \sum_{t=-10}^{10}t=0,\qquad
> \sum_{t=-10}^{10}t^2=2\sum_{t=1}^{10}t^2
> =2\cdot\frac{10\cdot11\cdot21}{6}=770.
> $$
> 因此
> $$
> A^TA=\begin{bmatrix}21&0\\0&770\end{bmatrix}.
> $$
> 数据只在 $t=0$ 取 $1$，所以
> $$
> A^Tb=\begin{bmatrix}1\\0\end{bmatrix}.
> $$
> 正规方程给出
> $$
> \begin{bmatrix}21&0\\0&770\end{bmatrix}
> \begin{bmatrix}\hat C\\\hat D\end{bmatrix}
> =\begin{bmatrix}1\\0\end{bmatrix},
> $$
> 从而
> $$\boxed{\hat C=\frac1{21},\qquad \hat D=0}.$$
> 最佳直线是所有测量的平均高度 $y=1/21$；对称性使斜率为零。
> <!-- bilingual-en:start -->
> The design matrix and data vector are the ones displayed above. Symmetry of the time points gives $\sum_i t_i=0$, while $\sum_i t_i^2=770$, $\sum_i b_i=1$, and $\sum_i t_ib_i=0$. Therefore the normal equations decouple and give
> $$\boxed{\hat C=\frac1{21},\qquad \hat D=0}.$$
> The best line is the average height of all measurements $y=1/21$; symmetry makes the slope zero.
> <!-- bilingual-en:end -->

### (b) 投影到哪个子空间？给出一个非零正交向量
<!-- bilingual-en:start -->
*(b) Onto which subspace is the projection? Give a nonzero orthogonal vector*
<!-- bilingual-en:end -->

> [!success]- 完整解答
> 投影发生在 $\mathbb R^{21}$，目标是 $C(A)$。一组基为
> $$
> c_1=(1,1,\dots,1)^T,\qquad
> c_2=(-10,-9,\dots,0,\dots,9,10)^T.
> $$
> 投影 $p=A\hat x=(1/21)c_1$。取残差
> $$
> e=b-p=\frac1{21}(-1,\dots,-1,20,-1,\dots,-1)^T,
> $$
> 其中中间分量为 $20/21$，两侧各十个 $-1/21$。它非零，且
> $$
> c_1^Te=\frac1{21}(-20+20)=0.
> $$
> 对 $c_2$，成对时刻 $t$ 与 $-t$ 的残差相等，乘积相消，中间时刻为零，所以 $c_2^Te=0$。因此 $e\in C(A)^\perp=N(A^T)$。
> <!-- bilingual-en:start -->
> The projection takes place in $\mathbb R^{21}$ and its target is $C(A)$, with basis vectors $c_1$ and $c_2$ as displayed above. The projection is $p=A\hat x=(1/21)c_1$, so the residual is $e=b-p$. Its middle component is $20/21$ and the other twenty components are $-1/21$, so $e\ne0$ and $c_1^Te=0$. For $c_2$, the residuals at $t$ and $-t$ are equal and their products cancel in pairs; the $t=0$ term vanishes. Hence $c_2^Te=0$ and $e\in C(A)^\perp=N(A^T)$.
> <!-- bilingual-en:end -->

> [!warning] 常见错误
> 目标子空间不是 $(t,y)$ 平面中的一条直线，而是 $\mathbb R^{21}$ 中由常数列和时间列张成的二维列空间。
> <!-- bilingual-en:start -->
> The target subspace is not a line in the $(t,y)$ plane, but a two-dimensional column space spanned by a constant column and a time column in the $\mathbb R^{21}$.
> <!-- bilingual-en:end -->

## Problem 3：Gram–Schmidt、两个投影矩阵与新向量
<!-- bilingual-en:start -->
*Problem 3:Gram-Schmidt, Two Projection Matrices and New Vectors*
<!-- bilingual-en:end -->

独立向量 $a_1,a_2,a_3\in\mathbb R^5$ 经 Gram–Schmidt 得到标准正交向量 $q_1,q_2,q_3$。令
<!-- bilingual-en:start -->
Applying Gram–Schmidt to the linearly independent vectors $a_1,a_2,a_3\in\mathbb R^5$ produces the orthonormal vectors $q_1,q_2,q_3$.
<!-- bilingual-en:end -->

$$
A=[a_1\ a_2\ a_3],\qquad Q=[q_1\ q_2\ q_3]\in\mathbb R^{5\times3}.
$$

### (a) 写出两个投影矩阵
<!-- bilingual-en:start -->
*(a) Writing out two projection matrices*
<!-- bilingual-en:end -->

> [!success]- 完整解答
> $A$ 满列秩，所以
> $$
> \boxed{P_A=A(A^TA)^{-1}A^T}.
> $$
> $Q^TQ=I_3$，所以
> $$
> \boxed{P_Q=Q(Q^TQ)^{-1}Q^T=QQ^T}.
> $$
> 两者都是 $5\times5$，因为它们把 $\mathbb R^5$ 中的向量映回 $\mathbb R^5$。
> <!-- bilingual-en:start -->
> $A$'s in full order, so
> $Q^TQ=I_3$, so
> Both are $5\times5$ because they map the vectors in $\mathbb R^5$ back to $\mathbb R^5$.
> <!-- bilingual-en:end -->

### (b) 比较 $P_A,P_Q$；求 $P_QQ$ 与 $\det P_Q$
<!-- bilingual-en:start -->
*(b) Comparing $P_A,P_Q$; seeking $P_QQ$ with $\det P_Q$*
<!-- bilingual-en:end -->

> [!success]- 完整解答
> Gram–Schmidt 不改变张成空间，所以
> $$C(A)=C(Q).$$
> 到同一个子空间的正交投影唯一，故
> $$\boxed{P_A=P_Q}.$$
> 再者
> $$
> P_QQ=QQ^TQ=QI_3=\boxed Q.
> $$
> $P_Q$ 是 $5\times5$ 但秩只有 $3$，因此奇异：
> $$\boxed{\det P_Q=0}.$$
> 等价地，$C(Q)^\perp$ 中的两个独立方向都是特征值 $0$ 的特征向量。
> <!-- bilingual-en:start -->
> Gram-Schmidt doesn't change the growth space, so
> $$C(A)=C(Q).$$
> The orthogonal projection to the same subspace is unique, so
> $$\boxed{P_A=P_Q}.$$
> furthermore
> $P_Q$ is $5\times5$ but the rank is only $3$, so it is odd:
> $$\boxed{\det P_Q=0}.$$
> Equivalently, the two independent directions in $C(Q)^\perp$ are the eigenvectors of the eigenvalue $0$.
> <!-- bilingual-en:end -->

### (c) 加入独立向量 $a_4$，哪一个是新的 $q_4$？
<!-- bilingual-en:start -->
*(c) Add the independent vector $a_4$, which is the new $q_4$?*
<!-- bilingual-en:end -->

> [!success]- 完整解答
> 先去掉 $a_4$ 在旧子空间 $C(A)=C(Q)$ 中的分量：
> $$
> u_4=a_4-P_Aa_4.
> $$
> 因 $a_4$ 与前三列独立，$u_4\ne0$。归一化得
> $$
> \boxed{q_4=\frac{a_4-P_Aa_4}{\|a_4-P_Aa_4\|}}.
> $$
> 即题目选项 3。逐项写成 $a_i$ 的简单投影系数通常错误，因为原始 $a_i$ 不正交；若改用 $q_i$，才可直接求和 $\sum(q_i^Ta_4)q_i$。
> <!-- bilingual-en:start -->
> First, we remove the component of $a_4$ in the old subspace $C(A)=C(Q)$:
> Because $a_4$ is separate from the first three columns, $u_4\ne0$.  normalized
> This is Question 3. Simply using the original $a_i$ as though their coefficients could be computed independently is usually wrong, because the $a_i$ are not orthogonal. With the orthonormal vectors $q_i$, however, the projection is obtained directly as $\sum(q_i^Ta_4)q_i$.
> <!-- bilingual-en:end -->

> [!warning] 常见错误
> $Q^TQ=I_3$ 不代表 $QQ^T=I_5$；后者是秩 3 投影，正因此行列式为零。
> <!-- bilingual-en:start -->
> $Q^TQ=I_3$ does not represent $QQ^T=I_5$; the latter is a rank 3 projection, so the determinant is zero.
> <!-- bilingual-en:end -->

## Problem 4：同一个参数占据第一行和第一列
<!-- bilingual-en:start -->
*Problem 4: The same parameter occupies the first row and column*
<!-- bilingual-en:end -->

一个 $4\times4$ 矩阵的第一行和第一列全部为参数 $x$，其余 $3\times3$ 块是任意常数。
<!-- bilingual-en:start -->
The first row and first column of a $4\times4$ matrix are all parameters $x$, and the remaining $3\times3$ blocks are arbitrary constants.
<!-- bilingual-en:end -->

### (a) $\det A$ 作为 $x$ 的多项式，最高可能几次？
<!-- bilingual-en:start -->
*(a) $\det A$ as a polynomial of $x$, up to how many times?*
<!-- bilingual-en:end -->

> [!success]- 完整解答
> 行列式大公式的每一项从每行、每列各取一个元素。
>
> - 若一项取左上角 $a_{11}=x$，第一行和第一列都已使用，不能再取其他 $x$，该项只含一次 $x$。
> - 若不取 $a_{11}$，最多可从第一行的某列 $j>1$ 取一个 $x$，再从第一列的某行 $i>1$ 取一个 $x$，共两次。
>
> 所以每项至多含 $x^2$，而适当选择其余元素确实能让二次项不消失。最高可能次数为
> $$\boxed 2.$$
> <!-- bilingual-en:start -->
> Each term in the Leibniz formula selects exactly one entry from every row and every column.
> - If a term selects the upper-left entry $a_{11}=x$, row one and column one are already used, so that term cannot contain another $x$; its degree in $x$ is one.
> - If a term does not select $a_{11}$, it can select at most one $x$ elsewhere in the first row and at most one $x$ elsewhere in the first column, for a total degree of two.
> Hence every term has degree at most two in $x$, and an appropriate choice of the remaining entries makes the quadratic coefficient nonzero. Therefore the maximum possible degree is
> $$\boxed 2.$$
> <!-- bilingual-en:end -->

### (b) 其余 $3\times3$ 块为 $I_3$ 时求行列式与奇异参数
<!-- bilingual-en:start -->
*(b) Determinants and singular parameters when the remaining $3\times3$ blocks are $I_3$*
<!-- bilingual-en:end -->

此时
<!-- bilingual-en:start -->
In this case,
<!-- bilingual-en:end -->

$$
A=\begin{bmatrix}
x&x&x&x\\
x&1&0&0\\
x&0&1&0\\
x&0&0&1
\end{bmatrix}.
$$

> [!success]- 完整解答
> 将第一行分别减去 $x$ 倍第 2、3、4 行；行加法不改变行列式：
> $$
> A\sim
> \begin{bmatrix}
> x-3x^2&0&0&0\\
> x&1&0&0\\
> x&0&1&0\\
> x&0&0&1
> \end{bmatrix}.
> $$
> 这是下三角矩阵，故
> $$
> \det A=(x-3x^2)\cdot1\cdot1\cdot1
> =\boxed{x(1-3x)}.
> $$
> 因而奇异参数为
> $$\boxed{x=0\quad\text{或}\quad x=\frac13}.$$
> 当 $x=0$ 时第一行、列造成零方向；当 $x=1/3$ 时则是非显然的行相关，由上面的消元直接暴露。
> <!-- bilingual-en:start -->
> Subtract $x$ times each of rows two, three, and four from row one. These row additions do not change the determinant:
> The resulting matrix is lower triangular, so
> Therefore the singular parameter values are
> $$\boxed{x=0\quad\text{or}\quad x=\frac13}.$$
> When $x=0$, the first row and column create an immediate null direction. When $x=1/3$, the row dependence is less obvious but is exposed directly by the elimination above.
> <!-- bilingual-en:end -->

> [!warning] 常见错误
> 参数出现七个位置，并不意味着次数可到 7；同一排列项受到“每行、每列只取一次”的严格限制。
> <!-- bilingual-en:start -->
> The seven positions of the argument do not mean that the number of times can be up to seven; the same permutation is strictly limited to "take once per row, once per column."
> <!-- bilingual-en:end -->

---

## Unit II 知识闭环
<!-- bilingual-en:start -->
*Unit II knowledge closed loop*
<!-- bilingual-en:end -->

$$
\text{四子空间正交}
\Longrightarrow
\text{投影与最小二乘}
\Longrightarrow
\text{正交基与 QR},
$$

$$
\text{行列式}
\Longrightarrow
\text{可逆性、cofactor、体积}
\Longrightarrow
\det(A-\lambda I)=0,
$$

$$
\text{特征向量基}
\Longrightarrow
\text{对角化}
\Longrightarrow
A^k,e^{At},\text{Markov 稳态与 Fourier 投影}.
$$

真正需要形成的不是公式清单，而是三种判断：目标是否是最近点；矩阵是否把体积压成零；是否存在一组让变换解耦的方向。
<!-- bilingual-en:start -->
The real goal is not to memorise a list of formulas, but to make three judgements: whether a candidate is the closest point, whether a matrix collapses volume to zero, and whether there is a basis of directions that decouples the transformation.
<!-- bilingual-en:end -->
