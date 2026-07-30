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

## 记号、空间与尺寸约定

- 除非特别说明，向量属于实数域，$A\in\mathbb R^{m\times n}$，$x\in\mathbb R^n$，$b\in\mathbb R^m$。
- $C(A)\subseteq\mathbb R^m$ 是列空间，$C(A^T)\subseteq\mathbb R^n$ 是行空间；$N(A)\subseteq\mathbb R^n$，$N(A^T)\subseteq\mathbb R^m$。
- $x^Ty$ 是欧氏内积，$\|x\|=\sqrt{x^Tx}$。本单元后半出现复向量时，内积必须改为 $x^*y$。
- $I_m$ 表示 $m\times m$ 单位矩阵；尺寸不含混时简写为 $I$。
- $\hat x$ 表示最小二乘解，$p=A\hat x$ 表示投影，$e=b-p$ 表示残差。

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

本节回答三个问题：怎样把“垂直”推广到任意维；四个基本子空间为什么成对正交；为什么 $A^TA$ 可逆恰好等价于 $A$ 的列向量线性无关。

前置知识是四个基本子空间、秩—零度定理以及矩阵乘法尺寸。若 $A\in\mathbb R^{m\times n}$，则 $A^TA\in\mathbb R^{n\times n}$，所以 $N(A)$ 与 $N(A^TA)$ 都是 $\mathbb R^n$ 的子空间，二者才可以比较。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S01_Lecture_Orthogonal_Vectors_and_Subspaces.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S01_Recitation_Problem_Solving_Orthogonal_Vectors_and_Subspaces.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.1prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.1sol.pdf#page=1)

### Lecture：从正交向量到四个基本子空间

两个向量 $x,y\in\mathbb R^n$ 的[[Orthogonality|正交性（orthogonality）]]定义为

$$
x\perp y\quad\Longleftrightarrow\quad x^Ty=0.
$$

它不是二维图形的偶然规则。由余弦公式 $x^Ty=\|x\|\|y\|\cos\theta$，非零向量的内积为零恰好对应 $\theta=90^\circ$。零向量与所有向量正交，但零向量没有可定义的方向或夹角。

若 $x\perp y$，则

$$
\begin{aligned}
\|x+y\|^2
&=(x+y)^T(x+y)\\
&=x^Tx+x^Ty+y^Tx+y^Ty\\
&=\|x\|^2+\|y\|^2.
\end{aligned}
$$

这就是高维勾股定理。后面“投影是最近点”的证明，只是在这个等式中把 $x,y$ 换成两个互相正交的误差分量。

两个子空间 $S,T\subseteq\mathbb R^n$ 正交，是指

$$
s^Tt=0\qquad\text{对所有 }s\in S, t\in T.
$$

要求是“任意一对”，不是只找到一对正交向量。例如三维空间中两个相交平面并不是正交子空间：它们的交线中有非零向量同时属于两者，该向量不可能与自身正交。

子空间 $S$ 的[[Orthogonal Complement|正交补（orthogonal complement）]]定义为

$$
S^\perp=\{x\in\mathbb R^n:x^Ts=0\text{ 对所有 }s\in S\}.
$$

$S^\perp$ 本身是子空间：若 $x,y\in S^\perp$ 且 $\alpha,\beta\in\mathbb R$，则对任意 $s\in S$，

$$
(\alpha x+\beta y)^Ts=\alpha x^Ts+\beta y^Ts=0.
$$

#### 四个基本子空间的两对正交关系

设 $A$ 的行向量为 $r_1^T,\dots,r_m^T$。若 $x\in N(A)$，则

$$
Ax=0\quad\Longrightarrow\quad r_i^Tx=0\quad(i=1,\dots,m).
$$

所以 $x$ 与每一行正交，也与行的任意线性组合正交：

$$
C(A^T)\perp N(A)\qquad(\text{都位于 }\mathbb R^n).
$$

把同一论证用于 $A^T$ 得

$$
C(A)\perp N(A^T)\qquad(\text{都位于 }\mathbb R^m).
$$

若 $\operatorname{rank}(A)=r$，则

$$
\dim C(A^T)=r,\quad \dim N(A)=n-r,
$$

两者维数相加为 $n$。因此它们不仅正交，而且互为正交补：

$$
C(A^T)^\perp=N(A),\qquad N(A)^\perp=C(A^T).
$$

把上面的逐行论证应用于 $A^T$，并使用 $\dim C(A)+\dim N(A^T)=r+(m-r)=m$，可得 $C(A)$ 与 $N(A^T)$ 在 $\mathbb R^m$ 中互为正交补。由此每个 $v\in\mathbb R^n$ 都能唯一写为

$$
v=r+n,\qquad r\in C(A^T),\quad n\in N(A).
$$

唯一性证明：若 $v=r_1+n_1=r_2+n_2$，则 $r_1-r_2=n_2-n_1$ 同时属于两个正交补；它与自身正交，只能是零向量，故 $r_1=r_2,n_1=n_2$。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-orthogonal-complements.png|760]]

#### 为什么 $N(A^TA)=N(A)$

> [!note] 定理
> 对任意实矩阵 $A\in\mathbb R^{m\times n}$，
> $$N(A^TA)=N(A).$$
> 因而 $\operatorname{rank}(A^TA)=\operatorname{rank}(A)$；$A^TA$ 可逆当且仅当 $A$ 满列秩。

**第一方向。** 若 $Ax=0$，左乘 $A^T$ 得 $A^TAx=0$，所以 $N(A)\subseteq N(A^TA)$。

**反方向。** 若 $A^TAx=0$，左乘 $x^T$：

$$
0=x^TA^TAx=(Ax)^T(Ax)=\|Ax\|^2.
$$

平方范数只有在向量为零时才为零，故 $Ax=0$，于是 $N(A^TA)\subseteq N(A)$。两边合并即得结论。

因为 $A^TA$ 是 $n\times n$ 方阵，它可逆等价于 $N(A^TA)=\{0\}$；由上式又等价于 $N(A)=\{0\}$，即 $A$ 的 $n$ 个列向量线性无关。这里不要求 $A$ 是方阵，只要求列数不超过可独立的维数，即必有 $n\le m$。

### Recitation：求 $S^\perp$ 并证明正交分解唯一

令

$$
S=\operatorname{span}\left\{
\begin{bmatrix}1\\2\\2\\3\end{bmatrix},
\begin{bmatrix}1\\3\\3\\2\end{bmatrix}
\right\}\subseteq\mathbb R^4.
$$

$x\in S^\perp$ 当且仅当它同时与两个生成向量正交，所以

$$
\begin{bmatrix}
1&2&2&3\\
1&3&3&2
\end{bmatrix}x=0.
$$

第二行减第一行，取 $x_3=a,x_4=b$：

$$
x_2=-a+b,\qquad x_1=-5b.
$$

因此

$$
x=a\begin{bmatrix}0\\-1\\1\\0\end{bmatrix}
+b\begin{bmatrix}-5\\1\\0\\1\end{bmatrix},
$$

故括号中的两个向量构成 $S^\perp$ 的一组基。原来两个生成向量线性无关，$\dim S=2$；现在 $\dim S^\perp=2$，两组基合起来给出 $\mathbb R^4$ 的四个线性无关向量，所以每个 $v\in\mathbb R^4$ 都有唯一的 $v=s+s_\perp$ 分解。

### Homework

> [!question]- Problem 16.1：用方程的线性组合得到矛盾 $0=1$
> 方程为 $x_1-x_2=1$、$x_2-x_3=1$、$x_1-x_3=1$。求 $y_1,y_2,y_3$，使三式加权后左边为 $0$、右边为 $1$。
>
> **解。** 取 $y=(1,1,-1)^T$。左边为
> $$
> (x_1-x_2)+(x_2-x_3)-(x_1-x_3)=0,
> $$
> 右边为 $1+1-1=1$。矩阵语言中，这说明 $A^Ty=0$ 而 $y^Tb=1$。若 $Ax=b$ 有解，则应有 $y^TAx=(A^Ty)^Tx=0$，但 $y^Tb=1$，矛盾；因此原系统无解。这正是 Fredholm alternative 在有限维情形的证书。

> [!question]- Problem 16.2：给定四条一维子空间，构造秩一矩阵
> 给定非零 $r,n,c,\ell\in\mathbb R^2$，希望它们分别成为 $C(A^T),N(A),C(A),N(A^T)$ 的基。条件是什么？给出一个 $A$。
>
> **解。** 必要条件是
> $$r^Tn=0,\qquad c^T\ell=0.$$
> 四个向量都非零，故四个空间都为一维，维数条件自动满足。取
> $$A=cr^T.$$
> $A$ 的每一列都是 $c$ 的倍数，所以 $C(A)=\operatorname{span}(c)$；每一行都是 $r^T$ 的倍数，所以 $C(A^T)=\operatorname{span}(r)$。又因零空间是对应行/列空间的正交补，便得到指定的 $n$ 与 $\ell$。任意非零标量倍 $\alpha cr^T$ 也可行。

### 边界、反例与易错点

- $S\perp T$ 强于 $S\cap T=\{0\}$；两条不垂直但不同的直线交集也是 $\{0\}$。
- 不可把 $C(A)\subseteq\mathbb R^m$ 与 $N(A)\subseteq\mathbb R^n$ 直接称为正交，除非 $m=n$ 且另有说明。
- $A^TA$ 总是对称，但未必可逆；公式 $(A^TA)^{-1}$ 只有在 $A$ **满列秩**时才存在。
- 从 $x^TA^TAx=0$ 推出 $Ax=0$ 用到了实数欧氏内积的正定性；复数情形应写 $x^*A^*Ax=\|Ax\|^2$。

### 三道自检

> [!question]- 1. 若 $A\in\mathbb R^{7\times4}$ 且 $\operatorname{rank}(A)=3$，四个基本子空间的维数分别是多少？
> $\dim C(A)=\dim C(A^T)=3$，$\dim N(A)=4-3=1$，$\dim N(A^T)=7-3=4$。

> [!question]- 2. 证明 $C(A^T)\cap N(A)=\{0\}$。
> 设 $z$ 同时属于两空间。因为二者正交，$z^Tz=0$，所以 $z=0$。

> [!question]- 3. 若 $A\in\mathbb R^{3\times5}$，$A^TA$ 能否可逆？
> 不能。五个列向量位于 $\mathbb R^3$，不可能线性无关，所以 $N(A)\ne\{0\}$；由 $N(A^TA)=N(A)$，$A^TA$ 奇异。

### 知识链

[[Column Space|列空间]]与[[Null Space|零空间]] → 正交补 → $N(A^TA)=N(A)$ → 下一节的[[Orthogonal Projection|正交投影]]。

---

## Session 2.2 Projections onto subspaces

### 本节问题、前置知识与尺寸

给定 $b\in\mathbb R^m$ 和子空间 $S\subseteq\mathbb R^m$，怎样严格找出 $S$ 中离 $b$ 最近的向量？若 $S=C(A)$ 且 $A\in\mathbb R^{m\times n}$ 满列秩，投影系数 $\hat x\in\mathbb R^n$，投影 $p=A\hat x\in\mathbb R^m$，残差 $e=b-p\in\mathbb R^m$。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S02_Lecture_Lecture_15_Projections_onto_Subspaces.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S02_Recitation_Problem_Solving_Projection_onto_Subspaces.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.2prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.2sol.pdf#page=1)

### Lecture：最近点由“残差正交”唯一决定

先投影到直线 $S=\operatorname{span}(a)$，其中 $a\ne0$。因为 $p\in S$，写成 $p=\hat xa$。正交投影要求误差 $e=b-p$ 与 $a$ 垂直：

$$
a^T(b-\hat xa)=0.
$$

解得

$$
\hat x=\frac{a^Tb}{a^Ta},\qquad
p=a\frac{a^Tb}{a^Ta}.
$$

于是投影矩阵为

$$
P=\frac{aa^T}{a^Ta}\in\mathbb R^{m\times m},\qquad p=Pb.
$$

分母 $a^Ta$ 是标量，分子 $aa^T$ 是 $m\times m$ 矩阵；把两者误写成 $a^Ta/(aa^T)$ 是尺寸错误。

#### 为什么正交条件保证最近

设 $p$ 满足 $p\in S$ 且 $e=b-p\perp S$。对任意 $s\in S$，

$$
b-s=(b-p)+(p-s)=e+(p-s).
$$

因为 $p-s\in S$ 而 $e\perp S$，勾股定理给出

$$
\|b-s\|^2=\|e\|^2+\|p-s\|^2\ge\|e\|^2=\|b-p\|^2.
$$

等号要求 $p-s=0$，即 $s=p$。因此 $p$ 不只是局部极小，而是唯一的全局最近点。

#### 投影到列空间

令 $S=C(A)$，所以 $p=A\hat x$。要求残差与 $A$ 的每一列正交：

$$
A^T(b-A\hat x)=0.
$$

整理得**正规方程（normal equations）**

$$
A^TA\hat x=A^Tb.
$$

若 $A$ 满列秩，$A^TA$ 可逆，于是

$$
\hat x=(A^TA)^{-1}A^Tb,\qquad
p=A(A^TA)^{-1}A^Tb.
$$

因此投影到 $C(A)$ 的[[Projection Matrix|投影矩阵（projection matrix）]]是

$$
P=A(A^TA)^{-1}A^T.
$$

逐项检查尺寸：$A^T b\in\mathbb R^n$，$(A^TA)^{-1}\in\mathbb R^{n\times n}$，最终 $Pb\in\mathbb R^m$。

#### 投影矩阵的结构

在满列秩条件下，

$$
P^T=P,
$$

因为 $A^TA$ 对称，其逆也对称；并且

$$
\begin{aligned}
P^2
&=A(A^TA)^{-1}A^TA(A^TA)^{-1}A^T\\
&=A(A^TA)^{-1}A^T=P.
\end{aligned}
$$

所以正交投影矩阵同时满足**对称**与**幂等**。还有

$$
C(P)=C(A),\qquad N(P)=N(A^T).
$$

第一式因为 $Pb$ 总在 $C(A)$ 中，且对 $p\in C(A)$ 有 $Pp=p$；第二式因为 $Pb=0$ 正好表示 $b$ 完全位于 $C(A)^\perp=N(A^T)$。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-projection.png|760]]

### Recitation：投影到平面 $x+y-z=0$

平面的法向量是 $n=(1,1,-1)^T$。可以选平面基

$$
a_1=(1,-1,0)^T,\qquad a_2=(1,0,1)^T,\qquad A=[a_1\ a_2],
$$

再代入 $A(A^TA)^{-1}A^T$。更短的办法是先投影到法线，再用互补投影：

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

### 边界、反例与易错点

- $(A^TA)^{-1}$ 只在 $A$ 满列秩时存在；列相关时应删去冗余列、换一组基，或在 Unit III 使用伪逆。
- $P^2=P$ 只说明“投影”意义上的幂等；若还要是**正交**投影，必须有 $P^T=P$。斜投影可以幂等但不对称。
- 一般不能把 $(A^TA)^{-1}$ 拆为 $A^{-1}(A^T)^{-1}$，因为长方矩阵 $A$ 没有双侧逆。
- 投影到同一个子空间的矩阵与所选基无关；系数 $\hat x$ 会随基改变，几何向量 $p$ 不变。

### 三道自检

> [!question]- 1. 把 $b=(3,1)^T$ 投影到 $a=(1,2)^T$ 张成的直线。
> $a^Tb=5,a^Ta=5$，故 $\hat x=1$，$p=(1,2)^T$，$e=(2,-1)^T$，并且 $a^Te=0$。

> [!question]- 2. 若 $P=P^T=P^2$，证明 $b=Pb+(I-P)b$ 是正交分解。
> 两项之和是 $b$；内积为
> $$
> (Pb)^T(I-P)b=b^TP(I-P)b=b^T(P-P^2)b=0.
> $$

> [!question]- 3. $A=[a\ 2a]$ 能否直接使用 $A(A^TA)^{-1}A^T$？怎样修复？
> 不能，两列相关使 $A^TA$ 奇异。删去第二列，用非零列 $a$ 作为同一列空间的一组基，再用 $aa^T/(a^Ta)$。

### 知识链

正交补 → [[Orthogonal Projection|正交投影]] → 正规方程 → 下一节的[[Least Squares|最小二乘]]与残差分析。

---

## Session 2.3 Projection matrices and least squares

### 本节问题、前置知识与尺寸

当 $Ax=b$ 因 $b\notin C(A)$ 而无解时，我们不伪造精确解，而是在所有 $Ax$ 中寻找离 $b$ 最近的一个。本节把这个几何问题写成[[Least Squares|最小二乘（least squares）]]，并解释正规方程的来源、解的唯一性条件和残差所属的子空间。

仍设 $A\in\mathbb R^{m\times n}$、$b\in\mathbb R^m$。若 $A$ 满列秩，$\hat x\in\mathbb R^n$ 唯一；无论坐标是否唯一，最佳拟合向量 $p=A\hat x\in C(A)$ 都是唯一的正交投影。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.3sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S03_Lecture_Lecture_16_Projection_Matrices_and_Least_Squares.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S03_Recitation_Problem_Solving_Least_Squares_Approximation.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.3prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.3sol.pdf#page=1)

### Lecture：把“拟合”翻译成列空间投影

最小二乘问题是

$$
\min_{x\in\mathbb R^n}\|Ax-b\|^2.
$$

$Ax$ 只能在 $C(A)$ 中移动，因此问题等价于：求 $b$ 在 $C(A)$ 上的投影 $p$，再解 $A\hat x=p$。最佳残差

$$
e=b-A\hat x
$$

必须属于 $C(A)^\perp=N(A^T)$，所以

$$
A^Te=0
\quad\Longleftrightarrow\quad
A^T(b-A\hat x)=0
\quad\Longleftrightarrow\quad
A^TA\hat x=A^Tb.
$$

这就是[[Least Squares via Normal Equations|用正规方程求最小二乘解]]。它不是对 $Ax=b$ 随意左乘 $A^T$，而是把“残差与全部列正交”同时写成一组方程。

若希望用微积分核对，令

$$
F(x)=\|Ax-b\|^2=(Ax-b)^T(Ax-b).
$$

展开为

$$
F(x)=x^TA^TAx-2x^TA^Tb+b^Tb.
$$

因为 $A^TA$ 对称，

$$
\nabla F(x)=2A^TAx-2A^Tb.
$$

令梯度为零同样得到正规方程。几何证明更强地说明这是全局最小：对任意 $x$，写

$$
b-Ax=(b-p)+(p-Ax)=e+A(\hat x-x),
$$

两项正交，因而

$$
\|b-Ax\|^2=\|e\|^2+\|A(\hat x-x)\|^2\ge\|e\|^2.
$$

#### 唯一性究竟在哪里

- 投影 $p$ 总是唯一，因为子空间中的最近点唯一。
- 若 $A$ 满列秩，则 $A^TA$ 正定且可逆，$\hat x$ 唯一：
  $$
  \hat x=(A^TA)^{-1}A^Tb.
  $$
- 若 $A$ 列相关，则可能有多个 $x$ 给出同一 $p$；任何两个最小二乘解之差都在 $N(A)$ 中。此时不能写 $(A^TA)^{-1}$。

投影矩阵与互补投影为

$$
P=A(A^TA)^{-1}A^T,\qquad I-P,
$$

并且

$$
p=Pb\in C(A),\qquad e=(I-P)b\in N(A^T),\qquad p^Te=0.
$$

### 课堂例题：三点的最佳拟合直线

拟合 $(1,1),(2,2),(3,2)$，设直线 $y=C+Dt$。则

$$
A=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\quad
x=\begin{bmatrix}C\\D\end{bmatrix},\quad
b=\begin{bmatrix}1\\2\\2\end{bmatrix}.
$$

原系统三式两未知，一般不相容。正规方程是

$$
\underbrace{\begin{bmatrix}3&6\\6&14\end{bmatrix}}_{A^TA}
\begin{bmatrix}\hat C\\\hat D\end{bmatrix}
=
\underbrace{\begin{bmatrix}5\\11\end{bmatrix}}_{A^Tb}.
$$

第一式乘 $2$ 后从第二式相减：$2\hat D=1$，所以 $\hat D=1/2$；代回得 $\hat C=2/3$。于是

$$
p=A\hat x=
\begin{bmatrix}7/6\\5/3\\13/6\end{bmatrix},\qquad
e=b-p=
\begin{bmatrix}-1/6\\1/3\\-1/6\end{bmatrix}.
$$

直接验算

$$
A^Te=
\begin{bmatrix}
-1/6+1/3-1/6\\
-1/6+2/3-3/6
\end{bmatrix}=0.
$$

第一行说明残差总和为零，第二行说明“时间加权残差”也为零；这两个条件分别来自常数列和时间列。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-least-squares-fit.png|780]]

### Recitation：过原点的最佳二次曲线，并修正转录算术

对点 $(1,1),(2,5),(-1,-2)$ 拟合 $y=ct+dt^2$。设计矩阵为

$$
A=\begin{bmatrix}1&1\\2&4\\-1&1\end{bmatrix},\qquad
b=\begin{bmatrix}1\\5\\-2\end{bmatrix}.
$$

按定义逐项计算：

$$
A^TA=\begin{bmatrix}6&8\\8&18\end{bmatrix},\qquad
A^Tb=\begin{bmatrix}13\\19\end{bmatrix}.
$$

因此

$$
\begin{bmatrix}6&8\\8&18\end{bmatrix}
\begin{bmatrix}\hat c\\\hat d\end{bmatrix}
=\begin{bmatrix}13\\19\end{bmatrix}.
$$

消元得 $22\hat d=5$，故 $\hat d=5/22$；再由 $6\hat c+8\hat d=13$ 得 $\hat c=41/22$。最佳拟合为

$$
y=\frac{41}{22}t+\frac5{22}t^2.
$$

> [!warning] 原 transcript 的算术错误
> 转录稿在这里把 $(A^TA)_{22}=1^2+4^2+1^2$ 记成了 $10$，并给出不满足正规方程的 $c=11/2,d=-5/2$。按题面数据正确值必须是 $18$，上面的解可由 $A^T(b-A\hat x)=0$ 直接验算。笔记保留课程思路，但不沿用这一算术错误。

### Homework

以下六题共用

$$
A=\begin{bmatrix}1&-1\\1&1\\1&2\end{bmatrix}
$$

（最后一题另建五行设计矩阵）。

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

> [!question]- Problem 16.2：求投影与残差，并解释 $Pe=0$
> **解。**
> $$
> p=A\hat x=\begin{bmatrix}5\\13\\17\end{bmatrix},\qquad
> e=b-p=\begin{bmatrix}2\\-6\\4\end{bmatrix}.
> $$
> 验算 $A^Te=(2-6+4,-2-6+8)^T=0$，故 $e\in N(A^T)=C(A)^\perp$。投影到 $C(A)$ 后为零，即 $Pe=0$；也可写成 $Pe=P(b-p)=Pb-Pp=p-p=0$。

> [!question]- Problem 16.3：把上一题的误差本身当成新数据
> **解。** 新 $b=e=(2,-6,4)^T$ 满足 $A^Tb=0$，正规方程右端为零。因 $A$ 满列秩，唯一解为 $\hat x=0$，最近直线是零函数，$p=0$。原因是 $b$ 已经垂直于整个 $C(A)$。

> [!question]- Problem 16.4：把上一题的投影本身当成数据
> **解。** $b=(5,13,17)^T=A(9,4)^T\in C(A)$，所以精确可解：$\hat x=(9,4)^T$，$p=b$，$e=0$。

> [!question]- Problem 16.5：$e,p,\hat x$ 分别属于哪个基本子空间？
> **解。** $e\in N(A^T)\subseteq\mathbb R^3$；$p\in C(A)\subseteq\mathbb R^3$；$\hat x\in\mathbb R^2=C(A^T)$，因为 $A$ 的秩为 $2$、行空间填满 $\mathbb R^2$。两列独立，故 $N(A)=\{0\}$。

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

### 边界、反例与易错点

- “least squares” 最小化的是残差平方和 $\sum e_i^2$，不是 $\sum e_i$；后者可由正负抵消。
- 拟合直线的几何图在 $(t,b)$ 平面，而投影 $b\mapsto p$ 发生在数据空间 $\mathbb R^m$；两种图景不能混成同一个空间。
- 离群点会因平方而获得更大权重，普通最小二乘并不稳健。
- 正规方程会把条件数近似平方；数值计算常优先用 QR，而不是显式形成 $A^TA$。

### 三道自检

> [!question]- 1. 若设计矩阵含常数列，为什么最小二乘残差的分量和为零？
> 常数列是 $\mathbf1$，正规方程中的正交条件给出 $\mathbf1^Te=\sum_i e_i=0$。

> [!question]- 2. 若 $b\in C(A)$ 且 $A$ 满列秩，最小二乘解是什么性质？
> 原方程精确可解且解唯一；$p=b,e=0$，最小残差为零。

> [!question]- 3. 两个最小二乘解 $x_1,x_2$ 为什么可能不同却有相同预测？
> 因最佳投影唯一，所以 $Ax_1=Ax_2=p$，从而 $x_1-x_2\in N(A)$。只有 $N(A)=\{0\}$ 时系数才唯一。

### 知识链

投影 → 正规方程 → 残差 $N(A^T)$ → 下一节用[[Gram-Schmidt Orthogonalization|Gram–Schmidt]]和 QR 更稳定地求投影系数。

---

## Session 2.4 Orthogonal matrices and Gram–Schmidt

### 本节问题、前置知识与尺寸

怎样把任意一组线性无关向量变成张成同一子空间的标准正交基？为什么在标准正交坐标中，投影、最小二乘和坐标恢复都会变简单？

设 $A=[a_1\ \cdots\ a_n]\in\mathbb R^{m\times n}$ 且列独立（$m\ge n$）。Gram–Schmidt 产生 $Q=[q_1\ \cdots\ q_n]\in\mathbb R^{m\times n}$，满足 $Q^TQ=I_n$，以及上三角 $R\in\mathbb R^{n\times n}$，使 $A=QR$。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S04_Lecture_Lecture_17_Orthogonal_Matrices_and_Gram_Schmidt.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S04_Recitation_Problem_Solving_Gram_Schmidt_Orthogonalization.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.4prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.4sol.pdf#page=1)

### Lecture：标准正交列为何特别方便

向量 $q_1,\dots,q_n$ **标准正交（orthonormal）**，是指

$$
q_i^Tq_j=\delta_{ij}=
\begin{cases}1,&i=j,\\0,&i\ne j.\end{cases}
$$

把它们作为列组成 $Q$，上式一次写成

$$
Q^TQ=I_n.
$$

若 $Q$ 是方阵，则称为[[Orthogonal Matrix|正交矩阵（orthogonal matrix）]]，且

$$
Q^{-1}=Q^T,\qquad QQ^T=Q^TQ=I.
$$

若 $Q$ 是高矩阵（$m>n$），只有 $Q^TQ=I_n$；此时

$$
QQ^T=P_{C(Q)}\ne I_m
$$

一般是秩为 $n$ 的投影矩阵。

正交矩阵保持内积和长度：

$$
(Qx)^T(Qy)=x^TQ^TQy=x^Ty,\qquad \|Qx\|=\|x\|.
$$

因此它只执行旋转、反射或这些变换的组合，不拉伸长度。

#### 标准正交向量必线性无关

若 $Qx=0$，左乘 $Q^T$：

$$
Q^TQx=Q^T0\quad\Longrightarrow\quad I_nx=0\quad\Longrightarrow\quad x=0.
$$

所以 $N(Q)=\{0\}$，列向量线性无关。注意 $Q$ 可以是长方矩阵，不能在第一步写 $Q^{-1}$。

#### Gram–Schmidt 的逐步构造

先取

$$
u_1=a_1,\qquad q_1=\frac{u_1}{\|u_1\|}.
$$

第二个向量减去在 $q_1$ 上的投影：

$$
u_2=a_2-(q_1^Ta_2)q_1,\qquad q_2=\frac{u_2}{\|u_2\|}.
$$

一般地，

$$
u_k=a_k-\sum_{j=1}^{k-1}(q_j^Ta_k)q_j,\qquad
q_k=\frac{u_k}{\|u_k\|}.
$$

验证 $u_k\perp q_i$（$i<k$）：

$$
q_i^Tu_k=q_i^Ta_k-\sum_{j<k}(q_j^Ta_k)q_i^Tq_j
=q_i^Ta_k-q_i^Ta_k=0.
$$

每一步只从 $a_k$ 中减去先前向量的线性组合，所以

$$
\operatorname{span}(q_1,\dots,q_k)
=\operatorname{span}(a_1,\dots,a_k).
$$

列独立保证 $u_k\ne0$；若某一步 $u_k=0$，正说明 $a_k$ 已落入前面列的张成空间。

#### 从 Gram–Schmidt 到 [[QR Decomposition|QR 分解]]

每个 $a_j$ 都能写成 $q_1,\dots,q_j$ 的组合：

$$
a_j=\sum_{i=1}^j r_{ij}q_i,\qquad r_{ij}=q_i^Ta_j.
$$

把这些系数排成上三角矩阵 $R$，便有

$$
A=QR,\qquad R=Q^TA.
$$

若约定 $r_{jj}=\|u_j\|>0$，满列秩矩阵的薄 QR 分解唯一。投影矩阵立即简化为

$$
P=QQ^T,
$$

最小二乘正规方程也可化为

$$
QR\hat x\approx b
\quad\Longrightarrow\quad
R\hat x=Q^Tb,
$$

只需回代上三角系统，避免显式形成 $A^TA$。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-gram-schmidt.png|760]]

### Recitation：完整 QR 计算

给定

$$
A=\begin{bmatrix}
1&2&4\\
0&0&5\\
0&3&6
\end{bmatrix}
=[a\ b\ c].
$$

第一列 $a=(1,0,0)^T$ 已是单位向量，所以 $q_1=(1,0,0)^T$。接着

$$
u_2=b-(q_1^Tb)q_1=(2,0,3)^T-2(1,0,0)^T=(0,0,3)^T,
$$

故 $q_2=(0,0,1)^T$。再算

$$
\begin{aligned}
u_3
&=c-(q_1^Tc)q_1-(q_2^Tc)q_2\\
&=(4,5,6)^T-4(1,0,0)^T-6(0,0,1)^T\\
&=(0,5,0)^T,
\end{aligned}
$$

所以 $q_3=(0,1,0)^T$。于是

$$
Q=\begin{bmatrix}1&0&0\\0&0&1\\0&1&0\end{bmatrix},\qquad
R=Q^TA=\begin{bmatrix}1&2&4\\0&3&6\\0&0&5\end{bmatrix},\qquad A=QR.
$$

$R$ 的第 $j$ 列正是 $a_j$ 在标准正交基 $q_i$ 下的坐标。

### Homework

> [!question]- Problem 17.1：用矩阵证明标准正交列自动独立
> **解。** 设 $Q$ 的列标准正交，所以 $Q^TQ=I$。若 $Qx=0$，则
> $$x=Ix=Q^TQx=Q^T0=0.$$
> 因齐次方程只有零解，$Q$ 的列线性无关。$Q$ 未必为方阵，因此不可假设 $Q^{-1}$ 存在。

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

### 边界、反例与易错点

- “orthonormal matrix” 常泛指列标准正交的长方矩阵；只有方阵才有 $Q^{-1}=Q^T$。
- Gram–Schmidt 必须减去在**所有已得到的 $q_j$** 上的投影，不能直接用尚未正交的 $a_j$ 代替。
- 原始 Gram–Schmidt 在列几乎相关时有数值误差；实际计算常用 modified Gram–Schmidt 或 Householder QR。
- $QR$ 中 $R$ 上三角来自“第 $j$ 列只需要前 $j$ 个 $q_i$”，不是事后巧合。

### 三道自检

> [!question]- 1. $Q\in\mathbb R^{5\times3}$ 且 $Q^TQ=I_3$。$QQ^T$ 的尺寸、秩与行列式分别是什么？
> $QQ^T$ 是 $5\times5$，秩为 $3$，是投影到 $C(Q)$ 的矩阵；因秩小于 $5$，行列式为 $0$。

> [!question]- 2. 对 $a_1=(1,0)^T,a_2=(1,1)^T$ 做 Gram–Schmidt。
> $q_1=(1,0)^T$；$u_2=a_2-(q_1^Ta_2)q_1=(0,1)^T$，故 $q_2=(0,1)^T$。

> [!question]- 3. 若 $A=QR$ 是薄 QR 且 $A$ 满列秩，为什么 $R$ 可逆？
> $R=Q^TA$ 是 $n\times n$ 上三角；其对角元是每一步非零残量的长度 $\|u_j\|>0$，所以所有主元非零，$R$ 可逆。

### 知识链

正交投影 → 标准正交基 → [[Gram-Schmidt Orthogonalization|Gram–Schmidt 正交化]] → QR → 更稳定的最小二乘。

---

## Session 2.5 Properties of determinants

### 本节问题、前置知识与尺寸

本节不把行列式当成待背的展开式，而是从三条基本性质推出所有计算规则。[[Determinant|行列式（determinant）]]只对方阵 $A\in\mathbb R^{n\times n}$ 定义，输出一个标量 $\det A$。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.5sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S05_Lecture_Lecture_18_Properties_of_Determinants.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S05_Recitation_Problem_Solving_Properties_of_Determinants.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.5prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.5sol.pdf#page=1)

### Lecture：三条公理推出整套规则

行列式由以下三条性质唯一确定：

1. $\det I=1$。
2. 交换两行，行列式变号。
3. 固定其他行时，行列式对某一行是线性的。例如
   $$
   \det\begin{bmatrix}\alpha u+\beta v\\\text{其余各行}\end{bmatrix}
   =\alpha\det\begin{bmatrix}u\\\text{其余各行}\end{bmatrix}
   +\beta\det\begin{bmatrix}v\\\text{其余各行}\end{bmatrix}.
   $$

#### 推论 1：有两行相同则行列式为零

若 $D$ 有两行相同，交换这两行后矩阵不变；但性质 2 说行列式变成 $-D$。因此 $D=-D$，在实数域中 $D=0$。

#### 推论 2：一行加上另一行的倍数不改变行列式

把第 $i$ 行 $r_i$ 换成 $r_i+cr_j$。按线性性，新行列式等于原行列式，加上 $c$ 乘以一个第 $i,j$ 行相同的行列式；后者为零，所以值不变。

#### 推论 3：零行、相关行与奇异矩阵

有零行时，从该行提出标量 $0$，得行列式为零。若行向量线性相关，可通过不改变行列式的消元产生零行，所以 $\det A=0$。反过来，若消元有 $n$ 个非零主元，便可还原 $A$，所以 $A$ 可逆且行列式非零。于是

$$
\det A\ne0
\Longleftrightarrow A\text{ 可逆}
\Longleftrightarrow \operatorname{rank}(A)=n
\Longleftrightarrow N(A)=\{0\}.
$$

#### 推论 4：三角矩阵的行列式是对角线乘积

对上三角矩阵，从左上角开始利用行线性和零元素，或沿用消元规则，得到

$$
\det U=u_{11}u_{22}\cdots u_{nn}.
$$

若 $A$ 通过不换行的消元得到 $U$，则 $\det A=\det U$；每换一次行额外乘 $-1$。若某行放大 $c$ 倍，行列式也放大 $c$ 倍。

#### 乘法与转置

重要结论为

$$
\det(AB)=\det A\det B,\qquad \det(A^T)=\det A.
$$

第一式可从“左乘初等矩阵对应一次行操作”证明：每类初等矩阵对行列式的影响与乘法相容，任意可逆 $A$ 都是初等矩阵的乘积；奇异情形两边同为零。由 $AA^{-1}=I$ 还得

$$
\det(A^{-1})=\frac1{\det A}.
$$

对标量 $c$，$cA$ 是把 **每一行** 放大 $c$，所以

$$
\det(cA)=c^n\det A,
$$

不是一般的 $c\det A$。

### Recitation：先识别结构，再决定算法

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

### Homework

> [!question]- Problem 18.1：行和揭示特征值与奇异性
> 若方阵 $A$ 每一行元素和为 $0$，证明 $\det A=0$。若每行和为 $1$，证明 $\det(A-I)=0$；这是否推出 $\det A=1$？
>
> **解。** 令 $\mathbf1=(1,\dots,1)^T\ne0$。行和为零意味着 $A\mathbf1=0$，所以 $A$ 有非零零空间，$\det A=0$。行和为一意味着 $A\mathbf1=\mathbf1$，故 $(A-I)\mathbf1=0$，所以 $\det(A-I)=0$；这只说明 $1$ 是 $A$ 的特征值，不说明其他特征值乘积为 $1$。反例
> $$
> A=\begin{bmatrix}0&1\\1&0\end{bmatrix}
> $$
> 每行和为 $1$，但 $\det A=-1$。

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

### 边界、反例与易错点

- 行列式不是逐元素线性，也不是对整个矩阵线性：通常 $\det(A+B)\ne\det A+\det B$。
- 行加法不改变行列式；行交换变号；行缩放会同比缩放。三种操作不要混记。
- $\det(AB)=\det A\det B$，但一般 $\det(A+B)$ 没有类似公式。
- $\det A=0$ 只告诉你至少一个方向被压扁；它不告诉你秩具体缺几维。

### 三道自检

> [!question]- 1. 若 $A$ 是 $4\times4$ 且 $\det A=3$，求 $\det(2A)$。
> $\det(2A)=2^4\det A=48$。

> [!question]- 2. 消元中交换两次行并把一行乘以 $5$，最后上三角对角为 $2,3,4$。原矩阵行列式是多少？
> 两次交换符号抵消；缩放后的矩阵行列式为 $2\cdot3\cdot4=24$，它是原值的 $5$ 倍，所以原值为 $24/5$。

> [!question]- 3. 奇数阶斜对称矩阵为什么一定不可逆？
> 因 $\det A=\det A^T=\det(-A)=(-1)^n\det A=-\det A$，故行列式为零。

### 知识链

消元与可逆性 → [[Determinant|行列式]]三公理 → 大公式与余子式 → 下一节的可计算公式。

---

## Session 2.6 Determinant formulas and cofactors

### 本节问题、前置知识与尺寸

三条性质说明“行列式应怎样变化”，本节把它们变成两个通用计算公式：含 $n!$ 项的排列公式，以及沿任意一行或一列展开的余子式公式。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S06_Lecture_Lecture_19_Determinant_Formulas_and_Cofactors.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S06_Recitation_Problem_Solving_Determinants.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.6prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.6sol.pdf#page=1)

### Lecture：排列大公式

从每一行取一个元素，并要求所取元素来自不同列。列指标 $(\sigma(1),\dots,\sigma(n))$ 必须是 $1,\dots,n$ 的一个排列。于是

$$
\det A=
\sum_{\sigma\in S_n}\operatorname{sgn}(\sigma)
\prod_{i=1}^n a_{i,\sigma(i)}.
$$

$S_n$ 是全部 $n!$ 个排列；$\operatorname{sgn}(\sigma)=+1$ 表示偶排列，$-1$ 表示奇排列。符号可由“恢复自然顺序所需交换次数”的奇偶性判断。

为什么每项不能在两行选同一列？若同一列重复，就会遗漏另一列；这样的项不能满足交换两行变号的交替性。也可以从行线性展开看出，只有每列恰好选一次的项能保留下来。

大公式适合稀疏且非零项很少的矩阵，但一般计算量为 $n!$，远逊于消元的约 $O(n^3)$。

### [[Cofactor Expansion|余子式展开]]与代数余子式

删去第 $i$ 行、第 $j$ 列所得的 $(n-1)\times(n-1)$ 矩阵记为 $M_{ij}$。对应的**代数余子式（cofactor）**是

$$
C_{ij}=(-1)^{i+j}\det M_{ij}.
$$

符号棋盘为

$$
\begin{bmatrix}
+&-&+&\cdots\\
-&+&-&\cdots\\
+&-&+&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{bmatrix}.
$$

把排列大公式按第一行所选列分组，就得到第一行展开：

$$
\det A=a_{11}C_{11}+a_{12}C_{12}+\cdots+a_{1n}C_{1n}.
$$

若改为按“第 $i$ 行选中了哪一列”给排列项分组，就得到第 $i$ 行展开；按“第 $j$ 列选中了哪一行”分组，则得到第 $j$ 列展开：

$$
\det A=\sum_{k=1}^n a_{ik}C_{ik}
=\sum_{k=1}^n a_{kj}C_{kj}.
$$

实际选择零最多的一行或一列，可显著减少子式数量。

### Recitation：按矩阵结构混合三种方法

第一类 $5\times5$ 稀疏循环矩阵在对角线上为 $x$，循环邻位为 $y$。沿第一列展开，只剩两个三角子式，得到

$$
\det A=x^5+y^5.
$$

第二类矩阵 $B$ 的对角线全为 $x$、非对角线全为 $y$。用相邻行相减，再累加列，可化为对角线包含 $x+4y$ 与四个 $x-y$ 的三角矩阵：

$$
\det B=(x+4y)(x-y)^4.
$$

这也可从特征方向预见：$\mathbf1$ 的特征值是 $x+4y$，其正交补上的特征值都是 $x-y$。此处特征值将在 Session 2.8 正式建立。

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

> [!question]- Problem 19.2：Pascal 矩阵末项减一为何使行列式从 $1$ 变成 $0$
> 已知 $n\times n$ 对称 Pascal 矩阵行列式为 $1$，其左上 $(n-1)\times(n-1)$ Pascal 子矩阵行列式也为 $1$。
>
> **解。** 只改变 $a_{nn}$，且行列式对该元素线性。$a_{nn}$ 的 cofactor 为
> $$
> C_{nn}=(-1)^{2n}\det M_{nn}=1\cdot1=1.
> $$
> 将 $a_{nn}$ 减 $1$ 会令整体行列式减少 $1\cdot C_{nn}=1$，故从 $1$ 变成 $0$。

### 边界、反例与易错点

- minor $\det M_{ij}$ 与 cofactor $C_{ij}$ 相差符号 $(-1)^{i+j}$。
- 沿某行展开时，每一项都使用**同一行元素**与其 cofactor；不可混用另一行。
- 大公式每项恰取每行、每列各一个元素；只检查行而忘记列会产生错误项。
- 余子式展开理论通用，但对稠密大矩阵递归计算极慢；数值计算应使用消元分解。

### 三道自检

> [!question]- 1. 求 $C_{23}$ 的符号。
> $(-1)^{2+3}=-1$，所以 $C_{23}=-\det M_{23}$。

> [!question]- 2. 为什么置换矩阵的行列式只能是 $\pm1$？
> 大公式中只有与该置换匹配的一项非零，所有选中元素均为 $1$，只剩排列符号 $\pm1$。

> [!question]- 3. $4\times4$ 行列式的大公式有多少项？每项含多少个矩阵元素？
> 有 $4!=24$ 项；每项从四行和四列各取一次，共含四个元素的乘积。

### 知识链

行列式性质 → 排列大公式 → cofactor 展开 → 下一节的 adjugate、[[Cramer's Rule|Cramer 法则]]与体积。

---

## Session 2.7 Cramer's rule, inverse matrix and volume

### 本节问题、前置知识与尺寸

本节把 cofactor 矩阵用于三个方向：构造逆矩阵、推导 Cramer 法则、解释坐标变换的面积或体积缩放。设 $A\in\mathbb R^{n\times n}$；逆矩阵与 Cramer 法则都要求 $\det A\ne0$。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S07_Lecture_Lecture_20_Cramer_s_Rule_Inverse_Matrix_and_Volume.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S07_Recitation_Problem_Solving_Determinants_and_Volume.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.7prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.7sol.pdf#page=1)

### Lecture：为什么 cofactor 的转置给出逆矩阵

令 $C=(C_{ij})$ 为 cofactor 矩阵。考察 $AC^T$ 的 $(i,j)$ 元素：

$$
(AC^T)_{ij}=\sum_{k=1}^n a_{ik}C_{jk}.
$$

- 若 $i=j$，这正是沿第 $i$ 行的 cofactor 展开，等于 $\det A$。
- 若 $i\ne j$，该和等于“把 $A$ 的第 $j$ 行替换为第 $i$ 行”所得矩阵沿第 $j$ 行展开。新矩阵有两行相同，行列式为零。

因此

$$
AC^T=(\det A)I.
$$

$C^T$ 称为伴随矩阵 $\operatorname{adj}(A)$。当 $\det A\ne0$ 时，两边除以 $\det A$，得到[[Matrix Inverse|逆矩阵]]公式：

$$
A^{-1}
=\frac{1}{\det A}C^T.
$$

这个公式适合理论推导和小矩阵，不适合大型数值求逆。

#### Cramer 法则

对 $Ax=b$，第 $j$ 个分量

$$
x_j=e_j^TA^{-1}b
=\frac1{\det A}\sum_{i=1}^n C_{ij}b_i.
$$

右侧分子正是把 $A$ 的第 $j$ 列替换为 $b$ 所得矩阵 $B_j$ 沿该列展开。因此

$$
x_j=\frac{\det B_j}{\det A}.
$$

这就是[[Cramer's Rule|Cramer 法则（Cramer's rule）]]。它清楚揭示解对数据的依赖，但求全部分量需要许多行列式，计算上通常不如消元。

#### 行列式的几何意义

矩阵 $A=[a_1\ \cdots\ a_n]$ 把单位立方体映成由列向量张成的平行多面体。其 $n$ 维体积为

$$
\operatorname{Vol}=|\det A|.
$$

绝对值给普通体积；符号记录定向是否翻转。这个结论满足行列式三公理：单位立方体体积为 $1$；交换两条边翻转定向；一条边变化时有向体积对该边线性。乘法公式则表示先做 $B$、再做 $A$ 时，体积缩放因子相乘。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-determinant-volume.png|780]]

### Recitation：四面体体积与“不改变高度”的行操作

四面体顶点为

$$
O=(0,0,0),\quad A_1=(2,2,-1),\quad A_2=(1,3,0),\quad A_3=(-1,1,4).
$$

三条边张成的平行六面体体积为

$$
\left|\det\begin{bmatrix}2&2&-1\\1&3&0\\-1&1&4\end{bmatrix}\right|=12.
$$

同底同高关系给出四面体体积是平行六面体的 $1/6$：

$$
V_T=\frac16\cdot12=2.
$$

若把 $A_3$ 移到 $A_3'=(-201,-199,104)$，则

$$
A_3'=A_3-100A_1.
$$

这对应第三行减第一行的 $100$ 倍，不改变行列式；几何上沿底面方向移动顶点，不改变到底面的高度。因此新四面体体积仍为 $2$。

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

### 边界、反例与易错点

- 逆矩阵公式使用 $C^T$，不是 $C$；转置来自 $(AC^T)_{ij}$ 中固定 cofactor 的“行”。
- Cramer 法则要求 $\det A\ne0$；奇异系统不能用分母为零的比值判断有解性。
- $\det A$ 是有向体积，普通体积必须取 $|\det A|$。
- Jacobian 的行列顺序改变会改变符号；体积元取绝对值，但若已限定坐标范围可判断符号。

### 三道自检

> [!question]- 1. 若 $\det A=-4$，$A$ 把单位立方体的体积与定向怎样改变？
> 体积放大为 $4$ 倍；负号表示定向翻转。

> [!question]- 2. 为什么 $AC^T$ 的非对角元为零？
> $(i,j)$ 非对角元等于把第 $j$ 行替换成第 $i$ 行后沿第 $j$ 行展开；所得矩阵有两行相同，行列式为零。

> [!question]- 3. 三维四面体由同一点出发的边向量组成列矩阵 $A$，体积公式是什么？
> 平行六面体体积为 $|\det A|$，四面体体积为 $|\det A|/6$。

### 知识链

cofactor → adjugate 与[[Matrix Inverse|逆矩阵]] → [[Cramer's Rule|Cramer 法则]] → 体积缩放 → 下一节由 $\det(A-\lambda I)=0$ 寻找特征方向。

---

## Session 2.8 Eigenvalues and eigenvectors

### 本节问题、前置知识与尺寸

一般向量被矩阵作用后会改变方向。哪些特殊方向只被缩放而不转向？设 $A\in\mathbb R^{n\times n}$；只有方阵才能在同一空间中比较 $x$ 与 $Ax$ 的方向。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.8sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S08_Lecture_Lecture_21_Eigenvalues_and_Eigenvectors.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S08_Recitation_Problem_Solving_Eigenvalues_and_Eigenvectors.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.8prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.8sol.pdf#page=1)

### Lecture：从 $Ax=\lambda x$ 到特征方程

非零向量 $x\ne0$ 若满足

$$
Ax=\lambda x,
$$

则 $x$ 是 $A$ 的[[Eigenvectors|特征向量（eigenvector）]]，$\lambda$ 是对应的[[Eigenvalues|特征值（eigenvalue）]]。必须排除 $x=0$，因为零向量会对任意 $\lambda$ 满足等式，却不代表任何方向。

移项得

$$
(A-\lambda I)x=0.
$$

要有非零解，$A-\lambda I$ 必须奇异，所以

$$
\det(A-\lambda I)=0.
$$

这称为[[Characteristic Polynomial|特征多项式（characteristic polynomial）]]对应的**特征方程**。具体计算可按[[Computing Eigenpairs|特征对计算流程]]执行：

1. 解标量多项式 $\det(A-\lambda I)=0$ 得特征值；
2. 对每个 $\lambda$ 解零空间 $N(A-\lambda I)$ 得特征向量。

> [!example] 一个 $2\times2$ 例子
> 对
> $$A=\begin{bmatrix}3&1\\1&3\end{bmatrix},$$
> $$
> \det(A-\lambda I)=(3-\lambda)^2-1=(\lambda-4)(\lambda-2).
> $$
> 当 $\lambda=4$，可取 $x_1=(1,1)^T$；当 $\lambda=2$，可取 $x_2=(1,-1)^T$。几何上，矩阵沿两条互相垂直的对角方向分别放大 $4$ 倍与 $2$ 倍。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-eigenvectors.png|760]]

#### 行列式、迹与特征值

把特征多项式写成

$$
\det(\lambda I-A)=\lambda^n-(\operatorname{tr}A)\lambda^{n-1}+\cdots+(-1)^n\det A.
$$

若在复数域计入[[Algebraic and Geometric Multiplicity|代数重数]]，根为 $\lambda_1,\dots,\lambda_n$，比较系数得

$$
\operatorname{tr}A=\sum_{i=1}^n\lambda_i,\qquad
\det A=\prod_{i=1}^n\lambda_i.
$$

因此 $A$ 可逆当且仅当没有零特征值；若 $A$ 可逆，$A^{-1}$ 在同一特征向量上的特征值是 $1/\lambda$：

$$
Ax=\lambda x
\Longrightarrow
x=\lambda A^{-1}x
\Longrightarrow
A^{-1}x=\lambda^{-1}x.
$$

更一般地，对多项式 $p$，

$$
p(A)x=p(\lambda)x,
$$

因为 $A^kx=\lambda^kx$，再按线性组合相加即可。

#### 不同特征值的特征向量线性无关

> [!note] 定理
> 对应于两两不同特征值 $\lambda_1,\dots,\lambda_k$ 的特征向量 $x_1,\dots,x_k$ 线性无关。

**证明。** 对 $k$ 归纳。$k=1$ 时 $x_1\ne0$，结论成立。假设前 $k-1$ 个独立，并设

$$
c_1x_1+\cdots+c_kx_k=0.
$$

左乘 $A$：

$$
c_1\lambda_1x_1+\cdots+c_k\lambda_kx_k=0.
$$

用第二式减去 $\lambda_k$ 倍第一式：

$$
c_1(\lambda_1-\lambda_k)x_1+\cdots+c_{k-1}(\lambda_{k-1}-\lambda_k)x_{k-1}=0.
$$

由归纳假设和 $\lambda_i-\lambda_k\ne0$，得 $c_1=\cdots=c_{k-1}=0$。代回原式，$c_kx_k=0$，因 $x_k\ne0$，故 $c_k=0$。所以全部系数为零，证毕。

### Recitation：不直接平方或求逆

取

$$
A=\begin{bmatrix}
1&2&3\\
0&1&-2\\
0&1&4
\end{bmatrix}.
$$

沿第一列展开：

$$
\det(A-\lambda I)
=(1-\lambda)\bigl((1-\lambda)(4-\lambda)+2\bigr)
=(1-\lambda)(\lambda-2)(\lambda-3).
$$

所以特征值为 $1,2,3$。相应可取

$$
x_1=\begin{bmatrix}1\\0\\0\end{bmatrix},\quad
x_2=\begin{bmatrix}1\\2\\-1\end{bmatrix},\quad
x_3=\begin{bmatrix}1\\-2\\2\end{bmatrix}.
$$

不用计算 $A^2$ 或 $A^{-1}$：

- $A^2$ 的特征值是 $1,4,9$，特征向量仍为 $x_i$；
- $A^{-1}-I$ 的特征值是 $1/\lambda_i-1$，即 $0,-1/2,-2/3$，特征向量仍为 $x_i$。

这展示了特征坐标的优势：矩阵函数在特征方向上退化为标量函数。

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

### 边界、反例与易错点

- 特征向量不能是零向量；同一特征空间中的任意非零倍数都是特征向量。
- $B$ 的特征值通常不能推出 $B^TB$ 的特征值；后者属于奇异值问题。
- 实矩阵可能没有足够的实特征值，例如二维旋转 $90^\circ$ 的特征值为 $\pm i$。
- 重复特征值未必提供足够多独立特征向量；代数重数与特征空间维数要分开。

### 三道自检

> [!question]- 1. 若 $Ax=3x$，求 $(A^2-4A+I)x$。
> 等于 $(3^2-4\cdot3+1)x=-2x$。

> [!question]- 2. 为什么 $0$ 是 $A$ 的特征值等价于 $A$ 奇异？
> $Ax=0x$ 有非零解恰好表示 $N(A)$ 非平凡，也恰好表示 $A$ 奇异。

> [!question]- 3. 一个 $3\times3$ 矩阵有三个互不相同的特征值，能否保证有一组特征向量基？
> 能。不同特征值对应的三个特征向量线性无关，在三维空间中构成一组基。

### 知识链

$\det(A-\lambda I)=0$ → [[Eigenvalues|特征值]]与[[Eigenvectors|特征向量]] → 独立特征方向 → 下一节的[[Diagonalization|对角化]]。

---

## Session 2.9 Diagonalization and powers of A

### 本节问题、前置知识与尺寸

什么时候能选一组特征向量作为整个空间的基？一旦做到，为什么 $A^k$ 会变成只对标量取幂？本节始终假设 $A\in\mathbb F^{n\times n}$，其中 $\mathbb F$ 至少包含所需的特征值。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.9sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S09_Lecture_Lecture_22_Diagonalization_and_Powers_of_A.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S09_Recitation_Problem_Solving_Powers_of_a_Matrix.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.9prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.9sol.pdf#page=1)

### Lecture：把全部特征方程并排放置

若 $A$ 有 $n$ 个线性无关特征向量 $x_1,\dots,x_n$，令

$$
S=[x_1\ \cdots\ x_n],\qquad
\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n).
$$

把 $Ax_i=\lambda_i x_i$ 并排写：

$$
AS=A[x_1\ \cdots\ x_n]
=[\lambda_1x_1\ \cdots\ \lambda_nx_n]
=S\Lambda.
$$

$S$ 的列独立，所以可逆。右乘 $S^{-1}$ 得

$$
A=S\Lambda S^{-1},\qquad S^{-1}AS=\Lambda.
$$

这称为[[Diagonalization|对角化（diagonalization）]]。它的含义不是把 $A$ 通过行操作变成对角矩阵，而是**换到特征向量基底**：$S^{-1}$ 把标准坐标换成特征坐标，$\Lambda$ 在各坐标独立缩放，$S$ 再换回标准坐标。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-diagonalization.png|780]]

#### 可对角化的条件

$$
A\text{ 可对角化}
\Longleftrightarrow
A\text{ 有 }n\text{ 个线性无关特征向量}.
$$

若有 $n$ 个互不相同的特征值，则上一节定理保证可对角化；反之不成立，例如 $I$ 只有一个不同特征值 $1$，却有任意基作为特征向量基。

重复特征值的关键是几何重数。矩阵

$$
\begin{bmatrix}1&1\\0&1\end{bmatrix}
$$

的特征值 $1$ 代数重数为 $2$，但 $N(A-I)$ 只有一维，因而不可对角化。

#### 矩阵幂

利用中间的 $S^{-1}S=I$：

$$
\begin{aligned}
A^k
&=(S\Lambda S^{-1})^k\\
&=S\Lambda(S^{-1}S)\Lambda\cdots(S^{-1}S)\Lambda S^{-1}\\
&=S\Lambda^kS^{-1}.
\end{aligned}
$$

其中

$$
\Lambda^k=\operatorname{diag}(\lambda_1^k,\dots,\lambda_n^k).
$$

对差分方程 $u_{k+1}=Au_k$，若

$$
u_0=c_1x_1+\cdots+c_nx_n,
$$

则

$$
u_k=A^ku_0=c_1\lambda_1^kx_1+\cdots+c_n\lambda_n^kx_n.
$$

[[Spectral Radius|谱半径（spectral radius）]] $\rho(A)=\max_i|\lambda_i|$ 给出长期增长的基本尺度：$|\lambda|<1$ 衰减，$|\lambda|>1$ 增长，$\lambda<0$ 伴随交替翻转，复特征值带来旋转振荡。若存在 Jordan 块，还要额外考虑多项式因子。

### Recitation：参数矩阵的 $k$ 次幂

令

$$
C=\begin{bmatrix}
2b-a&a-b\\
2b-2a&2a-b
\end{bmatrix}.
$$

它的特征值为 $a,b$，可取对应特征向量 $(1,2)^T,(1,1)^T$。因此

$$
S=\begin{bmatrix}1&1\\2&1\end{bmatrix},\quad
\Lambda=\begin{bmatrix}a&0\\0&b\end{bmatrix},\quad
S^{-1}=\begin{bmatrix}-1&1\\2&-1\end{bmatrix}.
$$

相乘得到

$$
C^k=S\Lambda^kS^{-1}
=\begin{bmatrix}
2b^k-a^k&a^k-b^k\\
2b^k-2a^k&2a^k-b^k
\end{bmatrix}.
$$

令 $k=1$ 可恢复原矩阵，这是必要的代数检查。若 $a=b=-1$，则 $C=-I$，所以 $C^{100}=I$。

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

### 边界、反例与易错点

- 行化简一般改变特征值；对角化是相似变换 $S^{-1}AS$，不是消元。
- $A=S\Lambda S^{-1}$ 的 $S$ 列顺序必须和 $\Lambda$ 对角元顺序一致。
- 有重复特征值不等于不可对角化；要检查对应特征空间总维数。
- $A^k=S\Lambda^kS^{-1}$ 要求可对角化；不可对角化矩阵需 Unit III 的 Jordan 方法。

### 三道自检

> [!question]- 1. 若 $A=S\operatorname{diag}(2,-1)S^{-1}$，$A^{10}$ 的特征值是什么？
> $2^{10}=1024$ 与 $(-1)^{10}=1$，特征向量不变。

> [!question]- 2. 为什么 $n$ 个不同特征值足以保证对角化？
> 对应的 $n$ 个特征向量线性无关，在 $n$ 维空间中形成基；以它们为 $S$ 的列即可。

> [!question]- 3. $A=I$ 只有一个不同特征值，为什么仍可对角化？
> 每个非零向量都是特征值 $1$ 的特征向量，可任取一组基；$A=SIS^{-1}$。

### 知识链

独立特征向量 → [[Diagonalization|对角化]] → $A^k$ 与差分方程 → 下一节把标量 $e^{\lambda t}$ 提升为[[Matrix Exponential|矩阵指数]]。

---

## Session 2.10 Differential equations and $e^{At}$

### 本节问题、前置知识与尺寸

如何求解耦合常系数系统

$$
\frac{du}{dt}=Au,\qquad u(0)=u_0,\qquad A\in\mathbb R^{n\times n}?
$$

核心思想是：特征向量方向把向量微分方程化为标量指数增长；把全部方向合起来得到[[Matrix Exponential|矩阵指数（matrix exponential）]] $e^{At}\in\mathbb R^{n\times n}$。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.10sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S10_Lecture_Lecture_23_Differential_Equations_and_expAt.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S10_Recitation_Problem_Solving_Differential_Equations_and_expAt.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.10prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.10sol.pdf#page=1)

### Lecture：每个特征方向是一种指数模式

若 $Ax=\lambda x$，尝试

$$
u(t)=e^{\lambda t}x.
$$

则

$$
u'(t)=\lambda e^{\lambda t}x=e^{\lambda t}Ax=A(e^{\lambda t}x)=Au(t).
$$

所以每一对特征值—特征向量都给出一个解。若 $A$ 有特征向量基 $x_1,\dots,x_n$，把初值分解为

$$
u_0=c_1x_1+\cdots+c_nx_n,
$$

由线性叠加得唯一解

$$
u(t)=c_1e^{\lambda_1t}x_1+\cdots+c_ne^{\lambda_nt}x_n.
$$

这与离散系统 $u_k=A^ku_0$ 完全平行：$\lambda_i^k$ 被 $e^{\lambda_i t}$ 取代。

#### 矩阵指数的定义与推导

标量指数的幂级数提示定义

$$
e^{At}=I+At+\frac{A^2t^2}{2!}+\frac{A^3t^3}{3!}+\cdots.
$$

逐项求导：

$$
\frac{d}{dt}e^{At}
=A+A^2t+\frac{A^3t^2}{2!}+\cdots
=Ae^{At}.
$$

并且 $e^{A\cdot0}=I$，所以

$$
u(t)=e^{At}u_0
$$

满足微分方程和初值。若 $A=S\Lambda S^{-1}$，则 $A^k=S\Lambda^kS^{-1}$，代入级数：

$$
e^{At}
=S\left(I+\Lambda t+\frac{\Lambda^2t^2}{2!}+\cdots\right)S^{-1}
=Se^{\Lambda t}S^{-1},
$$

其中

$$
e^{\Lambda t}=\operatorname{diag}(e^{\lambda_1t},\dots,e^{\lambda_nt}).
$$

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-matrix-exponential.png|780]]

必须区分：一般 $e^{(A+B)t}\ne e^{At}e^{Bt}$；只有 $AB=BA$ 时该标量指数法则才成立。另一方面，同一个 $A$ 的指数总满足

$$
e^{At}e^{As}=e^{A(t+s)},\qquad (e^{At})^{-1}=e^{-At}.
$$

### Recitation：三阶 ODE 化为一阶矩阵系统

考虑

$$
y'''+2y''-y'-2y=0.
$$

令

$$
u=\begin{bmatrix}y''\\y'\\y\end{bmatrix},\qquad
u'=\begin{bmatrix}y'''\\y''\\y'\end{bmatrix}
=
\underbrace{\begin{bmatrix}-2&1&2\\1&0&0\\0&1&0\end{bmatrix}}_{A}
u.
$$

特征多项式为

$$
\det(A-\lambda I)=(1-\lambda)(1+\lambda)(2+\lambda),
$$

故特征值为 $1,-1,-2$，可取

$$
x_1=\begin{bmatrix}1\\1\\1\end{bmatrix},\quad
x_2=\begin{bmatrix}1\\-1\\1\end{bmatrix},\quad
x_3=\begin{bmatrix}4\\-2\\1\end{bmatrix}.
$$

因此

$$
u(t)=c_1e^tx_1+c_2e^{-t}x_2+c_3e^{-2t}x_3,
$$

读第三个分量即得

$$
y(t)=c_1e^t+c_2e^{-t}+c_3e^{-2t}.
$$

令 $S=[x_1\ x_2\ x_3]$。$S^{-1}$ 第一列为 $(1/6,-1/2,1/3)^T$，因此 $e^{At}=Se^{\Lambda t}S^{-1}$ 的第一列为

$$
\frac16e^tx_1-\frac12e^{-t}x_2+\frac13e^{-2t}x_3.
$$

这里只计算所需的一列，体现了“先判断输出需求，再做最少矩阵乘法”的策略。

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

### 边界、反例与易错点

- $e^{At}$ 不是把 $A$ 的每个元素分别取指数；定义来自矩阵幂级数。
- 只有可对角化时才能直接写 $Se^{\Lambda t}S^{-1}$；级数定义本身对所有方阵都有效。
- $\lambda<0$ 表示该特征模式随 $t\to\infty$ 衰减；不是说解向量“为负”。
- 复共轭特征值产生实数形式的正弦—余弦振荡；不可简单丢弃虚部。

### 三道自检

> [!question]- 1. 若 $Ax=-2x$，初值 $u_0=3x$，求 $u(t)$。
> $u(t)=3e^{-2t}x$。

> [!question]- 2. 验证 $e^{At}u_0$ 满足初值。
> $e^{A0}u_0=Iu_0=u_0$；且 $\frac d{dt}(e^{At}u_0)=Ae^{At}u_0$。

> [!question]- 3. $e^{At}e^{Bt}=e^{(A+B)t}$ 总成立吗？
> 不成立。矩阵乘法不交换会使幂级数交叉项次序不同；当 $AB=BA$ 时才成立。

### 知识链

对角化 → 特征模式 $e^{\lambda t}x$ → [[Matrix Exponential|矩阵指数]] → 连续动力系统 → 下一节的概率稳态与正交函数展开。

---

## Session 2.11 Markov matrices and Fourier series

### 本节问题、前置知识与尺寸

本节把此前工具用于两个看似不同的问题：概率在状态之间转移时为什么会趋于稳态；函数怎样像向量一样投影到正交“坐标轴”上。二者共同核心是：选择能让变换解耦的基。

> [!info] 本地材料
> - [Session summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf#page=1)
> - [Lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf#page=1)
> - [Recitation transcript](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S11_Recitation_Problem_Solving_Markov_Matrices.pdf#page=1)
> - [Homework problems](MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.11prob.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.11sol.pdf#page=1)

### Lecture A：Markov 矩阵与稳态

课程采用“列随机”约定：[[Markov Matrix|Markov 矩阵（Markov matrix）]] $A=(a_{ij})$ 满足

$$
a_{ij}\ge0,\qquad \sum_{i=1}^n a_{ij}=1\quad\text{对每一列 }j.
$$

$a_{ij}$ 表示“当前在状态 $j$，下一步到状态 $i$”的概率。若 $p_k$ 是分量非负、和为 $1$ 的概率列向量，则

$$
p_{k+1}=Ap_k.
$$

概率总和保持：令 $\mathbf1=(1,\dots,1)^T$，列和为 $1$ 等价于

$$
\mathbf1^TA=\mathbf1^T.
$$

故

$$
\mathbf1^Tp_{k+1}=\mathbf1^TAp_k=\mathbf1^Tp_k=1.
$$

$\mathbf1^T$ 是左特征向量；由于 $A$ 与 $A^T$ 特征值相同，$1$ 是 $A$ 的特征值。满足

$$
Ap_*=p_*
$$

的概率向量称为**稳态（steady state）**。

若 $A$ 可对角化，特征值 $1$ 是单特征值，并且其他特征值均满足 $|\lambda_i|<1$，则

$$
p_k=A^kp_0=c_1p_*+\sum_{i\ge2}c_i\lambda_i^kx_i
\longrightarrow c_1p_*.
$$

归一化使极限分量和为 $1$。若 $1$ 的重数大于 $1$，稳态可能不唯一；若还存在 $\lambda=-1$ 或其他单位圆上的特征值，则可能周期振荡而不收敛。因此“Markov”本身并不自动保证收敛到唯一稳态。有限状态的正矩阵满足更强的 Perron--Frobenius 条件，本节图中的具体矩阵正是这一情形。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-markov-steady-state.png|760]]

### Recitation：两状态粒子的长期分布

粒子在 $A,B$ 两状态之间跳转：从 $A$ 留在 $A$ 的概率 $0.6$、到 $B$ 的概率 $0.4$；从 $B$ 到 $A$ 的概率 $0.2$、留在 $B$ 的概率 $0.8$。所以

$$
M=\begin{bmatrix}0.6&0.2\\0.4&0.8\end{bmatrix},\qquad p_0=\begin{bmatrix}1\\0\end{bmatrix}.
$$

一步后 $p_1=Mp_0=(0.6,0.4)^T$。特征值为 $1,0.4$，可取

$$
x_1=\begin{bmatrix}1\\2\end{bmatrix},\qquad x_2=\begin{bmatrix}1\\-1\end{bmatrix}.
$$

由

$$
p_0=\frac13x_1+\frac23x_2
$$

得到

$$
p_n=\frac13x_1+\frac23(0.4)^nx_2
=\frac13\begin{bmatrix}1+2(0.4)^n\\2-2(0.4)^n\end{bmatrix}.
$$

因此

$$
p_n\longrightarrow\begin{bmatrix}1/3\\2/3\end{bmatrix}.
$$

衰减因子 $(0.4)^n$ 精确描述“忘记初始状态”的速度。

### Lecture B：Fourier 级数是无限维投影

有限维中，若 $q_1,\dots,q_n$ 是标准正交基，

$$
v=\sum_i(q_i^Tv)q_i.
$$

函数空间中把内积定义为

$$
\langle f,g\rangle=\int_0^{2\pi}f(x)g(x)\,dx.
$$

函数 $1,\cos nx,\sin nx$ 两两正交，例如

$$
\int_0^{2\pi}\sin x\cos x\,dx=0,
$$

但它们尚未全部归一化：$\|1\|^2=2\pi$，$\|\cos nx\|^2=\|\sin nx\|^2=\pi$。因此[[Fourier Series|Fourier 级数（Fourier series）]]写成

$$
f(x)\sim a_0+\sum_{n=1}^{\infty}\bigl(a_n\cos nx+b_n\sin nx\bigr),
$$

其中投影系数为

$$
a_0=\frac1{2\pi}\int_0^{2\pi}f(x)\,dx,
$$

$$
a_n=\frac1\pi\int_0^{2\pi}f(x)\cos(nx)\,dx,\qquad
b_n=\frac1\pi\int_0^{2\pi}f(x)\sin(nx)\,dx.
$$

注意这里采用课程的“常数项写 $a_0$”约定；另一些教材写 $a_0/2$，相应地把 $a_0$ 定义为 $\frac1\pi\int f$。

截断到有限个三角函数时，Fourier 部分和就是在所张成函数子空间中的最小二乘投影；残差与每个保留的基函数正交。

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

> [!question]- Problem 24.2：矩阵分类与分解
> $$
> A=\begin{bmatrix}0&0&1\\0&1&0\\1&0&0\end{bmatrix},\qquad
> B=\frac13\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}.
> $$
>
> **矩阵 $A$。** $\det A=-1$，所以可逆；$A^TA=I$，所以正交；$A^2=I\ne A$，不是投影；每行每列恰有一个 $1$，所以是置换矩阵；$A=A^T$，所以可正交对角化；每列和为 $1$，所以也是 Markov。无换行的普通 $LU$ 因首主元为零而失败；QR、$S\Lambda S^{-1}$ 与 $Q\Lambda Q^T$ 均可行。
>
> **矩阵 $B$。** 秩为 $1$，不可逆、非正交；$B^2=B$，是投影；不是置换矩阵；$B=B^T$，故可正交对角化；每列和为 $1$，也是 Markov。可写退化的 $LU$；标准“满列秩薄 QR”不适用；$S\Lambda S^{-1}$ 与 $Q\Lambda Q^T$ 可行，特征值为 $1,0,0$。

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

### 边界、反例与易错点

- 本课用“列和为 $1$、概率列向量左乘”的约定；采用行向量的教材常定义行和为 $1$，两者互为转置。
- $\mathbf1^TA=\mathbf1^T$ 给的是左特征向量；稳态 $Ap_*=p_*$ 是右特征向量，除非矩阵对称，不可混同。
- Markov 矩阵可有周期而不收敛，例如交换矩阵的特征值含 $-1$。
- Fourier 基函数是正交但未按上述写法归一；系数分母来自各基函数的平方范数。

### 三道自检

> [!question]- 1. 为什么列随机矩阵保持概率总和？
> 列和为 $1$ 等价于 $\mathbf1^TA=\mathbf1^T$，所以 $\mathbf1^TAp=\mathbf1^Tp$。

> [!question]- 2. 若 Markov 矩阵特征值为 $1,0.8,-0.2$，长期收敛速度由哪个数控制？
> 除稳态外最大模为 $0.8$，误差通常按 $0.8^k$ 的量级衰减。

> [!question]- 3. 在 $[0,2\pi]$ 上，为什么 $\cos x$ 的投影系数分母是 $\pi$？
> 因 $\langle\cos x,\cos x\rangle=\int_0^{2\pi}\cos^2x\,dx=\pi$。

### 知识链

矩阵幂 → [[Markov Matrix|Markov 稳态]]；标准正交投影 → [[Fourier Series|Fourier 展开]]。两条线都体现“在合适的基中解耦”。

---

## Session 2.12 Exam 2 review

### 本节问题、前置知识与尺寸

本节不增加新定理，而是训练在有限时间内识别题型、先做结构判断、再选择计算工具。Exam 2 Review 串联三块内容：正交/投影/最小二乘/QR；行列式/cofactor/逆矩阵；特征值/对角化/矩阵幂。

> [!info] 本地材料
> - [Review summary](MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.12sum.pdf#page=1)
> - [Review lecture transcript](MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S12_Lecture_Exam_2_Review.pdf#page=1)
> - [Exam problem-solving recitation](MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S12_Recitation_Exam_2_Problem_Solving.pdf#page=1)
> - [Exam 2 problems](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=1) · [official solutions](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=1)

> [!note] Review 与所附样卷的范围不同
> Review lecture 会复习 eigenvalues、diagonalization 与 $A^k$，但下面完整解答的官方 Exam 2 样卷只覆盖正交/投影、最小二乘、Gram--Schmidt 与行列式。官网也明确说明 eigenvalue questions 会在随后出现；因此不能把 Review 的全部内容误称为这份样卷的题目范围。

### 题型分流表

| 题面信号 | 第一反应 | 必查条件 |
|---|---|---|
| closest / best fit / inconsistent | $A^TA\hat x=A^Tb$ 或 QR | $A$ 是否满列秩；残差是否满足 $A^Te=0$ |
| project onto $C(A)$ | $P=A(A^TA)^{-1}A^T$；若 $Q^TQ=I$ 则 $P=QQ^T$ | $P^T=P,P^2=P$；尺寸为 ambient space 的维数 |
| orthonormal / Gram–Schmidt | 逐项减投影并归一化；$A=QR$ | 长方 $Q$ 只有 $Q^TQ=I$ |
| determinant | 先看三角、零、相同行/列、稀疏 | 行交换、缩放、行加法影响不同 |
| cofactor / inverse column | $C_{ij}=(-1)^{i+j}\det M_{ij}$；$A^{-1}=C^T/\det A$ | 转置与符号 |
| $A^k$ / long run | 先找 eigen，再 $S\Lambda^kS^{-1}$ | 是否有足够独立特征向量 |
| matrix class | 分别查定义，不凭外观 | projection: $P^2=P$；orthogonal: $Q^TQ=I$；Markov: 非负且列和 1 |

### Review lecture：一题串起投影、秩、特征值和动力学

令 $a=(2,1,2)^T$，投影到 $\operatorname{span}(a)$：

$$
P=\frac{aa^T}{a^Ta}
=\frac19\begin{bmatrix}4&2&4\\2&1&2\\4&2&4\end{bmatrix}.
$$

- $\operatorname{rank}(P)=1$，$C(P)=\operatorname{span}(a)$。
- $a$ 是特征值 $1$ 的特征向量；$a^\perp$ 中的两个独立方向对应特征值 $0$。
- 因 $P^2=P$，对 $k\ge1$ 有 $P^k=P$。
- 若 $u_{k+1}=Pu_k$，只做一次投影后便稳定：$u_k=Pu_0$（$k\ge1$）。

这类综合题应先利用矩阵类别，而不是重新展开特征多项式或反复相乘。

另一个 review 例子是拟合过原点直线 $y=Dt$ 到 $(1,4),(2,5),(3,8)$：

$$
A=\begin{bmatrix}1\\2\\3\end{bmatrix},\quad
b=\begin{bmatrix}4\\5\\8\end{bmatrix},\quad
\hat D=\frac{A^Tb}{A^TA}=\frac{38}{14}=\frac{19}{7}.
$$

这里只有一个未知数，不应人为添加截距列。

### Recitation：15 分钟行列式综合题

考虑

$$
A=\begin{bmatrix}
1&2&3&4\\
5&6&7&8\\
0&0&9&10\\
0&0&11&12
\end{bmatrix}.
$$

大公式中，第三、四行只能从第三、四列取非零元素，前两行便只能从前两列取。因此仅有 $2\times2=4$ 个非零排列项，而不是盲目写满 $4!=24$ 项。更快地把它视为分块上三角：

$$
\det A
=\det\begin{bmatrix}1&2\\5&6\end{bmatrix}
\det\begin{bmatrix}9&10\\11&12\end{bmatrix}
=(-4)(-2)=8.
$$

第一行 cofactors 为

$$
C_{11}=-12,\qquad C_{12}=10,\qquad C_{13}=C_{14}=0.
$$

核对：$1(-12)+2(10)=8=\det A$。由逆矩阵公式，$A^{-1}$ 第一列是 $C^T$ 第一列，即 cofactor 矩阵第一行的转置除以行列式：

$$
(A^{-1})_{:,1}=\frac18\begin{bmatrix}-12\\10\\0\\0\end{bmatrix}.
$$

考试策略是让前一问的结果直接服务后一问，并在每阶段做一次低成本核对。

### 边界、反例与易错点

- 投影矩阵的行列式：若不是恒等投影，至少有一个零特征值，所以方阵投影的行列式一定为零。
- $P_A=P_Q$ 的原因是投影到同一子空间，不是因为 $A=Q$。
- 行列式大公式中的“最多次数”由每项每行每列各选一个元素限制。
- Exam 中写出尺寸和一行验算，常能及时发现把 $Q^TQ$ 与 $QQ^T$ 交换等错误。

### 三道自检

> [!question]- 1. $Q\in\mathbb R^{8\times3}$ 列标准正交，投影矩阵是什么？其秩和尺寸是什么？
> $P=QQ^T\in\mathbb R^{8\times8}$，秩为 $3$。

> [!question]- 2. $P^2=P$ 的特征值只能是多少？
> 若 $Px=\lambda x$，则 $P^2x=\lambda^2x$，又等于 $Px=\lambda x$，故 $\lambda(\lambda-1)=0$，只能是 $0$ 或 $1$。

> [!question]- 3. 若 $A$ 有特征值 $2,1/2,-1$，$A^k$ 长期一定收敛吗？
> 不一定且通常发散：特征值 $2$ 的分量指数增长；即使初值无该分量，$-1$ 分量也会交替而不收敛。

### 知识链

投影/QR、行列式/cofactor、eigen/diagonalization 三条线在 Unit II 与 Exam 2 Review 中汇合；下一单元将用对称性把它们进一步统一为正交对角化、正定性与 SVD。

---

# Exam 2

> [!info] 试卷与官方答案
> - [Exam 2 原题，第 1 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=1)
> - [Exam 2 原题，第 2 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=2)
> - [Exam 2 原题，第 3 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=3)
> - [Exam 2 原题，第 4 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2.pdf#page=4)
> - [官方答案，第 1 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=1) · [第 2 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=3) · [第 3 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=4) · [第 4 题](MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex2s.pdf#page=5)

## Problem 1：标准正交基构成的三个行列式

设 $q_1,q_2,q_3$ 是 $\mathbb R^3$ 中的标准正交向量。求下列行列式的所有可能值，并说明理由。

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

### (c) $\det[q_1\ q_2\ q_3]\det[q_2\ q_3\ q_1]$

> [!success]- 完整解答
> 第二个列次序由 $(1,2,3)$ 变为 $(2,3,1)$，是三循环，可由两次交换完成，所以是偶排列，行列式不变：
> $$
> \det[q_2\ q_3\ q_1]=\det[q_1\ q_2\ q_3].
> $$
> 乘积为 $(\det Q)^2=1$，故
> $$\boxed{1}.$$

> [!warning] 常见错误
> 不能因为列向量都是单位向量就直接断言行列式为 $1$；反射型标准正交基的行列式为 $-1$。只有乘积平方才固定为 $1$。

## Problem 2：21 个数据点的最小二乘直线

在 $t=-10,-9,\dots,9,10$ 共 21 个时刻测量。除 $t=0$ 的测量为 $1$ 外，其余测量全为 $0$。

### (a) 求最佳直线 $C+Dt$

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

### (b) 投影到哪个子空间？给出一个非零正交向量

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

> [!warning] 常见错误
> 目标子空间不是 $(t,y)$ 平面中的一条直线，而是 $\mathbb R^{21}$ 中由常数列和时间列张成的二维列空间。

## Problem 3：Gram–Schmidt、两个投影矩阵与新向量

独立向量 $a_1,a_2,a_3\in\mathbb R^5$ 经 Gram–Schmidt 得到标准正交向量 $q_1,q_2,q_3$。令

$$
A=[a_1\ a_2\ a_3],\qquad Q=[q_1\ q_2\ q_3]\in\mathbb R^{5\times3}.
$$

### (a) 写出两个投影矩阵

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

### (b) 比较 $P_A,P_Q$；求 $P_QQ$ 与 $\det P_Q$

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

### (c) 加入独立向量 $a_4$，哪一个是新的 $q_4$？

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

> [!warning] 常见错误
> $Q^TQ=I_3$ 不代表 $QQ^T=I_5$；后者是秩 3 投影，正因此行列式为零。

## Problem 4：同一个参数占据第一行和第一列

一个 $4\times4$ 矩阵的第一行和第一列全部为参数 $x$，其余 $3\times3$ 块是任意常数。

### (a) $\det A$ 作为 $x$ 的多项式，最高可能几次？

> [!success]- 完整解答
> 行列式大公式的每一项从每行、每列各取一个元素。
>
> - 若一项取左上角 $a_{11}=x$，第一行和第一列都已使用，不能再取其他 $x$，该项只含一次 $x$。
> - 若不取 $a_{11}$，最多可从第一行的某列 $j>1$ 取一个 $x$，再从第一列的某行 $i>1$ 取一个 $x$，共两次。
>
> 所以每项至多含 $x^2$，而适当选择其余元素确实能让二次项不消失。最高可能次数为
> $$\boxed 2.$$

### (b) 其余 $3\times3$ 块为 $I_3$ 时求行列式与奇异参数

此时

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

> [!warning] 常见错误
> 参数出现七个位置，并不意味着次数可到 7；同一排列项受到“每行、每列只取一次”的严格限制。

---

## Unit II 知识闭环

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
