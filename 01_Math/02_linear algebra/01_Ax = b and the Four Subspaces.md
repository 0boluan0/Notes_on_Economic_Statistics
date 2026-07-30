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

## 课程来源、约定与导航

- 官方课程：MIT OCW 18.06SC *Linear Algebra, Fall 2011*；总入口见 [[00_MIT OCW 18.06SC course map|课程总览]]。
- 本地资料索引：[[MIT_OCW_18.06SC_PDF/index|MIT 18.06SC PDF 索引]]。
- 本篇严格按官网逻辑顺序写：Geometry → Overview → Elimination → Inverse → LU → Vector Spaces → Column/Null Spaces → $Ax=0$ → $Ax=b$ → Basis/Dimension → Four Subspaces → Matrix Spaces → Graphs → Review → Exam 1。
- **编号提醒**：官网第二讲 Overview 的本地 summary 是 `Ses1.13sum.pdf`；官网第三至第十三讲依次使用本地 `Ses1.2–Ses1.12` 资料。
- 尺寸检查规则：若 $A$ 是 $m\times n$，则它有 $m$ 行、$n$ 列；$Ax$ 只有在 $x\in\mathbb F^n$ 时有定义，结果属于 $\mathbb F^m$。
- 除非特别说明，向量均写成列向量。$C(A)$、$N(A)$、$C(A^T)$、$N(A^T)$ 分别表示列空间、零空间、行空间和左零空间。

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

**问题**：同一个方程组为什么既能看成几何对象的交，又能看成列向量的线性组合？

**前置知识**：二元一次方程、向量加法与数乘。本节首次建立 [[Linear Algebra-hub|线性代数]] 的三个视角。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.1sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S01_Lecture_The_Geometry_of_Linear_Equations.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S01_Recitation_Geometry_of_Linear_Algebra.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.1prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.1sol.pdf#page=1|official solution p.1]]

### 1. 从 row picture 到 column picture

考虑课堂中的系统

$$
\begin{cases}
2x-y=0,\\
-x+2y=3.
\end{cases}
$$

**行图像（row picture）**把每一个方程看成 $xy$ 平面中的直线。第一条是 $y=2x$，第二条是 $y=(x+3)/2$；交点同时满足两个方程。联立得

$$
2x=\frac{x+3}{2}\Longrightarrow 4x=x+3\Longrightarrow x=1,
\qquad y=2.
$$

因此交点是 $(1,2)$。在 $m$ 个方程、$n$ 个未知数的一般情形中，每一行在 $\mathbb R^n$ 中给出一个超平面；解集是这些超平面的交。

**列图像（column picture）**把同一系统改写成

$$
x\begin{bmatrix}2\\-1\end{bmatrix}
+y\begin{bmatrix}-1\\2\end{bmatrix}
=\begin{bmatrix}0\\3\end{bmatrix}.
$$

未知数 $x,y$ 不再只是平面坐标，而是两列的组合系数。代入 $(x,y)=(1,2)$：

$$
\begin{bmatrix}2\\-1\end{bmatrix}
+2\begin{bmatrix}-1\\2\end{bmatrix}
=\begin{bmatrix}0\\3\end{bmatrix}.
$$

这引出 [[Column Space|列空间]]：$Ax=b$ 有解，当且仅当 $b$ 属于 $A$ 的列向量所张成的空间。

### 2. 矩阵乘向量的两种等价读法

设

$$
A=\begin{bmatrix}a_1&a_2&\cdots&a_n\end{bmatrix}\in\mathbb R^{m\times n},
\qquad x=\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix}\in\mathbb R^n.
$$

按列读：

$$
Ax=x_1a_1+\cdots+x_na_n\in\mathbb R^m.
$$

按行读：若 $r_i^T$ 是第 $i$ 行，则

$$
Ax=\begin{bmatrix}r_1^Tx\\\vdots\\r_m^Tx\end{bmatrix}.
$$

前者强调“输出是哪些列的组合”，后者强调“每个方程如何约束输入”。二者是同一次矩阵乘法，不是两个不同定义。

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

### 3. 三种解的几何命运

- **唯一解**：$b$ 可由列生成，而且系数表示唯一。
- **无解**：$b$ 不在列空间中。
- **无穷多解**：$b$ 可达，但存在非零向量 $z$ 使 $Az=0$；若 $Ax_p=b$，则 $A(x_p+tz)=b$。

这里已经预告了 [[Null Space|零空间]] 和 [[Linear system solution structure|线性方程组解结构]]。注意：方阵并不自动可逆；列必须既独立又张成整个输出空间。

### 4. Recitation 代表例题

Recitation 使用

$$
\begin{cases}
2x+y=3,\\
x-2y=-1.
\end{cases}
$$

第二式给 $x=2y-1$，代入第一式：$2(2y-1)+y=3$，所以 $5y=5$、$y=1$、$x=1$。列图像为

$$
x\begin{bmatrix}2\\1\end{bmatrix}
+y\begin{bmatrix}1\\-2\end{bmatrix}
=\begin{bmatrix}3\\-1\end{bmatrix}.
$$

右侧恰是两列各取一份。验证时既要代回两条原方程，也可直接做一次矩阵乘法。

### Homework：全部题目与逐步解答

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

> [!question]- Problem 1.3：矩阵乘法的尺寸
> **题目转述**：判断“$3\times2$ 矩阵 $A$ 乘 $2\times3$ 矩阵 $B$ 得到 $3\times3$ 矩阵 $AB$”是否正确。
>
> **解答**：正确。一般地，
> $$
> A_{m\times n}B_{n\times p}=(AB)_{m\times p}.
> $$
> 内侧尺寸 $n$ 必须相等，外侧尺寸 $m,p$ 成为结果的尺寸。

### 易错点、边界与反例

- 行图像位于未知数空间 $\mathbb R^n$；列图像位于输出空间 $\mathbb R^m$。当 $m\ne n$ 时，这两个空间甚至维数不同。
- “列数多”不等于“张成空间大”；重复列会带来冗余。
- $Ax=0$ 永远至少有零解；“无解”只可能发生在非齐次系统 $Ax=b$ 中。

### 三道自检题

> [!question]- 1. 尺寸题
> 若 $A\in\mathbb R^{4\times3}$，那么 $x$、$b$ 分别属于什么空间？
>
> **答案**：$x\in\mathbb R^3$，$b=Ax\in\mathbb R^4$。

> [!question]- 2. 结构题
> 若 $Az=0$ 且 $z\ne0$，为什么 $Ax=b$ 不可能有唯一解？
>
> **答案**：只要有一个特解 $x_p$，便有 $A(x_p+tz)=b$；不同 $t$ 给出不同解。若没有特解，则是无解，也不是唯一解。

> [!question]- 3. 计算题
> 把 $\begin{bmatrix}1&-1\\2&3\end{bmatrix}(4,1)^T$ 同时写成列组合并算出结果。
>
> **答案**：$4(1,2)^T+1(-1,3)^T=(3,11)^T$。

### 知识链小结

方程交点 → 列向量组合 → $b\in C(A)$ 决定存在性 → $N(A)$ 决定唯一性 → 下一步用消元系统地找出这些结构。

## Session 1.2 An overview of key ideas

### 本节问题与前置知识

**问题**：消元、子空间、正交、特征值和 SVD 为什么不是彼此无关的技巧？

**前置知识**：能从 Session 1.1 读懂 $Ax=b$ 的行图像和列图像。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.13sum.pdf#page=1|overview summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S02_Lecture_An_Overview_of_Linear_Algebra.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S02_Recitation_An_Overview_of_Key_Ideas.pdf#page=1|recitation transcript p.1]]

### 1. 整门课围绕一个输入—输出问题

矩阵 $A\in\mathbb R^{m\times n}$ 是线性映射

$$
A:\mathbb R^n\longrightarrow\mathbb R^m.
$$

课程不断追问四件事：

1. 哪些输入被送到 $0$？答案是 $N(A)$。
2. 哪些输出可以被达到？答案是 $C(A)$。
3. 不可达时，哪个可达输出离目标最近？答案由正交投影与最小二乘给出。
4. 哪些输入方向在作用后只改变长度、不改变方向，或对应最自然的输入—输出方向？答案通向特征向量和 SVD。

### 2. 差分矩阵展示可逆与奇异

普通差分矩阵

$$
D=\begin{bmatrix}
1&0&0\\
-1&1&0\\
0&-1&1
\end{bmatrix}
$$

把位置 $x=(x_1,x_2,x_3)^T$ 变为差分 $(x_1,x_2-x_1,x_3-x_2)^T$。它是下三角矩阵且三个对角元均为 $1$，因此可以从差分逐步恢复 $x$。

若首尾相连得到循环差分矩阵

$$
C=\begin{bmatrix}
-1&1&0\\
0&-1&1\\
1&0&-1
\end{bmatrix},
$$

则 $C(1,1,1)^T=0$。常数位移被完全丢失；并且每个输出 $b=Cx$ 都满足 $b_1+b_2+b_3=0$。这同时展示了奇异矩阵的两面：存在非零零空间方向，并且列空间不能覆盖整个输出空间。

### 3. 课程地图

- **Elimination**：把方程组变成容易读的形状。
- **Rank / basis / dimension**：数清独立方向。
- **Four fundamental subspaces**：统一输入端与输出端的可达、丢失结构。
- **Orthogonality / least squares**：处理无精确解时的最佳近似。
- **Determinant / eigenvalues**：描述体积缩放和反复作用。
- **SVD**：为任意 $m\times n$ 矩阵找最自然的正交输入、输出方向。

其中 [[Matrix Rank|秩]] 是贯穿全课的有效维数。若 $A$ 有 $n$ 列，秩—零度关系为

$$
\operatorname{rank}(A)+\dim N(A)=n.
$$

本式将在 Session 1.10 完整证明。

### 4. Recitation 反向推理例题

已知 $A$ 有三列，且

$$
Ax=b\quad\text{的全部解为}\quad
x=\begin{bmatrix}0\\1\\1\end{bmatrix}
+t\begin{bmatrix}0\\2\\1\end{bmatrix},
\qquad b=\begin{bmatrix}1\\4\\1\\1\end{bmatrix}.
$$

尺寸先行：$x\in\mathbb R^3$、$b\in\mathbb R^4$，故 $A\in\mathbb R^{4\times3}$。记列为 $c_1,c_2,c_3$。特解给出

$$
c_2+c_3=b,
$$

齐次方向给出

$$
2c_2+c_3=0.
$$

> [!note] Transcript 勘误
> 本地 transcript 的题首把齐次方向写成 $(1,2,1)^T$，后续计算却使用 $(0,2,1)^T$。原始 MIT Fall 1999 Quiz 1 Q4 与官方解答都确认后者正确。因此
> $$
> c_2+c_3=b,\qquad 2c_2+c_3=0
> $$
> 给出 $c_2=-b,c_3=2b$。又因全部解只有一个自由方向，nullity $=1$、rank $=2$；而 $c_2,c_3$ 只张成 $\operatorname{span}(b)$，故 $c_1$ 必须不是 $b$ 的倍数，才能使列空间达到二维。

### 5. 为什么“结构先于计算”

同一道题可有很多行运算路线，但秩、零空间维数、可解条件不会随路线改变。可靠解题顺序是：

1. 写尺寸；
2. 判断题目问存在性、唯一性还是参数化；
3. 再选择消元、子空间或分解；
4. 最后代回或做维数检查。

### 易错点与边界

- $N(A)$ 在输入空间 $\mathbb R^n$，$C(A)$ 在输出空间 $\mathbb R^m$，不能相加或直接比较，除非 $m=n$ 且另有语境。
- 秩下降同时影响存在性与唯一性，但两者不是同一句话：存在性由 $b\in C(A)$ 决定，唯一性由 $N(A)=\{0\}$ 决定。
- Overview 是官方第二讲；不要按本地 `Ses1.13` 文件名把它放到图论之后。

### 三道自检题

> [!question]- 1. 为什么循环差分矩阵不可能可逆？
> 因为 $C\mathbf1=0$ 且 $\mathbf1\ne0$，映射不是一一对应。

> [!question]- 2. 若 $A\in\mathbb R^{5\times3}$ 且 nullity $=1$，rank 是多少？
> 由 rank-nullity，$r=3-1=2$。

> [!question]- 3. 若 $N(A)=\{0\}$，是否保证每个 $b\in\mathbb R^m$ 都可解？
> 不保证。它只保证“至多一个解”；还需 $C(A)=\mathbb R^m$ 才保证存在。高矩阵可有独立列但不能覆盖全部 $\mathbb R^m$。

### 知识链小结

$Ax=b$ → 输入端 $N(A)$ 与输出端 $C(A)$ → rank 计数有效方向 → 正交、特征值与 SVD 将在后续单元继续刻画这些方向。

## Session 1.3 Elimination with matrices

### 本节问题与前置知识

**问题**：怎样用不改变解集的操作，把 $Ax=b$ 变成可回代的上三角系统？

**前置知识**：矩阵表示、矩阵乘向量、方程组等价。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.2sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S03_Lecture_Elimination_with_Matrices.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S03_Recitation_Recitation_Elimination_with_Matrices.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.2prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.2sol.pdf#page=1|official solution p.1]]

### 1. 三种基本行操作

[[Gaussian Elimination|高斯消元（Gaussian elimination）]]对增广矩阵 $[A\mid b]$ 反复使用三类可逆行操作：

1. 交换两行；
2. 一行乘非零标量；
3. 一行加上另一行的任意倍数。

> [!proof] 行操作为什么保持解集
> **目标**：说明每种行操作前后方程组等价。
>
> **构造与依据**：交换方程只改变书写顺序；把方程乘 $c\ne0$ 可再乘 $1/c$ 恢复；以 $R_i\leftarrow R_i-kR_j$ 替换第 $i$ 行，可用逆操作 $R_i\leftarrow R_i+kR_j$ 恢复。
>
> **边界**：一行乘 $0$ 不可逆，会丢掉原方程，因此不允许。
>
> **结论**：三种操作均可逆，故新旧系统有完全相同的解集；对应的系数矩阵彼此[[Row Equivalence|行等价（row-equivalent）]]。

行操作也可以写成左乘[[Elementary Matrix|初等矩阵（elementary matrix）]]。若

$$
E_{21}=\begin{bmatrix}1&0\\-3&1\end{bmatrix},
$$

则 $E_{21}A$ 把 $A$ 的第二行替换为 $R_2-3R_1$。左乘改变行，右乘改变列，不能混淆。

### 2. 主元、换行与消元失败

每一步用一个非零[[Pivot Position|pivot（主元）]]消去其下方元素。若预定主元是 $0$：

- 下方有非零数：交换行，将非零数换上来；
- 整列从当前位置向下全为 $0$：这一列没有主元，后来对应自由变量；
- 若增广列出现 $[0\ \cdots\ 0\mid c]$ 且 $c\ne0$：系统不相容。

主元个数就是矩阵的秩。主元的具体数值会随行缩放改变，但主元个数不会。

### 3. Recitation 完整消元例题

求解

$$
\begin{cases}
x-y-z+u=0,\\
2x+2z=8,\\
-y-2z=-8,\\
3x-3y-2z+4u=7.
\end{cases}
$$

增广矩阵尺寸是 $4\times5$：

$$
\left[\begin{array}{rrrr|r}
1&-1&-1&1&0\\
2&0&2&0&8\\
0&-1&-2&0&-8\\
3&-3&-2&4&7
\end{array}\right].
$$

先做 $R_2\leftarrow R_2-2R_1$、$R_4\leftarrow R_4-3R_1$：

$$
\left[\begin{array}{rrrr|r}
1&-1&-1&1&0\\
0&2&4&-2&8\\
0&-1&-2&0&-8\\
0&0&1&1&7
\end{array}\right].
$$

再做 $R_3\leftarrow R_3+\tfrac12R_2$：

$$
\left[\begin{array}{rrrr|r}
1&-1&-1&1&0\\
0&2&4&-2&8\\
0&0&0&-1&-4\\
0&0&1&1&7
\end{array}\right].
$$

第三个预定主元为 $0$，交换 $R_3,R_4$ 得上三角系统。由下往上回代：

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

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-elimination-lu.png|760]]

### Homework：全部题目与逐步解答

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

### 易错点、边界与反例

- 行操作必须同时作用于 $b$；只消 $A$ 会得到另一个方程组。
- “无主元”不一定无解：它可能只意味着自由变量；只有矛盾行才表示无解。
- 初等矩阵写在左边；$AE$ 一般执行的是列操作。
- 数值计算中若主元很小，实际算法会做 pivoting；本课先关注精确代数结构。

### 三道自检题

> [!question]- 1. 哪个矩阵实现 $R_3\leftarrow R_3+2R_1$？
> **答案**：$3\times3$ 单位矩阵的 $(3,1)$ 元改为 $2$，即 $E=I+2e_3e_1^T$。

> [!question]- 2. 行最简过程中出现 $[0\ 0\ 0\mid5]$ 表示什么？
> **答案**：方程 $0=5$，系统不相容，无解。

> [!question]- 3. 为什么交换两行不改变解？
> **答案**：只是交换两个必须同时满足的方程的书写顺序，且交换操作自身就是逆操作。

### 知识链小结

可逆行操作 → 上三角形 → 主元与 rank → 回代；下一节把行操作写成矩阵乘法，并研究对所有 $b$ 一次性求解的逆矩阵。

## Session 1.4 Multiplication and inverse matrices

### 本节问题与前置知识

**问题**：矩阵乘法如何表示变换复合？什么时候存在能撤销 $A$ 的矩阵？

**前置知识**：矩阵乘向量、初等矩阵与消元。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.3sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S04_Lecture_Multiplication_and_Inverse_Matrices.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S04_Recitation_Inverse_Matrices.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.3prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.3sol.pdf#page=1|official solution p.1]]

### 1. 矩阵乘法的四种读法

设 $A\in\mathbb R^{m\times n}$、$B\in\mathbb R^{n\times p}$，则 $AB\in\mathbb R^{m\times p}$。

1. **行乘列**：$(AB)_{ij}=\sum_{k=1}^n a_{ik}b_{kj}$。
2. **按列**：$AB$ 的第 $j$ 列是 $A$ 乘 $B$ 的第 $j$ 列。
3. **按行**：$AB$ 的第 $i$ 行是 $A$ 的第 $i$ 行乘 $B$。
4. **外积和**：
   $$
   AB=\sum_{k=1}^n A_{:k}B_{k:},
   $$
   每项都是列向量乘行向量的秩至多一矩阵。

矩阵乘法表示变换复合：$ABx=A(Bx)$，因此先做 $B$、后做 $A$。它满足结合律与分配律，但一般不满足交换律。

### 2. 逆矩阵

若方阵 $A\in\mathbb R^{n\times n}$ 存在 $A^{-1}$ 使

$$
A^{-1}A=AA^{-1}=I_n,
$$

则称 $A$ 可逆，$A^{-1}$ 是 [[Matrix Inverse|逆矩阵]]。于是 $Ax=b$ 的唯一解为 $x=A^{-1}b$。

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

### 3. Gauss–Jordan 求逆的原理

对增广矩阵 $[A\mid I]$ 做相同行操作，相当于左乘一串初等矩阵 $E_k\cdots E_1$。若左侧最终成为 $I$，则

$$
E_k\cdots E_1A=I,
$$

故 $E_k\cdots E_1=A^{-1}$，右侧同时变成 $A^{-1}$：

$$
[A\mid I]\longrightarrow[I\mid A^{-1}].
$$

若左侧无法产生 $n$ 个主元，$A$ 是 [[Singular Matrix|奇异矩阵]]，逆不存在。

### 4. 可逆等价链

对 $n\times n$ 方阵，下列命题等价：

- $A$ 可逆；
- 消元有 $n$ 个主元；
- $Ax=0$ 只有零解；
- 列向量线性无关；
- 列空间为 $\mathbb R^n$；
- 对每个 $b$，$Ax=b$ 有唯一解。

这组结论集中见 [[Invertible Matrix Equivalence Chain|可逆矩阵等价链]]。其逻辑核心是：无非零丢失方向保证一一性，覆盖整个输出空间保证满射；同维有限维空间中二者等价。

### 5. Recitation 的参数矩阵

Recitation 计算

$$
A=\begin{bmatrix}
a&b&b\\
a&a&b\\
a&a&a
\end{bmatrix}
$$

的可逆条件和逆矩阵。先用行差制造主元：

$$
R_3\leftarrow R_3-R_2=(0,0,a-b),
$$

$$
R_2\leftarrow R_2-R_1=(0,a-b,0),
$$

第一主元是 $a$，后两个主元是 $a-b$。若使用 Unit II 才会系统学习的行列式语言，也可把这一结果写成 $\det A=a(a-b)^2$。

因此恰在

$$
a\ne0,\qquad a\ne b
$$

时可逆。继续对 $[A\mid I]$ 做 Gauss–Jordan，得到

$$
A^{-1}=\begin{bmatrix}
\dfrac1{a-b}&0&-\dfrac{b}{a(a-b)}\\[6pt]
-\dfrac1{a-b}&\dfrac1{a-b}&0\\[6pt]
0&-\dfrac1{a-b}&\dfrac1{a-b}
\end{bmatrix}.
$$

例如第一行第一列的乘积为 $a/(a-b)-b/(a-b)=1$，第一行第二列为 $b/(a-b)-b/(a-b)=0$；其余位置同理可验证 $AA^{-1}=I$。这里的关键不是记住公式，而是先识别每个可能为零的主元，再在合法条件下相除。

### Homework：全部题目与逐步解答

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

### 易错点、边界与反例

- $AB\ne BA$ 一般成立；甚至 $AB$ 有定义时 $BA$ 也可能无定义。
- $(AB)^{-1}=B^{-1}A^{-1}$，顺序必须反转，因为撤销复合要先撤销最后执行的 $A$。
- $(A+B)^{-1}$ 一般不等于 $A^{-1}+B^{-1}$。
- 求逆只适用于方阵；解一般矩阵系统应使用消元和子空间语言。

### 三道自检题

> [!question]- 1. 若 $A$ 为 $2\times3$、$B$ 为 $3\times4$，$AB$ 的尺寸是什么？$BA$ 呢？
> **答案**：$AB$ 为 $2\times4$；$BA$ 的内侧尺寸 $4$ 与 $2$ 不合，未定义。

> [!question]- 2. 证明若 $A$ 可逆且 $Ax=0$，则 $x=0$。
> **答案**：左乘 $A^{-1}$：$x=I x=A^{-1}Ax=A^{-1}0=0$。

> [!question]- 3. 为什么 $[A\mid I]$ 左侧出现零行就不能得到逆？
> **答案**：零行表示主元数少于 $n$，无法经可逆行操作把左侧变为有 $n$ 个主元的 $I$。

### 知识链小结

矩阵乘法 = 变换复合 → 初等矩阵 = 可逆行操作 → Gauss–Jordan 同时求出撤销变换的 $A^{-1}$ → 下一节把整串消元压缩为 $A=LU$。

## Session 1.5 Factorization into A = LU

### 本节问题与前置知识

**问题**：消元产生的一整串操作，怎样保存为一次可复用的矩阵分解？

**前置知识**：初等矩阵、上三角回代、逆矩阵。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S05_Lecture_Factorization_into_A_LU.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S05_Recitation_LU_Decomposition.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.4prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.4sol.pdf#page=1|official solution p.1]]

### 1. 从消元到 [[LU Decomposition|LU 分解]]

设消元不需要换行。若初等矩阵依次满足

$$
E_k\cdots E_2E_1A=U,
$$

其中 $U$ 为上三角矩阵，则

$$
A=E_1^{-1}E_2^{-1}\cdots E_k^{-1}U=LU.
$$

$L$ 是单位下三角矩阵；在没有换行的标准消元中，$L$ 的下三角位置直接保存消元倍数。

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

### 2. 一个完整例子

令

$$
A=\begin{bmatrix}2&1&1\\4&-6&0\\-2&7&2\end{bmatrix}.
$$

第一列倍数为 $\ell_{21}=2$、$\ell_{31}=-1$：

$$
R_2\leftarrow R_2-2R_1,\qquad R_3\leftarrow R_3+R_1,
$$

得到

$$
\begin{bmatrix}2&1&1\\0&-8&-2\\0&8&3\end{bmatrix}.
$$

第二列倍数 $\ell_{32}=-1$，做 $R_3\leftarrow R_3+R_2$：

$$
U=\begin{bmatrix}2&1&1\\0&-8&-2\\0&0&1\end{bmatrix},\qquad
L=\begin{bmatrix}1&0&0\\2&1&0\\-1&-1&1\end{bmatrix}.
$$

检查 $LU$ 的三行：第一行等于 $U$ 第一行；第二行是 $2U_{1:}+U_{2:}=(4,-6,0)$；第三行是 $-U_{1:}-U_{2:}+U_{3:}=(-2,7,2)$，确实恢复 $A$。

### 3. 为什么 LU 对多个右端特别有用

求 $Ax=b$ 时，若 $A=LU$，则先解

$$
Lc=b
$$

（前代），再解

$$
Ux=c
$$

（回代）。$A$ 的消元只做一次；不同 $b$ 只需两次三角求解。对稠密 $n\times n$ 矩阵，分解约需 $O(n^3)$，每个新右端约需 $O(n^2)$。

### 4. 换行时的 $PA=LU$

若某一步必须交换行，用 [[Permutation Matrix|置换矩阵]] $P$ 记录交换。常用约定是

$$
PA=LU.
$$

例如 $A$ 第一主元为 $0$ 而下方非零，不能写无换行的标准 $A=LU$；先用 $P$ 把非零行换上来。实际数值计算还会主动选绝对值较大的主元以改善稳定性。

### Homework：全部题目与逐步解答

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

### 易错点、边界与反例

- $L$ 中保存的是消元倍数 $\ell_{ij}$，而消元矩阵中是 $-\ell_{ij}$。
- 若发生换行，不能只把 $P$ 忘掉后仍写 $A=LU$。
- $A=LU$ 不是逐元素乘法；必须用矩阵乘法验算。
- 三角矩阵可逆当且仅当所有对角元非零。

### 三道自检题

> [!question]- 1. $A=LU$ 时，为什么先解 $Lc=b$ 再解 $Ux=c$？
> **答案**：令 $c=Ux$，则原式 $LUx=b$ 变为 $Lc=b$；两步合起来等价于原式。

> [!question]- 2. 消元倍数 $\ell_{31}=-2$ 在 $L$ 的哪里？
> **答案**：$L_{31}=-2$；消元矩阵对应位置则是 $2$。

> [!question]- 3. 若 $A=\begin{bmatrix}0&1\\2&3\end{bmatrix}$，为什么标准无换行 LU 失败？
> **答案**：第一主元为 $0$，无法用它消去下方的 $2$；需交换两行，写 $PA=LU$。

### 知识链小结

消元矩阵 $E$ → $EA=U$ → 逆向恢复 $A=LU$ → 多个右端共享一次分解；接下来把矩阵本身放进更一般的向量空间语言。

## Session 1.6 Transposes, permutations, vector spaces

### 本节问题与前置知识

**问题**：转置和置换怎样交换行列？一个集合满足什么条件才真的是向量空间？

**前置知识**：矩阵乘法、逆矩阵、线性组合。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.5sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S06_Lecture_Transposes_Permutations_Vector_Spaces.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S06_Recitation_Subspaces_of_Three_Dimensional_Space.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.5prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.5sol.pdf#page=1|official solution p.1]]

### 1. 转置（transpose）

若 $A\in\mathbb R^{m\times n}$，其转置 $A^T\in\mathbb R^{n\times m}$ 定义为

$$
(A^T)_{ij}=A_{ji}.
$$

基本恒等式：

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

若 $A^T=A$，称 $A$ 为 [[Symmetric Matrix|对称矩阵]]；若 $A^T=-A$，称为斜对称矩阵，实数情形下其对角元必须为 $0$。

### 2. 置换矩阵

置换矩阵 $P$ 是把 $I$ 的行重新排列得到的矩阵。左乘 $PA$ 重排 $A$ 的行，右乘 $AP$ 重排 $A$ 的列。每一行、每一列恰有一个 $1$，其余为 $0$，并且

$$
P^{-1}=P^T.
$$

证明：$P$ 的列是标准正交基的重排，所以 $P^TP=I$。

### 3. 向量空间与子空间

一个 [[Vector Space|向量空间]] 必须对向量加法和标量乘法封闭，并满足通常的加法、数乘公理。若 $S\subseteq V$ 在继承 $V$ 的运算后仍为向量空间，称 $S$ 是 [[Subspace|子空间]]。

实用的子空间判别法：非空集合 $S$ 是子空间，当且仅当对任意 $u,v\in S$、任意标量 $\alpha,\beta$，有

$$
\alpha u+\beta v\in S.
$$

> [!proof] 为什么此判别已包含零向量和负向量
> 取 $\alpha=\beta=0$ 可得 $0\in S$；取 $\alpha=-1,\beta=0$ 可得 $-u\in S$；取 $\alpha=\beta=1$ 得加法封闭。因此线性组合封闭足够。

Recitation 在 $\mathbb R^3$ 中展示：一个非零向量的 span 是过原点的直线；两个不共线向量的 span 是过原点的平面。两条不同直线的并集通常不是子空间，因为分别取一条线上的向量后，它们的和一般不在并集中；但两条线的 span 是它们的和空间。

### Homework：全部题目与逐步解答

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

> [!question]- Problem 5.2：对称与斜对称矩阵的自由度
> **题目转述**：$4\times4$ 矩阵若为对称或斜对称，分别有多少个可独立选择的元素？
>
> **解答**：对称矩阵由对角线及上三角决定，共
> $$
> 4+\binom42=4+6=10.
> $$
> 斜对称矩阵对角元全为 $0$，下三角由上三角的负数决定，故有 $\binom42=6$ 个自由参数。

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

### 易错点、边界与反例

- 仿射平面 $ax+by+cz=1$ 不过原点，因此不是子空间；对应齐次平面 $ax+by+cz=0$ 才是。
- “集合里有很多向量”与“是子空间”无关；关键是所有线性组合是否仍在集合中。
- $(AB)^T$ 反序；只逐个转置但不反序是错误的。
- $P^T=P$ 并非所有置换矩阵都成立；正确恒等式是 $P^T=P^{-1}$。

### 三道自检题

> [!question]- 1. $S=\{(x,y):x+y=1\}$ 是子空间吗？
> **答案**：不是；$(0,0)$ 不满足方程。

> [!question]- 2. 所有 $3\times3$ 上三角矩阵构成子空间吗？
> **答案**：构成。和与数乘仍保持下三角位置为零。

> [!question]- 3. 若 $P$ 交换第一、第三行，写出 $P$。
> **答案**：$P=\begin{bmatrix}0&0&1\\0&1&0\\1&0&0\end{bmatrix}$，且 $P^{-1}=P^T=P$。

### 知识链小结

转置交换行列 → 置换矩阵实现可逆重排 → 子空间以线性组合封闭为核心 → 下一节证明列空间和零空间确实是子空间。

## Session 1.7 Column space and nullspace

### 本节问题与前置知识

**问题**：哪些右端 $b$ 可达到？哪些输入方向被 $A$ 压成 $0$？

**前置知识**：span、子空间判别、矩阵乘向量。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.6sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S07_Lecture_Column_Space_and_Nullspace.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S07_Recitation_Vector_Subspaces.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.6prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.6sol.pdf#page=1|official solution p.1]]

### 1. 定义与所在空间

对 $A\in\mathbb R^{m\times n}$：

$$
C(A)=\{Ax:x\in\mathbb R^n\}
=\operatorname{span}\{a_1,\ldots,a_n\}\subseteq\mathbb R^m,
$$

$$
N(A)=\{x\in\mathbb R^n:Ax=0\}\subseteq\mathbb R^n.
$$

列空间回答 $Ax=b$ 的**存在性**：

$$
Ax=b\text{ 有解}\iff b\in C(A).
$$

零空间回答齐次自由度，并控制非齐次解的**唯一性**。

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

### 2. 一个同时读出两空间的例子

令

$$
A=\begin{bmatrix}1&2&3\\2&4&6\end{bmatrix}.
$$

三列都是 $(1,2)^T$ 的倍数，所以

$$
C(A)=\operatorname{span}\left\{\begin{bmatrix}1\\2\end{bmatrix}\right\}\subset\mathbb R^2.
$$

$b=(b_1,b_2)^T$ 可解当且仅当 $b_2=2b_1$。零空间方程只有一条独立约束：

$$
x_1+2x_2+3x_3=0.
$$

令 $x_2=s,x_3=t$，则

$$
x=s\begin{bmatrix}-2\\1\\0\end{bmatrix}
+t\begin{bmatrix}-3\\0\\1\end{bmatrix}.
$$

所以 $N(A)$ 是 $\mathbb R^3$ 中的二维平面。输出仅剩一维、输入丢掉二维，预告 $1+2=3$ 的秩—零度关系。

### 3. Recitation 的子空间快速判别

若集合由齐次线性条件 $b_1+b_2-b_3=0$ 描述，它正是矩阵 $[1\ 1\ -1]$ 的零空间，必为子空间。若条件改为 $b_3=b_1b_2$，则数乘不封闭，例如 $(1,1,1)$ 满足但 $(2,2,2)$ 不满足。若集合是固定向量加一个 span，必须检查固定向量是否已在该 span 中；否则它是仿射平移，不含 $0$。

### Homework：全部题目与逐步解答

> [!question]- Problem 6.1：子空间之和与并集
> **题目转述**：若 $S,T$ 是 $V$ 的子空间，证明 $S+T=\{s+t:s\in S,t\in T\}$ 是子空间；并解释两条直线时 $S+T$、$S\cup T$ 的区别以及 $\operatorname{span}(S\cup T)=S+T$。
>
> **解答**：零向量 $0=0_S+0_T\in S+T$。若 $u=s_1+t_1$、$v=s_2+t_2$，则
> $$
> \alpha u+\beta v=(\alpha s_1+\beta s_2)+(\alpha t_1+\beta t_2)\in S+T,
> $$
> 因为 $S,T$ 各自封闭，故 $S+T$ 是子空间。若 $S,T$ 是不同直线，$S\cup T$ 只含两条线，取 $s\in S\setminus T$、$t\in T\setminus S$ 后 $s+t$ 通常不在并集中；而 $S+T$ 是它们张成的平面。$S+T$ 是包含 $S\cup T$ 的子空间，所以包含其 span；反过来 span 包含所有 $s+t$，故二者相等。

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

### 易错点、边界与反例

- 行操作通常改变列空间中的具体列，但保持行空间与零空间；找 $C(A)$ 的基必须回到原矩阵选 pivot columns。
- $C(A)$ 是列向量的 span，不是“列向量组成的有限集合”。
- 非齐次解集通常不是子空间，因为不含 $0$；它是零空间的仿射平移。

### 三道自检题

> [!question]- 1. $C(A)$ 位于哪里？
> **答案**：若 $A$ 为 $m\times n$，则 $C(A)\subseteq\mathbb R^m$。

> [!question]- 2. 为什么 $N(A)$ 一定含零向量？
> **答案**：线性性给 $A0=0$。

> [!question]- 3. 若两列相同，写出一个非零零空间向量。
> **答案**：若 $a_i=a_j$，则 $e_i-e_j\in N(A)$，因为 $A(e_i-e_j)=a_i-a_j=0$。

### 知识链小结

$C(A)$ = 可达输出 → $N(A)$ = 丢失输入 → 解的存在与唯一被拆开 → 下一节用 rref 为 $N(A)$ 构造可计算的基。

## Session 1.8 Solving Ax = 0: pivot variables and special solutions

### 本节问题与前置知识

**问题**：怎样从消元后的矩阵系统地写出 $N(A)$ 的一组基？

**前置知识**：消元、主元、列空间与零空间。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S08_Lecture_Solving_Ax_0_Pivot_Variables_Special_Solutions.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S08_Recitation_Solving_Ax_0.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.7prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.7sol.pdf#page=1|official solution p.1]]

### 1. 行最简形与变量分工

把 $A$ 化为 [[Reduced Row Echelon Form|行最简形]] $R$。由于行操作等价于左乘可逆矩阵 $E$，

$$
R=EA,\qquad Rx=0\iff EAx=0\iff Ax=0,
$$

所以行操作保持零空间。

- 主元列对应 **pivot variables（主元变量）**；
- 非主元列对应 **free variables（自由变量）**；
- 每个自由变量依次取 $1$、其余自由变量取 $0$，得到一个 **special solution（特殊解）**。

若 $A$ 有 $n$ 列、rank 为 $r$，自由变量数是 $n-r$，也就是 $\dim N(A)$。

### 2. 标准块形式

适当重排列后，rref 可写成

$$
R=\begin{bmatrix}I_r&F\\0&0\end{bmatrix},
$$

其中 $F$ 是主元方程里自由变量的系数块。把 $x$ 分成主元部分 $x_p$ 和自由部分 $x_f$：

$$
x_p+Fx_f=0\Longrightarrow x_p=-Fx_f.
$$

因此

$$
x=\begin{bmatrix}x_p\\x_f\end{bmatrix}
=\begin{bmatrix}-F\\I_{n-r}\end{bmatrix}x_f.
$$

矩阵 $\begin{bmatrix}-F\\I\end{bmatrix}$ 的列就是特殊解。

> [!proof] 特殊解为什么构成零空间的基
> **张成**：任意 $x_f$ 都可按标准基展开，公式表明对应 $x$ 是特殊解的相同系数组合。
>
> **无关**：若特殊解的线性组合为 $0$，观察其自由变量分量；这些分量正好组成单位矩阵，所以所有系数都必须为 $0$。
>
> **结论**：它们既张成又线性无关，构成 $N(A)$ 的基；数量为 $n-r$。

### 3. Recitation：齐次平面与仿射平面

非齐次平面

$$
x-5y+2z=9

$$

与齐次平面 $x-5y+2z=0$ 平行。齐次式以 $x$ 为主元变量、$y,z$ 为自由变量：

$$
x=5y-2z.
$$

特殊解为 $(5,1,0)^T$、$(-2,0,1)^T$，所以

$$
N(A)=\operatorname{span}\left\{
\begin{bmatrix}5\\1\\0\end{bmatrix},
\begin{bmatrix}-2\\0\\1\end{bmatrix}
\right\}.
$$

非齐次平面再加特解 $(9,0,0)^T$。这是下一节完整解结构的几何原型。

### Homework：全部题目与逐步解答

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

> [!question]- Problem 7.2：控制乘积的秩
> **题目转述**：令 $B=\begin{bmatrix}1&1\\1&1\end{bmatrix}$，找 $A_1,A_2$ 使 $\operatorname{rank}(A_1B)=1$、$\operatorname{rank}(A_2B)=0$。
>
> **解答**：取 $A_1=I_2$，则 $A_1B=B$，两列相同且非零，rank $=1$。取
> $$
> A_2=\begin{bmatrix}1&-1\\1&-1\end{bmatrix}.
> $$
> $B$ 的每一列都是 $(1,1)^T$，而 $A_2(1,1)^T=0$，所以 $A_2B=0$、rank $=0$。平凡选择 $A_2=0$ 也成立，但此选择更能显示 $C(B)\subseteq N(A_2)$ 才是乘积为零的结构原因。

### 易错点、边界与反例

- 求 $N(A)$ 可使用 rref，但求 $C(A)$ 的基不能直接拿 rref 的 pivot columns；行操作改变列本身。
- “一个自由变量”对应“一条特殊解”，不是只对应一个解；其任意标量倍数都在零空间中。
- 零空间是子空间，参数式必须包含 $x=0$；若不包含，说明你混入了非齐次特解。

### 三道自检题

> [!question]- 1. $A$ 有 7 列、4 个主元，$N(A)$ 的维数是多少？
> **答案**：$7-4=3$。

> [!question]- 2. 为什么主元变量不能任意取值？
> **答案**：它们由 rref 中的主元方程表示为自由变量的线性组合；任意选择会违反方程。

> [!question]- 3. 若 $R=[I_2\ F]$ 且 $F=\begin{bmatrix}2\\-3\end{bmatrix}$，写出特殊解。
> **答案**：$x=(-F,1)^T=(-2,3,1)^T$。

### 知识链小结

rref 保持 $N(A)$ → pivot/free 变量分工 → 特殊解给零空间基 → nullity $=n-r$ → 下一节把任意非齐次解写成“特解 + 零空间”。

## Session 1.9 Solving Ax = b: row reduced form R

### 本节问题与前置知识

**问题**：非齐次系统何时相容？相容时怎样一次写出全部解？

**前置知识**：rref、特殊解、列空间与零空间。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.8sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S09_Lecture_Solving_Ax_b_Row_Reduced_Form_R.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S09_Recitation_Solving_Ax_b.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.8prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.8sol.pdf#page=1|official solution p.1]]

### 1. 相容条件必须从增广矩阵读

对 $[A\mid b]$ 做相同行操作。若出现

$$
\begin{bmatrix}0&\cdots&0\mid c\end{bmatrix},\qquad c\ne0,
$$

即方程 $0=c$，则无解。否则系统相容。

等价的空间说法是

$$
Ax=b\text{ 相容}\iff b\in C(A).
$$

消元给可计算判据，列空间给结构解释。

### 2. 完整解 = 特解 + 零空间

先找任意一个 particular solution（特解）$x_p$ 满足 $Ax_p=b$，再求 $N(A)$。所有解是

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

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-affine-solution.png|760]]

### 3. Recitation：由 $b$ 决定的相容条件

考虑

$$
\begin{cases}
x-2y-2z=b_1,\\
2x-5y-4z=b_2,\\
4x-9y-8z=b_3.
\end{cases}
$$

对增广矩阵做 $R_2\leftarrow R_2-2R_1$、$R_3\leftarrow R_3-4R_1$，再消第二列，最后一行变为

$$
0=-2b_1-b_2+b_3.
$$

所以相容当且仅当

$$
b_3=2b_1+b_2.
$$

相容时 $z$ 自由，取 $z=0$ 得

$$
x_p=\begin{bmatrix}5b_1-2b_2\\2b_1-b_2\\0\end{bmatrix};
$$

齐次特殊解为 $(2,0,1)^T$，故

$$
x=x_p+t\begin{bmatrix}2\\0\\1\end{bmatrix}.
$$

验算策略：先验证 $Ax_p=b$，再验证 $A(2,0,1)^T=0$；线性性便保证整族都正确。

### Homework：全部题目与逐步解答

> [!question]- Problem 8.1：辨析完整解的三个错误说法
> **题目转述**：解释为什么以下说法都错：（a）完整解是 $x_p,x_n$ 的任意线性组合；（b）$Ax=b$ 至多有一个特解；（c）若 $A$ 可逆，零空间中没有解 $x_n$。
>
> **解答**：（a）$x_p$ 的系数必须是 $1$；若写 $\alpha x_p+x_n$，则其像为 $\alpha b$，只有 $\alpha=1$（或特殊的 $b=0$）才仍为 $b$。（b）若 $x_n\in N(A)$，则 $x_p+x_n$ 也是特解；只要零空间非平凡，就有无穷多个。（c）任何矩阵的零空间至少含 $x_n=0$；可逆只表示它不含非零向量。

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

> [!question]- Problem 8.3：相同解算子是否意味着矩阵相同
> **题目转述**：若对每个 $b$，$Ax=b$ 与 $Cx=b$ 有完全相同的解集，是否必有 $A=C$？
>
> **解答**：是。任取尺寸合适的 $y$，令 $b=Ay$。于是 $y$ 是 $Ax=b$ 的解；按假设也是 $Cx=b$ 的解，所以 $Cy=b=Ay$。这对所有 $y$ 成立，特别对每个标准基 $e_j$ 成立，于是 $A$、$C$ 的第 $j$ 列分别为 $Ae_j,Ce_j$ 且相等。因此 $A=C$。

### 易错点、边界与反例

- 特解不唯一；“particular”只是任选一个方便代表。
- 完整解不是 $\operatorname{span}\{x_p,N(A)\}$，因为那会允许改变 $x_p$ 的系数。
- 相容条件来自左端行之间的依赖关系；消元时必须带着 $b$。
- 若 $N(A)=\{0\}$，相容系统才唯一；它并不自动保证相容。

### 三道自检题

> [!question]- 1. 若 $x_1,x_2$ 都满足 $Ax=b$，$x_1-x_2$ 在哪里？
> **答案**：在 $N(A)$，因为 $A(x_1-x_2)=b-b=0$。

> [!question]- 2. 解集 $x=x_p+su+tv$ 的几何维数是多少？
> **答案**：若 $u,v$ 线性无关，则是二维仿射平面；其方向空间是 $\operatorname{span}\{u,v\}=N(A)$。

> [!question]- 3. 若 $Ax=b$ 有两个不同解，能否有恰好两个解？
> **答案**：不能。在 $\mathbb R$ 上，它们之差给非零零空间方向，$x_1+t(x_2-x_1)$ 对每个实数 $t$ 都是解，因此有无穷多个。

### 知识链小结

增广消元判相容 → 特解定位仿射平移 → 零空间给全部方向 → 完整解 = $x_p+N(A)$ → 下一节用线性无关、基和维数准确计数这些方向。

## Session 1.10 Independence, basis, and dimension

### 本节问题与前置知识

**问题**：一组生成向量中哪些方向真正不可替代？怎样用最少且不冗余的向量描述空间？

**前置知识**：span、零空间、主元与自由变量。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S10_Lecture_Independence_Basis_and_Dimension.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S10_Recitation_Basis_and_Dimension.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.9prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.9sol.pdf#page=1|official solution p.1]]

### 1. 线性无关

向量 $v_1,\ldots,v_k$ 称为 [[Linear Independence|线性无关]]，若

$$
c_1v_1+\cdots+c_kv_k=0

$$

只允许平凡系数 $c_1=\cdots=c_k=0$。把这些向量作为矩阵 $V$ 的列，则

$$
\{v_j\}\text{ 无关}\iff N(V)=\{0\}\iff V\text{ 的每一列都是主元列}.
$$

一组向量若包含零向量、重复向量，或某个向量是其余向量的组合，必然相关。

### 2. 基与维数

空间 $S$ 的一组 [[Basis|基]] 同时满足：

1. 张成 $S$；
2. 线性无关。

基实现“存在且唯一的坐标表示”。若 $v_1,\ldots,v_k$ 是基，每个 $s\in S$ 可唯一写成 $s=\sum c_iv_i$。

> [!proof] 为什么基坐标唯一
> 若 $s=\sum c_iv_i=\sum d_iv_i$，两式相减：
> $$
> \sum_i(c_i-d_i)v_i=0.
> $$
> 基向量线性无关，所以每个 $c_i-d_i=0$，即 $c_i=d_i$。

有限维空间任意两组基含有相同数量的向量，这个数量称为 [[Dimension|维数]]。零空间的基由特殊解给出；列空间的基由原矩阵的主元列给出。

### 3. [[Rank-Nullity Theorem|秩—零度定理]]

对 $A\in\mathbb R^{m\times n}$，设 rank $=r$：

$$
\dim C(A)=r,\qquad \dim N(A)=n-r.
$$

因此

$$
\boxed{\operatorname{rank}(A)+\operatorname{nullity}(A)=n}.
$$

> [!proof] [[Rank-Nullity Theorem Proof|目标—构造—计数证明]]
> **目标**：证明输入空间的维数 $n$ 被行空间有效方向与零空间丢失方向分成 $r+(n-r)$。
>
> **构造**：把 $A$ 化为 rref。恰有 $r$ 个主元列，因此有 $r$ 个主元变量；余下 $n-r$ 个变量自由。
>
> **逐步依据**：每个自由变量产生一条特殊解；上一节已证明这些特殊解构成 $N(A)$ 的基，所以 $\dim N(A)=n-r$。原矩阵的 $r$ 个主元列构成 $C(A)$ 的基，所以 $\dim C(A)=r$。
>
> **边界**：$r$ 可能为 $0$ 或 $\min(m,n)$，公式仍成立。
>
> **结论**：$r+(n-r)=n$。

### 4. 如何从生成集抽取基

若 $v_1,\ldots,v_k$ 是列向量，把它们组成 $A=[v_1\cdots v_k]$ 并消元。rref 的主元**位置**告诉你应从**原矩阵**选哪些列。行操作保持列之间的线性依赖关系，但不保持原列空间中的具体列，因此不能拿 rref 的主元列替代原列。

Recitation 也讨论把向量作为行消元：非零行可构成行空间的基；但若原问题问原列向量中的一个子集，必须按列放置并回到原列选择。

### Homework：全部题目与逐步解答

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

### 易错点、边界与反例

- “张成”不保证无关；“无关”也不保证张成目标空间。
- $k>n$ 个 $\mathbb R^n$ 向量必相关；$k<n$ 个向量不可能张成整个 $\mathbb R^n$。
- 不同基可以长得完全不同，但基向量数量相同。
- 行操作后选择列空间基，必须使用原矩阵对应列。

### 三道自检题

> [!question]- 1. $\mathbb R^5$ 中能否有 6 个线性无关向量？
> **答案**：不能；把它们作列得到 $5\times6$ 矩阵，rank 至多为 $5$。

> [!question]- 2. 两个向量张成一个平面需要什么条件？
> **答案**：二者都非零且不互为倍数，即线性无关。

> [!question]- 3. $A$ 有 8 列且 $N(A)$ 的基有 3 个向量，rank 是多少？
> **答案**：$8-3=5$。

### 知识链小结

无关 = 无冗余 → 基 = 无冗余且完整 → 维数 = 基的长度 → rank-nullity 计数输入自由度 → 下一节把同样的计数推广到四个基本子空间。

## Session 1.11 The four fundamental subspaces

### 本节问题与前置知识

**问题**：一个 $m\times n$ 矩阵在输入端和输出端分别决定哪四个空间？它们的维数和正交关系是什么？

**前置知识**：列空间、零空间、基、维数、转置。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S11_Lecture_The_Four_Fundamental_Subspaces.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S11_Recitation_Computing_the_Four_Fundamental_Subspaces.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.10prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.10sol.pdf#page=1|official solution p.1]]

### 1. 四空间总表

对 $A\in\mathbb R^{m\times n}$、rank $=r$：

| 空间 | 定义 | 所在环境 | 维数 |
|---|---|---:|---:|
| $C(A)$ | $A$ 的列的 span | $\mathbb R^m$ | $r$ |
| $N(A)$ | $Ax=0$ 的全部解 | $\mathbb R^n$ | $n-r$ |
| [[Row Space|行空间]] $C(A^T)$ | $A$ 的行的 span | $\mathbb R^n$ | $r$ |
| [[Left Nullspace|左零空间]] $N(A^T)$ | $A^Ty=0$ 的全部解 | $\mathbb R^m$ | $m-r$ |

行空间与列空间维数相同，都是 rank；这就是“行秩 = 列秩”。实践中，rref 的非零行给行空间基，原矩阵的主元列给列空间基。

### 2. 正交关系

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

这说明

$$
\mathbb R^n=C(A^T)\oplus N(A),\qquad
\mathbb R^m=C(A)\oplus N(A^T),
$$

其中 $\oplus$ 表示每个向量都有唯一的“两部分相加”表示；这里两直和还是正交直和。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-four-subspaces.png|760]]

### 3. $A$ 如何在四空间图中作用

- $A$ 把 $N(A)$ 中所有向量送到 $0$；
- $A$ 把行空间中的 $r$ 维有效输入一一映到列空间；
- $A^T$ 把左零空间送到 $0$；
- $A^T$ 把列空间中的有效输出映回行空间。

严格地说，限制映射

$$
A\big|_{C(A^T)}:C(A^T)\to C(A)
$$

是双射。它是满射，因为每个 $Ax$ 只依赖 $x$ 的行空间分量；它是单射，因为行空间与零空间交只有 $0$。

### 4. 从消元读四空间

见 [[Reading the Four Fundamental Subspaces from RREF|从 RREF 读四个基本子空间]]：

1. $C(A)$：取**原矩阵**的主元列；
2. $N(A)$：从 rref 的自由变量构造特殊解；
3. $C(A^T)$：取 rref 的非零行；
4. $N(A^T)$：解 $A^Ty=0$，或在完整消元矩阵 $E$ 中读取把 $A$ 消成零行的行组合。

Recitation 用 $B=LU$ 的 rank-2 例子说明：取 $L$ 中与 $U$ 的两个非零 pivot positions 对应的两列，可给出 $C(B)$ 的基；$U$ 给 $N(B)$ 与行空间，$E=L^{-1}$ 中对应 $U$ 零行的那一行给左零空间向量。

### Homework：全部题目与逐步解答

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

> [!question]- Problem 10.2：转置系统的存在与唯一
> **题目转述**：$A^Ty=d$ 在 $d$ 属于哪个基本子空间时可解？在什么空间只含零向量时解唯一？
>
> **解答**：$A^T$ 的列空间是 $C(A^T)$，也就是 $A$ 的行空间，所以
> $$
> A^Ty=d\text{ 可解}\iff d\in C(A^T).
> $$
> 两个解之差位于 $N(A^T)$，因此解唯一当且仅当左零空间 $N(A^T)=\{0\}$。

### 易错点、边界与反例

- $C(A)$ 与 $N(A^T)$ 都在 $\mathbb R^m$；$C(A^T)$ 与 $N(A)$ 都在 $\mathbb R^n$。正交只能在同一个环境空间内谈。
- 列空间的基来自原矩阵主元列，行空间基可以来自 rref 非零行。
- “四个空间互相正交”是错的；只有两对互为正交补。
- $N(A)$ 的维数是 $n-r$，左零空间的维数是 $m-r$，不要把 $m,n$ 对调。

### 三道自检题

> [!question]- 1. $A$ 为 $7\times5$、rank $3$，四空间维数各是多少？
> **答案**：$\dim C(A)=3$、$\dim N(A)=2$、$\dim C(A^T)=3$、$\dim N(A^T)=4$。

> [!question]- 2. 若 $y\in N(A^T)$，证明它与每列正交。
> **答案**：$A^Ty=0$ 的第 $j$ 个分量是 $a_j^Ty=0$。

> [!question]- 3. 为什么 $A$ 在整个 $\mathbb R^n$ 上不一定一一对应，但在行空间上是一一对应？
> **答案**：整个空间可能含非零 $N(A)$；行空间与 $N(A)$ 正交，交集仅有 $0$，故限制到行空间后核为零。

### 知识链小结

列空间/零空间 → 转置产生行空间/左零空间 → 两对正交补 → rank 同时给两个有效维数 → 四空间图统一 $A$ 的输入、输出结构。

## Session 1.12 Matrix spaces, rank 1, and small world graphs

### 本节问题与前置知识

**问题**：向量空间的“向量”能否本身就是矩阵？为什么秩一矩阵是一般矩阵的基本构件？

**前置知识**：子空间、基、维数、外积、rank。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.11sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S12_Lecture_Matrix_Spaces_Rank_1_Small_World_Graphs.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S12_Recitation_Matrix_Spaces.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.11prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.11sol.pdf#page=1|official solution p.1]]

### 1. 矩阵空间

所有 $m\times n$ 实矩阵组成向量空间 $M_{m\times n}$。标准基为

$$
E_{ij}\quad(1\le i\le m,\ 1\le j\le n),
$$

其中只有 $(i,j)$ 元为 $1$。任意矩阵唯一写成

$$
A=\sum_{i=1}^m\sum_{j=1}^n a_{ij}E_{ij},
$$

所以

$$
\dim M_{m\times n}=mn.
$$

对称矩阵、斜对称矩阵、上三角矩阵、trace 为零的方阵都可构成子空间；可逆矩阵集合不是子空间，因为不含零矩阵，且两个可逆矩阵的和可能奇异。

### 2. [[Rank-One Matrix|秩一矩阵]]与外积

若 $u\in\mathbb R^m$、$v\in\mathbb R^n$ 均非零，则

$$
A=uv^T
$$

是 $m\times n$ 矩阵，其第 $j$ 列为 $v_j u$，所有列都在同一直线上，所以 rank $=1$。

反过来，任一非零 rank-1 矩阵的所有列都是某个非零列 $u$ 的倍数；把倍数收进 $v$，便得 $A=uv^T$。

> [!proof] rank-$r$ 矩阵可分成 $r$ 个秩一矩阵
> 取 $A$ 的 $r$ 个主元列组成 $C\in\mathbb R^{m\times r}$。每一列都由这些主元列组合，所以存在 $R\in\mathbb R^{r\times n}$ 使 $A=CR$。按内维展开：
> $$
> A=\sum_{k=1}^r C_{:k}R_{k:}.
> $$
> 每项是列乘行，rank 至多 $1$。这给出 $r$ 个秩一构件；不能少于 $r$ 个，否则秩的次可加性会使总秩小于 $r$。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-rank-one.png|760]]

### 3. Recitation：带固定零空间向量的矩阵子空间

考虑所有满足

$$
A\begin{bmatrix}2\\1\\1\end{bmatrix}=0,\qquad A\in M_{2\times3}
$$

的矩阵。若 $A,B$ 满足条件，则 $(\alpha A+\beta B)v=0$，故它是子空间。每一行 $(a,b,c)$ 满足 $2a+b+c=0$，即

$$
(a,b,c)=a(1,0,-2)+b(0,1,-1).
$$

两行可独立选择，因此基可取

$$
\begin{bmatrix}1&0&-2\\0&0&0\end{bmatrix},
\begin{bmatrix}0&1&-1\\0&0&0\end{bmatrix},
\begin{bmatrix}0&0&0\\1&0&-2\end{bmatrix},
\begin{bmatrix}0&0&0\\0&1&-1\end{bmatrix},
$$

维数为 $4$。相比之下，“列空间包含固定非零向量 $(2,1)^T$”的矩阵集合不含零矩阵，不是子空间。

### 4. Small-world graph 的矩阵视角

图的邻接矩阵把连接关系变为矩阵。矩阵幂 $(A^k)_{ij}$ 可计数从节点 $i$ 到 $j$ 的长度 $k$ walk；局部边加上少量远程边可能显著缩短平均路径。这里的重点不是图论细节，而是同一套矩阵乘法能编码网络中的传播与连接。

### Homework：全部题目与逐步解答

> [!question]- Problem 11.1（Optional）：五个置换矩阵的基
> **题目转述**：把 $3\times3$ 单位矩阵写成其余五个置换矩阵的组合，并证明这五个矩阵线性无关；它们构成“所有行和、列和均相等”的矩阵子空间的一组基。
>
> **解答**：记三个换位矩阵为 $P_{21},P_{31},P_{32}$，两个三循环为 $P_{32}P_{21},P_{21}P_{32}$。前三者相加得到全一矩阵；后两者相加得到对角为 $0$、非对角为 $1$ 的矩阵，所以
> $$
> I=P_{21}+P_{31}+P_{32}-P_{32}P_{21}-P_{21}P_{32}.
> $$
> 若五者线性组合为零，查看三个对角位置可依次迫使三个换位矩阵的系数为零；再看剩余非对角位置迫使两个三循环系数为零，故五者无关。每个置换矩阵的各行和、列和都是 $1$，其组合具有共同的行和、列和。目标空间原有 $9$ 个参数；“三行和相等”给两个独立约束，“三列和相等”再给两个独立约束，而共同的行和与列和因总元素和相同而自动相等，所以维数是 $9-4=5$。已有五个无关矩阵，故它们确实是一组基。

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

### 易错点、边界与反例

- “rank 恰为 1 的矩阵集合”不是子空间：两个 rank-1 矩阵相加可能 rank 2，且不含零矩阵。
- $uv^T$ 的尺寸由 $u$ 的长度给行数、$v$ 的长度给列数。
- 矩阵空间中的线性算子也有核、像、rank-nullity；不要把输入维数误写成矩阵的行数或列数，$M_{m\times n}$ 的维数是 $mn$。

### 三道自检题

> [!question]- 1. $M_{2\times4}$ 的维数是多少？
> **答案**：$2\cdot4=8$。

> [!question]- 2. 若 $u\ne0,v\ne0$，为什么 $uv^T$ 不可能 rank 0？
> **答案**：取 $v_j\ne0$，第 $j$ 列 $v_ju\ne0$，矩阵非零；所有列共线，故 rank 恰为 1。

> [!question]- 3. 所有 $3\times3$ 可逆矩阵构成子空间吗？
> **答案**：不构成；零矩阵不可逆，且 $I+(-I)=0$。

### 知识链小结

向量空间可由矩阵充当元素 → rank-1 外积是矩阵的原子构件 → rank-$r$ 是 $r$ 个外积之和 → 矩阵算子仍满足核—像维数定理 → 下一节用关联矩阵编码图与网络。

## Session 1.13 Graphs, networks, and incidence matrices

### 本节问题与前置知识

**问题**：怎样用一个矩阵同时编码节点、边、势差、流量守恒与网络能量？

**前置知识**：四个基本子空间、转置、矩阵乘法。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.12sum.pdf#page=1|summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S13_Lecture_Graphs_Networks_Incidence_Matrices.pdf#page=1|lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S13_Recitation_Graphs_and_Networks.pdf#page=1|recitation transcript p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.12prob.pdf#page=1|homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.12sol.pdf#page=1|official solution p.1]]

### 1. 关联矩阵的定义

给有向图任意指定每条边的方向。若图有 $n$ 个节点、$m$ 条边，其 [[Incidence Matrix|关联矩阵]] $A\in\mathbb R^{m\times n}$ 每行对应一条边：边从节点 $i$ 指向节点 $j$，则该行在第 $i$ 列写 $-1$、第 $j$ 列写 $+1$，其余写 $0$。

方向只是记号选择；反转一条边只会把对应行乘 $-1$，不改变图的连通结构或 $A^TA$。

### 2. 四空间在网络中的含义

令节点势（potential）为 $x\in\mathbb R^n$，则

$$
e=Ax\in\mathbb R^m
$$

给出每条有向边的终点势减起点势。

- $N(A)$：所有边势差都为零的节点势。连通图中所有节点势相同，所以 $N(A)=\operatorname{span}\{\mathbf1\}$。
- $C(A)$：可由节点势产生的边势差。
- $N(A^T)$：满足每个节点净流量为零的边流，称为 cycle space；环流属于这里。
- $C(A^T)$：由边量累积到节点的净注入向量；其分量总和为零。

若图有 $c$ 个连通分量，则每个分量可有一个独立常势：

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

### 3. 传导、守恒与图 Laplacian

令 $C\in\mathbb R^{m\times m}$ 是对角 conductance（电导）矩阵，边流可按符号约定写成

$$
y=-CAx.
$$

若 $f\in\mathbb R^n$ 表示节点的**外部注入**，取“注入为正、网络净流出为正”的约定，则节点守恒写成

$$
-A^Ty=f.
$$

与 $y=-CAx$ 合并得

$$
A^TCAx=f.
$$

若把 $f$ 定义成外部净流出，或改用 $y=CAx$，守恒式中的符号会相应改变。重要的是从同一方向约定一致推导，不是死记正负号。矩阵

$$
L_G=A^TCA
$$

称加权[[Graph Laplacian|图 Laplacian（graph Laplacian）]]。它对称，并且

$$
x^TL_Gx=(Ax)^TC(Ax)=\sum_{e=1}^m c_e(\Delta x_e)^2\ge0.
$$

连通图中 $L_G\mathbf1=0$；势只确定到加一个常数，通常把一个节点接地来选定唯一代表。可解的注入必须满足 $\mathbf1^Tf=0$，即总流入等于总流出。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-incidence-network.png|760]]

### 4. Recitation：不用消元读出核

五节点六边的连通图，其关联矩阵为 $6\times5$。由连通性立刻得

$$
N(A)=\operatorname{span}\{(1,1,1,1,1)^T\},\qquad \operatorname{rank}(A)=4.
$$

故 $\dim N(A^T)=6-4=2$：两个独立基本环流生成全部平衡流。大环流可以写成两个小环流之和，所以不是新独立方向。

此外，$\operatorname{tr}(M)$ 表示方阵 $M$ 的对角元之和。于是

$$
\operatorname{tr}(A^TA)=\sum_{j=1}^n\|A_{:j}\|^2.
$$

关联矩阵第 $j$ 列每条相邻边贡献一个 $\pm1$，平方和等于节点度数。因此 trace 等于所有节点度数之和，也就是 $2m$；例中为 $12$。

### Homework：全部题目与逐步解答

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

### 易错点、边界与反例

- 关联矩阵的形状是“边数 $\times$ 节点数”，因为每一行对应边。
- 边方向可任取，但一旦选定，$A$、流量和势差的符号约定必须一致。
- 连通图的 $A^TA$ 不是可逆矩阵：常数向量总在核中。
- 环的数量不能凭肉眼数所有闭合路径；独立环维数是 $m-n+c$。

### 三道自检题

> [!question]- 1. 一棵有 $n$ 个节点的树有多少条边，关联矩阵 rank 多少？
> **答案**：边数 $n-1$；树连通，所以 rank $n-1$，且 $N(A^T)$ 维数为 $0$，没有独立环流。

> [!question]- 2. 为什么可解的节点注入必须总和为零？
> **答案**：$f\in C(A^T)=N(A)^\perp$，而连通图 $N(A)=\operatorname{span}\{\mathbf1\}$，故 $\mathbf1^Tf=0$。

> [!question]- 3. 反转一条边会怎样改变 $A^TA$？
> **答案**：只把 $A$ 对应行乘 $-1$；写成 $DA$，其中 $D^TD=I$，故 $(DA)^T(DA)=A^TA$，不变。

### 知识链小结

关联矩阵把节点势映为边差 → $N(A)$ 表示分量常势 → $N(A^T)$ 表示环流 → $A^TCA$ 汇总守恒与能量 → 四空间获得具体网络意义。

## Session 1.14 Exam 1 review

### 本节问题与前置知识

**问题**：如何把 Unit I 的算法线和结构线压缩为一条稳定的解题流程？

**前置知识**：Sessions 1.1–1.13 全部内容。

**本地资料**：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.14sum.pdf#page=1|review summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S14_Lecture_Exam_1_Review.pdf#page=1|review lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U1_S14_Recitation_Exam_1_Problem_Solving.pdf#page=1|problem-solving transcript p.1]]

### 1. Unit I 解题总流程

拿到 $Ax=b$ 后依次问：

1. **尺寸**：$A$ 是 $m\times n$ 吗？$x\in\mathbb R^n$、$b\in\mathbb R^m$ 吗？
2. **消元**：$[A\mid b]$ 的主元在哪里？是否出现矛盾行？
3. **存在性**：$b\in C(A)$ 吗？
4. **唯一性**：$N(A)=\{0\}$ 吗？
5. **完整解**：先取特解，再加所有特殊解。
6. **空间基**：列空间回原矩阵取 pivot columns；行空间取 rref 非零行；两个零空间解齐次系统。
7. **维数检查**：$r+(n-r)=n$、$r+(m-r)=m$。
8. **验算**：代回原矩阵，而不是只代回 rref。

### 2. 必会证明链

考前应能不查笔记完成：

- 行操作保持解集；
- 可逆矩阵的逆唯一；
- $Ax=b$ 完整解是 $x_p+N(A)$；
- 特殊解构成零空间基；
- rank-nullity；
- 行空间与零空间正交，列空间与左零空间正交；
- 连通图关联矩阵的零空间由常数向量张成。

### 3. Recitation 参数题完整闭环

令

$$
A=\begin{bmatrix}1&1&1\\1&2&3\\3&4&k\end{bmatrix},
\qquad b=\begin{bmatrix}2\\3\\7\end{bmatrix}.
$$

对增广矩阵依次做

$$
R_2\leftarrow R_2-R_1,\quad
R_3\leftarrow R_3-3R_1,\quad
R_3\leftarrow R_3-R_2,
$$

得到

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

消元倍数为 $1,3,1$，所以

$$
L=\begin{bmatrix}1&0&0\\1&1&0\\3&1&1\end{bmatrix},\qquad
U=\begin{bmatrix}1&1&1\\0&1&2\\0&0&k-5\end{bmatrix}.
$$

即使 $k=5$，分解仍成立，只是 $U$ 奇异。

### 4. 考试中的错误诊断

- **只报 rank 不报尺寸**：rank 必须同时满足 $r\le m,n$。
- **把 rref 主元列当列空间基**：应回原矩阵取同编号列。
- **参数题不分特殊值**：任何可能成为 $0$ 的主元都必须单独讨论。
- **只给一组解却声称完整**：有自由变量时必须写参数族。
- **把左零空间写进 $\mathbb R^n$**：它属于 $\mathbb R^m$。
- **用 $A^TA$ 消元替代 $A$**：会改变条件数且不是本单元必要步骤；直接消元更清楚。

### 三道自检题

> [!question]- 1. 若 $A$ 是 $4\times6$、rank $4$，对每个 $b\in\mathbb R^4$ 解的情况如何？
> **答案**：列空间是 $\mathbb R^4$，所以每个 $b$ 都可解；nullity $=2$，所以每个相容系统都有无穷多解。

> [!question]- 2. 若 $A$ 是 $6\times4$、rank $4$，解的情况如何？
> **答案**：$N(A)=\{0\}$，所以至多一个解；列空间是 $\mathbb R^6$ 中四维子空间，并非每个 $b$ 可解。

> [!question]- 3. 一个 $n\times n$ 方阵有 $n$ 个主元时，列出三条立即可得的结论。
> **答案**：可逆；$N(A)=\{0\}$；$C(A)=\mathbb R^n$；等价地每个 $b$ 有唯一解，任选三条即可。

### 知识链小结

尺寸 → 消元 → rank → 相容性 → 特解与零空间 → 四空间基与维数 → 代回验算；下面用 Exam 1 的四道题把这条链完整实践。

## Exam 1 完整题解

**本地试卷**：[[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex1.pdf#page=1|Unit 1 Exam p.1]]

**官方答案**：[[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex1s.pdf#page=1|Unit 1 Exam Solutions p.1]]

> [!warning] PDF 文本层说明
> 官方答案 PDF 的文本层编码损坏，但页面公式可正常阅读。本节按题目页面、答案页面与直接代数验算交叉核对；所有矩阵均额外做尺寸和乘积检查。

### Exam Problem 1：由存在性与唯一性反推尺寸和秩

题设：$A$ 为 $m\times n$，

$$
Ax=\begin{bmatrix}1\\1\\1\end{bmatrix}\text{ 无解},
\qquad
Ax=\begin{bmatrix}0\\1\\0\end{bmatrix}\text{ 恰有一解}.
$$

#### (a) 求 $m,n,r$ 的全部可能信息

**已知与目标**：两个右端都有三个分量，所以输出空间是 $\mathbb R^3$；要从一个无解和一个唯一解推断 rank。

**逐步推导**：

1. $Ax$ 有三个分量，故 $m=3$。
2. 第二个系统存在且唯一。若 $N(A)$ 含非零 $z$，则由一个解 $x_p$ 可产生 $x_p+tz$ 的无穷多个解，矛盾。因此 $N(A)=\{0\}$。
3. rank-nullity 给 $r+n-r=n$ 且 nullity $=0$，所以 $r=n$。
4. 第一个系统无解，说明 $C(A)\ne\mathbb R^3$，所以 $r<3$。
5. rank 为非负整数，而第二个非零右端可达，所以 $r\ge1$。

因此

$$
\boxed{m=3,\qquad r=n\in\{1,2\}.}
$$

#### (b) 求 $Ax=0$ 的全部解

由上一步 $N(A)=\{0\}$：

$$
\boxed{x=0\in\mathbb R^n.}
$$

#### (c) 给出一个例子

取 $n=r=1$：

$$
A=\begin{bmatrix}0\\1\\0\end{bmatrix}.
$$

则 $Ax=(0,x,0)^T$。右端 $(0,1,0)^T$ 唯一对应 $x=1$；$(1,1,1)^T$ 不在 $A$ 的一维列空间中。也可取 $n=r=2$ 的例子

$$
A=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}.
$$

#### 错误诊断

- 仅从“一个 $b$ 唯一可解”不能推出 $C(A)=\mathbb R^3$；它只推出核为零。
- $r=n$ 不等于 $r=m$；本题恰因 $r<m$ 才有不可达右端。

### Exam Problem 2：初等矩阵、逆矩阵与 LU

题设：$A$ 经以下顺序化为 $I$：

1. $E_{21}$：$R_2\leftarrow R_2-4R_1$；
2. $E_{31}$：$R_3\leftarrow R_3-3R_1$；
3. $E_{23}$：$R_2\leftarrow R_2-R_3$。

对应矩阵为

$$
E_{21}=\begin{bmatrix}1&0&0\\-4&1&0\\0&0&1\end{bmatrix},
$$

$$
E_{31}=\begin{bmatrix}1&0&0\\0&1&0\\-3&0&1\end{bmatrix},
\qquad
E_{23}=\begin{bmatrix}1&0&0\\0&1&-1\\0&0&1\end{bmatrix}.
$$

#### (a) 求 $A^{-1}$

因

$$
E_{23}E_{31}E_{21}A=I,
$$

所以

$$
A^{-1}=E_{23}E_{31}E_{21}.
$$

先乘后两项，再左乘 $E_{23}$（即把第二行减第三行）：

$$
A^{-1}
=\begin{bmatrix}
1&0&0\\
-1&1&-1\\
-3&0&1
\end{bmatrix}.
$$

#### (b) 求原矩阵 $A$

撤销操作时顺序反转且符号改变：

$$
A=E_{21}^{-1}E_{31}^{-1}E_{23}^{-1}
=\begin{bmatrix}
1&0&0\\
4&1&1\\
3&0&1
\end{bmatrix}.
$$

验算：

$$
AA^{-1}
=\begin{bmatrix}1&0&0\\4&1&1\\3&0&1\end{bmatrix}
\begin{bmatrix}1&0&0\\-1&1&-1\\-3&0&1\end{bmatrix}
=I_3.
$$

#### (c) 求 $A=LU$ 中的 $L$

LU 只消到上三角，不必把上方的 $1$ 再消掉。对 $A$：

$$
R_2\leftarrow R_2-4R_1,\qquad
R_3\leftarrow R_3-3R_1,
$$

得到

$$
U=\begin{bmatrix}1&0&0\\0&1&1\\0&0&1\end{bmatrix}.
$$

所以

$$
\boxed{L=\begin{bmatrix}1&0&0\\4&1&0\\3&0&1\end{bmatrix}.}
$$

直接计算 $LU=A$。第三个 Gauss–Jordan 操作 $E_{23}$ 是把 $U$ 进一步化为 $I$，不属于标准 LU 的下三角消元，因此其倍数不放进 $L$。

### Exam Problem 3：参数矩阵的列空间、零空间与完整解

题设

$$
A=\begin{bmatrix}
1&1&2&4\\
3&c&2&8\\
0&0&2&2
\end{bmatrix}.
$$

做

$$
R_2\leftarrow R_2-3R_1
$$

得到第二行 $(0,c-3,-4,-4)$；第三行是 $(0,0,2,2)$。特殊值只有 $c=3$。

#### (a) 对每个 $c$ 求列空间基

若 $c\ne3$，第二列产生第二个主元，第三列产生第三个主元，rank $=3$。取原矩阵前三个主元列：

$$
\boxed{
\left\{
\begin{bmatrix}1\\3\\0\end{bmatrix},
\begin{bmatrix}1\\c\\0\end{bmatrix},
\begin{bmatrix}2\\2\\2\end{bmatrix}
\right\}.}
$$

若 $c=3$，第二列与第一列相同，且消元后的第二、第三非零行互为倍数，rank $=2$。主元列为原第 1、3 列：

$$
\boxed{
\left\{
\begin{bmatrix}1\\3\\0\end{bmatrix},
\begin{bmatrix}2\\2\\2\end{bmatrix}
\right\}.}
$$

#### (b) 对每个 $c$ 求零空间基

齐次系统第三行给 $x_3=-x_4$。第二行化简为

$$
(c-3)x_2-4x_3-4x_4=(c-3)x_2=0.
$$

若 $c\ne3$，$x_2=0$；第一行给 $x_1=-2x_4$，所以

$$
\boxed{N(A)=\operatorname{span}\left\{
\begin{bmatrix}-2\\0\\-1\\1\end{bmatrix}
\right\}.}
$$

若 $c=3$，$x_2,x_4$ 都自由，且

$$
x_1=-x_2-2x_4,\qquad x_3=-x_4.
$$

因此

$$
\boxed{N(A)=\operatorname{span}\left\{
\begin{bmatrix}-1\\1\\0\\0\end{bmatrix},
\begin{bmatrix}-2\\0\\-1\\1\end{bmatrix}
\right\}.}
$$

维数检查：$c\ne3$ 时 $r+\text{nullity}=3+1=4$；$c=3$ 时 $2+2=4$。

#### (c) 求 $Ax=(1,c,0)^T$ 的完整解

容易验证

$$
x_p=\begin{bmatrix}0\\1\\0\\0\end{bmatrix}
$$

对所有 $c$ 都满足 $Ax_p=(1,c,0)^T$。因此把对应零空间加上即可。

若 $c\ne3$：

$$
\boxed{x=\begin{bmatrix}0\\1\\0\\0\end{bmatrix}
+t\begin{bmatrix}-2\\0\\-1\\1\end{bmatrix}.}
$$

若 $c=3$：

$$
\boxed{x=\begin{bmatrix}0\\1\\0\\0\end{bmatrix}
+s\begin{bmatrix}-1\\1\\0\\0\end{bmatrix}
+t\begin{bmatrix}-2\\0\\-1\\1\end{bmatrix}.}
$$

#### 错误诊断

- 参数矩阵必须在 $c=3$ 分情况；否则会非法除以 $c-3$。
- 列空间基必须取原矩阵列，不取消元结果的列。
- 特解对全部 $c$ 都成立，是本题最省计算的入口。

### Exam Problem 4：矩形矩阵、列关系与 RREF 空间

#### (a) $3\times5$ 矩阵的零空间信息

$A$ 有 $5$ 列而 rank $r\le3$，所以

$$
\dim N(A)=5-r\ge2.
$$

因此 $N(A)$ 是 $\mathbb R^5$ 的子空间，至少有两个线性无关的非零方向；$Ax=0$ 必有无穷多个解。

#### (b) 由给定 rref 推断原列关系

给定

$$
R=\operatorname{rref}(A)=
\begin{bmatrix}
1&4&0&0&0\\
0&0&0&1&0\\
0&0&0&0&1
\end{bmatrix}.
$$

主元列是 $1,4,5$，所以原矩阵的 $a_1,a_4,a_5$ 线性无关，并构成 $C(A)$ 的基。rank $=3=m$，故

$$
C(A)=\mathbb R^3.
$$

消元保持列之间的线性关系。由 $R$ 的列可见

$$
R_{:2}=4R_{:1},\qquad R_{:3}=0,
$$

因此原列满足

$$
\boxed{a_2=4a_1,\qquad a_3=0.}
$$

完整信息可概括为：三条 pivot columns $a_1,a_4,a_5$ 是 $\mathbb R^3$ 的一组基；其余两列分别是 $4a_1$ 与零列。

#### (c) 所有 $3\times3$ RREF 张成什么子空间

任何 rref 的第 $i$ 个主元位置至少在第 $i$ 列，且主元左侧为零、零行在底部。因此每个 $3\times3$ rref 都是上三角矩阵，故其 span 包含于上三角矩阵空间

$$
S=\left\{
\begin{bmatrix}a&b&c\\0&d&e\\0&0&f\end{bmatrix}:a,b,c,d,e,f\in\mathbb R
\right\}.
$$

反过来，六个上三角标准基矩阵均可由 rref 的差得到：

- $E_{11}$ 本身是 rref；
- $E_{12}=\begin{bmatrix}1&1&0\\0&0&0\\0&0&0\end{bmatrix}-E_{11}$，$E_{13}$ 同理；
- $E_{22}=\operatorname{diag}(1,1,0)-E_{11}$；
- $E_{23}$ 可由 $\begin{bmatrix}1&0&0\\0&1&1\\0&0&0\end{bmatrix}-\operatorname{diag}(1,1,0)$ 得到；
- $E_{33}=I-\operatorname{diag}(1,1,0)$。

所以所有上三角矩阵都在该 span 中，最终

$$
\boxed{S=\{3\times3\text{ 上三角矩阵}\},\qquad \dim S=6.}
$$

### Exam 1 题后复盘

四题分别检查了 Unit I 的四个层次：

1. 从存在性/唯一性反推维数结构；
2. 把行操作、逆与 LU 串起来；
3. 对参数值做 rank、四空间和完整解分流；
4. 把矩形矩阵及矩阵空间纳入统一的基与维数语言。

若某题计算正确却无法解释“为什么要分情况、这个向量位于哪个空间、维数是否闭合”，说明还停留在算法层，尚未完成 Unit I 的结构化理解。

## 本单元最终检查表

### 概念与尺寸

- [ ] 我能在任何 $m\times n$ 矩阵旁立刻写出四个基本子空间所在的 $\mathbb R^m$ 或 $\mathbb R^n$。
- [ ] 我能区分行图像、列图像与线性映射图像。
- [ ] 我能解释 rank 是有效方向数，nullity 是丢失的输入自由度。

### 算法

- [ ] 我能用增广矩阵消元，并在需要时换行。
- [ ] 我能从消元倍数构造 $L$，从结果读出 $U$，并处理 $PA=LU$。
- [ ] 我能由 rref 构造零空间特殊解，并从原矩阵挑列空间基。
- [ ] 我能把相容系统写成 $x_p+N(A)$，再代回原式验算。

### 证明

- [ ] 我能证明行操作保持解集、逆矩阵唯一、完整解公式和 rank-nullity。
- [ ] 我能证明两对基本子空间互为正交补。
- [ ] 我能证明连通图的关联矩阵零空间由常数向量张成。

### 下一单元接口

Unit II 将从

$$
C(A^T)\perp N(A),\qquad C(A)\perp N(A^T)
$$

出发，研究正交投影、最小二乘和正交基；也就是说，当 $b\notin C(A)$、精确方程无解时，我们将寻找 $C(A)$ 中离 $b$ 最近的向量。
