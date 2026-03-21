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

## 本单元主线

Unit II 解决的是两个更深的问题。第一，若 $Ax=b$ 无解，怎样在所有可能的 $Ax$ 中找到最接近 $b$ 的那个向量。第二，若反复施加同一个方阵，它会沿哪些方向拉伸、压缩、翻转或保持不变。前者导向正交、投影和最小二乘；后者导向行列式、特征值、对角化和矩阵指数。

这一单元最关键的桥梁是 [[Orthogonality]]。正交把“距离最小”“误差最小”“最佳逼近”“正交基”“傅里叶展开”都放进同一套语言里。于是你会看到：[[Projection Matrix]] 和 [[Least Squares]] 不只是求近似的技巧，而是整个函数逼近、数据拟合和信号展开的原型。

## Session 2.1 Orthogonal vectors and subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.1prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.1sol.pdf|solution]]

关联卡片：[[Orthogonality]]、[[Column Space]]、[[Null Space]]

### 正交的代数定义与几何意义

两个向量正交，当且仅当它们的内积为 0：
$$
x^Ty=0.
$$
这一定义之所以重要，是因为它把“垂直”从图像推广到任意维空间。只要内积为 0，就说明一个向量在另一个向量方向上的投影为 0，它们在几何上互不干扰。

### 勾股定理在线性代数里的写法

若 $x\perp y$，则
$$
\|x+y\|^2=(x+y)^T(x+y)=x^Tx+2x^Ty+y^Ty=\|x\|^2+\|y\|^2.
$$
这说明正交把长度分解成独立部分。后面的投影与最小二乘，实际上都在不断使用“把向量分解成相互正交的两部分”这一思想。

### 子空间的正交补

若 $S$ 是 $\mathbb{R}^n$ 的一个子空间，则它的正交补定义为
$$
S^\perp=\{x\in\mathbb{R}^n:x^Ts=0,\ \forall s\in S\}.
$$
对于矩阵 A，最重要的两对正交关系是
$$
C(A^T)\perp N(A),\qquad C(A)\perp N(A^T).
$$
第一条来自 $Ax=0$ 表明 x 与 A 的每一行都正交；第二条来自 $A^Ty=0$ 表明 y 与 A 的每一列都正交。

### 维数补齐

如果 $\operatorname{rank}(A)=r$，那么
$$
\dim C(A^T)=r,\qquad \dim N(A)=n-r.
$$
因此 row space 和 nullspace 不但正交，而且维数正好补满整个 $\mathbb{R}^n$。同理，column space 与 left nullspace 补满 $\mathbb{R}^m$。这是 Unit I 四个基本子空间在 Unit II 的自然延伸。

### 一个非常常用的结论

正规方程会频繁用到
$$
N(A^TA)=N(A).
$$
证明只需两步：如果 $A^TAx=0$，左乘 $x^T$ 得
$$
x^TA^TAx=(Ax)^T(Ax)=\|Ax\|^2=0,
$$
所以 $Ax=0$。反向包含显然成立。这个结论说明：$A^TA$ 不会制造新的零空间方向。

### 你要掌握

- 能用内积为 0 解释“正交”。
- 能证明 row space 与 nullspace 互相正交。
- 能说明为什么 $N(A^TA)=N(A)$。

## Session 2.2 Projections onto subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.2sol.pdf|solution]]

关联卡片：[[Orthogonal Projection]]、[[Projection Matrix]]、[[Orthogonality]]

### 投影的定义

给定一个子空间 $S$ 和向量 $b$，$b$ 在 $S$ 上的投影 $p$ 是离 $b$ 最近的 $S$ 中向量。关键不是“看起来最近”，而是误差
$$
e=b-p
$$
必须与整个子空间正交。也就是说，最佳逼近的判别条件是
$$
e\perp S.
$$

### 投影到一条直线

若 $S=\operatorname{span}(a)$，那么 $p$ 必须有形式 $p=\hat{x}a$。要求误差与 $a$ 正交：
$$
a^T(b-\hat{x}a)=0.
$$
解得
$$
\hat{x}=\frac{a^Tb}{a^Ta},\qquad
p=a\frac{a^Tb}{a^Ta}.
$$
这个公式值得熟记，因为它是所有高维投影的原型。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-projection.svg|720]]

图上最容易忽略的一点是：投影并不是“沿某个你随便选的方向压过去”，而是由“误差必须垂直于目标子空间”唯一决定的。也正因为这个垂直条件，投影和最小二乘才会自动导出一组线性方程，而不是停留在几何直觉上。

### 为什么“误差正交”就等于“距离最小”

若 $s\in S$，则
$$
b-s=(p-s)+e.
$$
这里 $p-s\in S$，而 $e\perp S$，因此
$$
\|b-s\|^2=\|p-s\|^2+\|e\|^2\ge \|e\|^2.
$$
等号只在 $s=p$ 时成立。这说明投影不是猜出来的，而是由勾股结构强制得到的。

### 投影到高维子空间

若子空间由矩阵 A 的列张成，投影点写成 $p=A\hat{x}$。误差与每一列都正交，于是
$$
A^T(b-A\hat{x})=0.
$$
这就是正规方程的来源。换句话说，投影问题一旦坐标化，立刻就变成线性方程组。

这里要刻意体会“正交条件如何逐列展开”。设 A 的列为 $a_1,\dots,a_n$，则
$$
a_i^T(b-A\hat{x})=0\quad (i=1,\dots,n).
$$
这不是 n 条互不相关的式子，而是在说：误差向量对列空间中的每一个方向都没有分量。把这 n 条式子收集起来，才得到紧凑的矩阵形式 $A^T(b-A\hat{x})=0$。因此正规方程的真正来源是“误差对整个子空间正交”，不是某个代数技巧。

### 你要掌握

- 能从“误差正交”推出投影公式。
- 能解释为什么投影是唯一的最佳逼近。
- 能把投影问题写成 $A^T(b-A\hat{x})=0$。

## Session 2.3 Projection matrices and least squares

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.3sol.pdf|solution]]

关联卡片：[[Projection Matrix]]、[[Least Squares]]、[[Orthogonal Projection]]

### 从无解系统到最佳近似

当 $Ax=b$ 无解时，我们不再要求精确等式成立，而是找一个 $\hat{x}$，使得
$$
\|A\hat{x}-b\|^2
$$
最小。几何上，这就是把 $b$ 投影到列空间 $C(A)$ 上；代数上，这就是最小二乘。

### 正规方程

令 $p=A\hat{x}$ 为投影点，误差 $e=b-p$ 必须满足 $A^Te=0$，因此
$$
A^T(b-A\hat{x})=0
\iff
A^TA\hat{x}=A^Tb.
$$
这就是正规方程。它把原来可能无解的 $m\times n$ 系统，变成了一个总是相容的 $n\times n$ 系统。

### 投影矩阵

若 A 的列线性无关，$A^TA$ 可逆，于是
$$
\hat{x}=(A^TA)^{-1}A^Tb,
\qquad
p=A\hat{x}=A(A^TA)^{-1}A^Tb.
$$
因此投影矩阵是
$$
P=A(A^TA)^{-1}A^T.
$$
它满足两个典型性质：
$$
P^T=P,\qquad P^2=P.
$$
前者表示投影没有偏向性，后者表示投影一次以后再投影不会再变。

### 线性回归例子

对数据点 $(1,1),(2,2),(3,2)$ 拟合直线 $C+Dt$，令
$$
A=\begin{bmatrix}
1&1\\
1&2\\
1&3
\end{bmatrix},
\qquad
x=\begin{bmatrix}C\\D\end{bmatrix},
\qquad
b=\begin{bmatrix}1\\2\\2\end{bmatrix}.
$$
正规方程为
$$
\begin{bmatrix}
3&6\\
6&14
\end{bmatrix}
\begin{bmatrix}\hat{C}\\\hat{D}\end{bmatrix}
=
\begin{bmatrix}5\\11\end{bmatrix}.
$$
解得
$$
\hat{C}=\frac23,\qquad \hat{D}=\frac12.
$$
这里拟合出来的不是“穿过所有点的直线”，而是让总平方误差最小的直线。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-least-squares-fit.svg|760]]

如果你把这个过程完全翻成“列空间语言”，其实是在做下面这件事：向量
$$
b=\begin{bmatrix}1\\2\\2\end{bmatrix}
$$
并不在由
$$
\begin{bmatrix}1\\1\\1\end{bmatrix},\qquad
\begin{bmatrix}1\\2\\3\end{bmatrix}
$$
张成的二维平面之外，它就在这个平面附近；最小二乘并不是“重新发明一个答案”，而是把 b 投影到这两列张成的平面上。于是线性回归和正交投影其实是一回事，只是一个来自数据分析，一个来自几何。

### 为什么 $A^TA$ 可逆

只要 A 的列独立，就有
$$
A^TAx=0 \implies x^TA^TAx=\|Ax\|^2=0 \implies Ax=0 \implies x=0.
$$
于是 $A^TA$ 零空间平凡，所以可逆。这个论证在最小二乘里几乎是必备的。

### 你要掌握

- 会从投影观点推导正规方程。
- 会写出并识别投影矩阵 $P$。
- 会解一个简单的数据拟合题。

## Session 2.4 Orthogonal matrices and Gram-Schmidt

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.4sol.pdf|solution]]

关联卡片：[[Orthogonal Matrix]]、[[Gram-Schmidt Orthogonalization]]

### 正交矩阵的结构

若矩阵 Q 的列是标准正交向量组，则
$$
Q^TQ=I.
$$
当 Q 是方阵时，这立刻给出
$$
Q^{-1}=Q^T.
$$
于是正交矩阵特别稳定：它保持长度、角度和内积不变，只做旋转和反射。

### 为什么正交基这么好用

若 A 的列已经标准正交，那么最小二乘问题极大简化，因为
$$
A^TA=I,\qquad \hat{x}=A^Tb.
$$
也就是说，各坐标互不干扰，直接取内积就能得到系数。这正是把问题转到“好基底”上的价值。

### Gram-Schmidt 的思想

[[Gram-Schmidt Orthogonalization]] 的任务是把任意一组独立向量改造成正交组。以两个向量 $a,b$ 为例，先保留
$$
u_1=a,
$$
再从 $b$ 中减去它在 $u_1$ 方向上的投影：
$$
u_2=b-\frac{u_1^Tb}{u_1^Tu_1}u_1.
$$
这样得到的 $u_2$ 与 $u_1$ 正交。最后再单位化，得到标准正交向量 $q_1,q_2$。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-gram-schmidt.svg|700]]

这张图解释了 Gram-Schmidt 最容易被“公式化”掩盖的本质：你并没有创造出新的信息，而只是把原来的第二个向量拆成“沿旧方向的部分”和“真正提供新方向的部分”。减掉前者以后，留下的才是新的正交方向。多维版本也完全一样，只不过你需要连续减掉它在前面所有已经建立好的方向上的投影。

### QR 分解

把 A 的列向量做 Gram-Schmidt，就得到
$$
A=QR,
$$
其中 Q 的列标准正交，R 是上三角矩阵。这个分解可以看成“正交版 LU”：LU 通过消元把 A 分成易解的三角结构，QR 通过正交化把 A 分成易于投影和最小二乘的结构。

### 你要掌握

- 能判断一个矩阵是否为 orthogonal matrix。
- 能手工完成二维或三维的 Gram-Schmidt。
- 知道 QR 分解为何对最小二乘有利。

## Session 2.5 Properties of determinants

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.5sol.pdf|solution]]

关联卡片：[[Determinant]]

### 行列式想衡量什么

[[Determinant]] 不是一个孤立的代数符号，它同时刻画两件事：

- A 是否可逆；
- A 把有向面积或体积缩放了多少。

因此 determinant 把“解方程的可逆性”和“空间几何的体积变化”压缩成了一个数。

### 定义性的三条性质

Strang 的讲法是先抓住三条最本质的性质，而不是先背公式：

- $\det(I)=1$；
- 交换两行，行列式变号；
- 行列式对每一行线性。

由此可以推出大量结论。

### 行操作如何影响 determinant

- 某行乘以常数 $c$，determinant 也乘以 $c$。
- 两行相同，determinant 为 0。
- 用某行减去另一行的倍数，不改变 determinant。
- 三角矩阵的 determinant 等于对角线元素乘积。

于是做消元时，只要记住有没有换行，就能快速求行列式。

### 乘法性

一个非常深但极好用的性质是
$$
\det(AB)=\det(A)\det(B).
$$
它意味着“先做 B 再做 A”的总体体积缩放，等于两次缩放的乘积。这和逆矩阵、矩阵幂、特征值乘积都会发生联系。

### 你要掌握

- 会根据行操作快速更新 determinant。
- 能解释“determinant 为 0”为什么等价于矩阵把空间压扁。
- 能从三角矩阵和消元读出 determinant。

## Session 2.6 Determinant formulas and cofactors

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.6sol.pdf|solution]]

关联卡片：[[Determinant]]

### minor 与 cofactor

删去第 $i$ 行第 $j$ 列后剩下的子式的行列式，叫做 minor $M_{ij}$；带上符号
$$
C_{ij}=(-1)^{i+j}M_{ij}
$$
就得到 cofactor。cofactor 是 determinant 展开的基本积木。

### 余子式展开

沿第 i 行展开有
$$
\det(A)=\sum_{j=1}^n a_{ij}C_{ij},
$$
沿某一列也同理。这个公式之所以成立，是因为 determinant 对每一行线性，并且遇到重复行就归零，所以只会留下“从每行每列各选一个元素”的项。

### 显式公式与计算现实

小维度时可以直接展开，但维数一大，determinant 的显式公式包含 $n!$ 项，计算成本很高。所以 cofactor expansion 更像理论公式，而不是数值计算的主力。实际求大矩阵 determinant，还是消元更稳、更快。

### 伴随矩阵

把 cofactor 排成矩阵并转置，就得到伴随矩阵 $\operatorname{adj}(A)$，满足
$$
A\,\operatorname{adj}(A)=\det(A)I.
$$
一旦 $\det(A)\neq 0$，便有
$$
A^{-1}=\frac{1}{\det(A)}\operatorname{adj}(A).
$$
这条公式理论上重要，因为它把 inverse 和 determinant 连起来；但实践里通常不如消元稳定。

### 你要掌握

- 会写出 minor、cofactor、cofactor expansion。
- 能解释为什么伴随矩阵能给出逆矩阵公式。
- 知道 cofactor 公式偏理论，消元偏计算。

## Session 2.7 Cramer's rule, inverse matrix, and volume

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.7sol.pdf|solution]]

关联卡片：[[Cramer's Rule]]、[[Matrix Inverse]]、[[Determinant]]

### Cramer's rule

当 A 可逆时，方程 $Ax=b$ 的第 i 个分量可写成
$$
x_i=\frac{\det(A_i)}{\det(A)},
$$
其中 $A_i$ 是把 A 的第 i 列换成 b 得到的矩阵。[[Cramer's Rule]] 适合小维度手算，也适合说明“解的每个分量都是由体积比给出的”。

### 体积解释

二维里 determinant 的绝对值是平行四边形面积缩放倍数，三维里是平行六面体体积缩放倍数，更高维依然表示有向体积缩放。符号记录方向是否翻转。因此
$$
\det(A^{-1})=\frac{1}{\det(A)}
$$
完全自然，因为逆变换要把体积缩放撤销回来。

### inverse 与 determinant 的统一理解

若 $\det(A)=0$，说明 A 把某个非零体积压成 0，于是变换不可逆；若 $\det(A)\neq 0$，说明 A 没有把空间压扁，因此可以一一对应地撤销。这里“体积不坍塌”和“线性方程唯一可解”再次成为同一件事的两个说法。

### 一个常见但重要的结论

若特征值为 $\lambda_1,\dots,\lambda_n$，则
$$
\det(A)=\lambda_1\cdots\lambda_n.
$$
你可以把 determinant 看成所有特征方向缩放因子的总乘积。这为下一讲的特征值做了准备。

### 你要掌握

- 会在 $2\times 2$ 或 $3\times 3$ 小题中使用 Cramer's rule。
- 能从几何上解释 determinant 的绝对值和符号。
- 能说明 determinant 与可逆性为何等价。

## Session 2.8 Eigenvalues and eigenvectors

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.8sol.pdf|solution]]

关联卡片：[[Eigenvalues]]、[[Eigenvectors]]

### 特征向量在捕捉什么

一般向量经过 A 作用后会改变方向，但某些特殊向量只会被拉伸或翻转：
$$
Av=\lambda v,\qquad v\neq 0.
$$
这里 v 是 [[Eigenvectors]]，$\lambda$ 是 [[Eigenvalues]]。它们刻画了矩阵最稳定、最内在的作用方向。

### 特征值方程

把式子移项得到
$$
(A-\lambda I)v=0.
$$
要有非零解，矩阵 $A-\lambda I$ 必须奇异，因此
$$
\det(A-\lambda I)=0.
$$
这给出 characteristic polynomial。其根就是全部特征值。

### 2×2 的实用关系

对
$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix},
$$
特征多项式为
$$
\lambda^2-(a+d)\lambda+(ad-bc)=0.
$$
也就是
$$
\lambda^2-\operatorname{trace}(A)\lambda+\det(A)=0.
$$
因此特征值之和等于 trace，特征值之积等于 determinant。

### 典型矩阵的特征值

- 三角矩阵：特征值就在对角线上。
- 投影矩阵：特征值只有 0 和 1。
- 旋转矩阵：可能没有实特征值，会出现复数。
- 若 $\lambda=0$ 是特征值，则对应的特征向量正是零空间里的非零向量。

### 你要掌握

- 会从 $\det(A-\lambda I)=0$ 求小矩阵特征值。
- 会从几何上解释特征向量为何是“不改方向”的方向。
- 知道 trace 与 determinant 分别控制特征值的和与积。

## Session 2.9 Diagonalization and powers of A

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.9sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.9prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.9sol.pdf|solution]]

关联卡片：[[Diagonalization]]、[[Eigenvalues]]、[[Eigenvectors]]

### 为什么要对角化

如果 A 有 n 个线性无关的特征向量，把它们排成矩阵
$$
S=\begin{bmatrix}v_1&\cdots&v_n\end{bmatrix},
$$
则有
$$
AS=S\Lambda,
$$
其中 $\Lambda$ 是对角矩阵。左右乘 $S^{-1}$ 得
$$
A=S\Lambda S^{-1}.
$$
这就是 [[Diagonalization]]。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit2-diagonalization.svg|820]]

图像上可以这样理解：在原坐标里，A 可能把各方向搅在一起；一旦切换到特征向量组成的基底里，A 的作用就只剩“沿每个特征方向单独放大或缩小”。对角化的价值，不是让矩阵看起来更整齐，而是把耦合问题拆成若干互不干扰的一维问题。

### 对角化带来的好处

一旦写成 $A=S\Lambda S^{-1}$，幂就变得非常容易：
$$
A^k=S\Lambda^kS^{-1}.
$$
而 $\Lambda^k$ 只需把每个特征值分别升到 k 次方。于是长期行为完全由特征值大小控制。

### 一个典型场景

若某个特征值绝对值最大，那么在不断乘以 A 的过程中，向量往往会越来越朝对应特征向量方向靠拢。这是很多迭代法和动力系统分析的核心直觉。

例如若
$$
A=S\begin{bmatrix}5&0\\0&2\end{bmatrix}S^{-1},
$$
则
$$
A^k=S\begin{bmatrix}5^k&0\\0&2^k\end{bmatrix}S^{-1}.
$$
随着 $k$ 增大，$5^k$ 会远快于 $2^k$，所以除非初始向量恰好没有第一特征方向分量，否则结果会越来越被第一特征方向支配。这就是“主特征值控制长期行为”的精确含义。

### 对角化失败说明什么

如果特征向量不够多，就不能对角化。此时不是“特征值方法失效”，而是矩阵内部还有耦合结构没被分离干净。这个缺口会在 Unit III 的 [[Jordan Form]] 中补上。

### 你要掌握

- 会判断一个矩阵是否可对角化。
- 会利用对角化快速计算 $A^k$。
- 知道“特征值大小决定长期行为”的基本图像。

## Session 2.10 Differential equations and $e^{At}$

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.10sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.10prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.10sol.pdf|solution]]

关联卡片：[[Matrix Exponential]]、[[Diagonalization]]

### 从标量指数函数到矩阵指数

对标量微分方程
$$
u'(t)=\lambda u(t),
$$
解是 $u(t)=e^{\lambda t}u(0)$。对向量系统
$$
x'(t)=Ax(t),
$$
自然猜想是
$$
x(t)=e^{At}x(0),
$$
其中 [[Matrix Exponential]] 定义为幂级数
$$
e^{At}=I+tA+\frac{t^2}{2!}A^2+\frac{t^3}{3!}A^3+\cdots.
$$

### 对角化时的求法

若 $A=S\Lambda S^{-1}$，则
$$
e^{At}=Se^{\Lambda t}S^{-1},
$$
而
$$
e^{\Lambda t}=
\operatorname{diag}(e^{\lambda_1 t},\dots,e^{\lambda_n t}).
$$
因此线性微分方程的求解又一次被还原成特征值问题。

### 稳定性

若所有特征值实部都为负，解会随时间衰减到 0；若存在实部为正的特征值，解会沿对应方向爆炸增长；若有纯虚特征值，则会出现振荡。这些定性行为只由特征值控制。

这里最好把微分方程和对角化再次并起来理解。若
$$
x(0)=c_1v_1+\cdots+c_nv_n,
$$
那么
$$
x(t)=c_1e^{\lambda_1 t}v_1+\cdots+c_ne^{\lambda_n t}v_n.
$$
也就是说，初始状态先被拆成各个特征方向上的分量，然后每个分量按自己的指数规律单独演化。稳定、爆炸、振荡都只是不同特征值在各自方向上的表现。

### 一个二维例子

若
$$
A=\begin{bmatrix}
2&0\\
0&-1
\end{bmatrix},
$$
则
$$
e^{At}=\begin{bmatrix}
e^{2t}&0\\
0&e^{-t}
\end{bmatrix}.
$$
同一系统里，一个方向指数增长，另一个方向指数衰减。这种“不同特征方向不同命运”的图像，是理解线性系统最核心的直觉。

### 你要掌握

- 会写出矩阵指数的定义。
- 会在可对角化时用特征分解求解 $x'(t)=Ax$。
- 能从特征值判断稳定、发散和振荡。

## Session 2.11 Markov matrices; Fourier series

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.11prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.11sol.pdf|solution]]

关联卡片：[[Markov Matrix]]、[[Fourier Series]]、[[Orthogonality]]

### Markov 矩阵

[[Markov Matrix]] 用来描述状态间的转移。若使用列概率向量约定，则矩阵各列和为 1，且元素非负。这样
$$
u_{k+1}=Au_k
$$
仍然是一个概率向量。Markov 矩阵总有特征值 1，对应长期稳态分布。

### 长期行为与特征值

如果系统足够“混合”，那么除了 1 之外其余特征值绝对值都小于 1，于是
$$
A^k u_0
$$
会逐渐靠近稳态特征向量。这是概率论版本的“主特征方向控制长期行为”。

### Fourier series 的线性代数本质

[[Fourier Series]] 不是一个孤立的分析技巧，而是把函数投影到一组正交基上。对区间上的函数，$\sin(nx)$、$\cos(nx)$ 扮演的角色，就像有限维空间里的正交向量组。系数由内积决定：
$$
c_n=\frac{\langle f,\phi_n\rangle}{\langle \phi_n,\phi_n\rangle}.
$$
这和投影到一条直线的公式完全同构。

### 为什么 Markov 和 Fourier 能放在同一讲

看起来一个是概率，一个是函数展开，但它们都在使用特征结构和正交结构：

- Markov 强调“反复施加矩阵后，哪个特征方向留下来”；
- Fourier 强调“把对象拆到一组彼此正交的模式上”。

两者都说明，选对基底以后，复杂问题会变成彼此独立的一维问题。

从课程结构上看，这一讲其实是在告诉你：线性代数的真正威力不在于会不会算某个行列式，而在于它不断把复杂对象改写成“正交模式”或“特征模式”的叠加。概率分布、函数、数据矩阵、微分方程解，都可以进入同一个框架。这也是为什么后面 SVD 和换基会显得如此自然。

### 你要掌握

- 知道 Markov 矩阵为何一定有特征值 1。
- 能用“投影到正交基”理解 Fourier series。
- 能看出 Markov 和 Fourier 与特征值、投影的统一性。

## Session 2.12 Exam 2 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.12sum.pdf|summary]]

### 本单元的主线回顾

这一单元不是两半毫无关系的内容，而是一条连续的链：

1. 正交让我们定义“最近”和“最佳逼近”。
2. 投影把最近点写成线性代数公式。
3. 最小二乘用正规方程解决无解系统。
4. 正交基和 QR 把计算进一步简化。
5. determinant 刻画可逆性与体积。
6. eigenvalues 和 diagonalization 刻画长期行为。
7. $e^{At}$、Markov 和 Fourier 是这些结构在动力系统与函数空间中的应用。

### 一定要会的题型

- 求投影、误差、投影矩阵，并验证误差正交。
- 推导和求解正规方程。
- 对一组向量做 Gram-Schmidt，并写出 QR。
- 用行操作或消元求 determinant。
- 计算小矩阵的特征值和特征向量。
- 用对角化求 $A^k$ 或 $e^{At}$。
- 判断 Markov 系统的稳态和长期行为。

### 常见混淆

- 投影矩阵是把 $b$ 投到列空间，不是把 $\hat{x}$ 投回参数空间。
- 正规方程永远相容，但它解的是最小二乘问题，不是原方程的精确解。
- determinant 为 0 说明矩阵奇异，不代表“某一步算错了”。
- 可对角化需要足够多的特征向量，不只是需要特征值存在。

## 本单元复习清单

- 我能从正交误差推出投影和最小二乘公式。
- 我能写出并验证投影矩阵 $P=A(A^TA)^{-1}A^T$ 的性质。
- 我能手工完成一个简短的 Gram-Schmidt 过程。
- 我能同时用代数和几何解释 determinant。
- 我能求特征值、判断可对角化，并利用它计算幂和矩阵指数。
- 我能解释 Markov 和 Fourier 为什么都是“选对基底后的简化”。
