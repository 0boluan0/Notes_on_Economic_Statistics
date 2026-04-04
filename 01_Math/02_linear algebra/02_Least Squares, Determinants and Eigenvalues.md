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

## 单元速览

- 这一单元真正回答的是两件事：原方程无解时怎样做最佳逼近；矩阵反复作用时哪些方向最关键。
- 你最后必须能把题目自动分流到三条线：
  - projection / least squares / QR。
  - determinant / cofactor / inverse / volume。
  - eigenvalues / diagonalization / matrix exponential / Markov。
- 如果复习时间很少，至少保住这 5 个抓手：
  - 误差最小 $\Leftrightarrow$ 误差与目标子空间正交。
  - 正规方程 `A^TA\hat{x}=A^Tb`。
  - 投影矩阵 `P=A(A^TA)^{-1}A^T`。
  - determinant 管奇异性和体积。
  - eigen / diagonalization 管长期行为与“选对基底”。

## 这页怎么用

- 先看每个 session 前面的 `快速回忆`，只要能口述，就不要直接沉回正文。
- 遇到题目时先判断它在问：
  - 最佳逼近？
  - 可逆性/体积？
  - 长期行为/特征结构？
- 若题目里出现 `best fit / closest / minimum error`，默认先去 projection / least squares。

## Session 回忆索引

- 2.1-2.4：正交、投影、least squares、Gram-Schmidt / QR。
- 2.5-2.7：determinant、cofactor、inverse、volume。
- 2.8-2.10：eigenvalues、diagonalization、$e^{At}$。
- 2.11：Markov / Fourier 作为“选对基底后的解耦”。
- 2.12：Exam 2 前的统一闭环。

## 本单元主线

Unit II 解决的是两个更深的问题。第一，若 $Ax=b$ 无解，怎样在所有可能的 $Ax$ 中找到最接近 $b$ 的那个向量。第二，若反复施加同一个方阵，它会沿哪些方向拉伸、压缩、翻转或保持不变。前者导向正交、投影和最小二乘；后者导向行列式、特征值、对角化和矩阵指数。

这一单元最关键的桥梁是 [[Orthogonality]]。正交把“距离最小”“误差最小”“最佳逼近”“正交基”“傅里叶展开”都放进同一套语言里。于是你会看到：[[Projection Matrix]] 和 [[Least Squares]] 不只是求近似的技巧，而是整个函数逼近、数据拟合和信号展开的原型。

## Session 2.1 Orthogonal vectors and subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.1prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.1sol.pdf|solution]]

关联卡片：[[Orthogonality]]、[[Column Space]]、[[Null Space]]、[[Linear Algebra Problem-Type Map]]

>[!note] 快速回忆
> - 这讲要回答：为什么正交会成为 Unit II 的底层语言。
> - 你要立刻想起：内积为 0、正交补、row space 与 nullspace 的正交关系、`N(A^TA)=N(A)`。
> - 典型题型：证明正交关系、解释维数补齐、说明某个向量为什么是最佳误差方向的候选。
> - 它接到下一讲：把“正交”具体落到 projection。

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



### 本讲知识点全景

- 正交的代数定义是内积为 0；它把“垂直”这个二维直觉推广到任意维空间。
- 若 $S$ 是子空间，则它的正交补 $S^\perp$ 也是子空间，并且维数会和 S 一起把整个空间补满。
- 对矩阵来说，最重要的两对关系是 $C(A^T)\perp N(A)$ 与 $C(A)\perp N(A^T)$；这说明 row space / nullspace 与 column space / left nullspace 是两组互补方向。
- 结论 $N(A^TA)=N(A)$ 非常关键，因为它保证 normal equations 不会凭空制造新的零空间方向。
- 从这一讲开始，“距离最小”“误差正交”“最佳逼近”会被放进同一套语言里。
- 把正交理解透，后面 projection、Gram-Schmidt、Fourier 和正交对角化都会显得自然。

>[!example] 例题
>
> 在 $\mathbb R^3$ 中，若
> $$
> S=\operatorname{span}\left\{\begin{bmatrix}1\\1\\0\end{bmatrix}\right\},
> $$
> 那么 $S^\perp$ 由所有满足 $x_1+x_2=0$ 的向量组成，是一个二维平面。这个例子说明：一个一维方向的正交补并不是“另外一条线”，而是把剩余维数全部装进来的子空间。

### 易错点与补充

- 正交补不是“随便找几个垂直向量”，而是与整个子空间都正交的全部向量集合。
- `N(A^TA)=N(A)` 的证明关键不是硬算，而是利用 $(Ax)^T(Ax)=\|Ax\|^2$。
- 这一讲里的正交关系都发生在正确的 ambient space 里，不要把 $C(A)$ 和 $N(A)$ 放在同一个空间里比较。
### 你要掌握

- 能用内积为 0 解释“正交”。
- 能证明 row space 与 nullspace 互相正交。
- 能说明为什么 $N(A^TA)=N(A)$。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.2 Projections onto subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.2sol.pdf|solution]]

关联卡片：[[Orthogonal Projection]]、[[Projection Matrix]]、[[Orthogonality]]

>[!note] 快速回忆
> - 这讲要回答：为什么“距离最小”会等价于“误差正交”。
> - 你要立刻想起：投影点 $p$ 的判别条件是 `b-p ⟂ S`；一维投影公式是所有高维投影的原型。
> - 典型题型：求线投影 / 子空间投影，并解释为什么它是 closest point。
> - 它接到下一讲：从几何投影写成正规方程与 least squares。

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



### 本讲知识点全景

- 投影的本质定义不是“看起来最近”，而是误差向量必须与目标子空间正交。
- 向量 $b$ 在子空间 S 上的投影 $p$ 满足分解 $b=p+e$，其中 $p\in S$ 且 $e\perp S$。
- 一维投影公式
> $$p=a\frac{a^Tb}{a^Ta}$$
  是所有高维投影的原型；高维只是在“对一条线正交”升级为“对整个子空间正交”。
- 为什么投影是 closest point，本质上来自勾股定理：任何别的 $s\in S$ 都会多出一段平行于 S 的误差。
- 若子空间由 A 的列张成，则投影点写成 $p=A\hat x$，误差正交就会立刻导出正规方程。
- 这一讲把几何直觉精确化，为 least squares 做好坐标化准备。

>[!example] 例题
>
> 将
> $$
> b=\begin{bmatrix}3\\1\end{bmatrix}
> $$
> 投到直线 $S=\operatorname{span}(a)$，其中
> $$
> a=\begin{bmatrix}1\\2\end{bmatrix}.
> $$
> 因为 $a^Tb=5$，$a^Ta=5$，所以
> $$
> p=a\frac{a^Tb}{a^Ta}=a=\begin{bmatrix}1\\2\end{bmatrix}.
> $$
> 于是误差
> $$
> e=b-p=\begin{bmatrix}2\\-1\end{bmatrix}
> $$
> 的确满足 $a^Te=0$。这个例子同时验证了“最近点”和“误差正交”是同一件事。

### 易错点与补充

- 投影到子空间时，正交的是误差 $e=b-p$，不是原向量 $b$ 本身。
- 只有当目标方向已经单位化时，投影系数才简化成单纯的内积。
- “closest” 是一个度量结论，不要和“沿某个方向压过去”混淆；真正决定投影的是正交条件。
### 你要掌握

- 能从“误差正交”推出投影公式。
- 能解释为什么投影是唯一的最佳逼近。
- 能把投影问题写成 $A^T(b-A\hat{x})=0$。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.3 Projection matrices and least squares

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.3sol.pdf|solution]]

关联卡片：[[Projection Matrix]]、[[Least Squares]]、[[Orthogonal Projection]]、[[Least Squares via Normal Equations]]

>[!note] 快速回忆
> - 这讲要回答：原系统无解时，最佳近似为什么由正规方程给出。
> - 你要立刻想起：`A^T(b-A\hat{x})=0`，`P=A(A^TA)^{-1}A^T`，且正交投影矩阵满足 `P^T=P`、`P^2=P`。
> - 典型题型：最小二乘、投影矩阵性质、线性回归 best fit。
> - 它接到下一讲：怎样更稳定地构造正交基与 QR。

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



### 本讲知识点全景

- 投影矩阵
> $$P=A(A^TA)^{-1}A^T$$
  把任意向量投到列空间 $C(A)$ 上；对应地，$I-P$ 把向量投到左零空间 $N(A^T)$ 上。
- `least squares` 的目标不是强行解无解系统，而是选择 $\hat x$ 使残差范数 $\|A\hat x-b\|$ 最小。
- 误差正交条件
> $$A^T(b-A\hat x)=0$$
  等价于正规方程
> $$A^TA\hat x=A^Tb.$$ 
  它把几何问题转成了可求解的方程组。
- 当 A 列独立时，$A^TA$ 可逆，因此 least squares 解唯一；这也是为什么“列独立”在这一讲里再次成为关键结构条件。
- 投影矩阵满足两条标志性性质：$P^T=P$ 和 $P^2=P$。前者表示正交投影，后者表示“投一次和投两次一样”。
- 线性回归只是这一讲最常见的应用场景，本质仍然是投影到列空间。

>[!example] 例题
>
> 对三点 $(1,1),(2,2),(3,2)$ 做直线拟合 $y=C+Dt$。写成矩阵后有
> $$
> A=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},
> \qquad
> b=\begin{bmatrix}1\\2\\2\end{bmatrix}.
> $$
> 正规方程给出
> $$
> A^TA\hat x=A^Tb,
> $$
> 解得 $\hat D=\tfrac12,\ \hat C=\tfrac23$。于是“最佳拟合”不再是几何直觉，而是正交投影的显式计算。

### 易错点与补充

- $\hat x$ 是参数空间里的最优系数，$p=A\hat x$ 才是数据空间里的投影点；两者不是同一个对象。
- 正规方程“总能写出”不等于“原方程组有解”；它解的是最佳逼近问题。
- $P=A(A^TA)^{-1}A^T$ 只在 A 列独立时可直接这样写；列不独立时要转向 QR 或 pseudoinverse 语言。
### 你要掌握

- 会从投影观点推导正规方程。
- 会写出并识别投影矩阵 $P$。
- 会解一个简单的数据拟合题。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.4 Orthogonal matrices and Gram-Schmidt

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.4sol.pdf|solution]]

关联卡片：[[Orthogonal Matrix]]、[[Gram-Schmidt Orthogonalization]]

>[!note] 快速回忆
> - 这讲要回答：为什么正交基会让 projection / least squares 的计算变简单。
> - 你要立刻想起：Gram-Schmidt 是不断减去已有方向上的投影；QR 让 least squares 变成更容易解的上三角系统。
> - 典型题型：手做 Gram-Schmidt、写 QR、解释 orthogonal matrix 的好处。
> - 它接到下一讲：从“长度与角度”切换到 determinant 的体积语言。

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



### 本讲知识点全景

- 正交矩阵 Q 的定义是 $Q^TQ=I$；它的列和行都是标准正交向量组，因此 $Q^{-1}=Q^T$。
- 正交矩阵保持长度、角度和内积不变，所以它是“不会扭曲几何”的最理想坐标变换。
- Gram-Schmidt 过程的目标是：把一组独立向量改造为一组张成同一子空间的正交或标准正交基。
- QR 分解把任意列独立矩阵写成 $A=QR$，其中 Q 负责给出正交基，R 负责记录原向量在该基下的坐标。
- 对 least squares，QR 的好处是把正规方程改写成更稳定的上三角问题：
> $$Rx=Q^Tb.$$ 
- 这一讲说明：正交不仅是几何概念，还是降低计算复杂度和提高稳定性的工具。

>[!example] 例题
>
> 取
> $$
> a_1=\begin{bmatrix}1\\1\end{bmatrix},\qquad a_2=\begin{bmatrix}1\\0\end{bmatrix}.
> $$
> Gram-Schmidt 先给出
> $$
> q_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}.
> $$
> 然后把 $a_2$ 在 $q_1$ 方向上的分量减掉，得到与 $q_1$ 正交的第二个方向，再标准化为 $q_2$。最终 Q 的列就是一组标准正交基，而 R 记录了 $a_1,a_2$ 如何由这组基组合出来。

### 易错点与补充

- Gram-Schmidt 先“去掉投影”，再“单位化”，这两个步骤不能颠倒。
- 正交矩阵与“列互相垂直”还差一步：每列还必须是单位长度。
- QR 里的 Q 给你的是新基，R 给你的是旧向量在新基下的坐标；不要把它们的角色反过来。
### 你要掌握

- 能判断一个矩阵是否为 orthogonal matrix。
- 能手工完成二维或三维的 Gram-Schmidt。
- 知道 QR 分解为何对最小二乘有利。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.5 Properties of determinants

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.5sol.pdf|solution]]

关联卡片：[[Determinant]]

>[!note] 快速回忆
> - 这讲要回答：determinant 到底在衡量什么。
> - 你要立刻想起：它同时刻画可逆性、体积缩放、orientation；三条定义性性质是后面一切公式的起点。
> - 典型题型：用行操作跟踪 determinant 变化，解释为何 determinant 为 0。
> - 它接到下一讲：从性质走向 cofactor 公式。

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



### 本讲知识点全景

- determinant 最应该记住的不是某个公式，而是它衡量矩阵对体积和方向的整体作用。
- 三种基本行操作对 determinant 的影响必须熟记：交换两行会变号；一行乘常数会把 determinant 乘同样常数；一行加另一行倍数不改变 determinant。
- 因此，做消元时 determinant 可以一路被追踪；对上三角矩阵，determinant 直接等于对角线元素乘积。
- 若 $\det(A)=0$，矩阵会把某个维度压扁，因而不可逆；反过来，可逆矩阵 determinant 必非零。
- 乘法公式
> $$\det(AB)=\det(A)\det(B)$$
  说明 determinant 真的是“整体缩放因子”，因为连续做两个变换时缩放会相乘。
- 这一讲把“可逆性”从消元语言翻译到了几何体积语言。

>[!example] 例题
>
> 对
> $$
> A=\begin{bmatrix}1&2&1\\0&3&4\\0&0&5\end{bmatrix},
> $$
> 因为它已经是上三角矩阵，所以
> $$
> \det(A)=1\cdot3\cdot5=15.
> $$
> 这比套一般公式快得多，也提醒你：determinant 很多时候应该靠结构读，而不是暴力展开。

### 易错点与补充

- determinant 不是“对角线求和”也不是“逐项相乘”；只有在特殊结构下才有简单读法。
- 行操作对 determinant 的影响和对解集的影响不是同一回事；前者要额外记账，后者不必。
- 不要把“determinant 为负”误解成“体积为负”；负号表示方向翻转，体积大小看绝对值。
### 你要掌握

- 会根据行操作快速更新 determinant。
- 能解释“determinant 为 0”为什么等价于矩阵把空间压扁。
- 能从三角矩阵和消元读出 determinant。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.6 Determinant formulas and cofactors

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.6sol.pdf|solution]]

关联卡片：[[Determinant]]

>[!note] 快速回忆
> - 这讲要回答：minor、cofactor、adjugate 为什么出现。
> - 你要立刻想起：cofactor expansion 给出显式公式，但计算上通常不如 elimination；它的价值在结构理解。
> - 典型题型：余子式展开、伴随矩阵、从 cofactor 推 inverse 公式。
> - 它接到下一讲：把 determinant 与 inverse / volume 合并看。

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



### 本讲知识点全景

- determinant 的全展开公式来自排列：每一项都从每行每列各取一个元素，再乘上排列的符号。
- `minor` 是删去某行某列后剩下的子行列式；`cofactor` 则是带上符号 $(-1)^{i+j}$ 的 minor。
- 按一行或一列做 cofactor expansion，是把高阶 determinant 递归地降成低阶 determinant。
- cofactor 语言不仅能算 determinant，也为 inverse 的伴随矩阵公式做准备。
- 从概念上看，cofactor 展开强调 determinant 对每一行都是线性的，同时又对交换两行非常敏感。
- 这一讲的重要性在于：它把“为什么 determinant 会长成这样”解释清楚，而不是只让你背三阶公式。

>[!example] 例题
>
> 对
> $$
> A=\begin{bmatrix}1&2&0\\3&4&5\\0&6&7\end{bmatrix},
> $$
> 第一行第三个元素是 0，因此沿第一行展开最省力：
> $$
> \det(A)=1\begin{vmatrix}4&5\\6&7\end{vmatrix}-2\begin{vmatrix}3&5\\0&7\end{vmatrix}.
> $$
> 当矩阵里有很多 0 时，选择合适的展开行列会极大降低计算量。

### 易错点与补充

- cofactor 的符号是棋盘格交替，不要漏掉负号。
- minor 还没带符号，cofactor 才带 $(-1)^{i+j}$；两者不要混写。
- cofactor expansion 在概念上重要，但高维数值计算通常不会用它直接算 determinant。
### 你要掌握

- 会写出 minor、cofactor、cofactor expansion。
- 能解释为什么伴随矩阵能给出逆矩阵公式。
- 知道 cofactor 公式偏理论，消元偏计算。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.7 Cramer's rule, inverse matrix, and volume

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.7sol.pdf|solution]]

关联卡片：[[Cramer's Rule]]、[[Matrix Inverse]]、[[Determinant]]

>[!note] 快速回忆
> - 这讲要回答：Cramer's rule、inverse、volume 为什么是同一组思想的不同投影。
> - 你要立刻想起：determinant 不只是一个数，它告诉你矩阵是否把空间压扁以及压缩了多少体积。
> - 典型题型：用 determinant 讨论 inverse 是否存在，解释 Cramer's rule 何时有意义。
> - 它接到下一讲：从体积切换到“不变方向”。

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



### 本讲知识点全景

- Cramer 法则告诉你：当 A 可逆时，每个未知量都能写成“把某一列换成 $b$ 后的 determinant 与 $\det(A)$ 的比值”。
- 这件事揭示了 determinant 与解的关系：若 $\det(A)=0$，不仅 inverse 不存在，连 Cramer 法则都失效。
- 伴随矩阵公式
> $$A^{-1}=\frac1{\det(A)}\operatorname{adj}(A)$$
  把 cofactor 组织成了 inverse 的统一表达式。
- determinant 的几何意义在这一讲被说得更完整：二维是平行四边形面积，三维是平行六面体体积，更高维是对应的超体积。
- 若 $|\det(A)|<1$，体积缩小；若大于 1，体积放大；若为 0，至少一个维度被压扁。
- 这一讲把 “解线性方程”“求 inverse”“看体积缩放” 三件事彻底绑定到 determinant 上。

>[!example] 例题
>
> 对系统
> $$
> \begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix}=\begin{bmatrix}5\\7\end{bmatrix},
> $$
> 有
> $$
> \det(A)=5.
> $$
> 若把第一列换成 $b$，得到
> $$
> A_1=\begin{bmatrix}5&1\\7&3\end{bmatrix},\qquad \det(A_1)=8,
> $$
> 所以 $x_1=\tfrac85$。同理可求 $x_2$。这个例子说明 Cramer 法则虽然不适合大规模计算，但它把“解的每个分量”直接和 determinant 连起来了。

### 易错点与补充

- Cramer 法则只适用于可逆方阵，不适用于秩亏或矩形矩阵。
- 体积看的是 $|\det(A)|$，而 determinant 本身还携带方向翻转信息。
- 伴随矩阵公式概念上很漂亮，但实际计算 inverse 时通常仍以消元或分解为主。
### 你要掌握

- 会在 $2\times 2$ 或 $3\times 3$ 小题中使用 Cramer's rule。
- 能从几何上解释 determinant 的绝对值和符号。
- 能说明 determinant 与可逆性为何等价。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.8 Eigenvalues and eigenvectors

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.8sol.pdf|solution]]

关联卡片：[[Eigenvalues]]、[[Eigenvectors]]

>[!note] 快速回忆
> - 这讲要回答：矩阵反复作用时，哪些方向会被保留下来。
> - 你要立刻想起：eigenvector 是“方向不变，只缩放”的向量；trace / determinant 在 2×2 中和 eigenvalues 有直接关系。
> - 典型题型：求 eigenvalues / eigenvectors，解释某个矩阵的几何作用。
> - 它接到下一讲：若有足够多 eigenvectors，就能 diagonalization。

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



### 本讲知识点全景

- 特征向量是矩阵作用下方向不变的向量，特征值是该方向上的缩放因子。
- 代数上通过
> $$Av=\lambda v$$
  寻找不变方向，等价于解
> $$\det(A-\lambda I)=0.$$
- `characteristic polynomial` 给出全部特征值；对于小矩阵，trace 与 determinant 往往能快速帮你校验结果。
- 不同特征值对应的特征向量必线性无关，因此若一个 $n\times n$ 矩阵有 n 个不同特征值，就一定可对角化。
- 特征值问题开始把“反复施加矩阵会怎样”说清楚，因为这些不变方向会在幂和微分方程中主导长期行为。
- 这一讲是从“解方程组”真正转向“研究变换自身结构”的转折点。

>[!example] 例题
>
> 对
> $$
> A=\begin{bmatrix}2&1\\0&3\end{bmatrix},
> $$
> 因为它是上三角矩阵，特征值直接就是对角线上的 $2$ 和 $3$。对应地，$\lambda=2$ 的特征向量满足 $(A-2I)v=0$，$\lambda=3$ 的特征向量满足 $(A-3I)v=0$。这个例子说明：特征值有时可以先从结构读，再回去解特征向量。

### 易错点与补充

- 特征值不是把某个矩阵元素拿出来看，而是使 $A-\lambda I$ 奇异的标量。
- 特征向量不能取零向量；零向量满足任何线性方程，但不提供方向信息。
- “有重复特征值” 不自动意味着 “不可对角化”；真正关键的是特征向量是否够多。
### 你要掌握

- 会从 $\det(A-\lambda I)=0$ 求小矩阵特征值。
- 会从几何上解释特征向量为何是“不改方向”的方向。
- 知道 trace 与 determinant 分别控制特征值的和与积。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.9 Diagonalization and powers of A

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.9sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.9prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.9sol.pdf|solution]]

关联卡片：[[Diagonalization]]、[[Eigenvalues]]、[[Eigenvectors]]

>[!note] 快速回忆
> - 这讲要回答：为什么 diagonalization 是处理矩阵幂的理想形式。
> - 你要立刻想起：`A=S\Lambda S^{-1}` 让复杂的矩阵幂变成标量幂；失败原因通常是特征向量不够。
> - 典型题型：判断是否可对角化、用对角化求 `A^k`。
> - 它接到下一讲：把同样思路推到 matrix exponential 与微分方程。

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



### 本讲知识点全景

- 若矩阵有一组基由特征向量组成，就能写成
> $$A=S\Lambda S^{-1}.$$
  这叫对角化。
- 对角化最有价值的地方在于把高次幂简化为
> $$A^k=S\Lambda^kS^{-1},$$
  因为对角矩阵的幂只需把每个对角元单独升幂。
- 在 eigenbasis 中，原本耦合的多个方向被解耦成互不干扰的一维模式；这就是 diagonalization 的结构含义。
- 不同特征值足以保证可对角化，但不是必要条件；关键是要有足够多的独立特征向量。
- 当某个特征值的绝对值最大时，它通常会控制 $A^k$ 的长期行为。
- 这一讲让你第一次看见“选择对的基底，复杂问题就会变简单”。

>[!example] 例题
>
> 若
> $$
> A=S\begin{bmatrix}2&0\\0&\tfrac12\end{bmatrix}S^{-1},
> $$
> 则
> $$
> A^k=S\begin{bmatrix}2^k&0\\0&(\tfrac12)^k\end{bmatrix}S^{-1}.
> $$
> 当 $k$ 很大时，第二个模式会迅速衰减，第一特征方向主导整体行为。这就是“主特征值控制长期行为”的最直接版本。

### 易错点与补充

- 对角化不是把矩阵“做个好看变形”，而是换到特征向量构成的坐标系里。
- 不能只因为特征值能算出来就默认可以对角化；还要检查特征向量是否形成一组基。
- $A^k$ 的长期行为看的是特征值的绝对值大小，不只是数值本身谁更大。
### 你要掌握

- 会判断一个矩阵是否可对角化。
- 会利用对角化快速计算 $A^k$。
- 知道“特征值大小决定长期行为”的基本图像。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.10 Differential equations and $e^{At}$

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.10sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.10prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.10sol.pdf|solution]]

关联卡片：[[Matrix Exponential]]、[[Diagonalization]]

>[!note] 快速回忆
> - 这讲要回答：为什么 $e^{At}$ 会自然出现在线性微分方程里。
> - 你要立刻想起：若能 diagonalize，就把矩阵指数降成对角线上逐项指数；稳定性看特征值实部。
> - 典型题型：用 eigen / diagonalization 求解二维线性系统。
> - 它接到下一讲：看 Markov 和 Fourier 这两类“选对基底就解耦”的应用。

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



### 本讲知识点全景

- 矩阵指数定义为幂级数
> $$e^{At}=I+At+\frac{(At)^2}{2!}+\cdots,$$
  它是标量指数函数在线性系统中的自然推广。
- 对常系数系统 $x'(t)=Ax(t)$，通解可写为
> $$x(t)=e^{At}x(0).$$
- 若 A 可对角化，
> $$e^{At}=Se^{\Lambda t}S^{-1},$$
  其中 $e^{\Lambda t}$ 只需对每个特征值单独指数化。
- 每个特征方向上的演化都像标量 ODE：实特征值给出增长或衰减，复特征值会带来旋转和振荡。
- 若出现 Jordan block，则除了指数项还会出现多项式因子；这为后面 Jordan form 埋下伏笔。
- 这一讲把特征值从“代数量”升级成了“动态行为的决定者”。

>[!example] 例题
>
> 若
> $$
> A=\begin{bmatrix}2&0\\0&-1\end{bmatrix},
> $$
> 则
> $$
> e^{At}=\begin{bmatrix}e^{2t}&0\\0&e^{-t}\end{bmatrix}.
> $$
> 因而第一坐标随时间爆炸增长，第二坐标指数衰减到 0。你应该把这理解成：不同特征方向各自按自己的特征值独立演化。

### 易错点与补充

- `e^{At}` 不是把矩阵每个元素逐个取指数，而是由幂级数定义的整体对象。
- 公式 $e^{A+B}=e^Ae^B$ 一般不成立，除非 A 与 B 可交换。
- 稳定性看的是特征值实部，而不仅仅是特征值本身是否为负数。
### 你要掌握

- 会写出矩阵指数的定义。
- 会在可对角化时用特征分解求解 $x'(t)=Ax$。
- 能从特征值判断稳定、发散和振荡。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.11 Markov matrices; Fourier series

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.11prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.11sol.pdf|solution]]

关联卡片：[[Markov Matrix]]、[[Fourier Series]]、[[Orthogonality]]

>[!note] 快速回忆
> - 这讲要回答：Markov 和 Fourier 为什么能放在同一讲。
> - 你要立刻想起：两者都在做“找到自然基底，让矩阵或算子分解得更简单”。
> - 典型题型：求 Markov 稳态、解释 Fourier basis 为什么正交。
> - 它接到下一讲：Exam 2 前把 projection、determinant、eigen 三线合并。

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



### 本讲知识点全景

- Markov 矩阵研究“重复施加同一个概率转移”后的长期分布，因此它本质上是一个 eigenvalue 问题。
- 在本课程的约定下，Markov 矩阵会保留概率向量结构，因此特征值 $1$ 对应稳态或守恒模式。
- 若除 $1$ 之外其余特征值绝对值都小于 1，反复乘 A 会把初始分布拉向稳态方向。
- Fourier series 把函数展开到一组正交基上；系数通过内积计算，本质就是无限维版本的 projection。
- Markov 和 Fourier 放在同一讲，不是巧合，而是在强调：找到自然基底后，复杂演化就会分解成一维模式的叠加。
- 这一讲是对 Unit II 的一次抬升，说明正交和特征结构并不只服务于有限维向量题。

>[!example] 例题
>
> 对两状态系统，若一步转移矩阵的稳态向量满足
> $$
> A\pi=\pi,
> $$
> 那么 $\pi$ 就是特征值 1 的特征向量。另一方面，在 Fourier 展开里，若函数写成
> $$
> f=\sum_n c_n\phi_n,
> $$
> 每个系数 $c_n$ 都是把 f 投影到正交模式 $\phi_n$ 上得到的。两个问题看似不同，核心却都是“选对模式后分别处理”。

### 易错点与补充

- Markov 矩阵有行随机和列随机两种常见约定，解题时必须先跟随题目所用 convention。
- Fourier 系数不是“凭感觉配出来”，而是由内积与正交性唯一确定。
- 这一讲真正要记住的不是某个概率例子，而是“模式分解”这条统一思想。
### 你要掌握

- 知道 Markov 矩阵为何一定有特征值 1。
- 能用“投影到正交基”理解 Fourier series。
- 能看出 Markov 和 Fourier 与特征值、投影的统一性。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 2.12 Exam 2 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.12sum.pdf|summary]]

>[!note] 快速回忆
> - 这讲要回答：Exam 2 前至少要保住哪三条线。
> - 你要立刻想起：`orthogonality/projection -> least squares/QR -> determinant -> eigen/diagonalization/$e^{At}$`。
> - 典型题型：projection、normal equations、Gram-Schmidt、determinant、eigen、Markov。
> - 复习时如果不会分题型，先回到本页开头的 `单元速览`。

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



### 高频题型展开

- `projection / least squares`：会不会从“误差正交”一路写到 normal equations 和 projection matrix，是 Unit II 最核心的诊断点。
- `QR / orthogonality`：常考标准正交基、Gram-Schmidt、以及为何正交基能让计算简单稳定。
- `determinant`：高频题不是死算，而是根据结构、行操作和三角形读 determinant，并解释它与 invertibility / volume 的关系。
- `eigen / diagonalization / e^{At}`：要能从特征值判断长期行为，从特征向量解释为什么 powers 和 ODE 会解耦。
- `Markov / Fourier`：通常作为压轴概念题，检查你是否真的理解“选对基底”这件事。

>[!example] 例题
>
> 若一道题同时出现 “best fit”“orthogonal error”“matrix with independent columns”，你应立刻联想到同一条路线：
> $$
> A^T(b-A\hat x)=0\ \Longrightarrow\ A^TA\hat x=A^Tb.
> $$
> 反过来，若题目提到“repeated powers”“steady state”“dominant direction”，则应立刻切换到 eigen / diagonalization 语言。Exam 2 的关键不是多会算，而是能否迅速识别题型入口。

### 易错点与补充

- Unit II 最大的误区是把正交、determinant、eigen 三部分看成三门课；实际上它们都在描述“矩阵如何作用于空间”。
- 复习时不要只背公式，要能说清“为什么这个公式对应这个几何结构”。
- 如果题型识别慢，优先回看每讲顶部的 `快速回忆` 和这里的题型串讲，而不是盲目刷题。
### 你要掌握

- 能把 Unit II 压缩成 `orthogonality -> projection -> least squares / QR -> determinant -> eigen / diagonalization / dynamics`。
- 能根据题目特征在 projection、determinant、eigen 这几条路线之间快速切换。
- 能说明这一单元为什么既讲最佳逼近，又讲长期行为。

### 回忆检查

- 不看正文，我能说出 Unit II 的三条核心题型：best fit、determinant/invertibility、eigen/dynamics。
- 我能写出 least squares、projection matrix、diagonalization 这三条线各自最关键的公式。
- 我知道自己若是不会分题型，应该回本页开头还是回某个具体 session。

## 本单元复习清单

- 我能从正交误差推出投影和最小二乘公式。
- 我能写出并验证投影矩阵 $P=A(A^TA)^{-1}A^T$ 的性质。
- 我能手工完成一个简短的 Gram-Schmidt 过程。
- 我能同时用代数和几何解释 determinant。
- 我能求特征值、判断可对角化，并利用它计算幂和矩阵指数。
- 我能解释 Markov 和 Fourier 为什么都是“选对基底后的简化”。
