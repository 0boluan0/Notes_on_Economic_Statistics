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

## 单元速览

- 这一单元真正回答的是：当矩阵不再只是“拿来解方程”时，怎样理解它的最佳结构、最佳坐标系和最佳逆。
- 你最后必须能把题目自动分流到四条线：
  - symmetric / positive definite / quadratic form。
  - similarity / Jordan。
  - SVD。
  - linear transformation / change of basis / pseudoinverse。
- 如果复习时间很少，至少保住这 5 个抓手：
  - 对称矩阵最适合正交特征分解。
  - positive definite 把二次型、极小值、最小二乘连起来。
  - diagonalization 失败时要改看 Jordan。
  - 任意矩阵统一由 SVD 处理。
  - 伪逆是“最佳逆”，不是随便补出的逆。

## 这页怎么用

- 先看每个 session 的 `快速回忆`，确认自己能说出“为什么这一讲存在”。
- 如果你只记公式但想不起课程逻辑，就回到本页开头和结尾的统一图景，不要直接沉进细节。
- 遇到题目时先判断：
  - 它是在问 symmetry / positivity？
  - 还是在问 coordinate change / representation？
  - 还是在问 general matrix decomposition？

## Session 回忆索引

- 3.1-3.3：对称矩阵、positive definite、quadratic form、minimum。
- 3.4：similarity 与 Jordan，解释为什么 diagonalization 不是总能成功。
- 3.5：SVD 统一任意矩阵。
- 3.6-3.7：linear transformation 与 change of basis。
- 3.8：left inverse / right inverse / pseudoinverse。
- 3.9：Exam 3 前的结构闭环。

## 本单元主线

Unit III 处理的是“把前两单元的工具彻底收束起来”。前面我们分别学过子空间、正交、最小二乘、行列式、特征值；这一单元开始问：对称矩阵为什么特别好，一般矩阵怎样被标准形式描述，线性变换与换基到底是什么，非方阵和秩亏矩阵的“最佳逆”又该怎样定义。

因此这一单元最重要的结论不是某一个公式，而是几个统一视角：

- 对称矩阵最适合用正交特征分解；
- 一般方阵若不能对角化，就用相似与 Jordan 形式描述缺陷；
- 一般矩阵则由 [[Singular Value Decomposition]] 统一处理；
- 线性变换、换基和伪逆把这些分解重新解释成“坐标选择”和“最佳恢复”。

## Session 3.1 Symmetric matrices and positive definiteness

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.1sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.1prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.1sol.pdf|solution]]

关联卡片：[[Symmetric Matrix]]、[[Positive Definite Matrix]]、[[Spectral Decomposition]]、[[Testing Positive Definiteness]]

>[!note] 快速回忆
> - 这讲要回答：为什么对称矩阵是“最好处理”的矩阵类。
> - 你要立刻想起：实特征值、正交特征向量基、正交对角化，以及 positive definite 的几种等价判别。
> - 典型题型：判断矩阵是否对称/正定，并说明依据。
> - 它接到下一讲：把特征值理论推进到复数域与 Fourier 基底。

### 为什么对称矩阵是“好矩阵”

若 $A=A^T$，则 A 是 [[Symmetric Matrix]]。对称矩阵有三个关键优点：

- 特征值全是实数；
- 不同特征值对应的特征向量彼此正交；
- 能选出一组标准正交特征向量作为整个空间的基。

于是对称矩阵不只是“可对角化”，而且可被正交对角化：
$$
A=Q\Lambda Q^T.
$$
这就是 [[Spectral Decomposition]]。

### 为什么特征值一定是实数

若 $Ax=\lambda x$，对复向量取共轭转置并利用 $A=A^T$（实矩阵时等于共轭转置），可得
$$
x^TAx=\lambda x^Tx
$$
以及
$$
x^TAx=\bar{\lambda}x^Tx.
$$
因为 $x\neq 0$ 时 $x^Tx>0$，所以 $\lambda=\bar{\lambda}$，即 $\lambda$ 为实数。这个证明的关键不是技巧，而是“对称性让左右两边的内积表达完全兼容”。

### 投影到特征方向

若 $q_1,\dots,q_n$ 是标准正交特征向量，则
$$
A=\lambda_1q_1q_1^T+\cdots+\lambda_nq_nq_n^T.
$$
这说明对称矩阵可以被看成若干个互相垂直的一维投影的加权和。每个 $q_iq_i^T$ 把向量投到第 i 个特征方向上，再由 $\lambda_i$ 决定该方向的伸缩。

### 正定性

[[Positive Definite Matrix]] 是对称矩阵中的最佳情形。它满足以下等价条件：

- 对所有非零向量 $x$，有
  $$
  x^TAx>0;
  $$
- 所有特征值都大于 0；
- 所有主元都大于 0；
- 所有顺序主子式都大于 0。

这些判据把 quadratic form、特征值、主元和 determinant 全部连接起来。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-positive-definite-ellipse.svg|760]]

上图对应的是二次型的等值线。对正定矩阵来说，等值线是一圈圈围绕唯一极小点的椭圆或高维椭球，这意味着你无论沿哪个方向走，函数值最终都会上升。若矩阵不是正定，图像就会出现平坦方向甚至马鞍方向；这正是“有零特征值”或“有负特征值”的几何表现。

>[!example] 例子
>
> 对
> $$
> A=\begin{bmatrix}
> 5&2\\
> 2&3
> \end{bmatrix},
> $$
> 它对称，且主元为 5 与 $11/5$，都为正，所以 A 正定。其特征值满足
> $$
> \lambda^2-8\lambda+11=0,
> $$
> 解为 $4\pm\sqrt{5}$，也都为正。
>
> 还可以直接看二次型：
> $$
> q(x_1,x_2)=5x_1^2+4x_1x_2+3x_2^2.
> $$
> 若把交叉项配方或转到特征向量基底，会发现它变成两个正平方项的和。这个过程再次说明：对称矩阵的“好”不是偶然，而是因为总能找到一组正交方向把它彻底拆开。
>


### 本讲知识点全景

- 对称矩阵最重要的结构优势是：它一定有实特征值，并且不同特征值对应的特征向量彼此正交。
- 这使得对称矩阵不仅可对角化，而且可被正交对角化：
> $$A=Q\Lambda Q^T.$$ 
  这就是 spectral theorem 的核心。
- `positive definite` 可以从多个等价视角判断：二次型 $x^TAx>0$、所有特征值为正、所有主元为正、所有顺序主子式为正。
- 二次型图像里的椭圆/椭球，就是正定矩阵“沿每个方向都向上”的几何表现。
- 对称性保证了“方向之间彼此独立”，正定性则保证了“每个方向都向上”。两者合起来就解释了为什么这类矩阵在优化和数值计算中格外好用。
- 这一讲是 Unit III 的起点，因为它给出了“最好处理的矩阵类”作为参照物。

>[!example] 例题
>
> 对
> $$
> A=\begin{bmatrix}5&2\\2&3\end{bmatrix},
> $$
> 你可以从三条路判断它正定：
> - 它对称；
> - 主元为 $5$ 与 $\tfrac{11}{5}$，都为正；
> - 特征值为 $4\pm\sqrt5$，都大于 0。
>
> 同一个结论能被多种判据支持，这正是正定矩阵值得单独学习的原因。

### 易错点与补充

- 仅有正对角元不够判断正定，非对角耦合项同样重要。
- 很多正定判据默认矩阵已经对称；离开对称语境，结论不一定成立。
- “可对角化” 比 “正交对角化” 弱得多；后者真正依赖对称性。
### 你要掌握

- 能证明对称矩阵特征值为实数。
- 会在“特征值判据 / 主元判据 / 二次型判据”间切换。
- 能解释正交特征分解为何比一般对角化更强。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 3.2 Complex matrices; fast Fourier transform

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.2sol.pdf|solution]]

关联卡片：[[Fourier Series]]、[[Orthogonality]]

>[!note] 快速回忆
> - 这讲要回答：为什么有时必须进入复数域，以及 FFT 在结构上到底做了什么。
> - 你要立刻想起：复特征值成共轭对出现；复内积必须带共轭；FFT 本质是结构化换基。
> - 典型题型：解释 unitary、说明 Fourier basis 为什么正交。
> - 它接到下一讲：从“好基底”回到 positive definite 与极小值问题。

### 为什么要进入复数域

实矩阵完全可能没有实特征值。最典型的是二维旋转矩阵，除了旋转角为 $0$ 或 $\pi$，实数域里根本没有不变方向。要让特征值理论完整，就必须允许复数。

对实矩阵，如果 $a+bi$ 是特征值，则 $a-bi$ 也是特征值；这来自特征多项式系数为实数，因此复根成共轭对出现。

### 复向量空间中的内积

复数情形下，正交要改成共轭内积：
$$
\langle z,w\rangle=z^*w.
$$
这里 $z^*$ 是共轭转置。只有这样才有
$$
\langle z,z\rangle=\sum |z_i|^2\ge 0.
$$
因此在复数世界里，正交矩阵的角色由 unitary matrix 承担。

### Fourier 基底

令
$$
\omega=e^{2\pi i/n},
$$
则 $1,\omega,\omega^2,\dots,\omega^{n-1}$ 是 n 次单位根。离散傅里叶矩阵可以写成
$$
F_{jk}=\frac{1}{\sqrt{n}}\omega^{jk},
$$
它的列彼此正交，因此是一个 unitary matrix。这个基底把“时域坐标”转换成“频域坐标”。

### FFT 的本质

FFT 不是新的数学对象，而是更聪明地计算傅里叶变换。它利用单位根的递归结构，把大问题拆成两个更小的问题，再不断递归，从而把复杂度从 $O(n^2)$ 降到 $O(n\log n)$。真正值得记住的是：FFT 的力量来自“选了一组特别好的基，并利用了这组基的结构”。

把它和 diagonalization 对照着看会更清楚。对角化是在找“使线性变换解耦的基”；FFT 则是在找“使卷积、周期结构和振荡模式变得简单的基”。它们在思想上是同源的，只是一个主要作用于矩阵动力系统，一个主要作用于信号和多项式运算。



### 本讲知识点全景

- 进入复数域的根本原因是：某些实矩阵在实数域里没有特征值，但在复数域里可以被完整描述，例如平面旋转。
- 复向量空间中的内积必须写成带共轭的形式 $x^*y$，这样长度和正交性的定义才保持良好性质。
- `unitary matrix` 是复数域里的正交矩阵对应物，满足 $U^*U=I$，同样保持长度和内积。
- Fourier matrix 的列由单位根构成，它给出了一组彼此正交的复指数模式，因此离散 Fourier 变换本质上是一次特殊的换基。
- FFT 并不是新的数学对象，而是高效计算 Fourier transform 的算法，它利用了单位根和偶奇拆分的结构重复。
- 这一讲把“选对基底”的想法从实数空间推进到了复数和频域。

>[!example] 例题
>
> 二维旋转矩阵
> $$
> R=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
> $$
> 在实数域里通常没有特征向量，但在复数域里有特征值 $e^{i\theta}$ 与 $e^{-i\theta}$。这说明：复数不是为了炫技，而是为了把“旋转”也放进 eigenvalue 框架。

### 易错点与补充

- 复内积里要有共轭，不能直接照抄实数域的 $x^Ty$。
- FFT 是算法，Fourier basis / Fourier matrix 才是结构对象。
- 这一讲真正要带走的是“有些结构只有在扩大的数域或更好的基底里才会显现”。
### 你要掌握

- 知道复特征值为什么在实矩阵中成共轭对出现。
- 能解释为什么复向量内积要带共轭。
- 能把 FFT 理解成“结构化换基”，而不是算法黑箱。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 3.3 Positive definite matrices and minima

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.3sol.pdf|solution]]

关联卡片：[[Positive Definite Matrix]]、[[Least Squares]]

>[!note] 快速回忆
> - 这讲要回答：为什么 positive definite 会自动带来唯一极小值。
> - 你要立刻想起：`f(x)=1/2 x^TAx-b^Tx` 的 Hessian 是 A；若 A 正定，则驻点就是唯一全局 minimum。
> - 典型题型：从二次型判断是否最小、解释 least squares 为什么唯一。
> - 它接到下一讲：从“最好处理的矩阵”转向“对角化失败时怎么办”。

### 二次型视角

正定矩阵最自然的载体是二次型
$$
q(x)=x^TAx.
$$
若 A 正定，则除了 $x=0$ 之外总有 $q(x)>0$。从几何上说，等值面是一族椭球；从优化上说，这意味着函数朝所有方向都是“向上弯”的。

### 极小值问题

考虑函数
$$
f(x)=\frac12x^TAx-b^Tx.
$$
其梯度为
$$
\nabla f(x)=Ax-b
$$
（当 A 对称时最干净），驻点满足
$$
Ax=b.
$$
若 A 正定，则这个驻点不仅存在唯一，而且是全局唯一极小值。于是“解线性方程”与“最小化二次函数”成为同一件事。

### 为什么 positive definite 意味着 unique minimum

对任意增量 $h$，
$$
f(x+h)-f(x)=h^T(Ax-b)+\frac12h^TAh.
$$
若 x 已满足 $Ax=b$，第一项消失，只剩
$$
\frac12h^TAh>0\qquad (h\neq 0).
$$
所以离开解点就一定增大。这就是“正定矩阵保证唯一极小值”的最直接证明。

### 与最小二乘的关系

在最小二乘中，目标函数是
$$
\|Ax-b\|^2=(Ax-b)^T(Ax-b).
$$
展开后其 Hessian 正是 $2A^TA$。当 A 列独立时，$A^TA$ 正定，于是最小二乘解唯一。这说明 Unit II 的正规方程，实际上已经把 positive definite 悄悄引入了。

这也是为什么很多优化问题最后都会回到“某个矩阵是否正定”。只要 Hessian 正定，你就不只是找到一个临界点，而是找到唯一可靠的极小点。从这个角度看，positive definite 不是一个局部章节概念，而是把线性代数、优化和统计连起来的枢纽。



### 本讲知识点全景

- 二次型
> $$q(x)=x^TAx$$
  把矩阵和几何曲面联系起来；它告诉你沿每个方向走，函数值会如何变化。
- 当 A 正定时，$q(x)>0$ 对所有非零 $x$ 成立，因此原点是严格极小点；更一般地，带线性项的二次函数会有唯一极小值。
- 这和优化问题直接相连：若
> $$f(x)=\tfrac12x^TAx-b^Tx,$$
  则临界点满足 $Ax=b$，而正定性保证该临界点真的是唯一极小值。
- $A^TA$ 总是正半定；当 A 列独立时，$A^TA$ 进一步正定。这把 least squares 与正定理论接起来了。
- 这一讲说明：正定不只是线性代数分类词，它会直接决定“最小值是否存在且唯一”。
- 从这里往后，正定矩阵会成为 SVD、pseudoinverse 与数值算法中的稳定核心。

>[!example] 例题
>
> 设
> $$
> f(x,y)=x^2+4xy+5y^2-2x.
> $$
> 其 Hessian 对应矩阵为
> $$
> A=\begin{bmatrix}2&4\\4&10\end{bmatrix}.
> $$
> 若 A 正定，则由一阶条件得到的临界点一定是唯一极小值。于是“求最小值”本质上变成了“解一个正定线性系统”。

### 易错点与补充

- 正半定只保证不向下，不保证严格向上；因此最小值可能不唯一。
- 判断极小值时，不要只看一阶条件，必须结合二阶结构或正定性。
- least squares 里的 $A^TA$ 之所以重要，不只是因为它可算，而是因为它把问题放进了正定框架。
### 你要掌握

- 能用二次型解释正定。
- 会把 $Ax=b$ 解释成二次目标函数的一阶条件。
- 知道正定性为什么自动带来唯一最小值。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 3.4 Similar matrices and Jordan form

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.4sol.pdf|solution]]

关联卡片：[[Similar Matrix]]、[[Jordan Form]]、[[Diagonalization]]、[[Choosing Matrix Decompositions]]

>[!note] 快速回忆
> - 这讲要回答：若矩阵不能被完整 diagonalize，应当怎样描述它。
> - 你要立刻想起：相似表示“同一变换的不同坐标表示”；Jordan form 描述的是特征向量不够时的缺陷结构。
> - 典型题型：判断相似、判断是否可对角化、解释 Jordan block 的含义。
> - 它接到下一讲：把一般方阵再扩展到任意矩阵，由 SVD 统一。

### 相似矩阵在说什么

若
$$
B=M^{-1}AM,
$$
则 A 与 B [[Similar Matrix|相似]]。这表示它们是同一个线性变换在不同基底下的矩阵表示。于是相似矩阵必然共享很多不变量：特征值、trace、determinant、rank 都一样。

### 对角化为何是理想情况

若 A 有足够多的独立特征向量，就能找到 M 使得
$$
M^{-1}AM=\Lambda
$$
为对角矩阵。这样每个方向完全解耦，所有动力学和幂运算都变得透明。

### 如果特征向量不够怎么办

有些矩阵有重复特征值，但独立特征向量数量不足，此时无法对角化。[[Jordan Form]] 给出的标准形式是由若干 Jordan block 组成：
$$
J=\begin{bmatrix}
\lambda&1&0&\cdots\\
0&\lambda&1&\cdots\\
\vdots& &\ddots&\ddots\\
0&\cdots&0&\lambda
\end{bmatrix}.
$$
对角线上是特征值，超对角线上的 1 记录“还差一点才能完全对角化”的耦合。

### Jordan 结构为什么重要

若
$$
A=SJS^{-1},
$$
则
$$
A^k=SJ^kS^{-1},\qquad e^{At}=Se^{Jt}S^{-1}.
$$
对于 Jordan block，$J^k$ 和 $e^{Jt}$ 会出现多项式因子，例如 $t e^{\lambda t}$，这正是对角化失败的动力学后果。

可以把 Jordan block 理解成“几乎已经沿特征方向解耦，但还残留了一点点沿链条传递的信息”。这点残余耦合在静态问题里不明显，但在幂和矩阵指数中会逐步积累，最后表现为比纯指数多出来的多项式因子。也正因此，Jordan form 更像是解释失败机制的工具，而不是日常数值计算的工具。



### 本讲知识点全景

- 相似矩阵描述的是“同一个线性变换在不同基底下的表示”，因此共享 eigenvalues、trace、determinant、rank 等不变量。
- 对角化是最理想的相似标准形；当特征向量不够时，Jordan form 则记录这种失败是如何发生的。
- Jordan block 在对角线上放特征值，在超对角线上放 1；这些 1 代表一般特征向量链之间仍有耦合。
- 正是因为这种耦合，$A^k$ 和 $e^{At}$ 会出现额外的多项式因子，而不再只是纯粹的特征值幂或指数。
- Jordan form 在概念上极强，因为它告诉你“为什么不能完全解耦”；但在数值上通常不如 SVD 或 QR 稳定。
- 这一讲为“不是所有矩阵都像对称矩阵那么好”给出最精确的描述。

>[!example] 例题
>
> 对
> $$
> J=\begin{bmatrix}\lambda&1\\0&\lambda\end{bmatrix},
> $$
> 有
> $$
> J^k=\begin{bmatrix}\lambda^k&k\lambda^{k-1}\\0&\lambda^k\end{bmatrix}.
> $$
> 这里额外出现的 $k\lambda^{k-1}$ 就是 Jordan block 残余耦合的直接后果。它告诉你：对角化失败不是一句抽象口号，而会真实影响动力学形状。

### 易错点与补充

- 相似不等于相等；它是同一对象在不同坐标系下的不同表示。
- 有重复特征值并不自动意味着有 Jordan block，关键仍是特征向量个数是否不足。
- Jordan form 更适合理解结构，而不是作为默认计算工具。
### 你要掌握

- 能解释相似为何等于换基。
- 知道对角化失败时 Jordan block 记录了什么信息。
- 明白 Jordan form 的意义主要是结构解释，不是机械计算表演。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 3.5 Singular value decomposition

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.5sol.pdf|solution]]

关联卡片：[[Singular Value Decomposition]]、[[Orthogonal Matrix]]、[[Pseudoinverse]]、[[Choosing Matrix Decompositions]]

>[!note] 快速回忆
> - 这讲要回答：为什么任意矩阵最后都该看 SVD。
> - 你要立刻想起：`A=U\Sigma V^T`，奇异值来自 `A^TA` 的特征值平方根，四个基本子空间会在 SVD 中归位。
> - 典型题型：解释 SVD 的来源、几何意义、rank 信息与低秩近似。
> - 它接到下一讲：从分解转回“矩阵只是线性变换的一种表示”。

### 为什么需要 SVD

特征分解要求方阵，正交特征分解还要求对称；但实际矩阵往往既不方也不对称。[[Singular Value Decomposition]] 给出了一种对任意矩阵都成立的标准形式：
$$
A=U\Sigma V^T,
$$
其中 U、V 为正交矩阵，$\Sigma$ 为非负对角矩阵（长方形时主对角线上放奇异值）。

### 从 $A^TA$ 出发推导

因为 $A^TA$ 总是对称半正定，所以可以正交对角化：
$$
A^TA=V\Sigma^T\Sigma V^T.
$$
于是 V 的列向量给出输入空间中的“最好方向”，而奇异值 $\sigma_i$ 就是 $A^TA$ 特征值的平方根。随后定义
$$
u_i=\frac{Av_i}{\sigma_i}
$$
即可得到输出空间中的正交方向 U。

### 几何解释

SVD 表示：先用 $V^T$ 把坐标轴转到最合适的输入方向，再由 $\Sigma$ 按各方向独立拉伸或压缩，最后由 U 再旋转到输出坐标系。任何矩阵都可以被理解为“旋转/反射 + 纯伸缩 + 再旋转/反射”。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-svd-geometry.svg|860]]

这张图是 SVD 最值得保留的直觉。输入空间里的单位圆先被转到最有意义的坐标，再被拉成一个轴对齐的椭圆，最后再整体旋转到输出空间。于是奇异值不是抽象数字，而是“在最自然方向上被放大多少”的度量。

### 四个基本子空间在 SVD 中的归位

若 rank 为 r，则

- $v_1,\dots,v_r$ 张成 row space；
- $v_{r+1},\dots,v_n$ 张成 nullspace；
- $u_1,\dots,u_r$ 张成 column space；
- $u_{r+1},\dots,u_m$ 张成 left nullspace。

因此 SVD 把 Unit I 的四个基本子空间一次性正交化了。

### rank-1 展开

SVD 还可写成
$$
A=\sigma_1u_1v_1^T+\cdots+\sigma_ru_rv_r^T.
$$
这说明一般矩阵是若干 rank-1 矩阵的正交叠加。奇异值越大，对矩阵整体作用贡献越大，这也是低秩逼近和压缩的基础。

这一表达特别适合连接前面的 rank-1 矩阵和后面的图像压缩。你可以把第一项看成“最重要的一层结构”，第二项看成“在第一层之外补充的第二重要结构”，依此类推。只保留前几项时，你丢掉的是细节，而不是主要轮廓。



### 本讲知识点全景

- SVD 给出任意矩阵的标准分解：
> $$A=U\Sigma V^T.$$ 
  它不要求方阵，也不要求可对角化。
- 奇异值来自 $A^TA$ 的特征值平方根；右奇异向量来自 $A^TA$，左奇异向量来自 $AA^T$。
- V 的列把输入空间分成 row space 与 nullspace 的正交基，U 的列把输出空间分成 column space 与 left nullspace 的正交基。
- 当 A 是对称正定矩阵时，SVD 会退化成正交特征分解；因此 SVD 可以看作对前面所有分解的统一升级。
- SVD 还直接暴露 rank、低秩近似和 pseudoinverse，因此它是课程后半段最统一的语言。
- 真正值得记住的不是公式本身，而是 “A 在某个正交输入基底上只做独立缩放，再转到某个正交输出基底”。

>[!example] 例题
>
> 对 rank 1 矩阵
> $$
> A=\begin{bmatrix}4&3\\8&6\end{bmatrix},
> $$
> 只有一个非零奇异值。于是 SVD 会告诉你：整个矩阵其实只保留了一个真正的输入模式，其余方向都被压进 nullspace。这个判断比逐项看矩阵更接近其结构本质。

### 易错点与补充

- 奇异值不是 eigenvalues，U 和 V 的角色也不同：V 管输入方向，U 管输出方向。
- SVD 适用于任意矩阵，这正是它比 eigen decomposition 更普适的原因。
- rank 信息直接体现在非零奇异值个数上；这一点在压缩和伪逆里极其重要。
### 你要掌握

- 能从 $A^TA$ 解释 V 和奇异值的来源。
- 能把 SVD 理解成“输入方向、拉伸强度、输出方向”的分解。
- 能说明为什么 SVD 同时给出四个基本子空间的正交基。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 3.6 Linear transformations and their matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.6sol.pdf|solution]]

关联卡片：[[Linear Transformation]]

>[!note] 快速回忆
> - 这讲要回答：为什么变换比矩阵本身更本质。
> - 你要立刻想起：矩阵列向量就是基向量的像；矩阵只是线性变换在某组基下的坐标表达。
> - 典型题型：从变换定义构造矩阵，或从矩阵解释变换作用。
> - 它接到下一讲：一旦换基，矩阵表示就会变化。

### 变换比矩阵更本质

[[Linear Transformation]] 的定义是
$$
T(v+w)=T(v)+T(w),\qquad T(cv)=cT(v).
$$
矩阵只是当你选定输入基和输出基之后，对 T 的坐标表示。也就是说，矩阵不是第一性对象，线性变换才是。

### 列向量从何而来

在标准基 $e_1,\dots,e_n$ 下，矩阵 A 的第 j 列正是 $T(e_j)$ 的坐标。因此只要知道基向量被送到哪里，就知道整个矩阵。因为任意向量
$$
x=x_1e_1+\cdots+x_ne_n,
$$
线性性立即给出
$$
T(x)=x_1T(e_1)+\cdots+x_nT(e_n).
$$

### 哪些映射是线性的

- 投影、旋转、反射、微分是线性的。
- 平移、取长度、加常数不是线性的。

这个判断很重要，因为它决定一个对象是否能被矩阵完整表示。

### 为什么这一讲重要

一旦把矩阵看成变换表示，相似、换基、对角化、SVD 就都不再是“变一个公式”，而是“换一种看同一变换的坐标系”。这会大幅减少机械记忆。



### 本讲知识点全景

- 线性变换首先是一个“把向量映到向量”的规则，满足可加性与齐次性；矩阵只是它在某组基底下的坐标表示。
- 一旦固定了定义域和值域的基底，矩阵的每一列就等于变换作用在相应基向量上的结果。
- kernel / image 是变换本身的几何对象；写成矩阵后，它们分别对应 nullspace / column space。
- 变换复合对应矩阵乘法，这解释了为什么乘法顺序不能乱。
- 这一讲强迫你区分“抽象线性映射”与“具体矩阵数组”，这是后面 change of basis 的前提。
- 真正掌握这讲后，你会把矩阵看成“某个变换在某个坐标系里的样子”，而不再把矩阵当作对象本身。

>[!example] 例题
>
> 设变换 $T:\mathbb R^2\to\mathbb R^2$ 把
> $$
> e_1\mapsto\begin{bmatrix}1\\2\end{bmatrix},\qquad e_2\mapsto\begin{bmatrix}3\\0\end{bmatrix}.
> $$
> 那么在标准基下，它的矩阵就是
> $$
> [T]=\begin{bmatrix}1&3\\2&0\end{bmatrix}.
> $$
> 这里“列就是像”这条规则，比死记矩阵更重要。

### 易错点与补充

- 矩阵依赖所选基底，线性变换本身不依赖。
- kernel/image 是变换层面的对象，不要把它们误以为是某个坐标系下才有的现象。
- 一旦基底变了，同一个变换的矩阵也会变；这不是变换变了，而是描述方式变了。
### 你要掌握

- 能从 $T(e_j)$ 写出矩阵。
- 能区分“变换本身”和“该变换在某组基下的矩阵”。
- 能快速判断一个映射是不是线性的。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 3.7 Change of basis; image compression

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.7sol.pdf|solution]]

关联卡片：[[Change of Basis]]、[[Singular Value Decomposition]]

>[!note] 快速回忆
> - 这讲要回答：换基为什么会改变矩阵，但不会改变变换本身。
> - 你要立刻想起：similarity 就是换基后的矩阵变换规则；低秩压缩本质上也依赖选对基底。
> - 典型题型：做一次换基计算，解释相似矩阵为什么代表同一个变换。
> - 它接到下一讲：再把“逆”推广到更一般的情形。

### 向量的换基

若新基底矩阵为 W，其列是新基向量，则任意向量都有
$$
x=Wc,
$$
其中 x 是旧基坐标，c 是新基坐标。换基不是改变向量本身，而是改变记录这个向量的坐标方式。

### 矩阵的换基

对同一个线性变换 T，若在旧基下的矩阵是 A，在新基下的矩阵是 B，则
$$
B=M^{-1}AM.
$$
这与相似矩阵完全一致。相似不是抽象代数游戏，而是换基公式。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-change-of-basis.svg|820]]

看这张图时要时刻提醒自己：变化的是坐标，不是向量本身。左边和右边标出的绿色箭头是同一个几何向量，只是投到不同基底后，坐标数字变了。很多“换基题”难，并不是计算真的复杂，而是把“对象本身”和“它在某套坐标里的数字”混在了一起。

### 为什么要换基

我们换基通常是为了三件事：

- 让矩阵更简单，如对角化或 Jordan form；
- 让系数更稀疏，如 Fourier/wavelet 表示；
- 让主要信息集中在少量坐标里，如图像压缩。

### 图像压缩与低秩近似

图像本质上是一个矩阵。若在像素标准基下表示，几乎所有像素都要单独记录；但若在 SVD 或某些频域基底下表示，往往只有少数主方向携带大部分信息。保留最大的若干奇异值与对应向量，就能得到视觉上仍很接近原图的低秩近似。

这个说法值得再翻译成更“课程化”的语言：SVD 找到的是图像最重要的输入模式和输出模式，奇异值告诉你这些模式的权重。于是压缩不是随便删像素，而是在删那些对整体能量贡献最小的模式。线性代数在这里第一次变得非常“工程”。



### 本讲知识点全景

- 换基的核心是：同一个向量、同一个变换，在不同坐标系下会有不同的坐标与矩阵表示。
- 若新基矩阵为 W，则向量坐标满足 $x=Wc$，也即 $c=W^{-1}x$；这就是“从几何对象到坐标”的翻译公式。
- 同一个线性变换在新基下的矩阵满足
> $$B=M^{-1}AM,$$
  这就是相似变换的来源。
- 选对基底可以把复杂矩阵变简单，例如在 eigenbasis 下对角化，在 Fourier basis 下看频率，在 SVD 基底下看主要模式。
- 图像压缩的线性代数本质，是把数据改写到更稀疏、更集中能量的基底上，再只保留最重要的模式。
- 这一讲把“换基不是换对象，而是换描述方式”真正落到了应用场景里。

>[!example] 例题
>
> 取基
> $$
> w_1=\begin{bmatrix}1\\1\end{bmatrix},\qquad w_2=\begin{bmatrix}1\\-1\end{bmatrix}.
> $$
> 若向量 $x=\begin{bmatrix}3\\1\end{bmatrix}$，则在新基下求的是系数 c，使 $x=Wc$。这一步告诉你：换基不是把向量改变了，而是把“如何拆成基向量”这件事改写了。

### 易错点与补充

- 向量本身没有变，变的是它在新基下的坐标。
- 不要把“换基后的矩阵”与“原矩阵做了新的线性变换”混为一谈；它们描述的是同一变换。
- 图像压缩里删掉的是小模式，不是随便删像素；这背后依赖的是基底选择与低秩思想。
### 你要掌握

- 会写向量换基公式 $x=Wc$。
- 会写矩阵换基公式 $B=M^{-1}AM$。
- 能解释换基在压缩和简化计算中的实际作用。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 3.8 Left and right inverses; pseudoinverse

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.8sol.pdf|solution]]

关联卡片：[[Left Inverse]]、[[Right Inverse]]、[[Pseudoinverse]]

>[!note] 快速回忆
> - 这讲要回答：当普通 inverse 不存在时，什么叫“最佳逆”。
> - 你要立刻想起：left inverse / right inverse 依赖满秩方向；伪逆由 SVD 给出，统一最小二乘解与最小范数解。
> - 典型题型：判断何时存在 left/right inverse，写出 pseudoinverse 公式并解释其意义。
> - 它接到下一讲：Exam 3 前把 symmetric、Jordan、SVD、change of basis、pseudoinverse 串起来。

### 双边逆只是方阵满秩时的特例

若 $A$ 是满秩方阵，当然存在普通逆
$$
A^{-1}A=I=AA^{-1}.
$$
但对矩形矩阵，左右两边的可逆性要分开讨论。

### 左逆

若 A 满列秩，即 $r=n$，则列独立、零空间平凡。此时
$$
A_{\text{left}}^{-1}=(A^TA)^{-1}A^T
$$
满足
$$
A_{\text{left}}^{-1}A=I_n.
$$
它能从输出中恢复唯一的输入参数，但 $AA_{\text{left}}^{-1}$ 只是投影到列空间，不会是整个 $\mathbb{R}^m$ 上的恒等。

### 右逆

若 A 满行秩，即 $r=m$，则行独立、左零空间平凡。此时
$$
A_{\text{right}}^{-1}=A^T(AA^T)^{-1}
$$
满足
$$
AA_{\text{right}}^{-1}=I_m.
$$
这意味着对每个 $b$ 都能找到某个解，但解通常不唯一，因为 nullspace 仍可能非平凡。

### 伪逆的动机

真正让逆矩阵失败的是非零 nullspace。某些信息一旦被 A 压成 0，就不可能被完全恢复。于是我们退一步：只在 row space 与 column space 之间做最自然的逆，并在其余方向上返回 0。这就是 [[Pseudoinverse]]。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-pseudoinverse.svg|840]]

这张图对应的关键判断是：A 并不是在整个输入空间上都可逆，它只在“没有被压扁的信息那一部分”上可逆。伪逆做的正是这件事，它不会假装把 nullspace 中已经丢失的信息恢复出来，而是老实地只在保留下来的 r 维信息子空间上做反向映射。

### 用 SVD 构造伪逆

若
$$
A=U\Sigma V^T,
$$
则定义
$$
A^+=V\Sigma^+U^T,
$$
其中 $\Sigma^+$ 把所有非零奇异值 $\sigma_i$ 变成 $1/\sigma_i$，零奇异值仍保留为 0。它的性质是：

- 给出最小二乘解；
- 在所有最小二乘解中选范数最小的那一个；
- 当 A 可逆时退化成普通逆。

如果把一个向量分解到 V 的基底上，那么伪逆的动作就非常透明：对那些被 A 真正保留下来的方向，伪逆把缩放因子倒回来；对那些被 A 压进 nullspace 的方向，伪逆保持为 0。于是“最小范数”并不是神秘附加条件，而是因为我们拒绝在已经失真的方向上凭空造信息。



### 本讲知识点全景

- 普通 inverse 只属于满秩方阵；对矩形矩阵，需要把“左边可逆”和“右边可逆”分开讨论。
- 若 A 满列秩，则存在 left inverse $(A^TA)^{-1}A^T$，它能在参数空间上恢复唯一输入。
- 若 A 满行秩，则存在 right inverse $A^T(AA^T)^{-1}$，它保证每个右端都至少能被某个输入打到。
- Moore-Penrose pseudoinverse 用 SVD 定义：
> $$A^+=V\Sigma^+U^T.$$ 
  它在非零奇异值方向上做真正的逆，在丢失信息的方向上保持为 0。
- pseudoinverse 同时统一了 least squares 解与最小范数解，因此它是“最佳逆”而不是“伪造的逆”。
- 这一讲把 Unit II 的 least squares、Unit III 的 SVD 和“矩形矩阵怎么办”全部收束到了一起。

>[!example] 例题
>
> 若 A 是 tall matrix 且列独立，则
> $$
> A^+=(A^TA)^{-1}A^T.
> $$
> 这恰好就是 Unit II 里 least squares 的公式。也就是说，least squares 并不是额外技巧，而是 pseudoinverse 在满列秩情形下的具体化。

### 易错点与补充

- pseudoinverse 不会把 nullspace 中已经丢失的信息“恢复出来”；它只在保留下来的模式上做逆。
- left inverse 与 right inverse 对应的是不同的秩条件，不能混用公式。
- 当 A 真正可逆时，$A^+$ 才退化成普通 inverse；否则它解决的是“最佳恢复”而非“完全恢复”。
### 你要掌握

- 能区分普通逆、左逆、右逆和伪逆。
- 能写出满列秩和满行秩时最自然的左右逆公式。
- 能用 SVD 解释伪逆为什么合理。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 3.9 Exam 3 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.9sum.pdf|summary]]

>[!note] 快速回忆
> - 这讲要回答：Exam 3 前最少必须保住哪几条结构线。
> - 你要立刻想起：`symmetric/positive definite -> similarity/Jordan -> SVD -> linear transformation/change of basis -> pseudoinverse`。
> - 典型题型：正定判别、Jordan 结构、SVD 解释、换基、左右逆与伪逆。
> - 复习时若公式都记得却说不出“为什么学这讲”，就回到本页开头的 `单元速览`。

### 本单元的统一图景

Unit III 真正要形成的是一个层次感：

- 对称矩阵最理想，能正交对角化；
- 正定矩阵进一步保证二次型总是向上，优化问题有唯一极小值；
- 一般方阵若可对角化，就用特征分解；若不可对角化，就看 Jordan form；
- 一般矩阵则统一由 SVD 处理；
- 线性变换和换基告诉你，这些分解本质上都是“找更好的坐标系”；
- 伪逆则把“逆矩阵”推广到非方阵和秩亏情形。

### 必练题型

- 判断矩阵是否对称、是否正定，并说明依据。
- 对可对角化矩阵写出特征分解，或说明为何不行。
- 对简单矩阵写出或解释 SVD。
- 从线性变换定义构造矩阵，或从矩阵反推变换含义。
- 做一次换基计算，并说明相似关系。
- 写出左右逆或伪逆的公式与适用条件。

### 常见混淆

- 正定必须先对称；仅有正特征值但不对称时，很多好性质并不自动成立。
- 相似和等于不是一回事；相似是同一变换在不同坐标系下的表示。
- SVD 与特征分解不同，前者适用于任意矩阵。
- 伪逆不是“随便补个逆”，而是 row space 到 column space 上最自然的逆。



### 高频题型展开

- `symmetric / positive definite`：常考多种判据切换，尤其是特征值、主元、二次型之间如何互证。
- `Jordan / SVD`：会不会区分“方阵但不够好”与“任意矩阵都能处理”的两条路线，是 Unit III 的核心分水岭。
- `linear transformation / change of basis`：高频概念题会专门检查你是否区分对象本身和坐标表示。
- `pseudoinverse`：压轴题通常要求你解释它为什么给出 least squares 或最小范数解，而不是只写公式。
- Unit III 真正考的是统一视角：你能否根据矩阵类型快速选择 spectral decomposition、Jordan form、SVD 或 pseudoinverse。

>[!example] 例题
>
> 若题目说“矩阵不对称、不是方阵、但要解释最重要模式和最佳逆”，你应立刻排除 diagonalization 和 Jordan，直接切换到
> $$
> A=U\Sigma V^T,\qquad A^+=V\Sigma^+U^T.
> $$
> 相反，若题目强调“对称且二次型严格向上”，则应优先回到 positive definite 与 spectral decomposition。Exam 3 的关键是选对语言。

### 易错点与补充

- Unit III 最常见的失误，是把所有分解都背成公式，却说不出“为什么该用这一种”。
- 复习时要不断问：题目是在问结构、坐标表示，还是最佳恢复；这三类问题的入口不同。
- 若只记住结论但忘了它在整门课中的角色，review session 的价值就还没有真正发挥出来。
### 你要掌握

- 能把 Unit III 压缩成 `symmetric / positive definite -> Jordan / SVD -> linear transformation / change of basis -> pseudoinverse`。
- 能说清楚为什么这单元是在给前两单元做结构收束，而不是另开新话题。
- 能根据矩阵类型快速判断该用 spectral decomposition、Jordan 还是 SVD。

### 回忆检查

- 不看正文，我能解释 Unit III 为什么必须同时讲 positive definite、Jordan、SVD 和 pseudoinverse。
- 我能区分“矩阵性质”“坐标系变化”“最佳逆”这三类问题各自该回哪一节。
- 如果我只记得公式，我也能说出每一讲在整门课里的角色。

## 本单元复习清单

- 我能解释为什么对称矩阵一定有正交特征向量基。
- 我能在二次型、特征值、主元三个视角下判断正定。
- 我能说明相似、换基和 Jordan form 的关系。
- 我能从 $A^TA$ 出发解释 SVD 的来源与几何意义。
- 我能把线性变换与矩阵表示区分开。
- 我能写出左逆、右逆、伪逆，并知道各自适用的秩条件。
