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

## 本单元主线

Unit III 处理的是“把前两单元的工具彻底收束起来”。前面我们分别学过子空间、正交、最小二乘、行列式、特征值；这一单元开始问：对称矩阵为什么特别好，一般矩阵怎样被标准形式描述，线性变换与换基到底是什么，非方阵和秩亏矩阵的“最佳逆”又该怎样定义。

因此这一单元最重要的结论不是某一个公式，而是几个统一视角：

- 对称矩阵最适合用正交特征分解；
- 一般方阵若不能对角化，就用相似与 Jordan 形式描述缺陷；
- 一般矩阵则由 [[Singular Value Decomposition]] 统一处理；
- 线性变换、换基和伪逆把这些分解重新解释成“坐标选择”和“最佳恢复”。

## Session 3.1 Symmetric matrices and positive definiteness

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.1sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.1prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.1sol.pdf|solution]]

关联卡片：[[Symmetric Matrix]]、[[Positive Definite Matrix]]、[[Spectral Decomposition]]

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

### 例子

对
$$
A=\begin{bmatrix}
5&2\\
2&3
\end{bmatrix},
$$
它对称，且主元为 5 与 $11/5$，都为正，所以 A 正定。其特征值满足
$$
\lambda^2-8\lambda+11=0,
$$
解为 $4\pm\sqrt{5}$，也都为正。

还可以直接看二次型：
$$
q(x_1,x_2)=5x_1^2+4x_1x_2+3x_2^2.
$$
若把交叉项配方或转到特征向量基底，会发现它变成两个正平方项的和。这个过程再次说明：对称矩阵的“好”不是偶然，而是因为总能找到一组正交方向把它彻底拆开。

### 你要掌握

- 能证明对称矩阵特征值为实数。
- 会在“特征值判据 / 主元判据 / 二次型判据”间切换。
- 能解释正交特征分解为何比一般对角化更强。

## Session 3.2 Complex matrices; fast Fourier transform

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.2sol.pdf|solution]]

关联卡片：[[Fourier Series]]、[[Orthogonality]]

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

### 你要掌握

- 知道复特征值为什么在实矩阵中成共轭对出现。
- 能解释为什么复向量内积要带共轭。
- 能把 FFT 理解成“结构化换基”，而不是算法黑箱。

## Session 3.3 Positive definite matrices and minima

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.3sol.pdf|solution]]

关联卡片：[[Positive Definite Matrix]]、[[Least Squares]]

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

### 你要掌握

- 能用二次型解释正定。
- 会把 $Ax=b$ 解释成二次目标函数的一阶条件。
- 知道正定性为什么自动带来唯一最小值。

## Session 3.4 Similar matrices and Jordan form

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.4sol.pdf|solution]]

关联卡片：[[Similar Matrix]]、[[Jordan Form]]、[[Diagonalization]]

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

### 你要掌握

- 能解释相似为何等于换基。
- 知道对角化失败时 Jordan block 记录了什么信息。
- 明白 Jordan form 的意义主要是结构解释，不是机械计算表演。

## Session 3.5 Singular value decomposition

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.5sol.pdf|solution]]

关联卡片：[[Singular Value Decomposition]]、[[Orthogonal Matrix]]、[[Pseudoinverse]]

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

### 你要掌握

- 能从 $A^TA$ 解释 V 和奇异值的来源。
- 能把 SVD 理解成“输入方向、拉伸强度、输出方向”的分解。
- 能说明为什么 SVD 同时给出四个基本子空间的正交基。

## Session 3.6 Linear transformations and their matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.6sol.pdf|solution]]

关联卡片：[[Linear Transformation]]

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

### 你要掌握

- 能从 $T(e_j)$ 写出矩阵。
- 能区分“变换本身”和“该变换在某组基下的矩阵”。
- 能快速判断一个映射是不是线性的。

## Session 3.7 Change of basis; image compression

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.7sol.pdf|solution]]

关联卡片：[[Change of Basis]]、[[Singular Value Decomposition]]

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

### 你要掌握

- 会写向量换基公式 $x=Wc$。
- 会写矩阵换基公式 $B=M^{-1}AM$。
- 能解释换基在压缩和简化计算中的实际作用。

## Session 3.8 Left and right inverses; pseudoinverse

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.8sol.pdf|solution]]

关联卡片：[[Left Inverse]]、[[Right Inverse]]、[[Pseudoinverse]]

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

### 你要掌握

- 能区分普通逆、左逆、右逆和伪逆。
- 能写出满列秩和满行秩时最自然的左右逆公式。
- 能用 SVD 解释伪逆为什么合理。

## Session 3.9 Exam 3 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.9sum.pdf|summary]]

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

## 本单元复习清单

- 我能解释为什么对称矩阵一定有正交特征向量基。
- 我能在二次型、特征值、主元三个视角下判断正定。
- 我能说明相似、换基和 Jordan form 的关系。
- 我能从 $A^TA$ 出发解释 SVD 的来源与几何意义。
- 我能把线性变换与矩阵表示区分开。
- 我能写出左逆、右逆、伪逆，并知道各自适用的秩条件。
