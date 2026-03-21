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

- Unit I 研究的是“方程能不能解”；Unit II 研究的是“无解时怎样找最佳近似，以及方阵的长期行为如何被特征结构控制”。
- 这部分最关键的桥梁是正交。正交把几何距离、最小二乘、投影矩阵、正交基和对角化连成一个系统。
- 这一单元开始，`A^TA`、determinant、eigenvalues 不再是孤立公式，而是不同层面的结构信息。
- 你可以把这一单元看成两段：
  - 前半段：正交、投影、least squares、Gram-Schmidt。
  - 后半段：determinant、eigenvalues、diagonalization、`e^{At}`、Markov / Fourier。

## Session 2.1 Orthogonal vectors and subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.1prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.1sol.pdf|solution]]

关联卡片：[[Orthogonality]]、[[Column Space]]、[[Null Space]]

- 两个向量正交等价于内积为零；零向量与所有向量都正交。
- 若 `x ⟂ y`，则
$$
\|x+y\|^2=\|x\|^2+\|y\|^2,
$$
  这正是勾股定理的代数写法。
- 两个子空间正交，意思是其中任一向量与另一个子空间中的任一向量都正交。
- 线性代数的大图景之一：[[Row Space|row space]] 与 [[Null Space|nullspace]] 互相正交，[[Column Space|column space]] 与 [[Left Nullspace|left nullspace]] 互相正交。
- 这两对正交关系不是偶然：
  - `Ax=0` 意味着 x 与 A 的每一行都点积为 0，所以 x 与整个 row space 正交。
  - `A^Ty=0` 意味着 y 与 A 的每一列都点积为 0，所以 y 与整个 column space 正交。
- 维数上也正好补齐：`r + (n-r)=n`，`r + (m-r)=m`，因此它们是 orthogonal complements。
- `N(A^TA)=N(A)` 是最小二乘理论的入口，它说明 `A^TA` 不会凭空消灭原来不在 nullspace 里的方向。
- `rank(A^TA)=rank(A)`，因此 `A^TA` 可逆当且仅当 A 的列独立。
- 本讲要会：
  - 用“点积为 0”与“空间正交补”两种语言描述正交。
  - 证明 row space ⟂ nullspace。

## Session 2.2 Projections onto subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.2sol.pdf|solution]]

关联卡片：[[Orthogonal Projection]]、[[Projection Matrix]]、[[Orthogonality]]

- 投影是“离某个子空间最近的点”，最关键的几何条件不是“平行”，而是“误差向量与子空间正交”。
- 对一条由向量 `a` 张成的直线，投影公式是
$$
p = a\frac{a^Tb}{a^Ta},\qquad e=b-p.
$$
- 这里 `p` 在目标子空间里，`e` 与目标子空间正交。
- 对更高维子空间，投影条件写成 `A^T(b-p)=0`，这会直接导向正规方程。
- 最佳近似不靠猜，而靠“正交误差”来刻画。
- 如果投影基是标准正交基，那么系数会极其简单：每个坐标就是 `q_i^Tb`。
- 投影不是近似的附属品，而是理解最小二乘与 [[Fourier Series|Fourier 展开]] 的母概念。
- 本讲要会：
  - 从“最近点”推导“误差正交”。
  - 熟练写出投影 `p` 与误差 `e`。

## Session 2.3 Projection matrices and least squares

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.3sol.pdf|solution]]

关联卡片：[[Projection Matrix]]、[[Least Squares]]、[[Orthogonal Projection]]

- 当 `Ax=b` 无解时，我们转而求 `Ax` 在 `Col(A)` 中最接近 `b` 的点。
- 若 `p=A\hat{x}` 是投影点，则误差 `e=b-p` 必须满足 `A^Te=0`，于是得到正规方程
$$
A^TA\hat{x}=A^Tb.
$$
- 投影矩阵写成
$$
P=A(A^TA)^{-1}A^T,
$$
  前提是 A 的列线性无关。
- 对应的误差投影矩阵是 `I-P`，它把向量送到 left nullspace。
- 典型例子是拟合直线 `b=C+Dt`。把数据点写成矩阵方程后，最小二乘就是解一个 `2×2` 的正规方程。
- summary 里的例子给出：
$$
3\hat C+6\hat D=5,\qquad 6\hat C+14\hat D=11,
$$
  解得 `\hat D=1/2, \hat C=2/3`。
- `A^TA` 为什么可逆？因为若 `A^TAx=0`，则 `(Ax)^T(Ax)=0`，所以 `Ax=0`，列独立时只能推出 `x=0`。
- 最小二乘的真正含义不是“凑一个答案”，而是把误差压到 `Col(A)` 的正交补里。
- 本讲要会：
  - 用 projection view 与 regression view 解释同一个 least squares 问题。
  - 推导并解正规方程。

## Session 2.4 Orthogonal matrices and Gram-Schmidt

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.4sol.pdf|solution]]

关联卡片：[[Orthogonal Matrix]]、[[Gram-Schmidt Orthogonalization]]

- 标准正交向量组满足
$$
q_i^Tq_j=
\begin{cases}
1,&i=j\\
0,&i\ne j
\end{cases}
$$
  因而一定线性无关。
- [[Orthogonal Matrix]] 满足 `Q^TQ=I`，所以它不会改变长度与角度，只会做旋转和反射。
- 如果 Q 是方阵，那么 `Q^{-1}=Q^T`；如果 Q 只是列正交，则 `Q^TQ=I` 但 `QQ^T` 是投影矩阵。
- 用标准正交基时，投影与坐标都极简：`x̂=Q^Tb`。
- [[Gram-Schmidt Orthogonalization]] 的作用是把任意一组基改造成正交或标准正交基。
- 两向量时公式最直观：
$$
A=a,\qquad B=b-\frac{A^Tb}{A^TA}A,
$$
  然后再做单位化。
- 三向量时继续减去在前面所有正交方向上的投影：
$$
C=c-\frac{A^Tc}{A^TA}A-\frac{B^Tc}{B^TB}B.
$$
- 这一讲直接连接到 QR 分解：若 A 的列向量经 Gram-Schmidt 得到 Q，则 `A=QR` 且 `R=Q^TA` 是上三角矩阵。
- 本讲要会：
  - 判断一个矩阵是否 orthogonal。
  - 手工完成 2 维或 3 维 Gram-Schmidt。
  - 解释 QR 为什么是“正交版的消元”。

## Session 2.5 Properties of determinants

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.5sol.pdf|solution]]

关联卡片：[[Determinant]]

- [[Determinant]] 是把“可逆性”和“体积缩放”编码成一个标量。
- 课程里重点不是先背大公式，而是先掌握 3 条定义性质：
  - `det(I)=1`
  - 交换两行会变号
  - determinant 对单行是线性的
- 从这 3 条性质可以推出整套常用规则：
  - 两行相同，则 determinant 为 0
  - 一行是零行，则 determinant 为 0
  - 某行减去另一行的倍数，不改变 determinant
  - 三角矩阵的 determinant 等于对角线乘积
  - `det(A)=0` 当且仅当 A 奇异
  - `det(AB)=det(A)det(B)`
  - `det(A^T)=det(A)`
- 从消元角度看，determinant 等于主元乘积乘上换行带来的符号修正。
- 这就是为什么计算大矩阵 determinant 时，实际方法通常是 elimination，而不是展开。
- 本讲要会：
  - 把行操作对 determinant 的影响说清楚。
  - 从三角矩阵和消元理解 determinant。

## Session 2.6 Determinant formulas and cofactors

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.6sol.pdf|solution]]

关联卡片：[[Determinant]]

- 真正计算时可以用余子式展开，但要先知道它为什么成立：它来自 determinant 的多线性与交替性。
- 对第 `i,j` 个元素，minor 是删去第 i 行第 j 列后的 determinant，cofactor 是
$$
C_{ij}=(-1)^{i+j}M_{ij}.
$$
- 沿一行或一列展开时，
$$
\det(A)=\sum_j a_{ij}C_{ij}.
$$
- 这给出了 determinant 的显式公式，也给出了 inverse matrix 的 cofactor / adjugate 公式来源。
- 行列式的大公式包含 `n!` 项，因此理论上重要、计算上通常不高效。
- 余子式矩阵和伴随矩阵为逆矩阵公式提供了显式表达，但在数值计算里通常不如消元稳定。
- 对做题而言，这一讲最重要的是理解“为什么能展开”，不是死记一个公式。
- 本讲要会：
  - 正确写出 minor、cofactor、cofactor expansion。
  - 解释为什么 cofactor 公式更多是理论工具而不是计算主力。

## Session 2.7 Cramer's rule, inverse matrix, and volume

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.7sol.pdf|solution]]

关联卡片：[[Cramer's Rule]]、[[Matrix Inverse]]、[[Determinant]]

- [[Cramer's Rule]] 给出 `Ax=b` 在 det 非零时的显式解：
$$
x_i=\frac{\det(A_i)}{\det(A)}.
$$
- 这里 `A_i` 是把 A 的第 i 列替换成 b 得到的矩阵。
- determinant 的几何意义在这里最直观：线性变换把面积或体积缩放了多少倍。
- `det(A^{-1})=1/det(A)`、`det(A^2)=det(A)^2` 都自然来自 `det(AB)=det(A)det(B)`。
- cofactor 进一步给出 inverse matrix 的显式公式，这说明 inverse 和 determinant 并不是两条无关的线。
- 逆矩阵公式、体积缩放与可逆性其实在说同一件事：A 是否把空间压扁了。
- 在方阵问题里，determinant 是“结构是否坍塌”的一个一眼可见的标记。
- 本讲要会：
  - 在小维度题里熟练使用 Cramer's rule。
  - 用体积缩放解释 determinant 的绝对值，用符号解释方向翻转。

## Session 2.8 Eigenvalues and eigenvectors

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.8sol.pdf|solution]]

关联卡片：[[Eigenvalues]]、[[Eigenvectors]]

- [[Eigenvectors]] 是在矩阵作用下方向不变的非零向量，[[Eigenvalues]] 是对应的缩放因子。
- 若 `\lambda=0`，那么 eigenvectors 恰好就是 nullspace 中的非零向量。
- 求 eigenvalues 的标准方法是：
$$
Av=\lambda v
\iff
(A-\lambda I)v=0
\iff
\det(A-\lambda I)=0.
$$
- 对 `2×2` 矩阵，有非常实用的关系：
$$
\lambda^2-\operatorname{trace}(A)\lambda+\det(A)=0.
$$
- 因而“特征值之和等于 trace，特征值之积等于 determinant”。
- 三类特别重要的例子：
  - projection matrix 的特征值只有 0 和 1；
  - triangular matrix 的特征值就在对角线上；
  - real matrix 也可能有 complex eigenvalues。
- 如果矩阵是反对称的，特征值会是纯虚数；如果矩阵是对称的，后面会看到特征值一定实数。
- 从这一讲开始，线性代数不再只处理“求解一次”，而是开始研究“反复作用很多次”的行为。
- 本讲要会：
  - 计算特征方程。
  - 从 `(A-\lambda I)v=0` 找 eigenvectors。
  - 用 trace 与 determinant 快速检查答案。

## Session 2.9 Diagonalization and powers of A

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.9sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.9prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.9sol.pdf|solution]]

关联卡片：[[Diagonalization]]、[[Eigenvalues]]、[[Eigenvectors]]

- 若 A 有 n 个线性无关特征向量，就可写成 `A=S\Lambda S^{-1}`。
- 这个等式的含义是：在 eigenvector basis 下，A 的作用只剩下按坐标分别缩放。
- 这使得
$$
A^k=S\Lambda^kS^{-1},
$$
  矩阵幂的复杂性被压缩成对角元的幂。
- 因此差分方程 `u_{k+1}=Au_k` 的长期行为完全由特征值的大小控制。
- 是否可对角化，取决于是否有足够多的独立特征向量，而不是特征值是否互异的表面形式。
- 如果没有足够多的特征向量，就要进入 Unit III 的 Jordan 视角。
- 本讲要会：
  - 判断一个矩阵能否 diagonalize。
  - 用 diagonalization 计算 `A^k`。

## Session 2.10 Differential equations and exp(At)

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.10sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.10prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.10sol.pdf|solution]]

关联卡片：[[Matrix Exponential]]、[[Diagonalization]]

- 系统 `du/dt=Au` 的解是 `u(t)=e^{At}u(0)`。
- 这里
$$
e^{At}=I+At+\frac{(At)^2}{2!}+\cdots
$$
  是 [[Matrix Exponential]]。
- 如果 A 可对角化，那么
$$
e^{At}=Se^{\Lambda t}S^{-1},
$$
  于是问题被拆成多个一维指数函数 `e^{\lambda_i t}`。
- 特征值的实部控制增长或衰减，虚部控制振荡；所以稳定性问题本质上就是 eigenvalue problem。
- 若 eigenvalues 是复数，对应的真实解会体现出旋转与振荡。
- 这讲把“特征值”从静态代数对象变成了动态系统中的稳定性指标。
- 本讲要会：
  - 从 diagonalization 推出 `e^{At}`。
  - 用 eigenvalues 判断系统长期行为。

## Session 2.11 Markov matrices; Fourier series

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.11prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.11sol.pdf|solution]]

关联卡片：[[Markov Matrix]]、[[Fourier Series]]、[[Orthogonal Projection]]

- [[Markov Matrix]] 的每一列和为 1，所以它总有特征值 1；长期稳态对应这个特征值的特征向量。
- 若其他特征值的绝对值都小于 1，那么反复迭代 `u_{k+1}=Au_k` 后，系统会收敛到 steady state。
- [[Fourier Series]] 则展示了另一种投影思想：把函数投影到正交基上，系数由内积给出。
- 在 Fourier 里，复杂信号被拆成不同频率模式；在 Markov 里，复杂状态分布被拆成不同衰减模式。
- Markov 与 Fourier 看似风马牛不相及，实则都在讲“选择合适基之后，复杂系统被分解成简单模式”。
- 本讲要会：
  - 从 Markov 矩阵读 steady state。
  - 理解 Fourier 系数为什么本质上就是投影坐标。

## Session 2.12 Exam 2 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.12sum.pdf|summary]]

- 必会公式：投影、最小二乘、`Q^TQ=I`、determinant 的基本性质、`det(A-\lambda I)=0`、对角化与矩阵幂。
- 必会判断：A 的列是否独立、`A^TA` 是否可逆、矩阵是否可对角化。
- 必会应用：best fit、[[Gram-Schmidt Orthogonalization|Gram-Schmidt]]、[[Markov Matrix|Markov]] 长期行为、用特征值分析迭代。
- 这一单元最重要的统一结构是：
  - 正交让距离可计算；
  - 投影让近似有最优性；
  - determinant 让可逆性有标量刻画；
  - eigenvalues 让长期行为可分解。

## 本单元复习清单

- 我能把“最近点问题”写成正交条件，再写成正规方程。
- 我能解释 projection matrix 为什么满足 `P^2=P`。
- 我能从正交基角度理解 QR，而不是只背 Gram-Schmidt 步骤。
- 我能把 determinant、eigenvalues、diagonalization 视为同一个方阵结构的不同切面。
- 我能从“子空间 + 正交 + 特征结构”三条线同时看同一道题。
