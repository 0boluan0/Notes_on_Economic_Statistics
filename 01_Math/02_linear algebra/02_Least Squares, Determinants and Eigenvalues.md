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

## Session 2.1 Orthogonal vectors and subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.1prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.1sol.pdf|solution]]

关联卡片：[[Orthogonality]]、[[Column Space]]、[[Null Space]]

- 两个向量正交等价于内积为零；零向量与所有向量都正交。
- 两个子空间正交，意思是其中任一向量与另一个子空间中的任一向量都正交。
- 线性代数的大图景之一：[[Row Space|row space]] 与 [[Null Space|nullspace]] 互相正交，[[Column Space|column space]] 与 [[Left Nullspace|left nullspace]] 互相正交。
- `N(A^TA)=N(A)` 是最小二乘理论的入口，它说明 `A^TA` 不会凭空消灭原来不在 nullspace 里的方向。

## Session 2.2 Projections onto subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.2sol.pdf|solution]]

关联卡片：[[Orthogonal Projection]]、[[Projection Matrix]]、[[Orthogonality]]

- 投影是“离某个子空间最近的点”，最关键的几何条件不是“平行”，而是“误差向量与子空间正交”。
- 对一条由向量 `a` 张成的直线，投影公式是 `p = a(a^Tb)/(a^Ta)`。
- 对更高维子空间，投影条件写成 `A^T(b-p)=0`，这会直接导向正规方程。
- 投影不是近似的附属品，而是理解最小二乘与 [[Fourier Series|Fourier 展开]] 的母概念。

## Session 2.3 Projection matrices and least squares

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.3sol.pdf|solution]]

关联卡片：[[Projection Matrix]]、[[Least Squares]]、[[Orthogonal Projection]]

- 当 `Ax=b` 无解时，我们转而求 `Ax` 在 `Col(A)` 中最接近 `b` 的点。
- 正规方程是 `A^TA\hat{x}=A^Tb`，它把“距离最小”变成了一个可解的方程组。
- 投影矩阵写成 `P=A(A^TA)^{-1}A^T`，前提是 A 的列线性无关。
- 最小二乘的真正含义不是“凑一个答案”，而是把误差压到 `Col(A)` 的正交补里。

## Session 2.4 Orthogonal matrices and Gram-Schmidt

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.4sol.pdf|solution]]

关联卡片：[[Orthogonal Matrix]]、[[Gram-Schmidt Orthogonalization]]

- [[Orthogonal Matrix]] 满足 `Q^TQ=I`，所以它不会改变长度与角度，只会做旋转和反射。
- 正交列向量使投影、求坐标、计算逆都变得简单，因为 `Q^{-1}=Q^T`。
- [[Gram-Schmidt Orthogonalization]] 的作用是把任意一组基改造成正交或标准正交基。
- 这一讲直接连接到 QR 分解：把一个矩阵的列分解成“正交方向 + 上三角系数”。

## Session 2.5 Properties of determinants

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.5sol.pdf|solution]]

关联卡片：[[Determinant]]

- [[Determinant]] 是把“可逆性”和“体积缩放”编码成一个标量。
- 三条定义 determinant 的核心性质：对行交换变号、对单行线性、对单位矩阵取值为 1。
- 从消元角度看，determinant 等于主元乘积乘上换行带来的符号修正。
- 一旦 det 为 0，说明体积塌陷、矩阵不可逆、存在非零 nullspace。

## Session 2.6 Determinant formulas and cofactors

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.6sol.pdf|solution]]

关联卡片：[[Determinant]]

- 真正计算时可以用余子式展开，但要先知道它为什么成立：它来自 determinant 的多线性与交替性。
- 行列式的大公式包含 `n!` 项，因此理论上重要、计算上通常不高效。
- 余子式矩阵和伴随矩阵为逆矩阵公式提供了显式表达，但在数值计算里通常不如消元稳定。
- 对做题而言，这一讲最重要的是理解“为什么能展开”，不是死记一个公式。

## Session 2.7 Cramer's rule, inverse matrix, and volume

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.7sol.pdf|solution]]

关联卡片：[[Cramer's Rule]]、[[Matrix Inverse]]、[[Determinant]]

- [[Cramer's Rule]] 给出 `Ax=b` 在 det 非零时的显式解，但它更适合作为理论窗口而不是计算主力。
- determinant 的几何意义在这里最直观：线性变换把面积或体积缩放了多少倍。
- 逆矩阵公式、体积缩放与可逆性其实在说同一件事：A 是否把空间压扁了。
- 在方阵问题里，determinant 是“结构是否坍塌”的一个一眼可见的标记。

## Session 2.8 Eigenvalues and eigenvectors

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.8sol.pdf|solution]]

关联卡片：[[Eigenvalues]]、[[Eigenvectors]]

- [[Eigenvectors]] 是在矩阵作用下方向不变的非零向量，[[Eigenvalues]] 是对应的缩放因子。
- 方程 `Av=\lambda v` 可改写成 `(A-\lambda I)v=0`，因此特征值来自 `det(A-\lambda I)=0`。
- 一旦找到特征方向，复杂矩阵作用就变成各方向上的独立缩放。
- 从这一讲开始，线性代数不再只处理“求解一次”，而是开始研究“反复作用很多次”的行为。

## Session 2.9 Diagonalization and powers of A

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.9sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.9prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.9sol.pdf|solution]]

关联卡片：[[Diagonalization]]、[[Eigenvalues]]、[[Eigenvectors]]

- 若 A 有 n 个线性无关特征向量，就可写成 `A=S\Lambda S^{-1}`。
- 这使得 `A^k=S\Lambda^kS^{-1}`，矩阵幂的复杂性被压缩成对角元的幂。
- 是否可对角化，取决于是否有足够多的独立特征向量，而不是特征值是否互异的表面形式。
- 许多离散时间系统、差分方程和稳定性判断都靠这一步完成。

## Session 2.10 Differential equations and exp(At)

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.10sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.10prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.10sol.pdf|solution]]

关联卡片：[[Matrix Exponential]]、[[Diagonalization]]

- 系统 `du/dt=Au` 的解是 `u(t)=e^{At}u(0)`。
- 如果 A 可对角化，那么 `e^{At}=Se^{\Lambda t}S^{-1}`，于是问题被拆成多个一维指数函数。
- 特征值的实部控制增长或衰减，虚部控制振荡。
- 这讲把“特征值”从静态代数对象变成了动态系统中的稳定性指标。

## Session 2.11 Markov matrices; Fourier series

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses2.11prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses2.11sol.pdf|solution]]

关联卡片：[[Markov Matrix]]、[[Fourier Series]]、[[Orthogonal Projection]]

- [[Markov Matrix]] 的每一列和为 1，所以它总有特征值 1；长期稳态对应这个特征值的特征向量。
- 其余特征值若绝对值小于 1，对应分量会在多次迭代后衰减掉。
- [[Fourier Series]] 则展示了另一种投影思想：把函数投影到正交基上，系数由内积给出。
- Markov 与 Fourier 看似风马牛不相及，实则都在讲“选择合适基之后，复杂系统被分解成简单模式”。

## Session 2.12 Exam 2 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.12sum.pdf|summary]]

- 必会公式：投影、最小二乘、`Q^TQ=I`、determinant 的基本性质、`det(A-\lambda I)=0`、对角化与矩阵幂。
- 必会判断：A 的列是否独立、`A^TA` 是否可逆、矩阵是否可对角化。
- 必会应用：best fit、[[Gram-Schmidt Orthogonalization|Gram-Schmidt]]、[[Markov Matrix|Markov]] 长期行为、用特征值分析迭代。

## 本单元复习清单

- 我能把“最近点问题”写成正交条件，再写成正规方程。
- 我能解释 projection matrix 为什么满足 `P^2=P`。
- 我能从正交基角度理解 QR，而不是只背 Gram-Schmidt 步骤。
- 我能把 determinant、eigenvalues、diagonalization 视为同一个方阵结构的不同切面。
