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

- 这一单元把前两单元的工具全部汇总起来：正交、特征值、行列式、秩、子空间、矩阵分解。
- 它回答两个更高层的问题：什么矩阵“结构最好”；一般矩阵和一般线性变换如何被分解、近似和表示。
- 这部分的终点是 [[Singular Value Decomposition]]、[[Linear Transformation]]、[[Change of Basis]]、[[Pseudoinverse]]。

## Session 3.1 Symmetric matrices and positive definiteness

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.1sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.1prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.1sol.pdf|solution]]

关联卡片：[[Symmetric Matrix]]、[[Positive Definite Matrix]]、[[Spectral Decomposition]]

- 实对称矩阵是线性代数里的“好矩阵”：特征值全实、特征向量可以选成正交基。
- 因此对称矩阵可写成 `A=Q\Lambda Q^T`，这就是 [[Spectral Decomposition]]。
- 正定矩阵是更强的情形：所有特征值都为正，所有主元也都为正。
- 这说明对称、主元、特征值、二次型这些此前分散的概念已经开始合流。

## Session 3.2 Complex matrices; fast Fourier transform

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.2sol.pdf|solution]]

关联卡片：[[Fourier Series]]、[[Orthogonality]]

- 实矩阵也可能有复特征值，因此复数不是可选装饰，而是一般理论的一部分。
- 复向量空间中的内积要带共轭，这直接影响“正交”的写法。
- FFT 的核心思想并不是神秘算法，而是利用特殊正交基把卷积或频域运算变快。
- 这一讲提醒我们：换一组基，计算复杂度与可解释性都会变化。

## Session 3.3 Positive definite matrices and minima

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.3sol.pdf|solution]]

关联卡片：[[Positive Definite Matrix]]

- 二次型 `x^TAx` 是研究正定矩阵最直接的方式。
- 若 A 正定，则 `x^TAx>0` 对所有非零 x 成立；这使得很多优化问题有唯一极小值。
- 从几何上看，正定矩阵定义了一个被拉伸旋转后的椭球。
- 从分析上看，正定性是稳定、唯一、可逆、局部凸性这些性质的共同来源。

## Session 3.4 Similar matrices and Jordan form

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.4sol.pdf|solution]]

关联卡片：[[Similar Matrix]]、[[Jordan Form]]、[[Diagonalization]]

- [[Similar Matrix]] 表示的是“同一个线性变换在不同基下的矩阵表示”，所以相似矩阵有相同特征值。
- 当矩阵没有足够多的特征向量时，对角化失败，[[Jordan Form]] 提供最接近对角化的标准形式。
- Jordan block 解释了为什么有些系统除了指数项外还会冒出多项式因子。
- 这讲的核心不是机械求 Jordan form，而是理解“失败的对角化长什么样”。

## Session 3.5 Singular value decomposition

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.5sol.pdf|solution]]

关联卡片：[[Singular Value Decomposition]]、[[Orthogonal Matrix]]、[[Pseudoinverse]]

- [[Singular Value Decomposition]] 把任意矩阵写成 `A=U\Sigma V^T`。
- `V` 来自 `A^TA` 的特征向量，`U` 来自 `AA^T` 的特征向量，`\Sigma` 的对角元是奇异值。
- [[Singular Value Decomposition|SVD]] 统一了 rank、四个基本子空间、最佳低秩逼近和伪逆。
- 对非方阵而言，[[Singular Value Decomposition|SVD]] 是比特征分解更普适的结构工具。

## Session 3.6 Linear transformations and their matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.6sol.pdf|solution]]

关联卡片：[[Linear Transformation]]

- 矩阵不是第一性对象，[[Linear Transformation]] 才是；矩阵只是给定基之后的坐标表示。
- 同一个变换在不同基下会有不同矩阵，但变换本身不变。
- 这能解释为什么相似、换基、对角化本质上都是“重新选择观察坐标”。
- 一旦把矩阵看成变换的代表，很多“公式”会转化成更自然的几何理解。

## Session 3.7 Change of basis; image compression

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.7sol.pdf|solution]]

关联卡片：[[Change of Basis]]、[[Singular Value Decomposition]]

- [[Change of Basis]] 的目标不是换记号，而是找到更适合计算或解释的坐标系。
- 在新基下，复杂矩阵可能变成对角、块对角或更稀疏的形式。
- 图像压缩是换基思想的现实应用：只保留最重要的奇异值和对应方向，就能保留主要信息。
- 这讲把“基”从抽象定义变成了真正可操作的建模选择。

## Session 3.8 Left and right inverses; pseudoinverse

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.8sol.pdf|solution]]

关联卡片：[[Left Inverse]]、[[Right Inverse]]、[[Pseudoinverse]]

- 对非方阵而言，左逆和右逆不再等价，它们分别对应满列秩与满行秩。
- [[Pseudoinverse]] 则给出统一的“最佳逆”：在可逆时退化成普通逆，在不可逆或非方阵时给出最小二乘意义下的自然替代。
- [[Singular Value Decomposition|SVD]] 是理解伪逆最干净的方式，因为只要把非零奇异值逐个取倒数即可。
- 这讲把“逆矩阵”从方阵特例推广为一般线性映射的稳定求解框架。

## Session 3.9 Exam 3 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.9sum.pdf|summary]]

- Exam 3 的重心是：eigen / diagonalization、`e^{At}`、symmetric / positive definite、similar matrices、SVD。
- 线性变换、change of basis、pseudoinverse 会更多地留给 final。
- 真正要掌握的不是单个技巧，而是何时该用对称谱分解、何时该用相似、何时必须上 SVD。

## 本单元复习清单

- 我能解释为什么对称矩阵一定能正交对角化。
- 我能用特征值或主元判断正定，而不只会背定义。
- 我能解释相似与换基的关系。
- 我能说明 SVD 为什么比特征分解更普适。
- 我能把伪逆理解成“最小二乘 + 最小范数”的统一答案。
