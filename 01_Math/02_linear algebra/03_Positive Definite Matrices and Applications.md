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
- 如果说 Unit I 是“解方程”，Unit II 是“最佳逼近与特征结构”，那么 Unit III 就是在问：这些结构最终如何统一。

## Session 3.1 Symmetric matrices and positive definiteness

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.1sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.1prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.1sol.pdf|solution]]

关联卡片：[[Symmetric Matrix]]、[[Positive Definite Matrix]]、[[Spectral Decomposition]]

- 实对称矩阵是线性代数里的“好矩阵”：特征值全实、特征向量可以选成正交基。
- 证明特征值实数的关键技巧是比较 `x^TAx` 与它的共轭转置形式，从而推出 `\lambda=\bar\lambda`。
- 因此对称矩阵可写成 `A=Q\Lambda Q^T`，这就是 [[Spectral Decomposition]]。
- 另一种等价写法是 `A=\lambda_1q_1q_1^T+\cdots+\lambda_nq_nq_n^T`，也就是把 A 拆成若干相互垂直的一维投影。
- 正定矩阵是更强的情形：所有特征值都为正，所有主元也都为正。
- 对称矩阵正定的常见等价判据：
  - `x^TAx>0` 对所有非零 x；
  - 所有 eigenvalues 正；
  - 所有 pivots 正；
  - 所有 leading principal minors 正。
- 这说明对称、主元、特征值、二次型这些此前分散的概念已经开始合流。
- 本讲要会：
  - 用对称性推出 orthogonal diagonalization。
  - 在 eigenvalue / pivot / quadratic form 三种判据间切换。

## Session 3.2 Complex matrices; fast Fourier transform

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.2sol.pdf|solution]]

关联卡片：[[Fourier Series]]、[[Orthogonality]]

- 实矩阵也可能有复特征值，因此复数不是可选装饰，而是一般理论的一部分。
- 复向量空间中的内积要带共轭，这直接影响“正交”的写法。
- 对实矩阵来说，若 `a+bi` 是特征值，则 `a-bi` 也是特征值。
- FFT 的核心思想并不是神秘算法，而是利用特殊的 Fourier basis 把原本昂贵的运算拆成许多便宜的小块。
- 也就是说，FFT 的本质仍然是“选了一组特别好的 basis”。
- 这一讲提醒我们：换一组基，计算复杂度与可解释性都会变化。
- 本讲要会：
  - 接受 complex eigenvalues 是正常现象。
  - 用“basis selection”而不是“算法黑箱”理解 FFT。

## Session 3.3 Positive definite matrices and minima

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.3sol.pdf|solution]]

关联卡片：[[Positive Definite Matrix]]

- 二次型 `x^TAx` 是研究正定矩阵最直接的方式。
- 若 A 正定，则 `x^TAx>0` 对所有非零 x 成立；这使得很多优化问题有唯一极小值。
- 正定矩阵因此自然出现在最小二乘、最优化、统计协方差矩阵和能量函数里。
- 从几何上看，正定矩阵定义了一个被拉伸旋转后的椭球；从代数上看，它是“各个特征方向都朝正方向拉伸”。
- 若存在 0 特征值，则只是半正定；若有负特征值，则二次型会沿某些方向下降。
- 从分析上看，正定性是稳定、唯一、可逆、局部凸性这些性质的共同来源。
- 本讲要会：
  - 用二次型视角解释正定。
  - 看见 minima / energy / quadratic objective 时主动想到 positive definite。

## Session 3.4 Similar matrices and Jordan form

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.4sol.pdf|solution]]

关联卡片：[[Similar Matrix]]、[[Jordan Form]]、[[Diagonalization]]

- [[Similar Matrix]] 表示的是“同一个线性变换在不同基下的矩阵表示”，所以相似矩阵有相同特征值。
- [[Similar Matrix]] 表示的是“同一个线性变换在不同基下的矩阵表示”，所以相似矩阵有相同特征值、trace、determinant。
- 对角化是最理想的相似变换：把矩阵变成完全对角。
- 但如果特征向量不够多，对角化失败，[[Jordan Form]] 提供最接近对角化的标准形式。
- Jordan block 在对角线上是同一个特征值，在超对角线上多出 1，编码的正是“还差了一点对角化”的缺陷。
- 这也解释了为什么某些矩阵幂与矩阵指数会出现多项式因子而不只是纯指数项。
- 这讲的核心不是机械求 Jordan form，而是理解“失败的对角化长什么样”。
- 本讲要会：
  - 解释相似为什么等价于换基。
  - 解释 Jordan block 的结构含义。

## Session 3.5 Singular value decomposition

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.5sol.pdf|solution]]

关联卡片：[[Singular Value Decomposition]]、[[Orthogonal Matrix]]、[[Pseudoinverse]]

- [[Singular Value Decomposition]] 把任意矩阵写成 `A=U\Sigma V^T`。
- `V` 来自 `A^TA` 的特征向量，`U` 来自 `AA^T` 的特征向量，`\Sigma` 的对角元是奇异值。
- 计算上，先看 `A^TA=V\Sigma^T\Sigma V^T`，因此 V 来自 `A^TA` 的 eigenvectors，singular values 是 `A^TA` eigenvalues 的平方根。
- 同理，`AA^T` 的 eigenvectors 给出 U。
- 对 rank-deficient 矩阵，`\Sigma` 后半段自动出现 0，这正是 nullspace 与 left nullspace 的位置。
- 因而 SVD 同时给出四个基本子空间的正交基：
  - `v_1,\dots,v_r` 张成 row space；
  - `v_{r+1},\dots,v_n` 张成 nullspace；
  - `u_1,\dots,u_r` 张成 column space；
  - `u_{r+1},\dots,u_m` 张成 left nullspace。
- [[Singular Value Decomposition|SVD]] 统一了 rank、四个基本子空间、最佳低秩逼近和伪逆。
- 对非方阵而言，[[Singular Value Decomposition|SVD]] 是比特征分解更普适的结构工具。
- 本讲要会：
  - 从 `A^TA` 和 `AA^T` 理解 U、V、`\Sigma` 的来源。
  - 解释零奇异值与 nullspaces 的关系。

## Session 3.6 Linear transformations and their matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.6sol.pdf|solution]]

关联卡片：[[Linear Transformation]]

- 矩阵不是第一性对象，[[Linear Transformation]] 才是；矩阵只是给定基之后的坐标表示。
- 线性变换的定义是 `T(v+w)=T(v)+T(w)` 与 `T(cv)=cT(v)`。
- projection、rotation、reflection、derivative 都是线性变换；平移和取长度则不是。
- 只要知道一组 basis 向量在 T 下的像，就能知道任意向量在 T 下的像。
- 因而矩阵的每一列，本质上就是“某个 basis vector 经过 T 后在输出基下的坐标”。
- 同一个变换在不同基下会有不同矩阵，但变换本身不变。
- 这能解释为什么相似、换基、对角化本质上都是“重新选择观察坐标”。
- 一旦把矩阵看成变换的代表，很多“公式”会转化成更自然的几何理解。
- 本讲要会：
  - 用 basis images 构造 transformation matrix。
  - 区分“变换本身”和“变换的矩阵表示”。

## Session 3.7 Change of basis; image compression

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.7sol.pdf|solution]]

关联卡片：[[Change of Basis]]、[[Singular Value Decomposition]]

- [[Change of Basis]] 的目标不是换记号，而是找到更适合计算或解释的坐标系。
- 对向量来说，如果新基矩阵是 W，则 `x=Wc`，这里 x 是旧基坐标，c 是新基坐标。
- 对线性变换来说，如果同一变换在两组基下的矩阵分别是 A 与 B，那么 `B=M^{-1}AM`。
- 这再次说明“相似矩阵 = 同一变换在不同基下的表示”。
- 图像压缩是换基思想的现实应用：
  - 像素 basis 直观但通常不稀疏；
  - Fourier / wavelet basis 更能把信息集中到少量系数里；
  - 把小系数砍掉，就能近似保真地压缩。
- 一个“好基”的标准通常是：变换和逆变换都快，且大多数系数都很小。
- 这讲把“基”从抽象定义变成了真正可操作的建模选择。
- 本讲要会：
  - 写出向量与矩阵的换基公式。
  - 用 image compression 解释“为什么要换基”。

## Session 3.8 Left and right inverses; pseudoinverse

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.8sol.pdf|solution]]

关联卡片：[[Left Inverse]]、[[Right Inverse]]、[[Pseudoinverse]]

- 对非方阵而言，左逆和右逆不再等价，它们分别对应满列秩与满行秩。
- 若 A 满列秩，则存在 favorite left inverse：`(A^TA)^{-1}A^T`。
- 若 A 满行秩，则存在 favorite right inverse：`A^T(AA^T)^{-1}`。
- 这两种 inverse 都是“部分可逆”：只能在某一侧恢复单位矩阵。
- [[Pseudoinverse]] 则给出统一的“最佳逆”：在可逆时退化成普通逆，在不可逆或非方阵时给出最小二乘意义下的自然替代。
- 深层结构是：A 把 row space 一一对应地映到 column space；真正不可逆的麻烦来自 nullspace。
- 若 `A=U\Sigma V^T`，则 `A^+=V\Sigma^+U^T`，只需把非零奇异值逐个取倒数即可。
- [[Singular Value Decomposition|SVD]] 是理解伪逆最干净的方式，因为它把“可逆部分”和“被压扁部分”拆开了。
- 这讲把“逆矩阵”从方阵特例推广为一般线性映射的稳定求解框架。
- 本讲要会：
  - 区分 two-sided inverse、left inverse、right inverse、pseudoinverse。
  - 知道 `A^+` 与 least squares 的关系。

## Session 3.9 Exam 3 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.9sum.pdf|summary]]

- Exam 3 的重心是：eigen / diagonalization、`e^{At}`、symmetric / positive definite、similar matrices、SVD。
- 线性变换、change of basis、pseudoinverse 会更多地留给 final。
- 真正要掌握的不是单个技巧，而是何时该用对称谱分解、何时该用相似、何时必须上 SVD。
- 本单元真正的统一图景是：
  - 对称矩阵对应最干净的特征结构；
  - 相似/Jordan 处理对角化失败；
  - SVD 处理非方阵与一般矩阵；
  - 线性变换与换基把这些计算放回几何本体。

## 本单元复习清单

- 我能解释为什么对称矩阵一定能正交对角化。
- 我能用特征值或主元判断正定，而不只会背定义。
- 我能解释相似与换基的关系。
- 我能说明 [[Singular Value Decomposition|SVD]] 为什么比特征分解更普适。
- 我能把伪逆理解成“最小二乘 + 最小范数”的统一答案。
- 我能把 Unit I、II 的内容都在 SVD 和 linear transformation 视角下重新理解一遍。
