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

## 本单元主线

- 这一单元回答的是线性代数最根本的问题：`Ax=b` 什么时候有解，解为什么可能唯一、可能不存在、也可能有无穷多。
- 真正的结构不是“一个方程组”，而是矩阵 `A` 所对应的四个子空间：[[Column Space]]、[[Null Space]]、[[Row Space|row space]]、[[Left Nullspace|left nullspace]]。
- 做题时始终把三个视角连在一起：row picture、column picture、matrix picture。

## 使用资料

- 总入口：[[00_MIT OCW 18.06SC course map|课程总览]]
- 题目索引：[[MIT_OCW_18.06SC_PDF/index|MIT OCW 18.06SC 题目与资料索引]]
- 旧稿素材：[[00_perface]]、[[01_matrices and Gaussian Elimination]]、[[02_vector spaces and subspace]]
- 注意：本地 summary PDF 的 `An Overview of Key Ideas` 存在编号错位，对应文件是 `Ses1.13sum.pdf`，但本文按 syllabus 把它放在 Session 1.2。

## Session 1.1 The geometry of linear equations

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.1sum.pdf|summary]]

关联卡片：[[Linear system solution structure]]、[[Column Space]]、[[Vector Space]]

- 核心问题：把线性方程组看成几何对象时，解到底是什么。
- Row picture：每个方程给出一条直线或一个平面，解是这些几何对象的公共交点。
- Column picture：把右端项 `b` 写成矩阵列向量的线性组合，系数就是未知数。
- Matrix picture：用 `Ax=b` 统一行图像和列图像，矩阵乘法既是行与向量的点积，也是列向量的线性组合。
- 第一个重要判断：如果 A 的列向量不能覆盖整个目标空间，那么某些 `b` 一定无解，这就是“奇异”的早期征兆。

## Session 1.2 An overview of key ideas

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.13sum.pdf|summary]]

关联卡片：[[Linear Algebra-hub|线性代数 Hub]]、[[Vector Space]]、[[Matrix Rank]]

- 这一讲是整门课的路线图：消元告诉我们如何解方程，子空间解释为什么有解或无解，[[Orthogonality|正交]] 与 [[Orthogonal Projection|投影]] 处理无解情形，[[Determinant|行列式]] 和 [[Eigenvalues|特征值]] 刻画方阵的深层结构，[[Singular Value Decomposition|SVD]] 则把这些工具统一起来。
- “矩形矩阵”不是边角料，而是主角。四个基本子空间的维数关系要同时理解方阵与非方阵。
- 一条贯穿全课的主线是：rank 决定了空间被压缩成什么样，nullity 记录了丢失了多少自由度。
- 这一讲的作用不是记公式，而是建立地图：后面的概念都服务于“矩阵如何作用于空间”。

## Session 1.3 Elimination with matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.2sol.pdf|solution]]

关联卡片：[[Matrix Rank]]、[[LU Decomposition]]

- 高斯消元的目标是把矩阵变成上三角或更进一步的 row echelon form，从而暴露主元位置。
- 主元（pivot）对应真正独立的信息；没有主元的列最终会变成自由变量来源。
- 每一步消元都可以写成左乘一个消元矩阵，所以“行变换”本身也是矩阵运算。
- 做题时最重要的不是机械消元，而是每一步都知道自己在保留什么：保留解集，但改变表示。

## Session 1.4 Multiplication and inverse matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.3sol.pdf|solution]]

关联卡片：[[Matrix Inverse]]、[[Singular Matrix]]

- 矩阵乘法至少要会四种理解：行乘列、列组合、行组合、分块乘法。
- `AB` 的列来自 A 作用在 B 的列上，说明矩阵乘法本质上是“复合线性作用”。
- [[Matrix Inverse]] 的含义不是公式，而是“撤销 A 的作用”；存在逆矩阵等价于 A 可逆、无零特征方向、无非零 nullspace。
- `AB` 可逆时 `(AB)^{-1}=B^{-1}A^{-1}`，顺序反过来，因为复合变换要逆向拆解。

## Session 1.5 Factorization into A = LU

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.4sol.pdf|solution]]

关联卡片：[[LU Decomposition]]、[[Permutation Matrix]]

- 消元不仅能求解，还能把矩阵拆成结构化因子：`A=LU`。
- `U` 是消元后的上三角矩阵，`L` 记录消元乘子，所以它是单位下三角矩阵。
- 一旦需要换行，就不再是单纯 `A=LU`，而要写成 `PA=LU`，其中 `P` 是 [[Permutation Matrix]]。
- LU 的真正价值在于“分解一次，反复求解”：同一个 A 对多个不同右端 `b` 时效率极高。

## Session 1.6 Transposes, permutations, vector spaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.5sol.pdf|solution]]

关联卡片：[[Permutation Matrix]]、[[Vector Space]]、[[Subspace]]

- 转置 `A^T` 交换了行和列，因此 [[Row Space|row space]] 与 [[Column Space|column space]] 会在转置下互换。
- 置换矩阵本质上是“重新排列基向量”的单位矩阵，左乘做换行，右乘做换列。
- [[Vector Space]] 的本质是对加法与数乘封闭；[[Subspace]] 则是其中仍保留全部线性结构的子集。
- 这一步把“求解方程”正式提升到“研究空间中的线性结构”。

## Session 1.7 Column space and nullspace

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.6sol.pdf|solution]]

关联卡片：[[Column Space]]、[[Null Space]]、[[Subspace]]

- [[Column Space]] 由 A 的列向量张成，决定 `Ax=b` 可能落到哪里。
- [[Null Space]] 由所有满足 `Ax=0` 的向量组成，决定解的不唯一性来自哪里。
- `b` 有解的必要充分条件是 `b∈Col(A)`；齐次解空间 `N(A)` 永远经过原点。
- 这一讲开始，列空间和零空间不再只是定义，而成为“有没有解、解有几维”的核心工具。

## Session 1.8 Solving Ax = 0: pivot variables, special solutions

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.7sol.pdf|solution]]

关联卡片：[[Null Space]]、[[Matrix Rank]]

- 解齐次方程的关键是区分主变量和自由变量。
- 消元后每个自由变量都对应一个 special solution；这些特解张成整个 [[Null Space]]。
- 如果 `rank(A)=r` 且 A 有 n 列，那么自由变量个数是 `n-r`，这就是零空间的维数。
- 实战中最常见错误是：找到一个特解就停下。真正的答案必须写成“特解的线性组合”。

## Session 1.9 Solving Ax = b: row reduced form R

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.8sol.pdf|solution]]

关联卡片：[[Column Space]]、[[Null Space]]、[[Linear system solution structure]]

- 非齐次方程的通解结构是：`x = x_p + x_n`，其中 `x_p` 是一个特解，`x_n ∈ N(A)`。
- 如果增广矩阵在最后一列出现主元，则系统无解；这等价于 `b` 不在 [[Column Space]] 中。
- row reduced form 让“是否有解”和“自由度来自哪里”一目了然。
- 齐次方程给出的是方向结构，非齐次方程给出的是“平移后的同一几何结构”。

## Session 1.10 Independence, basis, and dimension

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.9prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.9sol.pdf|solution]]

关联卡片：[[Linear Independence]]、[[Vector Space]]、[[Matrix Rank]]

- 线性无关的意义是：没有冗余方向。
- 基（basis）是一组既能张成整个空间、又没有冗余的向量；维数是基向量个数。
- 消元中的 pivot columns 给出 column space 的一个基，但要注意它们来自原矩阵 A，而不是消元后的 U。
- 线性无关、张成、基、维数这四个概念是整门课后续所有“坐标表示”的基础。

## Session 1.11 The four fundamental subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.10prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.10sol.pdf|solution]]

关联卡片：[[Column Space]]、[[Null Space]]、[[Matrix Rank]]

- 对任意 `m×n` 矩阵 A，都要同时盯住四个空间：`C(A)`、`N(A)`、`C(A^T)`、`N(A^T)`。
- 维数关系是第一层：`dim C(A)=dim C(A^T)=rank(A)=r`，`dim N(A)=n-r`，`dim N(A^T)=m-r`。
- 结构关系是第二层：后续正交单元会告诉我们 `[[Row Space|row space]] ⟂ [[Null Space|nullspace]]`，`[[Column Space|column space]] ⟂ [[Left Nullspace|left nullspace]]`。
- 线性代数的很多题，本质上都可以还原成“你到底在问哪一个基本子空间”。

## Session 1.12 Matrix spaces; rank 1; small world graphs

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.11sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.11prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.11sol.pdf|solution]]

关联卡片：[[Matrix Rank]]

- 向量空间不一定长得像 `R^n`，矩阵本身也能构成向量空间。
- rank-1 矩阵通常可写成 `uv^T`，它把所有输入都压到一条直线方向上，是理解低秩结构的最好起点。
- small world graph 的出现提醒我们：线性代数不是纯代数工具，它可以直接编码网络结构。
- 一旦开始把“矩阵当成向量”，很多分解与逼近问题就会变得自然。

## Session 1.13 Graphs, networks, incidence matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.12sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.12prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.12sol.pdf|solution]]

关联卡片：[[Incidence Matrix]]、[[Column Space]]、[[Null Space]]

- 网络问题把矩阵的结构暴露得特别明显：节点、边、流量、约束都可以编码进 incidence matrix。
- [[Incidence Matrix]] 的列通常对应边，行对应节点，每列恰好包含一个 `+1` 和一个 `-1`。
- 图中的“环”会落到 [[Null Space|nullspace]]，图中的“连通性约束”会落到 [[Column Space|column space]] / [[Left Nullspace|left nullspace]] 的关系里。
- 这一讲把抽象子空间与真实系统连接起来，是后续 [[Markov Matrix|Markov]]、FFT、图像压缩等应用的预演。

## Session 1.14 Exam 1 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.14sum.pdf|summary]]

- 必会判断：`Ax=b` 是否可解、是否唯一、是否无穷多解。
- 必会操作：消元、找 pivot / free variables、写 special solutions、写一般解。
- 必会结构：column space、nullspace、basis、dimension、rank、四个基本子空间。
- 必会应用：把 network / incidence matrix 题还原成子空间问题。

## 本单元复习清单

- 我能在 row picture、column picture、matrix picture 三种视角之间切换。
- 我能从 rref 直接读出主变量、自由变量、nullspace 的基与通解。
- 我能解释“有解”为什么等价于 `b∈Col(A)`。
- 我能用 rank 和四个基本子空间组织所有题目，而不是只会机械计算。
