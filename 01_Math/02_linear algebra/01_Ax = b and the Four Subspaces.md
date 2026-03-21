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
- 这一单元的关键词是：消元、主元、自由变量、basis、dimension、rank、four fundamental subspaces。
- 学完整个 unit 之后，你应该能把任何“解方程组”的题自动翻译成“某个子空间上的结构题”。

## 使用资料

- 总入口：[[00_MIT OCW 18.06SC course map|课程总览]]
- 题目索引：[[MIT_OCW_18.06SC_PDF/index|MIT OCW 18.06SC 题目与资料索引]]
- 注意：本地 summary PDF 的 `An Overview of Key Ideas` 存在编号错位，对应文件是 `Ses1.13sum.pdf`，但本文按 syllabus 把它放在 Session 1.2。

## Session 1.1 The geometry of linear equations

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.1sum.pdf|summary]]

关联卡片：[[Linear system solution structure]]、[[Column Space]]、[[Vector Space]]

- 核心问题：把线性方程组看成几何对象时，解到底是什么。
- 典型例子是
$$
\begin{cases}
2x-y=0\\
-x+2y=3
\end{cases}
$$
  两条直线的交点就是解 `(x,y)=(1,2)`。
- Row picture：每个方程给出一条直线或一个平面，解是这些几何对象的公共交点。三元一次方程时，对应的是三个平面的交点。
- Column picture：把右端项 `b` 写成矩阵列向量的线性组合。这里
$$
x\begin{bmatrix}2\\-1\end{bmatrix}
+y\begin{bmatrix}-1\\2\end{bmatrix}
=\begin{bmatrix}0\\3\end{bmatrix},
$$
  所以“求解”就是问 `b` 是否在这两列向量张成的平面里，以及它的坐标是多少。
- Matrix picture：把系统统一写成 `Ax=b`。这一步把“方程组”升级成“矩阵作用在向量上”的问题。
- 矩阵乘法有两个同时成立的理解：
  - 行视角：每一行与 `x` 做点积，得到方程左边。
  - 列视角：A 的列按 `x` 的系数组合成 `b`。
- 第一个结构判断：如果 A 的列向量不能覆盖整个目标空间，那么某些 `b` 一定无解；如果列向量之间有冗余，则会出现不唯一性。
- 本讲要会：
  - 在 row / column / matrix 三种视角间切换。
  - 解释“解是交点”与“解是列坐标”其实是同一件事。

## Session 1.2 An overview of key ideas

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.13sum.pdf|summary]]

关联卡片：[[Linear Algebra-hub|线性代数 Hub]]、[[Vector Space]]、[[Matrix Rank]]

- 这一讲是整门课的路线图：消元告诉我们如何解方程，子空间解释为什么有解或无解，[[Orthogonality|正交]] 与 [[Orthogonal Projection|投影]] 处理无解情形，[[Determinant|行列式]] 和 [[Eigenvalues|特征值]] 刻画方阵的深层结构，[[Singular Value Decomposition|SVD]] 则把这些工具统一起来。
- 课程最早从向量开始：向量可以加、减、数乘，因此会出现 linear combination。
- 一旦把若干向量放到矩阵的列里，`Ax` 就自然变成“列向量的线性组合”。
- 这讲通过两个例子对比“可逆”和“不可逆”：
  - 差分矩阵 A 的列独立，所以 `Ax=b` 对每个 `b` 有唯一解。
  - 循环差分矩阵 C 的列相关，所以 `Cx=0` 有非零解，`Cx=b` 还要求 `b_1+b_2+b_3=0`。
- 从这里开始，subspace 的语言正式出现：
  - 所有列组合组成 [[Column Space]]。
  - 所有齐次解组成 [[Null Space]]。
  - n 个独立向量张成整个 `R^n` 时，它们就构成 basis。
- “矩形矩阵”不是边角料，而是主角。四个基本子空间的维数关系要同时理解方阵与非方阵。
- 一条贯穿全课的主线是：rank 决定了空间被压缩成什么样，nullity 记录了丢失了多少自由度。
- 这一讲的作用不是记公式，而是建立地图：后面的概念都服务于“矩阵如何作用于空间”。
- 本讲要会：
  - 说清楚 basis、subspace、invertible matrix 之间的关系。
  - 解释为什么“列向量独立”会对应“系统唯一解”。

## Session 1.3 Elimination with matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.2sol.pdf|solution]]

关联卡片：[[Matrix Rank]]、[[LU Decomposition]]

- 高斯消元的目标是把矩阵变成上三角或更进一步的 row echelon form，从而暴露主元位置。
- 典型计算：
$$
\begin{bmatrix}
1&2&1\\
3&8&1\\
0&4&1
\end{bmatrix}
\to
\begin{bmatrix}
1&2&1\\
0&2&-2\\
0&0&5
\end{bmatrix}.
$$
  三个主元分别是 `1,2,5`。
- 增广矩阵把 `b` 一起带着消元，因此 `Ax=b` 被变成更容易解的 `Ux=c`。
- 上三角系统用 back substitution 反推：先求最后一个变量，再逐步往回。
- 主元（pivot）对应真正独立的信息；没有主元的列最终会变成自由变量来源。
- 如果某个主元位置是 0，就必须换行；如果该列下面也全是 0，则矩阵不可逆，系统不可能对所有 `b` 唯一可解。
- 每一步消元都可以写成左乘一个 elimination matrix，例如 `E_{21}` 表示“用第 1 行消掉第 2 行的首项”。
- elimination matrix 是可逆的，因此消元过程本质上是在做一串可逆变换；这就是“为什么消元不改变解集”的根本理由。
- 置换矩阵负责换行，左乘换行、右乘换列。
- 本讲要会：
  - 通过消元读出 pivots。
  - 写出消元矩阵与置换矩阵的作用。
  - 解释“可逆”为什么等价于“每一步都能找到非零主元”。

## Session 1.4 Multiplication and inverse matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.3sol.pdf|solution]]

关联卡片：[[Matrix Inverse]]、[[Singular Matrix]]

- 矩阵乘法至少要会四种理解：行乘列、列组合、行组合、分块乘法。
- `Ax` 是 A 的列按 `x` 的系数组合；`y^TA` 是 A 的行按 `y` 的系数组合。
- 两个矩阵相乘 `AB` 的最本质含义是“先做 B，再做 A”的复合变换。
- [[Matrix Inverse]] 的含义不是公式，而是“撤销 A 的作用”；若 `A^{-1}A=I`，则 A 的每个输出都能唯一追溯回输入。
- 方阵 A 可逆的常见等价条件：
  - `Ax=b` 对每个 `b` 都有唯一解。
  - `Ax=0` 只有零解。
  - A 的列线性无关。
  - A 没有零主元。
- 若 A 不可逆，则它是 [[Singular Matrix]]；奇异的根本含义是空间被压扁，某些方向信息被丢掉。
- `AB` 可逆时 `(AB)^{-1}=B^{-1}A^{-1}`，顺序反过来，因为复合变换要逆向拆解。
- 对做题而言，本讲要会：
  - 用“复合变换”解释乘法顺序。
  - 用“可逆性等价条件”在题目中来回切换。

## Session 1.5 Factorization into A = LU

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.4sol.pdf|solution]]

关联卡片：[[LU Decomposition]]、[[Permutation Matrix]]

- 消元不仅能求解，还能把矩阵拆成结构化因子：`A=LU`。
- `U` 是消元后的上三角矩阵，`L` 记录消元乘子，所以它是单位下三角矩阵。
- 例如从
$$
A=\begin{bmatrix}
1&2&1\\
3&8&1\\
0&4&1
\end{bmatrix}
$$
  消元得到 `U`，而乘子 `3` 和 `2` 则记录在 `L` 中。
- 一旦需要换行，就不再是单纯 `A=LU`，而要写成 `PA=LU`，其中 `P` 是 [[Permutation Matrix]]。
- 从求解角度看，`Ax=b` 先变成 `LUx=b`，于是先解 `Ly=b`，再解 `Ux=y`。
- 从结构角度看，LU 是把“算法过程”固化成“矩阵分解”。
- 更进一步还可以分成 `A=LDU`，把对角主元单独提出来。
- LU 的真正价值在于“分解一次，反复求解”：同一个 A 对多个不同右端 `b` 时效率极高。
- 本讲要会：
  - 知道 L、U 各自记录什么信息。
  - 在需要换行时主动写出 `PA=LU`。

## Session 1.6 Transposes, permutations, vector spaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.5sol.pdf|solution]]

关联卡片：[[Permutation Matrix]]、[[Vector Space]]、[[Subspace]]

- 转置 `A^T` 交换了行和列，因此 [[Row Space|row space]] 与 [[Column Space|column space]] 会在转置下互换。
- 对任意矩阵，`(AB)^T=B^TA^T`；顺序翻转是后续对称矩阵、正交矩阵的重要基础。
- 置换矩阵本质上是“重新排列基向量”的单位矩阵，左乘做换行，右乘做换列。
- [[Vector Space]] 的本质是对加法与数乘封闭；[[Subspace]] 则是其中仍保留全部线性结构的子集。
- 在 `R^2` 和 `R^3` 里，典型子空间只有：原点、过原点的直线、过原点的平面、整个空间。
- “必须经过原点”是判断一个集合是不是线性子空间的第一个快速筛选标准。
- 这一步把“求解方程”正式提升到“研究空间中的线性结构”。
- 本讲要会：
  - 判断一个集合是不是 subspace。
  - 熟练切换 row/column 与 transpose 的关系。

## Session 1.7 Column space and nullspace

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.6sol.pdf|solution]]

关联卡片：[[Column Space]]、[[Null Space]]、[[Subspace]]

- [[Column Space]] 由 A 的列向量张成，决定 `Ax=b` 可能落到哪里。
- [[Null Space]] 由所有满足 `Ax=0` 的向量组成，决定解的不唯一性来自哪里。
- 如果某一列是其他列的线性组合，那么删掉它不会改变 column space，只会暴露出变量间冗余。
- `b` 有解的必要充分条件是 `b∈Col(A)`；齐次解空间 `N(A)` 永远经过原点。
- 从几何上看，column space 生活在目标空间里，nullspace 生活在输入空间里；这两个空间不是同一个空间里的对象。
- 这一讲的核心不是记定义，而是熟悉这两个空间分别回答什么问题：
  - column space：哪些右端项能被 A 打出来。
  - nullspace：哪些输入会被 A 压成 0。
- 本讲要会：
  - 从方程组问题里识别“这是在问 column space 还是 nullspace”。
  - 解释“dependent columns 为什么会带来不唯一性”。

## Session 1.8 Solving Ax = 0: pivot variables, special solutions

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.7sol.pdf|solution]]

关联卡片：[[Null Space]]、[[Matrix Rank]]

- 解齐次方程的关键是区分主变量和自由变量。
- 当 rref 写成
$$
R=\begin{bmatrix}I&F\\0&0\end{bmatrix},
$$
  就说明前面是 pivot columns，后面是 free columns。
- 每个自由变量都对应一个 special solution；把某个自由变量设成 1，其他自由变量设成 0，就能解出一个特解。
- 所有这些 special solutions 线性组合起来，才是整个 [[Null Space]]。
- 如果 `rank(A)=r` 且 A 有 n 列，那么自由变量个数是 `n-r`，这就是零空间的维数。
- 这里最常见的错有两个：
  - 把 U 的 pivot columns 当成 A 的 pivot columns。
  - 找到一个 special solution 就误以为完事。
- 本讲要会：
  - 从 rref 直接写出 nullspace 的基。
  - 用 `n-r` 解释 nullspace 的维数。

## Session 1.9 Solving Ax = b: row reduced form R

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.8sol.pdf|solution]]

关联卡片：[[Column Space]]、[[Null Space]]、[[Linear system solution structure]]

- 非齐次方程的通解结构是：`x = x_p + x_n`，其中 `x_p` 是一个特解，`x_n ∈ N(A)`。
- 若 A 的某些行有线性关系，那么 `b` 也必须满足同样的兼容条件；否则系统无解。
- 检查 solvability 的标准方法是看增广矩阵消元后，是否出现形如
$$
[0\ 0\ \cdots\ 0\ |\ \ast]
$$
  的矛盾行。
- 找 particular solution 的一个省力方法是：先把所有自由变量设为 0，再解出主变量。
- 完整解集是一个平移后的子空间：nullspace 是经过原点的平面/直线，而 `x_p+N(A)` 是平移后的同维几何对象。
- rank 决定了解的四种典型情形：
  - `r=m=n`：每个 `b` 唯一解。
  - `r=n<m`：有些 `b` 无解，有解时唯一。
  - `r=m<n`：每个 `b` 都有解，但通常无穷多解。
  - `r<m,n`：要么无解，要么无穷多解。
- 本讲要会：
  - 看见 `Ax=b` 就想到“particular + nullspace”。
  - 用 rank 和 rref 分类讨论解的个数。

## Session 1.10 Independence, basis, and dimension

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.9prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.9sol.pdf|solution]]

关联卡片：[[Linear Independence]]、[[Vector Space]]、[[Matrix Rank]]

- 线性无关的意义是：没有冗余方向。
- 如果一组向量能张成整个空间且又线性无关，那么它们就是 basis。
- basis 的个数不唯一，但所有 basis 的向量个数相同，这个共同个数就是 dimension。
- 在矩阵语言里：
  - pivot columns 给出 column space 的基。
  - row reduced form 的非零行给出 row space 的基。
- “independent columns” 这个判断会反复出现：它既决定是否可逆，也决定 `A^TA` 是否可逆，还决定 least squares 是否唯一。
- 线性无关、张成、基、维数这四个概念是整门课后续所有“坐标表示”的基础。
- 本讲要会：
  - 在“向量组 / 子空间 / 矩阵列”三种语言里说同一件事。
  - 解释为什么“最小生成组”等价于 basis。

## Session 1.11 The four fundamental subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.10prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.10sol.pdf|solution]]

关联卡片：[[Column Space]]、[[Null Space]]、[[Matrix Rank]]

- 对任意 `m×n` 矩阵 A，都要同时盯住四个空间：`C(A)`、`N(A)`、`C(A^T)`、`N(A^T)`。
- 维数关系是第一层：
  - `dim C(A)=r`
  - `dim N(A)=n-r`
  - `dim C(A^T)=r`
  - `dim N(A^T)=m-r`
- 这四个数字加起来正好对应输入维数 `n` 与输出维数 `m`。
- column space 的基来自原矩阵 A 的 pivot columns。
- nullspace 的基来自 special solutions。
- row space 的基可以直接从 rref 的非零行读取，因为 row operations 不改变 row space。
- left nullspace 的基可以从 `EA=R` 中 E 的底部零行对应的行向量读出。
- 结构关系是第二层：后续正交单元会告诉我们 `[[Row Space|row space]] ⟂ [[Null Space|nullspace]]`，`[[Column Space|column space]] ⟂ [[Left Nullspace|left nullspace]]`。
- 线性代数的很多题，本质上都可以还原成“你到底在问哪一个基本子空间”。
- 本讲要会：
  - 记住四个空间分别住在哪个 ambient space 里。
  - 用 rank 一次性写出它们的维数。

## Session 1.12 Matrix spaces; rank 1; small world graphs

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.11sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.11prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.11sol.pdf|solution]]

关联卡片：[[Matrix Rank]]

- 向量空间不一定长得像 `R^n`，矩阵本身也能构成向量空间。
- 例如所有 `3×3` 矩阵组成一个向量空间，其中：
  - 所有上三角矩阵是一个子空间。
  - 所有对称矩阵是一个子空间。
  - 所有对角矩阵也是一个子空间，且是前两者的交。
- rank-1 矩阵通常可写成 `uv^T`，它把所有输入都压到一条直线方向上，是理解低秩结构的最好起点。
- rank-1 的意义是：column space 与 row space 都只有一维。
- small world graph 的出现提醒我们：线性代数不是纯代数工具，它可以直接编码网络结构。
- 一旦开始把“矩阵当成向量”，很多分解与逼近问题就会变得自然。
- 本讲要会：
  - 把“矩阵集合”识别成向量空间或子空间。
  - 解释 rank-1 为什么是最简单的非零矩阵结构。

## Session 1.13 Graphs, networks, incidence matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.12sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.12prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.12sol.pdf|solution]]

关联卡片：[[Incidence Matrix]]、[[Column Space]]、[[Null Space]]

- 网络问题把矩阵的结构暴露得特别明显：节点、边、流量、约束都可以编码进 incidence matrix。
- [[Incidence Matrix]] 的列通常对应边，行对应节点，每列恰好包含一个 `+1` 和一个 `-1`。
- 这样做的好处是：
  - 列空间编码“允许的净流入/净流出模式”。
  - 零空间编码“沿环流动而不在节点堆积的流”。
  - 左零空间则编码网络的整体平衡约束。
- 图中的“环”会落到 [[Null Space|nullspace]]，图中的“连通性约束”会落到 [[Column Space|column space]] / [[Left Nullspace|left nullspace]] 的关系里。
- 物理系统中的平衡、守恒、网络流与图论结构都能被这一套语言统一表达。
- 这一讲把抽象子空间与真实系统连接起来，是后续 [[Markov Matrix|Markov]]、FFT、图像压缩等应用的预演。
- 本讲要会：
  - 看见 graph/network 题时，主动想到 incidence matrix。
  - 把“流”“环”“平衡条件”翻译成子空间语言。

## Session 1.14 Exam 1 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.14sum.pdf|summary]]

- 必会判断：`Ax=b` 是否可解、是否唯一、是否无穷多解。
- 必会操作：消元、找 pivot / free variables、写 special solutions、写一般解。
- 必会结构：column space、nullspace、basis、dimension、rank、四个基本子空间。
- 必会应用：把 network / incidence matrix 题还原成子空间问题。
- 本单元真正的压轴不是某个公式，而是以下统一结构：
  - 消元给出 rank 与 pivots。
  - pivots 决定 basis 与 dimension。
  - basis 与 dimension 决定四个基本子空间。
  - 四个基本子空间决定解结构。

## 本单元复习清单

- 我能在 row picture、column picture、matrix picture 三种视角之间切换。
- 我能从 rref 直接读出主变量、自由变量、nullspace 的基与通解。
- 我能解释“有解”为什么等价于 `b∈Col(A)`。
- 我能用 rank 和四个基本子空间组织所有题目，而不是只会机械计算。
- 我能把每道题都压缩成一句话：它究竟在问哪一个子空间、哪一个维数、哪一种可解性。
