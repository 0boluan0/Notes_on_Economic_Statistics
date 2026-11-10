---
aliases:
  - Jordan canonical form
  - Jordan normal form
  - Jordan 标准形
  - Jordan 形
student_os: knowledge-atom
atom_id: LA-EIG-033
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
requires:
  - "[[Jordan块]]"
  - "[[相似矩阵]]"
related:
  - "[[Jordan链]]"
  - "[[Jordan标准形的域条件]]"
  - "[[Jordan形唯一性]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Jordan canonical form
<!-- bilingual-en:end -->

*Jordan 标准形*

<!-- bilingual-en:start -->
**Jordan canonical form** represents a square matrix in coordinates arranged into chains. Within each chain, the action has the simple pattern of a [[Jordan块|Jordan block]]; different chains occupy separate blocks. It extends diagonalization to matrices that do not have enough independent ordinary eigenvectors. The map stays the same, while the coordinate basis changes.
<!-- bilingual-en:end -->

**Jordan 标准形**把方阵写在按链组织的坐标中。每条链内部都采用一个 [[Jordan块]]的简单模式，不同链占据不同块。即使普通特征向量不够组成一组基，这种形式仍可能接替对角化。改变的是坐标基，映射本身没有改变。

<!-- bilingual-en:start -->
## A change of coordinates exposes a chain
<!-- bilingual-en:end -->

*换一组坐标，链结构就显现出来*

<!-- bilingual-en:start -->
Take
<!-- bilingual-en:end -->

取

$$A=\begin{pmatrix}2&1\\-1&4\end{pmatrix},\qquad
v_1=\begin{pmatrix}1\\1\end{pmatrix},\qquad
v_2=\begin{pmatrix}0\\1\end{pmatrix}.$$

<!-- bilingual-en:start -->
Direct multiplication gives $Av_1=3v_1$ and $Av_2=v_1+3v_2$. The first vector is an [[特征对|eigenvector]]: its image is three times itself. The second also contributes one copy of $v_1$. Thus $(A-3I_2)v_2=v_1$ and $(A-3I_2)v_1=0$, where $I_2$ is the [[单位矩阵|identity matrix]]. This is a length-two [[Jordan链|Jordan chain]].
<!-- bilingual-en:end -->

直接相乘可得 $Av_1=3v_1$、$Av_2=v_1+3v_2$。第一个向量是[[特征对|特征向量]]，像等于自身三倍；第二个向量还多贡献了一份 $v_1$。因此 $(A-3I_2)v_2=v_1$、$(A-3I_2)v_1=0$，其中 $I_2$ 为[[单位矩阵]]。这是一条长度为二的 [[Jordan链]]。

<!-- bilingual-en:start -->
The vectors $v_1,v_2$ form a [[基|basis]]: any vector $x$ can be written uniquely as $x=c_1v_1+c_2v_2$. Put them into the columns of $S$ and collect the new coordinates into $c=(c_1,c_2)^T$, where $T$ means [[矩阵转置|transpose]] and writes the list as a column. Then
<!-- bilingual-en:end -->

$v_1,v_2$ 构成一组[[基]]：任意向量 $x$ 都能唯一写成 $x=c_1v_1+c_2v_2$。把它们放进 $S$ 的各列，并把新坐标收集成 $c=(c_1,c_2)^T$；这里 $T$ 表示[[矩阵转置|转置]]，将坐标列表写成列。于是有

$$S=\begin{pmatrix}1&0\\1&1\end{pmatrix},\qquad
x=Sc=\begin{pmatrix}c_1\\c_1+c_2\end{pmatrix},\qquad
S^{-1}=\begin{pmatrix}1&0\\-1&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
The [[逆矩阵|inverse]] $S^{-1}$ converts ordinary coordinates back to the chain coordinates. Since
<!-- bilingual-en:end -->

[[逆矩阵]] $S^{-1}$ 把普通坐标转回链坐标。因为

$$A(c_1v_1+c_2v_2)=(3c_1+c_2)v_1+3c_2v_2,$$

<!-- bilingual-en:start -->
the new-coordinate input $(c_1,c_2)^T$ produces the new-coordinate output $(3c_1+c_2,3c_2)^T$. Its matrix is therefore
<!-- bilingual-en:end -->

新坐标输入 $(c_1,c_2)^T$ 产生的新坐标输出为 $(3c_1+c_2,3c_2)^T$。因此新坐标中的矩阵是

$$J=S^{-1}AS=\begin{pmatrix}3&1\\0&3\end{pmatrix}=J_2(3).$$

<!-- bilingual-en:start -->
Read $S^{-1}AS$ from right to left: convert the input using $S$, apply $A$, then convert the output using $S^{-1}$. This is a [[相似矩阵|similarity transformation]], using the same new basis for input and output. It is not merely row reduction of $A$.
<!-- bilingual-en:end -->

从右向左读 $S^{-1}AS$：先用 $S$ 转换输入坐标，施加 $A$，再用 $S^{-1}$ 转换输出坐标。这是[[相似矩阵|相似变换]]，输入输出采用同一组新基，并非只对 $A$ 做行化简。

> [!important] The whole matrix consists of separate blocks · 整个矩阵由分开的块组成
> <!-- bilingual-en:start -->
> Let $A$ be an $n\times n$ matrix over a chosen field, with $n\ge1$. A Jordan canonical form of $A$ is a matrix $J$ for which an invertible matrix $S$ over the same field satisfies
> <!-- bilingual-en:end -->
>
> 设 $A$ 是所选数域上的 $n\times n$ 矩阵，$n\ge1$。若存在同一数域上的可逆矩阵 $S$，使下式成立，就称 $J$ 是 $A$ 的一个 Jordan 标准形：
>
> $$S^{-1}AS=J=\operatorname{diag}\bigl(J_{r_1}(\lambda_1),\ldots,J_{r_q}(\lambda_q)\bigr),\qquad
> r_1+\cdots+r_q=n.$$
>
> <!-- bilingual-en:start -->
> The positive integer $q$ counts blocks; each positive integer $r_i$ is a block size, and $\lambda_i$ is its eigenvalue. Here $\operatorname{diag}$ puts entire square blocks along the diagonal and zero blocks everywhere else. The $\lambda_i$ need not be different. Equivalently, $A=SJS^{-1}$ or $AS=SJ$.
> <!-- bilingual-en:end -->
>
> 正整数 $q$ 是块的个数，每个正整数 $r_i$ 是相应块的尺寸，$\lambda_i$ 是它的特征值。这里 $\operatorname{diag}$ 表示沿主对角线摆放整个方块，其余块全为零。不同块的 $\lambda_i$ 不必不同。等价地，也可以写成 $A=SJS^{-1}$ 或 $AS=SJ$。

<!-- bilingual-en:start -->
## The columns of S explain the blocks
<!-- bilingual-en:end -->

*从 S 的列读出各个块*

<!-- bilingual-en:start -->
Arrange the columns of $S$ chain by chain. If one chain is $v_1,\ldots,v_r$ with eigenvalue $\lambda$, its columns satisfy $Av_1=\lambda v_1$ and $Av_{j+1}=\lambda v_{j+1}+v_j$. These are precisely the columns of the identity $AS=SJ$ for that block. The chain's [[张成|span]], meaning all its linear combinations, is an [[不变子空间|invariant subspace]]: applying $A$ keeps the output in the same span. Different blocks do not exchange coordinates in this chosen basis. The chain bases need not be perpendicular.
<!-- bilingual-en:end -->

把 $S$ 的列按链分组。若一条属于 $\lambda$ 的链是 $v_1,\ldots,v_r$，相应列满足 $Av_1=\lambda v_1$、$Av_{j+1}=\lambda v_{j+1}+v_j$。这些恰好是 $AS=SJ$ 在该块对应各列上的等式。链的[[张成|张成空间]]包含其全部线性组合，并且是[[不变子空间]]：经过 $A$ 后仍留在同一张成空间中。在所选的基里，不同块之间不交换坐标；各链的基并不要求彼此垂直。

<!-- bilingual-en:start -->
One chain covers only its own span. To describe all of $A$, the columns of $S$ must together be a basis of the entire $n$-dimensional space, not merely a collection of individually valid chains. Each block supplies one independent ordinary eigenvector. All blocks have size one exactly when the matrix can be [[对角化|diagonalized]] over the chosen field. Repeated eigenvalues alone do not prevent this: a diagonal matrix may repeat the same value several times.
<!-- bilingual-en:end -->

一条链只覆盖自己的张成空间。要表示整个 $A$，$S$ 的全部列必须共同构成整个 $n$ 维空间的一组基，不能只收集若干条各自合法的链。每个块提供一个独立普通特征向量。当且仅当全部块尺寸都为一时，矩阵才在所选数域上[[对角化|可对角化]]。重复特征值本身不阻止这一点：对角矩阵也可以多次出现同一个值。

> [!example] Same eigenvalues, different chain structure · 特征值相同，链结构仍可不同
> <!-- bilingual-en:start -->
> Compare
> $$J=\operatorname{diag}\!\left(\begin{pmatrix}3&1\\0&3\end{pmatrix},[1]\right)
> =\begin{pmatrix}3&1&0\\0&3&0\\0&0&1\end{pmatrix},\qquad
> D=\begin{pmatrix}3&0&0\\0&3&0\\0&0&1\end{pmatrix}.$$
> Both list the eigenvalues $3,3,1$. For eigenvalue 3, $J$ has one independent eigenvector, while $D$ has two. In $J$, those two coordinates form one chain of length two; in $D$, they form two chains of length one. The matrices are not similar. The precise statement that block sizes, not just eigenvalues, are fixed up to reordering is [[Jordan形唯一性|uniqueness of Jordan form]].
> <!-- bilingual-en:end -->
>
> 两者列出的特征值都是 $3,3,1$。对于特征值3，$J$ 只有一个独立特征向量，$D$ 却有两个。在 $J$ 中，这两个坐标组成一条长度二的链；在 $D$ 中，它们组成两条长度一的链。因此二者不相似。块尺寸而不只是特征值在忽略排列后保持唯一，精确结论见 [[Jordan形唯一性]]。

<!-- bilingual-en:start -->
## Existence depends on the field; the basis need not be unique
<!-- bilingual-en:end -->

*存在性取决于数域，基本身未必唯一*

<!-- bilingual-en:start -->
Every complex square matrix has a Jordan form. Over a specified field, ordinary Jordan form exists exactly when the [[特征多项式|characteristic polynomial]] $\det(tI-A)$ splits into linear factors over that field; here $t$ is a polynomial variable. Thus a real matrix with nonreal eigenvalues need not have an ordinary Jordan form using real entries. See [[Jordan标准形的域条件|the field condition]] for the theorem and the real rotation example. This condition concerns existence; the displayed formula above defines what the resulting form is.
<!-- bilingual-en:end -->

每个复方阵都有 Jordan 形。对指定数域，普通 Jordan 形存在，当且仅当[[特征多项式]] $\det(tI-A)$ 在该域上分裂为一次因子的乘积；这里 $t$ 是多项式变量。因此，含非实特征值的实矩阵未必能用全实数条目写出普通 Jordan 形。定理及实平面旋转的例子见 [[Jordan标准形的域条件]]。这个条件负责存在性；前面的显示公式定义的是得到的形式是什么。

<!-- bilingual-en:start -->
Even when the block data are fixed, $S$ is generally not unique. Multiplying every vector of one chain by the same nonzero scalar preserves its chain equations and the same block. Permuting whole chains reorders the blocks. Thus a canonical block structure does not mean a unique coordinate basis. Also, exact Jordan structure can change under tiny perturbations; [[Jordan、Schur与SVD用途边界|the comparison with Schur and SVD]] explains why symbolic classification and numerical computation may need different representations.
<!-- bilingual-en:end -->

即使块资料固定，$S$ 一般也不唯一。把同一条链的全部向量乘以同一个非零标量，链方程及相应块仍然不变；交换整条链的顺序，则会重排块。因此，规范的块结构不意味着唯一的坐标基。另外，微小扰动也可能改变精确 Jordan 结构；[[Jordan、Schur与SVD用途边界]]解释了为什么符号分类与数值计算可能需要不同表示。

> [!question]- Do three blocks mean three different eigenvalues? · 三个块是否代表三个不同特征值？
> <!-- bilingual-en:start -->
> No. For instance, $\operatorname{diag}(J_2(3),J_1(3),J_1(1))$ has three blocks but only two distinct eigenvalues. Block count records the number of independent eigenvectors in a full Jordan basis, not the number of distinct eigenvalues.
> <!-- bilingual-en:end -->
>
> 不一定。例如 $\operatorname{diag}(J_2(3),J_1(3),J_1(1))$ 有三个块，却只有两个不同特征值。块数记录完整 Jordan 基中的独立普通特征向量个数，不是不同特征值的个数。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 4th ed., §8C, definition 8.44 and theorems 8.45–8.46, printed pp.322–324 / PDF pp.336–338](https://linear.axler.net/LADR4e.pdf#page=336): Jordan bases, block-diagonal chain representation and existence over the complex field. The displayed change-of-basis example is verified directly.
- [University of Colorado, Math 6140, §12.3, pp.1–2](https://math.colorado.edu/~mayr/teaching/math6140spring25/PID3.pdf): the characteristic-polynomial splitting condition, uniqueness up to permutation of blocks and the diagonalizable case. The necessity of splitting also follows by taking the characteristic polynomial of the displayed blocks.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=3|MIT 18.06SC Session 3.4 summary, pp.3–4]]: block sizes distinguish similarity classes even when eigenvalues agree.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf#page=15|MIT Lecture 28 transcript, pp.15,17–21]]: the purpose of Jordan classification and its sensitivity in numerical work. Statements about the “number of eigenvectors” are read as the number of independent directions.
<!-- bilingual-en:end -->

- Axler 第8C节定义8.44、定理8.45–8.46，印刷322–324页／PDF336–338页：核对 Jordan 基、按链的块对角表示及复数域存在性；显示的换基例子直接核算。
- 科罗拉多大学 Math6140 第12.3节第1–2页：核对特征多项式分裂条件、块排列之外的唯一性及可对角化情形；分裂的必要性也由显示的块特征多项式直接推出。
- MIT Session3.4 摘要第3–4页：核对即使特征值相同，块尺寸仍能区分相似类。
- MIT Lecture28 讲稿第15、17–21页：核对 Jordan 分类的目的与数值敏感性；“特征向量个数”按独立方向数理解。
