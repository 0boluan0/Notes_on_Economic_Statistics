---
aliases:
- Jordan, Schur and SVD comparison
- Jordan、Schur 与 SVD 的用途边界
student_os: knowledge-atom
atom_id: LA-EIG-015
atom_set: eigenvalues-linear-dynamics
atom_type: distinction
status: source-checked
requires:
- '[[Jordan标准形]]'
- '[[Schur分解]]'
- '[[Schur分解存在性]]'
- '[[奇异值分解]]'
related:
- '[[Normal 矩阵谱定理]]'
- '[[Jordan标准形的域条件]]'
part_of:
- '[[特征值、对角化与线性动力系统.canvas]]'
---

<!-- bilingual-en:start -->
# Choosing between Jordan, Schur and SVD
<!-- bilingual-en:end -->

*Jordan、Schur 与 SVD 的用途边界*

<!-- bilingual-en:start -->
A matrix factorization is useful because it makes a particular question easier to answer. Jordan form describes exact generalized eigenvector chains, Schur form represents a square eigenproblem in perpendicular coordinates, and SVD describes directional stretching between input and output spaces. They simplify matrices under different constraints, so first decide what information the question needs.
<!-- bilingual-en:end -->

矩阵分解之所以有用，是因为它让某一类问题更容易回答。Jordan 形描述精确的广义特征向量链结构，Schur 形在垂直坐标中表示方阵特征问题，SVD 则描述输入到输出的方向伸缩。它们遵守不同的简化约束，所以要先明确问题需要什么信息。

<!-- bilingual-en:start -->
Two transformations organize the choice. A [[相似矩阵|similarity transformation]] $S^{-1}AS$ changes the basis of one space on both the input and output sides; it preserves eigenvalues and the behavior of repeated applications. A [[奇异值分解|singular value decomposition]] instead allows a separate orthonormal basis on each side, giving $U^*AV=\Sigma$. Here $*$ is [[矩阵转置|conjugate transpose]], and orthonormal means perpendicular unit basis vectors. The two-sided freedom also works for rectangular matrices.
<!-- bilingual-en:end -->

两种变换可以帮助选择。[[相似矩阵|相似变换]] $S^{-1}AS$ 在同一空间的输入输出两侧一起换基，保留特征值及反复作用的结构。[[奇异值分解|SVD]]则允许两侧分别使用标准正交基，得到 $U^*AV=\Sigma$。星号表示[[矩阵转置|共轭转置]]，标准正交基由彼此垂直的单位基向量组成；这种两侧自由也适用于长方矩阵。

<!-- bilingual-en:start -->
## Choose the structure needed by the question
<!-- bilingual-en:end -->

*按问题需要的结构选择*

<!-- bilingual-en:start -->
For a complex square matrix, [[Jordan标准形|Jordan form]] writes $A=SJS^{-1}$ with $S$ invertible. Each [[Jordan块|Jordan block]] records one eigenvalue and a [[Jordan链|chain of generalized eigenvectors]], allowing structure to be described even when there is no eigenvector basis. The eigenvalues and block sizes classify matrices up to similarity, apart from block ordering. This is an exact algebraic statement. Over the reals, the usual scalar Jordan form requires eigenvalues to lie in the field; see [[Jordan标准形的域条件|the field condition]]. A symbolic problem about exact block sizes is a natural setting for Jordan form.
<!-- bilingual-en:end -->

对复方阵，[[Jordan标准形]]写作 $A=SJS^{-1}$，其中 $S$ 可逆。每个 [[Jordan块]]记录一个特征值和一条[[Jordan链|广义特征向量链]]，即使没有完整特征向量基也能描述结构。特征值与块尺寸在忽略块顺序后，给出相似分类。这是精确代数结论；若在实数域工作，通常的标量 Jordan 形要求特征值属于所用的域，见[[Jordan标准形的域条件]]。询问精确块尺寸的符号问题，适合使用 Jordan 形。

<!-- bilingual-en:start -->
For a complex square matrix, [[Schur分解|Schur form]] writes $A=QTQ^*$, where $Q$ is unitary ($Q^*Q=I$) and $T$ is upper triangular. Its diagonal lists the eigenvalues. Every complex square matrix admits this form, even if it is not diagonalizable; see [[Schur分解存在性|Schur existence]]. For a real matrix one can keep real arithmetic using an orthogonal $Q$ and an upper **quasi-triangular** $T$, whose diagonal blocks have size 1 or 2. The $2\times2$ blocks represent nonreal conjugate eigenvalue pairs; their diagonal entries alone are not those eigenvalues.
<!-- bilingual-en:end -->

对复方阵，[[Schur分解]]写作 $A=QTQ^*$，其中 $Q$ 酉，即 $Q^*Q=I$，而 $T$ 为上三角矩阵，对角线上列出特征值。每个复方阵都有这种形式，即使不可对角化也成立，见[[Schur分解存在性]]。对实矩阵，可以保留实数运算：取正交 $Q$，令 $T$ 为上**准三角**矩阵，对角块尺寸为1或2。$2\times2$ 块表示非实共轭特征值对，不能仅把块的对角元素当成这两个特征值。

<!-- bilingual-en:start -->
Schur form is useful when computing eigenvalues and invariant subspaces numerically. An [[不变子空间|invariant subspace]] is a set of vectors that the map keeps inside the same subspace. In complex Schur form, the first $k$ columns of $Q$ span an invariant subspace because $T$ is upper triangular. For real Schur form, stop only at a diagonal-block boundary. The columns are not generally individual eigenvectors. LAPACK normally first reduces a general matrix to Hessenberg form and then computes its Schur form; perpendicular coordinates avoid reliance on an ill-conditioned eigenvector basis.
<!-- bilingual-en:end -->

数值计算特征值与不变子空间时，Schur 形很有用。[[不变子空间]]指其中向量经过映射后仍留在同一子空间的集合。复 Schur 形中，$T$ 的上三角结构使 $Q$ 的前 $k$ 列张成不变子空间；实 Schur 形必须在完整对角块的边界处停止。各列通常并非逐个特征向量。LAPACK 通常先把一般矩阵约化为 Hessenberg 形，再计算 Schur 形；垂直坐标避免依赖病态的特征向量基。

<!-- bilingual-en:start -->
For a real or complex $m\times n$ matrix, SVD writes $A=U\Sigma V^*$. Its [[奇异值|nonnegative singular values]] measure stretching, and its [[奇异向量|left and right directions]] describe where each input component goes. It supports rank, Euclidean conditioning, [[线性最小二乘|least squares]] and [[低秩近似|low-rank approximation]]. It exists for square and rectangular matrices but is generally not a similarity transformation. The diagonal entries of $\Sigma$ therefore do not generally give the eigenvalues of $A$; the precise comparison is [[SVD与特征分解|SVD versus eigendecomposition]].
<!-- bilingual-en:end -->

对实或复 $m\times n$ 矩阵，SVD 写作 $A=U\Sigma V^*$。非负的[[奇异值]]衡量伸缩，[[奇异向量|左右方向]]描述每个输入分量送往何处。它支持秩、欧氏条件性、[[线性最小二乘]]与[[低秩近似]]，对方阵和长方矩阵都存在，但一般不是相似变换。因此 $\Sigma$ 的对角元素通常不是 $A$ 的特征值，精确比较见[[SVD与特征分解]]。

> [!example] A tiny perturbation changes exact Jordan blocks / 微小扰动可以改变精确 Jordan 块
> <!-- bilingual-en:start -->
> Consider
> $$A_0=\begin{pmatrix}1&1\\0&1\end{pmatrix},\qquad
> A_\varepsilon=\begin{pmatrix}1&1\\0&1+\varepsilon\end{pmatrix}.$$
> $A_0$ has a single Jordan block of size 2. For every real $\varepsilon\ne0$, $A_\varepsilon$ has distinct eigenvalues $1$ and $1+\varepsilon$, so its Jordan form consists of two blocks of size 1. Eigenvectors can be chosen as $(1,0)^T$ and $(1,\varepsilon)^T$; these directions become almost parallel when $\varepsilon$ is small. Yet both matrices are already valid real Schur forms with $Q=I$.
> <!-- bilingual-en:end -->
>
> 考虑显示的两个矩阵。$A_0$ 有一个尺寸为2的 Jordan 块；任何非零实数 $\varepsilon$ 都使 $A_\varepsilon$ 有两个不同特征值 $1$ 与 $1+\varepsilon$，Jordan 形因此变为两个尺寸为1的块。可取特征向量 $(1,0)^T$ 与 $(1,\varepsilon)^T$，当 $\varepsilon$ 很小时，这两方向几乎平行。但这两个矩阵本身已经是有效实 Schur 形，直接取 $Q=I$ 即可。

<!-- bilingual-en:start -->
If entries come from rounded measurements, deciding whether $\varepsilon$ is exactly zero may not be justified by the data. An arbitrarily small perturbation can therefore change the requested Jordan block structure. Exact symbolic data can still justify a Jordan calculation. Schur reduction avoids asking for that discontinuous block decision, but it does **not** guarantee that every eigenvalue, eigenvector or chosen invariant subspace is insensitive to perturbations. Good coordinates and a well-conditioned underlying question are different requirements.
<!-- bilingual-en:end -->

若矩阵元素来自舍入后的测量，数据可能不足以支持“$\varepsilon$ 恰好等于零”的判断；任意微小扰动都可能改变所问的 Jordan 块结构。精确符号数据仍可支持 Jordan 计算。Schur 约化避免要求这种不连续的块判定，但**不保证**每个特征值、特征向量或选定不变子空间都对扰动不敏感。良好的坐标与问题本身良态，是不同要求。

> [!tip] Start from the requested output / 先看最终要回答什么
> <!-- bilingual-en:start -->
> For exact similarity classes or generalized eigenvector chains, use Jordan structure. For numerical eigenvalues and invariant subspaces of a general square matrix, use Schur-based methods. For stretch, rank, least-squares recovery or low-rank reconstruction, use SVD. Additional structure, such as real symmetry or complex normality, may allow a simpler [[Normal 矩阵谱定理|orthonormal eigendecomposition]].
> <!-- bilingual-en:end -->
>
> 若要精确相似分类或广义特征向量链，使用 Jordan 结构；若要一般方阵的数值特征值与不变子空间，使用基于 Schur 的方法；若要伸缩、秩、最小二乘恢复或低秩重构，使用 SVD。若还有实对称或复正规结构，可能进一步采用更简单的[[Normal 矩阵谱定理|标准正交特征分解]]。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf|MIT Lecture 28 transcript]], Jordan discussion: exact block classification and numerical sensitivity to repeated eigenvalues and rank. The displayed perturbation example is checked directly.
- [LAPACK Users’ Guide, Eigenvalues, Eigenvectors and Schur Factorization](https://netlib.org/lapack/lug/node50.html) and [Nonsymmetric Eigenproblems](https://netlib.org/lapack95/lug95/node34.html): real/complex Schur forms, Hessenberg-to-Schur route and invariant-subspace use. The real invariant-subspace statement is restricted here to whole diagonal blocks.
- [Axler, *Linear Algebra Done Right*, 4th ed., 8.44–8.46](https://linear.axler.net/LADR4e.pdf#page=336): Jordan bases and existence over the complex field.
- [LAPACK SVD](https://netlib.org/lapack/lug/node32.html): rectangular two-basis factorization and nonnegative stretches.
<!-- bilingual-en:end -->

- MIT Lecture28 的 Jordan 讨论核对精确块分类及重根、秩的数值敏感性；显示的扰动例子直接核算。
- LAPACK 的 Schur 与非对称特征问题说明核对实／复 Schur 形、Hessenberg 到 Schur 的路线及不变子空间用途。本卡把实数不变子空间的前列选择限定在完整对角块边界。
- Axler 8.44–8.46核对 Jordan 基及复数域上的存在性。
- LAPACK SVD 说明核对长方形两侧基分解与非负伸缩量。
