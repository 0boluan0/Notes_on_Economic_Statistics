---
aliases:
- 奇异值分解存在定理
- Existence of the singular value decomposition
student_os: knowledge-atom
atom_id: LA-SVD-020
atom_set: singular-value-decomposition-low-rank
atom_type: theorem
status: source-checked
requires:
- '[[奇异值分解]]'
- '[[Hermitian 谱定理]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
leads_to:
- '[[SVD手算流程]]'
- '[[奇异向量]]'
related:
- '[[四个基本子空间]]'
---

<!-- bilingual-en:start -->
# Existence of the singular value decomposition
<!-- bilingual-en:end -->

*SVD 存在性*

<!-- bilingual-en:start -->
The [[奇异值分解|SVD]] describes a matrix through perpendicular input/output directions and nonnegative stretches. The existence theorem guarantees this description for **every finite real or complex matrix**. The matrix need not be square, invertible or diagonalizable by its own [[特征对|eigenvectors]], directions mapped to scalar multiples of themselves. We obtain the directions from a better-behaved square companion.
<!-- bilingual-en:end -->

[[奇异值分解]]用垂直的输入输出方向与非负伸缩描述矩阵。存在定理保证**每个有限维实或复矩阵**都能得到这种描述，无须方形、可逆，或能由自己的[[特征对|特征向量]]对角化；特征向量指被映到自身某个标量倍的非零方向。方向来自一个性质更好的方阵伙伴。

> [!example] No eigenvector basis is needed / 不需要原矩阵有完整特征向量基
> <!-- bilingual-en:start -->
> The matrix $A=\begin{bmatrix}0&1\\0&0\end{bmatrix}$ sends $(x_1,x_2)^T$ to $(x_2,0)^T$. Its only eigenvalue is zero; all its eigenvectors lie on the first coordinate axis, so there is no eigenvector basis. Nevertheless,
> $$A=I_2\begin{bmatrix}1&0\\0&0\end{bmatrix}
> \begin{bmatrix}0&1\\1&0\end{bmatrix}^{T}.$$
> This SVD swaps the input coordinates, keeps the first of the swapped coordinates, and leaves the output basis unchanged. $I_2$ is the [[单位矩阵|identity]] and $T$ is [[矩阵转置|transpose]].
> <!-- bilingual-en:end -->
>
> 上述 $A$ 把 $(x_1,x_2)^T$ 送到 $(x_2,0)^T$。它只有零特征值，特征向量都在第一坐标轴上，没有完整特征向量基；显示的三个因子却给出 SVD：先交换输入坐标，再保留交换后的第一坐标，输出基不变。$I_2$ 是[[单位矩阵]]，$T$ 是[[矩阵转置|转置]]。

<!-- bilingual-en:start -->
For $A\in\mathbb F^{m\times n}$ with $\mathbb F=\mathbb R$ or $\mathbb C$ and $m,n\ge1$, there exist square unitary factors and a nonnegative rectangular diagonal factor with
$$A=U\Sigma V^*,\qquad U:m\times m,\quad\Sigma:m\times n,\quad V:n\times n.$$
In the real case the factors can be real and orthogonal. The star denotes conjugate transpose: transpose and replace each $a+bi$ by $a-bi$, with $i^2=-1$. Unitary columns form an [[标准正交组|orthonormal basis]]: unit lengths and zero inner products between distinct columns.
<!-- bilingual-en:end -->

对 $A\in\mathbb F^{m\times n}$，$\mathbb F=\mathbb R$ 或 $\mathbb C$，$m,n\ge1$，总存在上述尺寸的方形酉因子及非负长方对角因子，使 $A=U\Sigma V^*$。实数时可取实正交因子。星号表示共轭转置，转置后把 $a+bi$ 换成 $a-bi$，其中 $i^2=-1$。酉矩阵的列组成[[标准正交组|标准正交基]]，各自长为1，不同列内积为零。

<!-- bilingual-en:start -->
## Find a guaranteed input basis
<!-- bilingual-en:end -->

*先找到一定存在的输入基*

<!-- bilingual-en:start -->
Form the $n\times n$ [[Gram矩阵正定判据|Gram matrix]] $G=A^*A$. It is [[Hermitian 矩阵|Hermitian]] because $G^*=G$, and [[半正定矩阵|positive semidefinite]] because
$$x^*Gx=\|Ax\|_2^2\ge0.$$
Here $\|\cdot\|_2$ is [[欧氏范数|Euclidean length]]. The [[Hermitian 谱定理|spectral theorem]] supplies an orthonormal eigenbasis $v_1,\ldots,v_n$ and real eigenvalues $\lambda_1\ge\cdots\ge\lambda_n\ge0$, with $Gv_i=\lambda_i v_i$. This step remains valid even when $A$ has no eigenvector basis.
<!-- bilingual-en:end -->

构造 $n\times n$ 的 [[Gram矩阵正定判据|Gram 矩阵]] $G=A^*A$。它因 $G^*=G$ 而为 [[Hermitian 矩阵]]，又因显示的长度平方非负而[[半正定矩阵|半正定]]，$\|\cdot\|_2$ 是[[欧氏范数|欧氏长度]]。[[Hermitian 谱定理]]给出标准正交特征基 $v_1,\ldots,v_n$ 及非负实特征值 $\lambda_1\ge\cdots\ge\lambda_n\ge0$，满足 $Gv_i=\lambda_i v_i$。即使 $A$ 无完整特征向量基，这一步仍成立。

<!-- bilingual-en:start -->
If $\lambda_i=0$, then $\|Av_i\|_2^2=v_i^*Gv_i=0$, so $Av_i=0$. More generally, $N(G)=N(A)$, where the [[零空间|nullspace]] consists of inputs sent to zero. The [[秩零度定理|rank–nullity theorem]] shows that the number $r$ of positive eigenvalues, counted with multiplicity, equals $\operatorname{rank}(A)$. Thus $r\le\min(m,n)$, and the zero eigenvalues already supply all right nullspace directions.
<!-- bilingual-en:end -->

若 $\lambda_i=0$，则 $\|Av_i\|_2^2=v_i^*Gv_i=0$，所以 $Av_i=0$。一般地 $N(G)=N(A)$，[[零空间]]包含被映到零的输入。由[[秩零度定理]]，按重数计的正特征值个数 $r$ 等于 $\operatorname{rank}(A)$，所以 $r\le\min(m,n)$。零特征值已经给出全部右零空间方向。

<!-- bilingual-en:start -->
## Build the matching output directions
<!-- bilingual-en:end -->

*再构造配对输出方向*

<!-- bilingual-en:start -->
For $1\le i\le r$, define $\sigma_i=\sqrt{\lambda_i}>0$ and $u_i=Av_i/\sigma_i$. Only positive values are used as divisors. For two such indices,
$$u_i^*u_j=\frac{v_i^*A^*Av_j}{\sigma_i\sigma_j}
=\frac{\lambda_j}{\sigma_i\sigma_j}v_i^*v_j
=\begin{cases}1,&i=j,\\0,&i\ne j.\end{cases}$$
The normalized images are therefore perpendicular unit vectors. They span the [[列空间|column space]] $C(A)$: expand any input in the $v_i$ basis, and only its first $r$ components survive under $A$.
<!-- bilingual-en:end -->

对 $1\le i\le r$，定义 $\sigma_i=\sqrt{\lambda_i}>0$、$u_i=Av_i/\sigma_i$，只以正值作分母。上面的内积计算说明归一化后的像确实彼此垂直且长度为1。它们张成[[列空间]] $C(A)$：任意输入按 $v_i$ 展开后，经过 $A$ 只剩前 $r$ 项。

<!-- bilingual-en:start -->
Complete $u_1,\ldots,u_r$ to an orthonormal basis of the output space. The extra $m-r$ directions belong to $C(A)^\perp=N(A^*)$, the [[左零空间|left nullspace]]. Put the full bases into $U=[u_1\ \cdots\ u_m]$ and $V=[v_1\ \cdots\ v_n]$. Fill the first $r$ diagonal positions of an $m\times n$ zero matrix $\Sigma$ with $\sigma_1,\ldots,\sigma_r$.
The column identities $Av_i=\sigma_i u_i$ for $i\le r$, and $Av_i=0$ otherwise, say exactly that $AV=U\Sigma$. Since $VV^*=I_n$, multiplying by $V^*$ proves $A=U\Sigma V^*$.
<!-- bilingual-en:end -->

把 $u_1,\ldots,u_r$ 补成输出空间的标准正交基，新增 $m-r$ 条方向属于 $C(A)^\perp=N(A^*)$，即[[左零空间]]。把完整基收进 $U=[u_1\ \cdots\ u_m]$、$V=[v_1\ \cdots\ v_n]$；在一个 $m\times n$ 零矩阵的前 $r$ 个对角位置填入 $\sigma_1,\ldots,\sigma_r$，得到 $\Sigma$。前 $r$ 列满足 $Av_i=\sigma_i u_i$，其余列满足 $Av_i=0$，合起来恰为 $AV=U\Sigma$。因为 $VV^*=I_n$，右乘 $V^*$ 就证明 $A=U\Sigma V^*$。

<!-- bilingual-en:start -->
If $A=0$, choose any two orthonormal bases and $\Sigma=0$; no division is needed. For a $5\times3$ matrix of rank 2, use two positive pairs, one extra right nullspace direction and three extra left nullspace directions. Neither the rectangular shape nor the missing rank obstructs the construction.
<!-- bilingual-en:end -->

若 $A=0$，两侧任取标准正交基并令 $\Sigma=0$，无须相除。对秩2的 $5\times3$ 矩阵，使用两对正方向，再补一条右零空间方向、三条左零空间方向。长方形尺寸与缺失的秩都不妨碍构造。

<!-- bilingual-en:start -->
Existence does not make every returned direction unique: repeated positive values allow coordinated basis changes, and [[SVD零块补基不唯一|nullspace completions]] have their own freedom. Nor does the proof prescribe a stable numerical algorithm. The [[SVD手算流程|hand calculation]] follows the construction for small examples; numerical routines can avoid explicitly forming the Gram matrix.
<!-- bilingual-en:end -->

存在性不保证返回的每条方向唯一：重复正值允许两侧协调换基，[[SVD零块补基不唯一|零空间补基]]还有自己的自由。证明也不是稳定算法的处方。[[SVD手算流程]]沿构造处理小例子；数值程序可以避免显式形成 Gram 矩阵。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 7.64, 7.68 and 7.70–7.74, PDF pp.284–288](https://linear.axler.net/LADR4e.pdf#page=284): Gram positivity/nullspace, positive-pair construction, orthonormality and full-basis completion.
- [LAPACK, SVD](https://netlib.org/lapack/lug/node32.html): full real/complex dimensions and the matrix convention with min(m,n) diagonal positions.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT18.06SC Session3.5]], PDF pp.2–4: local Gram calculation and rank-deficient example. The non-diagonalizable example is checked directly.
<!-- bilingual-en:end -->

- Axler 第7.64、7.68、7.70–7.74条核对 Gram 正性与零空间、正方向构造、标准正交性及补基。
- LAPACK 核对完整实复尺寸及 min(m,n) 个对角位置口径。
- MIT Session3.5 第2–4页核对课程 Gram 计算与秩亏例；不可对角化例子直接核验。
