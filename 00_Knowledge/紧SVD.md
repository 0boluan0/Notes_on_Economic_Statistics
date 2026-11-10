---
aliases:
- Compact singular value decomposition
- Rank-sized SVD
student_os: knowledge-atom
atom_id: LA-SVD-013
atom_set: singular-value-decomposition-low-rank
atom_type: definition
status: source-checked
requires:
- '[[奇异值分解]]'
- '[[奇异值与秩]]'
- '[[奇异向量]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
leads_to:
- '[[SVD秩一展开]]'
- '[[截断SVD]]'
related:
- '[[四个基本子空间]]'
- '[[SVD与四个基本子空间]]'
- '[[SVD零块补基不唯一]]'
---

<!-- bilingual-en:start -->
# Compact singular value decomposition
<!-- bilingual-en:end -->

*紧 SVD*

<!-- bilingual-en:start -->
A full [[奇异值分解|SVD]] carries complete coordinate bases, including directions that contribute nothing to the output. The **compact SVD** keeps only positive stretches and their paired directions. It still reconstructs the original matrix exactly: it removes zero contributions. Here “compact” always means **rank-sized**.
<!-- bilingual-en:end -->

完整[[奇异值分解]]携带完整坐标基，包括不贡献输出的方向。**紧 SVD**只保留正伸缩及其配对方向，仍精确重构原矩阵，因为删除的贡献为零。这里固定将“紧”用于**按秩取尺寸**的分解。

> [!example] One direction is enough / 一条方向就够了
> <!-- bilingual-en:start -->
> Let
> $$A=\begin{bmatrix}3&0\\4&0\\0&0\end{bmatrix},\quad
> u_1=\frac15\begin{bmatrix}3\\4\\0\end{bmatrix},\quad
> v_1=\begin{bmatrix}1\\0\end{bmatrix},\quad \sigma_1=5.$$
> Then $A=u_1\,5\,v_1^T$ exactly: read the first input coordinate, stretch it by 5 and place it along $u_1$. The second input coordinate disappears. Both displayed vectors have length one. The compact factors have sizes $3\times1$, $1\times1$ and $2\times1$; the last is transposed in the product. $T$ denotes [[矩阵转置|transpose]].
> <!-- bilingual-en:end -->
>
> 上述 $A=u_1\,5\,v_1^T$ 完全精确：读出第一输入坐标，放大5倍，再沿 $u_1$ 放回输出空间；第二输入坐标消失。两向量长度均为1。紧因子尺寸为 $3\times1$、$1\times1$、$2\times1$，乘积中的最后因子须转置，$T$ 表示[[矩阵转置|转置]]。

<!-- bilingual-en:start -->
## Retain all positive directions
<!-- bilingual-en:end -->

*保留全部正奇异方向*

<!-- bilingual-en:start -->
Let $A$ be a real or complex $m\times n$ matrix with $m,n\ge1$ and [[矩阵秩|rank]] $r>0$. Rank counts surviving independent directions and equals the [[奇异值与秩|number of positive singular values]]. Its compact SVD is
$$A=U_r\Sigma_rV_r^*,\qquad
U_r\in\mathbb F^{m\times r},\quad
\Sigma_r=\operatorname{diag}(\sigma_1,\ldots,\sigma_r),\quad
V_r\in\mathbb F^{n\times r},\qquad
\sigma_1\ge\cdots\ge\sigma_r>0.$$
Here $\mathbb F$ is $\mathbb R$ or $\mathbb C$, and $\operatorname{diag}$ puts the listed values on the diagonal with zeros elsewhere. The star means conjugate transpose: transpose and change $a+bi$ to $a-bi$, where $i^2=-1$; for real matrices it is $T$.
<!-- bilingual-en:end -->

设 $A$ 为实或复 $m\times n$ 矩阵，$m,n\ge1$，[[矩阵秩|秩]]为 $r>0$。秩计数保留下来的独立方向，等于[[奇异值与秩|正奇异值个数]]。紧 SVD 如上；$\mathbb F$ 为实数域或复数域，$\operatorname{diag}$ 表示对角线上放所列值、其余为零。星号是共轭转置，即转置并把 $a+bi$ 换成 $a-bi$，其中 $i^2=-1$；实数时就是 $T$。

<!-- bilingual-en:start -->
Each column family is [[标准正交组|orthonormal]], so $U_r^*U_r=V_r^*V_r=I_r$, with $I_r$ the [[单位矩阵|identity]]. These factors are usually rectangular. One must not infer $U_rU_r^*=I_m$ or $V_rV_r^*=I_n$ from orthonormal columns.
<!-- bilingual-en:end -->

两侧列各自为[[标准正交组]]，因此 $U_r^*U_r=V_r^*V_r=I_r$，$I_r$ 是[[单位矩阵]]。因子通常为长方矩阵，不能由列标准正交反推出 $U_rU_r^*=I_m$ 或 $V_rV_r^*=I_n$。

<!-- bilingual-en:start -->
A full SVD expands as $A=\sum_{i=1}^{p}\sigma_i u_i v_i^*$ with $p=\min(m,n)$. Every term after $r$ is zero. Removing them leaves
$$A=\sum_{i=1}^{r}\sigma_i u_i v_i^*,$$
which is exactly the compact product. The retained $u_i$ span the [[列空间|attainable output space]] $C(A)$; the retained $v_i$ span $C(A^*)=N(A)^\perp$, the input directions perpendicular to the [[零空间|nullspace]]. Hence $U_rU_r^*$ and $V_rV_r^*$ [[标准正交基投影公式|project onto those spaces]], rather than usually being whole-space identities. The omitted bases describe the right and left nullspaces; see [[SVD与四个基本子空间|the four subspaces in SVD]].
<!-- bilingual-en:end -->

完整 SVD 展开为 $A=\sum_{i=1}^{p}\sigma_i u_i v_i^*$，$p=\min(m,n)$；第 $r$ 项后均为零。删除这些项，便得到显示的前 $r$ 项之和，恰好是紧乘积。保留的 $u_i$ 张成[[列空间|可达输出空间]] $C(A)$，$v_i$ 张成 $C(A^*)=N(A)^\perp$，即垂直于[[零空间]]的输入方向。因此 $U_rU_r^*$、$V_rV_r^*$ 是[[标准正交基投影公式|对应空间的投影]]，通常不是整个空间的单位矩阵。被省略的基描述左右零空间，见 [[SVD与四个基本子空间]]。

<!-- bilingual-en:start -->
## Compare rank-sized, economy-sized and truncated factors
<!-- bilingual-en:end -->

*比较按秩、按形状与截断*

<!-- bilingual-en:start -->
The terms “reduced” and “thin” vary across sources and software. A common economy-sized SVD retains $p=\min(m,n)$ columns on each side, even if some associated values are zero. LAPACK DGESDD with JOBZ='S', for example, returns this many left columns and rows of the transposed right factor. When $r<p$, that is larger than the compact SVD defined here. Check dimensions and retained values rather than relying only on a label.
<!-- bilingual-en:end -->

“reduced”和“thin”随资料与软件有不同用法。常见经济型 SVD 在两侧保留 $p=\min(m,n)$ 列，即使相关的值为零。例如 LAPACK DGESDD 取 JOBZ='S' 时，返回这么多左列及转置后的右行。$r<p$ 时，它比本卡定义的紧 SVD 更大。应检查尺寸与保留的值，不能只依赖名称。

> [!note] A dimension comparison / 对照尺寸
> <!-- bilingual-en:start -->
> For a $7\times4$ matrix of rank 2, full $U,\Sigma,V$ have sizes $7\times7$, $7\times4$, $4\times4$. Economy-sized factors have sizes $7\times4$, $4\times4$, $4\times4$. Compact factors have sizes $7\times2$, $2\times2$, $4\times2$. All reconstruct the same matrix exactly; the factor called $V$ is conjugate-transposed in the product.
> <!-- bilingual-en:end -->
>
> 秩2的 $7\times4$ 矩阵：完整 $U,\Sigma,V$ 为 $7\times7$、$7\times4$、$4\times4$；经济型为 $7\times4$、$4\times4$、$4\times4$；紧因子为 $7\times2$、$2\times2$、$4\times2$。三者均精确重构同一矩阵，最后因子 $V$ 在乘积中都取共轭转置。

<!-- bilingual-en:start -->
A [[截断SVD|truncated SVD]] that keeps only $k<r$ positive directions changes the matrix. Compact SVD retains every positive direction, however small its stretch. Treating a tiny positive value as zero is a [[数值秩|numerical-rank decision]]; the resulting compact decomposition is exact for a modified matrix, not necessarily for the original one.
<!-- bilingual-en:end -->

只保留 $k<r$ 条正方向的[[截断SVD]]会改变矩阵。紧 SVD 则保留每条正方向，无论伸缩多小。把很小的正值视为零，是[[数值秩]]决策；得到的紧分解精确对应修改后的矩阵，未必对应原矩阵。

<!-- bilingual-en:start -->
If $A=0$, then $r=0$. The compact convention can use empty factors of sizes $m\times0$, $0\times0$ and $n\times0$, whose product is the zero $m\times n$ matrix. Implementations can handle zero as a separate case. For $r>0$, the positive diagonal block is invertible, which supports the [[伪逆SVD公式|pseudoinverse formula]]. Compactness does not remove the basis freedom associated with repeated singular values.
<!-- bilingual-en:end -->

若 $A=0$，则 $r=0$；紧口径可用 $m\times0$、$0\times0$、$n\times0$ 的空因子，乘积为 $m\times n$ 零矩阵，程序也可单独处理零。$r>0$ 时，正对角块可逆，可用于[[伪逆SVD公式]]。紧表示并不会消除重复奇异值对应的换基自由。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 7.68 and 7.70–7.74, PDF pp.286–288](https://linear.axler.net/LADR4e.pdf#page=286): positive-value count, exact positive-pair expansion and optional basis completion. This card names that rank-sized matrix form “compact”.
- [LAPACK DGESDD original source, Purpose and JOBZ](https://netlib.org/lapack/lapack_routine/dgesdd.f): full versus first-min(m,n) output dimensions, including the transposed right factor.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT18.06SC Session3.5]], PDF pp.3–4: positive directions and nullspace completion. The rank-one and dimension examples are checked directly.
<!-- bilingual-en:end -->

- Axler 第7.68、7.70–7.74条核对正值个数、正方向精确展开与可选补基；本卡明确将按秩的矩阵形式称为紧 SVD。
- LAPACK DGESDD 原码 Purpose、JOBZ 段核对完整与前 min(m,n) 项输出尺寸，包括转置后的右因子。
- MIT Session3.5 第3–4页核对正方向与零空间补基；秩1和尺寸算例直接核验。
