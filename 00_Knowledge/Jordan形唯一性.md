---
aliases:
  - Uniqueness of Jordan form
  - Jordan 形唯一性
  - Jordan canonical form uniqueness
student_os: knowledge-atom
atom_id: LA-EIG-034
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
  - "[[Jordan标准形]]"
  - "[[相似矩阵]]"
  - "[[Jordan标准形的域条件]]"
related:
  - "[[Jordan链]]"
  - "[[相似保持特征空间维数]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Uniqueness of Jordan form
<!-- bilingual-en:end -->

*Jordan 形的唯一性*

<!-- bilingual-en:start -->
A [[Jordan标准形|Jordan form]] breaks a matrix into chains, one [[Jordan块|Jordan block]] per chain. Different people may choose different chain vectors or list the chains in a different order. The uniqueness theorem says that they must still obtain the same eigenvalue attached to each block and the same collection of block sizes. These data describe the matrix itself, rather than a particular choice of coordinates.
<!-- bilingual-en:end -->

[[Jordan标准形]]把矩阵拆成若干条链，每条链对应一个 [[Jordan块]]。不同的人可能选择不同链向量，或按不同顺序排列各条链。唯一性定理说明：每个块对应的特征值，以及各特征值下的块尺寸集合，仍必须完全相同。这些资料属于矩阵本身，而不取决于某次坐标选择。

<!-- bilingual-en:start -->
## Two eigenvector directions do not determine the chain lengths
<!-- bilingual-en:end -->

*两个特征方向，还不能确定链有多长*

<!-- bilingual-en:start -->
Consider two four-coordinate matrices with only eigenvalue zero:
<!-- bilingual-en:end -->

考虑两个都只有零特征值的四阶矩阵：

$$A=\operatorname{diag}(J_3(0),J_1(0))
=\begin{pmatrix}0&1&0&0\\0&0&1&0\\0&0&0&0\\0&0&0&0\end{pmatrix},\qquad
B=\operatorname{diag}(J_2(0),J_2(0))
=\begin{pmatrix}0&1&0&0\\0&0&0&0\\0&0&0&1\\0&0&0&0\end{pmatrix}.$$

<!-- bilingual-en:start -->
For a column with coordinates $x_1,x_2,x_3,x_4$, $A$ outputs $(x_2,x_3,0,0)$ and $B$ outputs $(x_2,0,x_4,0)$. Both [[零空间|nullspaces]] have dimension two: each equation $Ax=0$ or $Bx=0$ leaves two free coordinates. Thus both matrices have two independent ordinary eigenvector directions. But applying the matrices twice distinguishes them: $A^2x=(x_3,0,0,0)$, whereas $B^2x=0$ for every $x$. The first squared map has a three-dimensional nullspace; the second has a four-dimensional one.
<!-- bilingual-en:end -->

对坐标为 $x_1,x_2,x_3,x_4$ 的列向量，$A$ 输出 $(x_2,x_3,0,0)$，$B$ 输出 $(x_2,0,x_4,0)$。两个[[零空间]]都为二维：$Ax=0$ 与 $Bx=0$ 各留下两个自由坐标。因此，两者都有两个独立普通特征方向。但再作用一次便能区分：$A^2x=(x_3,0,0,0)$，$B^2x$ 却对每个 $x$ 都为零。前者平方后的零空间为三维，后者则为四维。

<!-- bilingual-en:start -->
The $3+1$ chain pattern and the $2+2$ chain pattern cannot be changed into one another by an invertible change of coordinates. The reason is that such a change preserves the number of independent inputs killed after each number of applications. We can use these counts to recover every block size.
<!-- bilingual-en:end -->

$3+1$ 与 $2+2$ 的链模式不能通过可逆换基互相转换。原因是：这种换基会保留“作用指定次数后被消去的独立输入有多少”这一数量。用这些数量，就能逐项恢复全部块尺寸。

> [!important] Unique block data, with order ignored · 忽略排列后，块资料唯一
> <!-- bilingual-en:start -->
> Fix a field $\mathbb F$ and an $n\times n$ matrix $A$, with $n\ge1$, whose characteristic polynomial [[Jordan标准形的域条件|splits over that field]]. For each eigenvalue $\lambda$, every Jordan form of $A$ contains exactly the same block sizes, with the same repetitions. The only freedom in the resulting Jordan matrix is to permute whole blocks. Two such matrices over the same field are [[相似矩阵|similar]] exactly when these eigenvalue-labelled block data agree.
> <!-- bilingual-en:end -->
>
> 固定数域 $\mathbb F$，设 $A$ 为 $n\times n$ 方阵，$n\ge1$，且特征多项式[[Jordan标准形的域条件|在该域中分裂]]。对每个特征值 $\lambda$，$A$ 的任意 Jordan 形都有相同的块尺寸及其重复次数。所得 Jordan 矩阵只允许交换整块的排列。对同一域中满足该条件的两个矩阵，它们[[相似矩阵|相似]]当且仅当这些带特征值标签的块资料相同。

<!-- bilingual-en:start -->
## Count what disappears after repeated applications
<!-- bilingual-en:end -->

*数一数重复作用后消失了多少方向*

<!-- bilingual-en:start -->
Fix one eigenvalue $\lambda$. For each integer $j\ge1$, define
<!-- bilingual-en:end -->

固定一个特征值 $\lambda$。对每个整数 $j\ge1$，定义

$$d_j=\dim\ker\bigl((A-\lambda I_n)^j\bigr),\qquad d_0=0.$$

<!-- bilingual-en:start -->
Here $I_n$ is the [[单位矩阵|identity matrix]], $\ker$ denotes the nullspace, and [[维数|dimension]] counts independent directions in a space. Subtracting $\lambda I_n$ removes the scalar part of a Jordan block; raising this difference to power $j$ applies the remaining action $j$ times. Thus $d_j$ counts the independent directions killed by then. In particular, $d_1$ is the dimension of the ordinary [[特征空间|eigenspace]]. The $j=0$ convention matches the fact that the zeroth power is $I_n$, whose nullspace is just zero.
<!-- bilingual-en:end -->

这里 $I_n$ 为[[单位矩阵]]，$\ker$ 表示零空间，[[维数]]计算一个空间中的独立方向数。减去 $\lambda I_n$ 会移除相应 Jordan 块的标量部分，再取 $j$ 次幂，就是连续作用 $j$ 次。因此 $d_j$ 数的是到这一步已经归零的独立方向。特别地，$d_1$ 是普通[[特征空间]]的维数。$j=0$ 的约定也符合零次幂为 $I_n$、其零空间只有零向量这一事实。

<!-- bilingual-en:start -->
These counts cannot change under similarity. If $B=M^{-1}AM$ with $M$ invertible, then
<!-- bilingual-en:end -->

这些数量在相似变换下不变。若 $B=M^{-1}AM$、$M$ 可逆，则

$$(B-\lambda I_n)^j=M^{-1}(A-\lambda I_n)^jM.$$

<!-- bilingual-en:start -->
Whenever $(A-\lambda I_n)^jv=0$, the vector $M^{-1}v$ is killed by $(B-\lambda I_n)^j$. Multiplication by $M$ reverses this correspondence. It is a one-to-one linear map between the two nullspaces, so their dimensions agree. This is the same reasoning as [[相似保持特征空间维数|eigenspace-dimension invariance]], now applied to every power rather than just the first.
<!-- bilingual-en:end -->

只要 $(A-\lambda I_n)^jv=0$，$M^{-1}v$ 就被 $(B-\lambda I_n)^j$ 消去；乘以 $M$ 又能反向还原。这在两个零空间之间给出一一对应的线性映射，所以维数相同。推理与[[相似保持特征空间维数]]相同，只是从一次作用推广到每一个幂。

<!-- bilingual-en:start -->
## The increments reveal the blocks
<!-- bilingual-en:end -->

*每次增加的维数，怎样揭示块尺寸*

<!-- bilingual-en:start -->
In one block of size $s$ for eigenvalue $\lambda$, the shifted matrix is the one-step shift $N=J_s(\lambda)-\lambda I_s$. After $j$ applications, exactly the first $\min(j,s)$ coordinate directions have been sent to zero. This block contributes $\min(j,s)$ to $d_j$. A block for a different eigenvalue $\mu\ne\lambda$ contributes zero: its shift by $\lambda I$ has nonzero diagonal $\mu-\lambda$, is invertible, and so are its powers.
<!-- bilingual-en:end -->

对于属于 $\lambda$、尺寸为 $s$ 的一个块，移去标量部分后得到一步移位 $N=J_s(\lambda)-\lambda I_s$。连续作用 $j$ 次，恰好把前 $\min(j,s)$ 个坐标方向消去，因此这个块对 $d_j$ 贡献 $\min(j,s)$。属于另一特征值 $\mu\ne\lambda$ 的块贡献为零：它减去 $\lambda I$ 后，对角元为非零的 $\mu-\lambda$，因此可逆，每个幂也可逆。

<!-- bilingual-en:start -->
Let $s_1,\ldots,s_q$ be the sizes of the blocks belonging to this fixed $\lambda$; $q$ counts those blocks only. Adding their contributions gives
<!-- bilingual-en:end -->

令 $s_1,\ldots,s_q$ 为属于这个固定 $\lambda$ 的各块尺寸，$q$ 只计这些块。将贡献相加，得到

$$d_j=\sum_{\ell=1}^{q}\min(j,s_\ell).$$

<!-- bilingual-en:start -->
Going from $j-1$ to $j$, a block supplies one new nullspace direction if its size is at least $j$, and supplies none if it has already ended. Consequently
<!-- bilingual-en:end -->

从 $j-1$ 增加到 $j$，一个块若至少有 $j$ 阶，就新增一个归零方向；若链已经结束，就不再贡献。因此

$$b_j:=d_j-d_{j-1}
=\#\{\text{blocks for }\lambda\text{ of size at least }j\}.$$

<!-- bilingual-en:start -->
The symbol $\#$ means “number of.” Subtract the blocks that continue beyond length $j$ to obtain the number $c_j$ of blocks of size exactly $j$:
<!-- bilingual-en:end -->

符号 $\#$ 表示“个数”。再减去那些长度超过 $j$ 的块，便得到尺寸恰为 $j$ 的块数 $c_j$：

$$c_j=b_j-b_{j+1}=2d_j-d_{j-1}-d_{j+1}.$$

<!-- bilingual-en:start -->
Since all $d_j$ are fixed by $A$, every $c_j$ is fixed. This proves uniqueness of the block data. The nonnegative increments $b_j$ are nonincreasing and eventually reach zero. Every block has size at most $n$, so the counts stabilize by $j=n$; for the last difference, $d_{n+1}=d_n$. If $\lambda$ is not an eigenvalue, all these counts are zero.
<!-- bilingual-en:end -->

由于全部 $d_j$ 都由 $A$ 确定，每个 $c_j$ 也随之确定，这就证明了块资料的唯一性。非负的增量 $b_j$ 会逐渐不增，最终变成零。每块至多有 $n$ 阶，因此到 $j=n$ 时数量已稳定；计算最后一个差时使用 $d_{n+1}=d_n$。若 $\lambda$ 不是特征值，这些数量全部为零。

> [!example] Recover 3 + 1 and 2 + 2 · 从维数恢复两种分组
> <!-- bilingual-en:start -->
> For the opening matrices and $\lambda=0$, the sequences from $j=0$ through $4$ are
> $$\begin{array}{c|ccccc}
> j&0&1&2&3&4\\\hline
> d_j(A)&0&2&3&4&4\\
> d_j(B)&0&2&4&4&4
> \end{array}$$
> For $A$, the increments are $2,1,1,0$: two blocks have size at least one, one has size at least two, and one has size at least three. Thus the sizes are $3,1$. For $B$, the increments are $2,2,0,0$, giving two blocks of size two. Both have the same characteristic polynomial $t^4$ and the same $d_1=2$; $d_2$ is the first count that separates them.
> <!-- bilingual-en:end -->
>
> 对开头矩阵与 $\lambda=0$，上表从 $j=0$ 列到 $4$。$A$ 的增量为 $2,1,1,0$：至少一阶有两个块，至少二阶有一个，至少三阶仍有一个，因此尺寸为 $3,1$。$B$ 的增量为 $2,2,0,0$，得到两个二阶块。二者特征多项式都是 $t^4$，且 $d_1=2$ 相同；最先区分它们的是 $d_2$。

<!-- bilingual-en:start -->
## What uniqueness does not fix
<!-- bilingual-en:end -->

*唯一性没有固定什么*

<!-- bilingual-en:start -->
Reordering whole blocks changes the displayed matrix, but a permutation of the chain basis gives a similarity between the two displays. For example, $\operatorname{diag}(J_2(3),J_1(1))$ and $\operatorname{diag}(J_1(1),J_2(3))$ describe the same block data. “Different-looking Jordan matrices are never similar” would therefore be too strong.
<!-- bilingual-en:end -->

重排整块会改变显示的矩阵，但相应地重排链基，就得到两种显示之间的相似变换。例如 $\operatorname{diag}(J_2(3),J_1(1))$ 与 $\operatorname{diag}(J_1(1),J_2(3))$ 表示相同块资料。因此不能说“长得不同的 Jordan 矩阵必不相似”。

<!-- bilingual-en:start -->
The change-of-basis matrix $S$ is generally not unique either. For a length-two [[Jordan链|Jordan chain]] $v_1,v_2$, replacing $v_2$ by $v_2+a v_1$ for any scalar $a$ preserves $(A-\lambda I)v_2=v_1$, because $(A-\lambda I)v_1=0$. The new pair is still a basis of the same chain span and gives the same block. The theorem fixes the eigenvalue-labelled block sizes, not the individual chain vectors.
<!-- bilingual-en:end -->

换基矩阵 $S$ 一般也不唯一。对长度二的 [[Jordan链]] $v_1,v_2$，把 $v_2$ 改成 $v_2+a v_1$，其中 $a$ 为任意标量，仍保留 $(A-\lambda I)v_2=v_1$ 的链关系，因为 $(A-\lambda I)v_1=0$。新的一对向量仍是同一链张成空间的基，并给出相同块。定理固定的是带特征值标签的块尺寸，而不是链向量本人。

<!-- bilingual-en:start -->
These are exact algebraic statements. Matching eigenvalues or estimated ranks approximately in numerical data is a different problem; the [[Jordan、Schur与SVD用途边界|Jordan–Schur–SVD comparison]] explains the computational boundary.
<!-- bilingual-en:end -->

这些都是精确代数结论。数值数据中特征值近似相同、或估计出的秩相同，是另一类问题；计算边界见[[Jordan、Schur与SVD用途边界]]。

## 来源与核验

<!-- bilingual-en:start -->
- [University of Colorado, Math 6140, §12.3, pp.1–2](https://math.colorado.edu/~mayr/teaching/math6140spring25/PID3.pdf): uniqueness up to permutation of Jordan blocks under the splitting condition.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S04_Recitation_Problem_Solving_Similar_Matrices.pdf#page=1|MIT 18.06SC Session 3.4 recitation, pp.1–2]]: similarity of matrix polynomials and the use of nullspaces of powers to distinguish Jordan structure. Its “different Jordan matrices” shorthand is interpreted up to block reordering, as required by the precise uniqueness theorem. The formulas for $d_j,b_j,c_j$ above are derived directly from a single block's shift action.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=3|MIT Session 3.4 summary, pp.3–4]] and [[03_Positive Definite Matrices and Applications#3.4.5 课件与 Recitation 例子|permanent course notes, §3.4.5]]: the $3+1$ versus $2+2$ example and the successive-nullspace method. All displayed actions, dimensions and basis changes are checked directly.
<!-- bilingual-en:end -->

- 科罗拉多大学第12.3节第1–2页核对分裂条件下、忽略块排列的唯一性。
- MIT Session3.4 Recitation 第1–2页核对矩阵多项式的相似性，以及用幂的零空间区分 Jordan 结构；其“不同 Jordan 矩阵”的简写须按精确唯一性定理排除块重排。正文的 $d_j,b_j,c_j$ 公式由单个块的移位作用直接推导。
- MIT Session3.4 摘要第3–4页及永久课程第3.4.5节核对 $3+1$ 与 $2+2$ 的例子、连续零空间方法；显示的作用、维数和换基均直接计算。
