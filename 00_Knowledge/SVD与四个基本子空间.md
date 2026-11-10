---
aliases:
  - Four fundamental subspaces from the SVD
student_os: knowledge-atom
atom_id: LA-SVD-022
atom_set: singular-value-decomposition-low-rank
atom_type: classification
status: source-checked
requires:
  - "[[奇异向量]]"
  - "[[四个基本子空间]]"
  - "[[奇异值与秩]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[SVD零块补基不唯一]]"
related:
  - "[[紧SVD]]"
---

<!-- bilingual-en:start -->
# The four subspaces from an SVD
<!-- bilingual-en:end -->

*SVD 与四个基本子空间*

<!-- bilingual-en:start -->
A matrix reads some input directions and loses others. On the output side, it reaches some directions and cannot reach their perpendicular complement. An SVD supplies perpendicular unit bases for all four spaces at once. Keeping track of **which side a vector belongs to** prevents the input null space from being confused with the left null space.
<!-- bilingual-en:end -->

矩阵会读取部分输入方向，消去另一部分；在输出侧，它能到达一些方向，却到不了与它们垂直的补空间。SVD 同时为四个空间提供相互垂直的单位基。先分清**向量属于输入侧还是输出侧**，才不会混淆零空间与左零空间。

<!-- bilingual-en:start -->
## See both sides of a rectangular map
<!-- bilingual-en:end -->

*先看矩形映射的两侧*

<!-- bilingual-en:start -->
Consider

$$A=\begin{pmatrix}3&0&0&0\\0&2&0&0\\0&0&0&0\end{pmatrix},\qquad
Ax=(3x_1,2x_2,0)^T.$$

Inputs have four coordinates and outputs have three. Let $f_1,\ldots,f_4$ be the standard input coordinate vectors and $e_1,e_2,e_3$ the output coordinate vectors: each has a single 1 in the indicated position. Then $Af_1=3e_1$, $Af_2=2e_2$, and $Af_3=Af_4=0$. The superscript $T$ denotes [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

考虑上面的 $A$：输入有四个坐标，输出有三个。用 $f_1,\ldots,f_4$ 表示输入侧标准坐标向量，用 $e_1,e_2,e_3$ 表示输出侧标准坐标向量，即指定位置为1、其余为0。于是 $Af_1=3e_1$、$Af_2=2e_2$，而 $Af_3=Af_4=0$；$T$ 表示[[矩阵转置|转置]]。

> [!example] Useful inputs and unreachable outputs / 有效输入与不可达输出
> <!-- bilingual-en:start -->
> The [[零空间|null space]] is all combinations of $f_3,f_4$: changing those inputs leaves the output unchanged. The complementary active input space is spanned by $f_1,f_2$. The [[列空间|column space]] is the output plane spanned by $e_1,e_2$. Its perpendicular direction $e_3$ satisfies $A^Te_3=0$, so it spans the [[左零空间|left null space]]. A target with nonzero third coordinate is unattainable. The output vector $e_3$ is tested by $A^T$, not by $A$.
> <!-- bilingual-en:end -->
>
> [[零空间]]是 $f_3,f_4$ 的全部线性组合，改变这些输入不改变输出；互补的有效输入空间由 $f_1,f_2$ 张成。[[列空间]]是 $e_1,e_2$ 张成的输出平面，垂直方向 $e_3$ 满足 $A^Te_3=0$，所以张成[[左零空间]]。目标第三坐标非零时不可达；$e_3$ 属于输出侧，应由 $A^T$ 检验，不能当作 $A$ 的输入。

<!-- bilingual-en:start -->
## Read the bases from a complete decomposition
<!-- bilingual-en:end -->

*从完整分解读取四组基*

<!-- bilingual-en:start -->
Let $A$ be a real or complex $m\times n$ matrix, with $m,n\ge1$ and [[矩阵秩|rank]] $r$. A complete [[奇异值分解|SVD]] is $A=U\Sigma V^*$, where $U$ is $m\times m$ and $V$ is $n\times n$. Their columns $u_1,\ldots,u_m$ and $v_1,\ldots,v_n$ are [[标准正交组|orthonormal bases]]: unit vectors perpendicular to each other. The star means conjugate transpose, ordinary transpose for real entries. Arrange the $r$ positive singular values first.

Write $C(M)$ for column space, $N(M)$ for null space, and $\mathbb F$ for $\mathbb R$ or $\mathbb C$.
<!-- bilingual-en:end -->

设 $A$ 为实或复 $m\times n$ 矩阵，$m,n\ge1$，[[矩阵秩|秩]]为 $r$。完整[[奇异值分解]]是 $A=U\Sigma V^*$，其中 $U$ 为 $m\times m$，$V$ 为 $n\times n$；各自的列 $u_1,\ldots,u_m$ 与 $v_1,\ldots,v_n$ 构成[[标准正交组|标准正交基]]，即相互垂直的单位向量。星号表示共轭转置，实数时为普通转置，把 $r$ 个正奇异值排在前面。

记 $C(M)$ 为列空间，$N(M)$ 为零空间，$\mathbb F$ 表示 $\mathbb R$ 或 $\mathbb C$。

<!-- bilingual-en:start -->
| Space | Ambient side | Orthonormal basis | Dimension |
|---|---|---|---:|
| Active input space $C(A^*)$ | Input $\mathbb F^n$ | $v_1,\ldots,v_r$ | $r$ |
| Null space $N(A)$ | Input $\mathbb F^n$ | $v_{r+1},\ldots,v_n$ | $n-r$ |
| Attainable output space $C(A)$ | Output $\mathbb F^m$ | $u_1,\ldots,u_r$ | $r$ |
| Left null space $N(A^*)$ | Output $\mathbb F^m$ | $u_{r+1},\ldots,u_m$ | $m-r$ |
<!-- bilingual-en:end -->

输入侧：前 $r$ 个右奇异向量张成 $C(A^*)$，其余 $n-r$ 个张成 $N(A)$。输出侧：前 $r$ 个左奇异向量张成可达的 $C(A)$，其余 $m-r$ 个张成 $N(A^*)$。

<!-- bilingual-en:start -->
For real matrices, $C(A^T)$ is the [[行空间|row space]] written with column-vector coordinates. Over complex numbers, use $C(A^*)$, including conjugation rather than copying the un-conjugated rows. These are the [[四个基本子空间|four fundamental subspaces]], now with bases chosen by the SVD.
<!-- bilingual-en:end -->

实矩阵的 $C(A^T)$ 是把坐标排成列后的[[行空间]]；复数时使用 $C(A^*)$，必须取共轭，不能直接照搬未共轭的行坐标。这就是[[四个基本子空间]]，SVD 又为它们选出具体的标准正交基。

<!-- bilingual-en:start -->
## Why the lists span exactly these spaces
<!-- bilingual-en:end -->

*为什么这些列恰好张成所需空间*

<!-- bilingual-en:start -->
For $i\le r$, the pairing equations are

$$Av_i=\sigma_i u_i,\qquad A^*u_i=\sigma_i v_i,\qquad \sigma_i>0.$$

Thus the first $r$ left vectors are attainable outputs, and the first $r$ right vectors belong to $C(A^*)$. An arbitrary input expands as $x=\sum_{i=1}^n c_i v_i$, with scalar coordinates $c_i=v_i^*x$. Its output is

$$Ax=\sum_{i=1}^r \sigma_i c_i u_i.$$

These outputs span exactly $C(A)$, because each positive factor can be divided out. The perpendicular $u_i$ cannot cancel each other, so $Ax=0$ holds exactly when $c_1=\cdots=c_r=0$. The remaining $v_i$ therefore form a basis of $N(A)$. Applying the same reasoning to $A^*$ gives the other two spaces. The vectors $\sigma_i u_i$ are perpendicular but generally not unit vectors: $A$ stretches basis directions rather than preserving all their lengths.
<!-- bilingual-en:end -->

$i\le r$ 时有上述配对式，所以前 $r$ 个左向量确实是可达输出，前 $r$ 个右向量属于 $C(A^*)$。任意输入可展开为 $x=\sum_{i=1}^n c_i v_i$，标量坐标为 $c_i=v_i^*x$，输出便是所示前 $r$ 项之和。每个正因子都可除掉，因此这些输出恰好张成 $C(A)$。

相互垂直的 $u_i$ 不能彼此抵消，故 $Ax=0$ 当且仅当前 $r$ 个坐标全为零，其余 $v_i$ 就构成 $N(A)$ 的基。对 $A^*$ 使用相同论证，得到另外两个空间。注意 $\sigma_i u_i$ 相互垂直，却通常不是单位向量；$A$ 会伸缩基方向，并非总保持长度。

<!-- bilingual-en:start -->
The complete bases give orthogonal [[直和|direct sums]]:
<!-- bilingual-en:end -->

完整基给出正交[[直和]]：

$$\mathbb F^n=C(A^*)\oplus N(A),\qquad
\mathbb F^m=C(A)\oplus N(A^*).$$

<!-- bilingual-en:start -->
Each vector splits uniquely into perpendicular components in the two displayed spaces on its side. A [[紧SVD|compact SVD]] lists only the positive directions, not the two null-space bases. Their sizes can differ: for a $7\times4$ matrix of rank 2, the null-space bases are $v_3,v_4$ and $u_3,\ldots,u_7$.

At rank zero, both active spaces are $\{0\}$ and have an empty basis; all columns of $V$ and $U$ belong to their respective null spaces. Full column rank gives $N(A)=\{0\}$; full row rank gives $N(A^*)=\{0\}$. The spaces are fixed by $A$, but the basis vectors need not be unique. In particular, [[SVD零块补基不唯一|null-space completions]] can change without changing the matrix.
<!-- bilingual-en:end -->

每一侧的向量都唯一分成所示两个空间中的垂直分量。[[紧SVD]]只列出正方向，不列出两个零空间的基；两组数量也可能不同：$7\times4$、秩2时，分别为 $v_3,v_4$ 和 $u_3,\ldots,u_7$。

秩零时，两个有效空间都是 $\{0\}$，基为空；$V,U$ 的全部列分别属于相应零空间。满列秩时 $N(A)=\{0\}$，满行秩时 $N(A^*)=\{0\}$。子空间由 $A$ 固定，具体基却未必唯一，尤其[[SVD零块补基不唯一|零空间补基]]可以变化而不改变矩阵。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT18.06SC Session3.5, PDF pp.3–4]]: the rank-one example and correct four-basis list at the end of p.4.
- [Axler, *Linear Algebra Done Right*, 4th ed., §7E, Theorems7.70/7.75 and completion discussion, PDF287–289; Exercise8, PDF292](https://linear.axler.net/LADR4e.pdf#page=287): real/complex singular directions, adjoint pairing and orthonormal completion. [LAPACK's SVD definition](https://netlib.org/lapack/lug/node32.html) fixes rectangular dimensions. Space identities and examples follow directly from the displayed pairing equations.
<!-- bilingual-en:end -->

- MIT18.06SC Session3.5第3–4页：核对秩一例子与第4页末四组基的对应。
- Axler 第7E节定理7.70/7.75及补基讨论，PDF287–289；习题8在PDF292：核对实复奇异方向、伴随配对及标准正交补基。LAPACK 定义核对矩形尺寸，空间恒等式与例子由正文配对式推出。
