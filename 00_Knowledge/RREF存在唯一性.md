---
aliases:
  - Existence and uniqueness of reduced row echelon form
  - Uniqueness of RREF
student_os: knowledge-atom
atom_id: LA-SYS-047
atom_set: linear-systems-four-subspaces
atom_type: theorem
status: source-checked
requires:
  - "[[行最简形]]"
  - "[[行等价]]"
  - "[[高斯消元]]"
  - "[[行空间]]"
  - "[[维数]]"
related:
  - "[[主元]]"
leads_to:
  - "[[RREF读取子空间基]]"
  - "[[Gauss-Jordan求逆]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

<!-- bilingual-en:start -->
# Existence and uniqueness of RREF
<!-- bilingual-en:end -->
*行最简形的存在性与唯一性*

<!-- bilingual-en:start -->
Row reduction allows choices: one person may swap rows first, while another starts by subtracting a multiple of a row. The theorem says that these choices do not change the final [[行最简形|reduced row echelon form (RREF)]], provided both calculations use exact arithmetic and finish all the reduction steps. RREF is therefore a canonical representative: one agreed matrix for the whole [[行等价|row-equivalence class]].
<!-- bilingual-en:end -->

行化简允许选择：有人先换行，有人先减去另一行的倍数。这个定理说明，只要两人都进行精确运算并完成全部化简步骤，最终的[[行最简形|行最简形（RREF）]]就相同。因此，RREF 是规范代表：同一个[[行等价|行等价类]]统一对应一张矩阵。

> [!important] One final matrix · 一个最终矩阵
> <!-- bilingual-en:start -->
> For positive integers $m,n$, every $m\times n$ matrix $A$ over a field $\mathbb F$ is row equivalent to exactly one RREF matrix $R$. The field may be $\mathbb R$ or $\mathbb C$; the argument also works over other fields. The uniqueness concerns $R$, not the sequence of row operations.
> <!-- bilingual-en:end -->
>
> 对正整数 $m,n$，域 $\mathbb F$ 上每个 $m\times n$ 矩阵 $A$ 都行等价于唯一一个 RREF 矩阵 $R$。数域可以是 $\mathbb R$ 或 $\mathbb C$，证明也适用于其他域。唯一的是 $R$，不是行操作的顺序。

<!-- bilingual-en:start -->
## See what the reduction fixes
<!-- bilingual-en:end -->
*先看化简固定了什么*

<!-- bilingual-en:start -->
Over the real numbers, consider
<!-- bilingual-en:end -->

在实数域上，考虑

$$
A=\begin{bmatrix}1&2&3\\2&4&7\end{bmatrix}
\xrightarrow{R_2\leftarrow R_2-2R_1}
\begin{bmatrix}1&2&3\\0&0&1\end{bmatrix}
\xrightarrow{R_1\leftarrow R_1-3R_2}
R=\begin{bmatrix}1&2&0\\0&0&1\end{bmatrix}.
$$

<!-- bilingual-en:start -->
Here $R_i$ in an operation denotes row $i$. The [[主元|pivots]] are the leading $1$s in columns $1$ and $3$; each is the only nonzero entry in its column. Column $2$ is allowed to contain $2$. RREF does not require every nonpivot column to be zero. A different legal elimination route must reach this same $R$.
<!-- bilingual-en:end -->

操作中的 $R_i$ 表示第 $i$ 行。[[主元]]是第 $1$、$3$ 列的首个 $1$，各自所在列的其他元素均为零。第 $2$ 列允许出现 $2$：RREF 不要求非主元列全为零。换一条合法消元路线，最终也必须得到这张 $R$。

<!-- bilingual-en:start -->
## Why a reduced form exists
<!-- bilingual-en:end -->
*为什么一定能得到行最简形*

<!-- bilingual-en:start -->
[[高斯消元|Gaussian elimination]] scans columns from left to right. In the rows not yet used for pivots, skip a column if all its entries are zero. Otherwise move a nonzero entry to the next pivot row, and subtract suitable multiples of that row to clear the entries below it. Continue on the remaining rows and later columns. Each pivot uses a new row and a later column, so there can be at most $\min(m,n)$ pivots. If the remaining block is zero, stop the forward stage.
<!-- bilingual-en:end -->

[[高斯消元]]从左向右扫描列。在尚未用作主元行的各行中，如果某列全为零，就跳过这一列；否则把一个非零元素换到下一个主元位置，再减去该行的适当倍数，消去主元下方的元素。随后处理剩余各行和更靠后的列。每个主元使用新的一行和更靠后的一列，因此至多出现 $\min(m,n)$ 个主元；剩余块全为零时，向前阶段结束。

<!-- bilingual-en:start -->
Divide each nonzero row by its pivot, and work upwards from the last pivot to clear the entries above each pivot. Division is legal because the pivot is nonzero. These reversible operations leave zero rows at the bottom and produce exactly the RREF conditions. This establishes existence without assuming uniqueness.
<!-- bilingual-en:end -->

把每个非零行除以本行的主元，再从最后一个主元向上逐个清除其上方元素。主元非零，所以除法合法。这些可逆操作把零行留在底部，并满足 RREF 的全部条件。这就证明了存在性，没有预先使用唯一性。

<!-- bilingual-en:start -->
## Why the final form is unique
<!-- bilingual-en:end -->
## 为什么最终形式唯一

<!-- bilingual-en:start -->
The [[行空间|row space]] $W$ is the set of all linear combinations of the rows. Reversible row operations leave it unchanged: new rows are combinations of old rows, and reversing the operations expresses old rows as combinations of new rows. We will recover both the pivot positions and the individual RREF rows from $W$ alone.
<!-- bilingual-en:end -->

[[行空间]] $W$ 是各行的所有线性组合构成的集合。可逆行操作保持它不变：新行是旧行的组合，反向操作又把旧行写成新行的组合。下面只根据 $W$，依次恢复主元位置和 RREF 的每个非零行。

<!-- bilingual-en:start -->
For the displayed example, $W=\{(s,2s,t):s,t\in\mathbb R\}$. Looking only at the first coordinate reveals one freely varying number. Looking at the first two coordinates still reveals just one, because the second is twice the first. Adding the third coordinate reveals a second free number. This is how the row space identifies columns $1$ and $3$ as pivot columns.
<!-- bilingual-en:end -->

前例中，$W=\{(s,2s,t):s,t\in\mathbb R\}$。只看第一个坐标，有一个可自由变化的数；看前两个坐标时仍只有一个，因为第二个是第一个的两倍；再加入第三个坐标，才增加第二个自由数。行空间正是这样识别出第 $1$、$3$ 列为主元列。

<!-- bilingual-en:start -->
For a general $W\subseteq\mathbb F^n$, let $\pi_j$ retain the first $j$ coordinates of each vector, and let $d_j$ be the [[维数|dimension]] of the resulting space. Set $d_0=0$, corresponding to retaining no coordinates. If an RREF has $r$ nonzero rows with pivot columns $p_1<\cdots<p_r$, then
<!-- bilingual-en:end -->

一般地，对 $W\subseteq\mathbb F^n$，令 $\pi_j$ 保留向量的前 $j$ 个坐标，并令 $d_j$ 为所得空间的[[维数]]。约定 $d_0=0$，对应一个坐标也不保留。如果某个 RREF 有 $r$ 个非零行，主元列依次为 $p_1<\cdots<p_r$，则

$$
d_j=\dim\pi_j(W)=\#\{i:p_i\le j\},
\qquad j=0,1,\ldots,n.
$$

<!-- bilingual-en:start -->
To justify this count, truncate the RREF rows to their first $j$ coordinates. Rows whose pivots lie after column $j$ become zero. The other truncated rows are independent, because restricting them to their pivot coordinates gives an identity matrix, and they span $\pi_j(W)$ because the full rows span $W$. Thus the dimension is exactly the stated count. Consequently, column $j$ is a pivot column precisely when $d_j-d_{j-1}=1$. Every $d_j$ depends only on $W$, so the pivot positions are fixed by $W$.
<!-- bilingual-en:end -->

这个计数的理由如下：把 RREF 各行截取到前 $j$ 个坐标。主元在第 $j$ 列之后的行变成零向量；其余截短行在各自主元坐标上组成单位矩阵，因此线性无关。完整各行张成 $W$，截短行也就张成 $\pi_j(W)$，故维数正好是上面的数量。因此，第 $j$ 列是主元列当且仅当 $d_j-d_{j-1}=1$。每个 $d_j$ 只由 $W$ 决定，所以主元位置也被 $W$ 固定。

<!-- bilingual-en:start -->
It remains to fix every entry of each row. Assume $r>0$, write the nonzero RREF rows as $w_1,\ldots,w_r$, and extract just the pivot coordinates using the map
<!-- bilingual-en:end -->

还要固定每一行的全部元素。设 $r>0$，把 RREF 的非零行写成 $w_1,\ldots,w_r$，再用下面的映射只提取主元坐标：

$$
P:W\longrightarrow\mathbb F^r,
\qquad P(w)=(w_{p_1},\ldots,w_{p_r}).
$$

<!-- bilingual-en:start -->
The rows $w_i$ form a [[基|basis]] of $W$: they span the row space, and the pivot coordinates make them independent. Let $e_i$ denote the vector in $\mathbb F^r$ with a $1$ in position $i$ and zeros elsewhere. The RREF conditions give $P(w_i)=e_i$. For arbitrary scalars $c_1,\ldots,c_r$,
<!-- bilingual-en:end -->

各行 $w_i$ 构成 $W$ 的一组[[基]]：它们张成行空间，主元坐标又保证线性无关。令 $e_i$ 表示 $\mathbb F^r$ 中第 $i$ 个坐标为 $1$、其他坐标为零的向量。RREF 条件给出 $P(w_i)=e_i$。对任意标量 $c_1,\ldots,c_r$，

$$
P(c_1w_1+\cdots+c_rw_r)=(c_1,\ldots,c_r).
$$

<!-- bilingual-en:start -->
So each list of pivot coordinates comes from exactly one vector in $W$: its coordinates directly reveal its basis coefficients. In particular, row $i$ must be the unique vector $P^{-1}(e_i)$. Both $W$ and the pivot positions are already fixed, hence these rows are fixed. Any two row-equivalent RREF matrices of the same $m\times n$ size have these same $r$ nonzero rows in pivot order, followed by $m-r$ zero rows. If $r=0$, $W=\{0\}$ and every row is zero, so uniqueness is immediate as well.
<!-- bilingual-en:end -->

因此，每一组主元坐标恰好来自 $W$ 中一个向量：这些坐标直接给出了它的基系数。特别地，第 $i$ 行必须是唯一的 $P^{-1}(e_i)$。$W$ 和主元位置都已固定，各行也就固定了。同为 $m\times n$ 且行等价的两个 RREF，必有相同的 $r$ 个非零行，按主元次序排列，后面再接 $m-r$ 个零行。如果 $r=0$，则 $W=\{0\}$，所有行只能是零行，同样唯一。

<!-- bilingual-en:start -->
## What uniqueness does and does not say
<!-- bilingual-en:end -->
## 边界

<!-- bilingual-en:start -->
An ordinary echelon form is not unique. Over $\mathbb R$, for example, both $\begin{bmatrix}1&2\\0&1\end{bmatrix}$ and $\begin{bmatrix}2&4\\0&1\end{bmatrix}$ are row-equivalent echelon matrices. Normalizing the pivots and clearing their entire columns remove this freedom. The theorem makes [[RREF读取子空间基|structural readings from RREF]] independent of the legal elimination route; it does not make every possible basis or every elimination history identical. Floating-point roundoff and different thresholds for treating a small number as zero are separate numerical issues, not counterexamples to exact uniqueness.
<!-- bilingual-en:end -->

普通阶梯形不唯一，例如在 $\mathbb R$ 上， $\begin{bmatrix}1&2\\0&1\end{bmatrix}$ 与 $\begin{bmatrix}2&4\\0&1\end{bmatrix}$ 都是彼此行等价的阶梯形。归一化主元并清空整个主元列，才消除了这项自由。定理保证[[RREF读取子空间基|从 RREF 读取结构]]不依赖合法消元路线，并不要求所有可能的基或全部消元历史都相同。浮点舍入以及把小数当作零的不同阈值属于数值计算问题，不是精确唯一性的反例。

> [!question]- Why should two completed reductions agree? · 为什么两份完整化简应当相同？
> <!-- bilingual-en:start -->
> Both outputs are RREF matrices row equivalent to the same original matrix. They have the same row space, which fixes their pivot positions and then every row. Compare the final matrices, not the lists of intermediate operations.
> <!-- bilingual-en:end -->
>
> 两个结果都是与同一原矩阵行等价的 RREF。它们具有相同行空间，后者先固定主元位置，再固定每一行。因此比较的是最终矩阵，而不是中间操作清单。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf|MIT 18.06SC Session 1.7 summary]], pp. 1–2: RREF construction, normalized pivot columns, and free columns.
- [Sam Schiavone, MIT 18.700 Lesson 1](https://samschiavone.github.io/courses/18-700/Lectures/LessonPlan1.pdf), pp. 4–6: elementary operations, the existence-and-uniqueness statement, and the distinction between an echelon form and RREF.
- [Interactive Linear Algebra, §1.2](https://textbooks.math.gatech.edu/ila/row-reduction.html), §§1.2.2–1.2.3: precise shape conditions and a finite reduction algorithm. The latter two references state uniqueness without proving it; the coordinate-prefix and pivot-coordinate argument above supplies the complete proof here.
<!-- bilingual-en:end -->

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf|MIT 18.06SC Session 1.7 摘要]]，第 1–2 页：核对 RREF 构造、归一化主元列与自由列。
- [Sam Schiavone，MIT 18.700 Lesson 1](https://samschiavone.github.io/courses/18-700/Lectures/LessonPlan1.pdf)，第 4–6 页：核对初等操作、存在唯一性陈述及普通阶梯形与 RREF 的区别。
- [Interactive Linear Algebra，§1.2](https://textbooks.math.gatech.edu/ila/row-reduction.html)，§§1.2.2–1.2.3：核对精确形状条件与有限步算法。后两项资料陈述唯一性而未给出证明；本文用前缀坐标维数和主元坐标给出完整证明。
