---
aliases:
  - Leibniz determinant formula
  - Permutation formula for determinants
student_os: knowledge-atom
atom_id: LA-DET-008
atom_set: determinants
atom_type: formula
status: source-checked
requires:
  - "[[行列式]]"
part_of:
  - "[[行列式.canvas]]"
---

<!-- bilingual-en:start -->
# Leibniz determinant formula
<!-- bilingual-en:end -->
*Leibniz 行列式公式*

<!-- bilingual-en:start -->
The **Leibniz formula** computes a determinant by adding signed products of matrix entries. Each product selects one entry from **every row and every column**, with no column reused. The sign depends on the order of the selected column labels, not on whether the chosen entries themselves are positive or negative.
<!-- bilingual-en:end -->

**Leibniz 公式**把矩阵元素组成的带符号乘积相加，得到行列式。每个乘积必须从**每一行、每一列各取一个元素**，不能重复使用某列。项前的符号由所选列标签的顺序决定，不是由所选元素本身的正负决定。

<!-- bilingual-en:start -->
## Choose columns without repetition
<!-- bilingual-en:end -->
*选取列时不重复*

<!-- bilingual-en:start -->
For $A=\begin{pmatrix}a&b\\c&d\end{pmatrix}$, the column choices in row order can only be $(1,2)$ or $(2,1)$. They select $ad$ and $bc$. The first order is the natural order and has sign $+1$; the second reverses it and has sign $-1$. Thus $\det A=ad-bc$. Choosing $(1,1)$ would produce $ac$, but it uses column $1$ twice and misses column $2$, so it is not a Leibniz term.
<!-- bilingual-en:end -->

对 $A=\begin{pmatrix}a&b\\c&d\end{pmatrix}$，按行排列的列选择只能是 $(1,2)$ 或 $(2,1)$，分别选出 $ad$ 和 $bc$。前者是自然顺序，取符号 $+1$；后者将次序反过来，取 $-1$，因此 $\det A=ad-bc$。选择 $(1,1)$ 会产生 $ac$，但它重复使用第1列、遗漏第2列，不属于 Leibniz 项。

<!-- bilingual-en:start -->
For $n$ rows, a [[排列|permutation]] is a list $\sigma(1),\ldots,\sigma(n)$ containing each of $1,\ldots,n$ exactly once. Equivalently, $\sigma$ is a bijection of those labels: it assigns each row a different column. The set of all these choices is $S_n$. There are $n!=n(n-1)\cdots1$ choices: $n$ columns for the first row, $n-1$ remaining columns for the second, and so on.
<!-- bilingual-en:end -->

对 $n$ 行而言，一个[[排列]]就是列表 $\sigma(1),\ldots,\sigma(n)$，其中 $1,\ldots,n$ 各出现一次。等价地，$\sigma$ 是这组标签到自身的双射：给每一行分配不同的列。全部选择组成集合 $S_n$，数量为 $n!=n(n-1)\cdots1$：第一行有 $n$ 列可选，第二行还剩 $n-1$ 列，依此类推。

<!-- bilingual-en:start -->
An **inversion** is a pair of positions $i<j$ whose selected labels are reversed: $\sigma(i)>\sigma(j)$. Let $N(\sigma)$ count these pairs, and define $\operatorname{sgn}(\sigma)=(-1)^{N(\sigma)}$. An even count gives $+1$; an odd count gives $-1$. For $(2,3,1)$, the reversed pairs are $2>1$ and $3>1$, so the sign is positive. For $(3,2,1)$ there are three reversed pairs, so it is negative. Swapping adjacent labels changes inversion parity; sorting by adjacent inverted pairs removes one inversion at each step. Hence this sign also records the parity of the swaps needed to restore natural order. Any swap of two positions is an odd number of adjacent swaps, so the parity is independent of the chosen swap sequence.
<!-- bilingual-en:end -->

一个**逆序对**指位置 $i<j$，但所选标签反过来满足 $\sigma(i)>\sigma(j)$。令 $N(\sigma)$ 数出这些对，定义 $\operatorname{sgn}(\sigma)=(-1)^{N(\sigma)}$：偶数个取 $+1$，奇数个取 $-1$。例如 $(2,3,1)$ 有 $2>1$、$3>1$ 两对逆序，符号为正；$(3,2,1)$ 有三对，符号为负。交换相邻标签会改变逆序数的奇偶性；逐次交换相邻逆序标签排序，每一步恰好减少一个逆序。因此这个符号也记录恢复自然顺序所需交换次数的奇偶性。任意两位置的交换都可分解为奇数次相邻交换，所以无论选哪条交换路线，奇偶性都相同。

<!-- bilingual-en:start -->
## Read the formula through a three-row example
<!-- bilingual-en:end -->
*用三行例子读懂公式*

> [!important] One selected entry from each row and column · 每行每列各选一项
> <!-- bilingual-en:start -->
> For an $n\times n$ matrix $A=(a_{ij})$ over one field $\mathbb F$, with integer $n\ge1$,
> <!-- bilingual-en:end -->
>
> 对同一数域 $\mathbb F$ 上的 $n\times n$ 矩阵 $A=(a_{ij})$，其中 $n\ge1$ 为整数，
>
> $$\boxed{\det A=\sum_{\sigma\in S_n}\operatorname{sgn}(\sigma)
> \prod_{i=1}^n a_{i,\sigma(i)}.}$$
>
> <!-- bilingual-en:start -->
> Here $a_{ij}$ is the entry in row $i$, column $j$; $\prod$ multiplies the $n$ selected entries, and $\sum$ adds the signed results over all permutations. This is ordinary multiplication over $\mathbb R$ or $\mathbb C$, with no conjugation. The same formula works over any field, interpreting $\pm1$ in that field.
> <!-- bilingual-en:end -->
>
> $a_{ij}$ 表示第 $i$ 行、第 $j$ 列的元素；$\prod$ 把选出的 $n$ 个元素相乘，$\sum$ 再对全部排列的带符号结果求和。实数或复数情形都用普通乘法，不取共轭；公式也适用于任意域，只须把 $\pm1$ 解释为该域中的元素。

<!-- bilingual-en:start -->
Take $A=\begin{pmatrix}1&2&3\\0&4&5\\1&0&6\end{pmatrix}$. The six column-choice lists give the following six terms:
<!-- bilingual-en:end -->

取 $A=\begin{pmatrix}1&2&3\\0&4&5\\1&0&6\end{pmatrix}$。六种列选择给出以下六项：

$$\begin{array}{c|r|c|r}
(\sigma(1),\sigma(2),\sigma(3))&N(\sigma)&\operatorname{sgn}(\sigma)\prod_i a_{i,\sigma(i)}&\text{value}\\\hline
(1,2,3)&0&+1\cdot4\cdot6&24\\
(1,3,2)&1&-1\cdot5\cdot0&0\\
(2,1,3)&1&-2\cdot0\cdot6&0\\
(2,3,1)&2&+2\cdot5\cdot1&10\\
(3,1,2)&2&+3\cdot0\cdot0&0\\
(3,2,1)&3&-3\cdot4\cdot1&-12
\end{array}$$

<!-- bilingual-en:start -->
Thus $\det A=24+10-12=22$. The list $(2,3,1)$ means select positions $(1,2),(2,3),(3,1)$; it does not mean permute the numerical values $2,3,1$ inside the matrix. Some valid terms are zero because a selected entry is zero. There are still six permutation terms in the formula.
<!-- bilingual-en:end -->

因此 $\det A=24+10-12=22$。列表 $(2,3,1)$ 表示选位置 $(1,2),(2,3),(3,1)$，不是把数值 $2,3,1$ 在矩阵中重新排列。某些合法项会因为选到了零元素而为零，但公式仍有六个排列项。

<!-- bilingual-en:start -->
## Derive uniqueness, then verify existence
<!-- bilingual-en:end -->
*先推唯一性，再核验存在性*

<!-- bilingual-en:start -->
The [[行列式|determinant definition]] asks for normalization at the identity, separate row linearity, and zero whenever two rows agree. Suppose a function $D$ has those properties. Write row $i$ as $\sum_{j=1}^n a_{ij}e_j$, where $e_j$ is the row with $1$ in position $j$ and zeros elsewhere. Expanding all rows gives a sum over $n^n$ choices before zero terms are removed. If two chosen indices agree, two coordinate rows agree and that term vanishes. The surviving choices are exactly the permutations. The matrix with rows $e_{\sigma(1)},\ldots,e_{\sigma(n)}$ becomes the identity after sorting these rows, so its value is $\operatorname{sgn}(\sigma)$. This proves that every such $D$ must equal the displayed sum.
<!-- bilingual-en:end -->

[[行列式|行列式定义]]要求在单位矩阵处归一化、逐行线性，以及任意两行相同时归零。假设函数 $D$ 满足这些性质，把第 $i$ 行写成 $\sum_{j=1}^n a_{ij}e_j$，其中 $e_j$ 是第 $j$ 个位置为 $1$、其余位置为零的坐标行。展开全部行，在去掉零项以前有 $n^n$ 种选择；若两个下标相同，就会出现两条相同坐标行，该项为零。留下来的选择恰好是排列。把行依次为 $e_{\sigma(1)},\ldots,e_{\sigma(n)}$ 的矩阵排序成单位矩阵，便知其取值是 $\operatorname{sgn}(\sigma)$。因此任何这样的 $D$ 都必须等于所示求和。

<!-- bilingual-en:start -->
For existence, call the explicit sum $L(A)$, without assuming a determinant already exists. Each product uses exactly one entry of each row, so distributing a linear combination in any fixed row proves separate linearity. For the [[单位矩阵|identity matrix]], only the natural-order selection survives, giving $L(I_n)=1$. If rows $r$ and $s$ are equal, pair each permutation with the one obtained by swapping its assigned columns at these two rows. The paired permutations are different, give equal products, and have opposite signs. Every pair therefore sums to zero, proving repeated-row vanishing. This pairing is valid even in characteristic two: additive opposites sum to zero although $-1=1$. Thus $L$ satisfies all three defining properties. It is the determinant, and the previous paragraph proves uniqueness.
<!-- bilingual-en:end -->

证明存在性时，先把显式求和叫作 $L(A)$，不预设行列式已经存在。每个乘积恰好使用每行的一个元素，因此在固定行内分配线性组合，就证明了逐行线性。对[[单位矩阵]]，只有自然顺序的选择留下，故 $L(I_n)=1$。若第 $r,s$ 行相同，把每个排列与“交换这两行所分配列标签”的排列配成一对：它们互不相同，乘积相等，符号相反，因此每对之和为零，证明了重复行归零。这种配对在特征二时也成立：虽然 $-1=1$，互为加法逆元的两项仍相加为零。因此 $L$ 满足定义的三条性质，就是行列式；上一段已证明其唯一性。

<!-- bilingual-en:start -->
## When this formula helps
<!-- bilingual-en:end -->
## 何时用它

<!-- bilingual-en:start -->
The formula explains the connection between determinants and permutations, and it is useful when most products vanish. For a [[置换矩阵|permutation matrix]], each row and column contains exactly one $1$ and all other entries are $0$. Exactly one term survives, so the determinant is $+1$ or $-1$; in characteristic two those two scalars coincide. For example, choosing columns $(2,3,1)$ gives sign $+1$.
<!-- bilingual-en:end -->

公式解释了行列式与排列的联系；当大部分乘积为零时，它也便于手算。[[置换矩阵]]的每行每列恰有一个 $1$，其余为 $0$，所以只有一个非零项，行列式为 $+1$ 或 $-1$；特征二时这两个标量相同。例如，列选择 $(2,3,1)$ 给出符号 $+1$。

<!-- bilingual-en:start -->
The full formula has $n!$ terms, each with $n$ entries: for $n=4$ there are $24$ terms containing four entries apiece. Directly evaluating every term becomes expensive; [[消元计算行列式|elimination]] instead uses a number of arithmetic operations of order $n^3$ for a general dense matrix, while tracking row exchanges and any scaling. The formula is an exact identity, not a claim that factorial enumeration is a good numerical algorithm.
<!-- bilingual-en:end -->

完整公式有 $n!$ 项，每项含 $n$ 个矩阵元素；例如 $n=4$ 时有 $24$ 项，每项四个元素。逐项直接计算很快就会变得昂贵；对一般稠密矩阵，[[消元计算行列式|消元]]所需算术操作数量为 $n^3$ 量级，同时须记录换行和任何缩放的影响。公式是一条精确恒等式，并不表示枚举阶乘数量的项是好的数值算法。

<!-- bilingual-en:start -->
## Check the row and column labels
<!-- bilingual-en:end -->
## 易错边界

<!-- bilingual-en:start -->
Selecting one entry per row is insufficient if a column is repeated. Conversely, a legitimate permutation may select zero or negative entries: the permutation sign still comes from the column order, and then multiplies those entries. The familiar $3\times3$ diagonal mnemonic is a compact way to list its six terms; it does not extend directly to $4\times4$, where there are already $24$ terms. For $n=1$, the sole permutation has sign $+1$, giving $\det[a]=a$.
<!-- bilingual-en:end -->

如果重复选某一列，仅仅“每行选一个”仍不够。反过来，合法排列完全可以选到零或负元素：排列符号仍由列次序决定，再乘这些元素。常见的三阶斜线记忆法，只是紧凑地列出六项，不能直接推广到已有 $24$ 项的四阶情形。$n=1$ 时只有一个符号为 $+1$ 的排列，所以 $\det[a]=a$。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf#page=1|MIT 18.06SC Session 2.6, PDF pp.1–2]] and [[02_Least Squares, Determinants and Eigenvalues#Lecture：排列大公式|the permanent course section]]: row expansion, permutation selections, factorial term count and the three-dimensional mnemonic boundary. The six-entry table above is independently computed.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf#page=2|MIT Session 1.4, p.2, “How expensive is elimination?”]]: the cubic arithmetic-operation count used to contrast elimination with factorial enumeration.
- [Miličić, *Determinants*, §1.3 and §2.1, PDF pp.3–7](https://www.math.utah.edu/~milicic/Math_2270/determinants.pdf#page=3): inversion parity, swap parity, and the existence/uniqueness argument over $\mathbb R$ or $\mathbb C$; its column-based construction is written here by rows. The direct repeated-row pairing above also covers characteristic two.
- [Mathlib, *Determinant of a matrix*](https://leanprover-community.github.io/mathlib4_docs/Mathlib/LinearAlgebra/Matrix/Determinant/Basic.html): `det_apply'` gives the equivalent product $\prod_i a_{\sigma(i),i}$; replacing a permutation by its inverse gives the row-indexed form above with the same sign. `detRowAlternating` and `det_zero_of_row_eq` verify the all-field alternating construction.
<!-- bilingual-en:end -->

- MIT Session 2.6，PDF1–2及永久课程段：核对逐行展开、排列选择、阶乘项数和三阶记忆法的适用范围；正文六项表独立计算。
- MIT Session 1.4，第2页“How expensive is elimination?”：核对消元的三次量级算术操作数，用于与阶乘枚举比较。
- Miličić 第1.3节和第2.1节，PDF3–7：核对逆序奇偶性、交换次数奇偶性及实数或复数域上的存在唯一性论证；正文将其按列构造改写为按行，并直接配对重复行项以覆盖特征二。
- Mathlib 的 `det_apply'` 写成等价乘积 $\prod_i a_{\sigma(i),i}$；用逆排列重编号且保持符号，即得正文按行指标的形式。`detRowAlternating` 和 `det_zero_of_row_eq` 核对适用于任意域的交替构造。
