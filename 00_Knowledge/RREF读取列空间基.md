---
aliases:
  - Reading a column-space basis from RREF
student_os: knowledge-atom
atom_id: LA-SYS-025
atom_set: linear-systems-four-subspaces
atom_type: method
status: source-checked
requires:
  - "[[行最简形]]"
  - "[[基]]"
  - "[[主元与自由变量]]"
  - "[[列空间]]"
  - "[[行操作与四个子空间]]"
related:
  - "[[RREF读取子空间基]]"
  - "[[四个基本子空间]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

<!-- bilingual-en:start -->
# A column-space basis from RREF
<!-- bilingual-en:end -->
*从行最简形读取列空间基*

<!-- bilingual-en:start -->
To describe every output a matrix can produce, we want a few independent columns that generate all its other columns. Row reduction tells us **which column positions to choose**. The vectors we keep must come from the original matrix, because row operations can move its outputs to a different space.
<!-- bilingual-en:end -->

要描述矩阵能够产生的全部输出，我们希望选出少数几个无关列，让它们能够生成其余所有列。行化简告诉我们**应选哪些列号**。实际保留的向量要从原矩阵取出，因为行操作可能把输出移到另一个空间。

<!-- bilingual-en:start -->
## Follow the positions back to the original matrix
<!-- bilingual-en:end -->
*沿着列号返回原矩阵*

<!-- bilingual-en:start -->
The [[列空间|column space]] $C(A)$ consists of all linear combinations of the columns of $A$. A [[基|basis]] must both generate this entire space and be [[线性无关|linearly independent]]: only zero coefficients can combine its vectors into zero. Consider
<!-- bilingual-en:end -->

[[列空间]] $C(A)$ 是 $A$ 各列的全部线性组合。一组[[基]]既要生成整个空间，又要[[线性无关]]：只有全零系数才能把这些向量组合成零。考虑

$$
A=\begin{pmatrix}1&0&1\\0&1&1\\1&1&2\end{pmatrix}
\quad\longrightarrow\quad
R=\begin{pmatrix}1&0&1\\0&1&1\\0&0&0\end{pmatrix}.
$$

<!-- bilingual-en:start -->
Subtract the first two rows from the third to obtain $R$. It is the [[行最简形|reduced row echelon form]], abbreviated RREF. Each nonzero row starts with a leading $1$, called a [[主元|pivot]], and each pivot is the only nonzero entry in its column. Here the pivot columns are columns $1$ and $2$. Take those positions from $A$:
<!-- bilingual-en:end -->

用第三行减去前两行，就得到 $R$。它是[[行最简形]]，英文缩写为 RREF。每个非零行的首个非零元素为 $1$，称为[[主元]]；每个主元又是其所在列唯一的非零元素。这里的主元列为第 $1$、$2$ 列。按这两个位置从 $A$ 取出

$$a_1=(1,0,1)^T,\qquad a_2=(0,1,1)^T.$$

<!-- bilingual-en:start -->
The superscript $T$ means [[矩阵转置|transpose]], writing the coordinate lists as columns. These two vectors are independent because the first two coordinates of $s a_1+t a_2=0$ force $s=t=0$. The remaining column is $a_3=a_1+a_2$, so they also span all columns. Thus
<!-- bilingual-en:end -->

上标 $T$ 表示[[矩阵转置|转置]]，把坐标写成列。由 $s a_1+t a_2=0$ 的前两个坐标可得 $s=t=0$，因此两列无关；剩下的列满足 $a_3=a_1+a_2$，所以它们又能张成全部列。因此

$$C(A)=\{(s,t,s+t)^T:s,t\in\mathbb R\},\qquad
\{a_1,a_2\}\text{ is a basis of }C(A).$$

> [!warning] The reduced columns describe a different output space · 化简后的列可能属于另一个空间
> <!-- bilingual-en:start -->
> The first two columns of $R$ span $\{(s,t,0)^T\}$, a different plane. For instance, $(1,0,0)^T$ belongs to $C(R)$ but not to $C(A)$, whose third coordinate must equal the sum of the first two. A shared pivot pattern does not make the column spaces equal.
> <!-- bilingual-en:end -->
>
> $R$ 的前两列张成 $\{(s,t,0)^T\}$，这是另一个平面。例如，$(1,0,0)^T$ 属于 $C(R)$，却不属于要求第三坐标等于前两坐标之和的 $C(A)$。主元位置相同，不代表列空间相同。

<!-- bilingual-en:start -->
## Why the rule always works
<!-- bilingual-en:end -->
*为什么这条规则成立*

<!-- bilingual-en:start -->
Let $A\in\mathbb F^{m\times n}$, where $m,n$ are positive integers and $\mathbb F$ is a field, such as $\mathbb R$ or $\mathbb C$. Exact elementary row operations give $R=EA$ with an [[逆矩阵|invertible]] $m\times m$ matrix $E$. Writing the columns as $a_j$ and $r_j$ gives $r_j=Ea_j$. Therefore every coefficient relation is preserved in both directions:
<!-- bilingual-en:end -->

设 $A\in\mathbb F^{m\times n}$，其中 $m,n$ 为正整数，$\mathbb F$ 是一个域，例如 $\mathbb R$ 或 $\mathbb C$。精确初等行操作给出 $R=EA$，$E$ 是[[逆矩阵|可逆]]的 $m\times m$ 矩阵。把两矩阵的列分别记为 $a_j,r_j$，则 $r_j=Ea_j$，所以系数关系在两个方向都保持：

$$\sum_{j=1}^n c_j a_j=0
\quad\Longleftrightarrow\quad
\sum_{j=1}^n c_j r_j=0.$$

<!-- bilingual-en:start -->
One implication follows by multiplying by $E$; the other follows by multiplying by $E^{-1}$. This preserves relations among columns, rather than each column vector itself.
<!-- bilingual-en:end -->

一个方向是左乘 $E$，反方向是左乘 $E^{-1}$。保持的是各列之间的关系，不是每一列向量本身。

<!-- bilingual-en:start -->
Suppose the pivot positions of $R$ are $j_1,\ldots,j_r$. Its pivot columns are the first $r$ standard unit columns, with zeros in the remaining rows, so they are independent. Every other column is their combination: its entries in the $r$ nonzero rows give the coefficients. The preserved relations show that the corresponding original columns are also independent and generate every original column.
<!-- bilingual-en:end -->

设 $R$ 的主元列号为 $j_1,\ldots,j_r$。这些主元列就是前 $r$ 根标准单位列，其余行坐标为零，因此无关。每根其他列都由它们组合而成，系数就是它在 $r$ 个非零行中的元素。列关系保持，说明对应的原列也无关，并且能够生成原矩阵的每一列。

> [!important] Keep the original indexed columns · 保留对应的原列
> $$\boxed{\{a_{j_1},\ldots,a_{j_r}\}\text{ is a basis of }C(A).}$$
>
> <!-- bilingual-en:start -->
> Reduce a copy of $A$, record its pivot-column indices, and return to $A$ for the actual vectors. Their number $r$ is the [[矩阵秩|rank]], the dimension of the column space of attainable outputs.
> <!-- bilingual-en:end -->
>
> 化简 $A$ 的副本，记下主元列号，再回到 $A$ 取出向量。所取向量数 $r$ 就是[[矩阵秩|秩]]，也就是可达输出所成列空间的维数。

<!-- bilingual-en:start -->
## Boundaries of the procedure
<!-- bilingual-en:end -->
*使用边界*

<!-- bilingual-en:start -->
If $A=0$, there are no pivots. The selected list is empty and is a basis of the zero subspace $\{0\}$; the column space itself is not empty. If columns were permuted during a calculation, undo that permutation when interpreting the indices. In an augmented matrix $[A\mid b]$, the added target column is not a column of $A$ and must not be included in its column-space basis.
<!-- bilingual-en:end -->

若 $A=0$，就没有主元；选出的空列表是零子空间 $\{0\}$ 的一组基，列空间本身并不为空。若计算中交换过列，解释列号时要还原这次排列。增广矩阵 $[A\mid b]$ 中新增的目标列不属于 $A$，不能选入 $A$ 的列空间基。

<!-- bilingual-en:start -->
The proof is exact and works over the stated field. A very small computed entry is not automatically an exact zero; deciding which directions to discard under finite precision belongs to [[数值秩|numerical rank]]. For the row space, the rule is different: [[RREF读取行空间基|use the nonzero reduced rows themselves]].
<!-- bilingual-en:end -->

这是精确代数结论，在所声明的域上成立。计算所得的很小元素不自动等于精确的零；有限精度下舍弃哪些方向，属于[[数值秩]]的判断。行空间采用另一条规则：[[RREF读取行空间基|直接使用化简后的非零行]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf#page=2|MIT 18.06SC, independence, basis and dimension, p.2]]: pivot columns of the original matrix provide an independent spanning list.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf#page=2|MIT four-subspace summary, pp.1–2]]: row reduction can change the column space, while reversible row operations give $R=EA$. Its printed identity and elimination-matrix size labels on p.2 should be $m\times m$.
- [Axler, *Linear Algebra Done Right*, 3.52–3.57, PDF pp.91–92](https://linear.axler.net/LADR4e.pdf#page=91): column span, basis selection, ordinary transpose and the distinction between row and column rank. The preserved-relations proof and displayed plane example are verified directly.
<!-- bilingual-en:end -->

- MIT 无关、基与维数摘要第2页：核对原矩阵主元列既无关又张成的规则。
- MIT 四子空间摘要第1–2页：核对行化简可能改变列空间，以及可逆行操作给出 $R=EA$；第2页的单位阵、消元矩阵尺寸标签应为 $m\times m$。
- Axler 3.52–3.57，PDF91–92页：核对列张成、选基、普通转置与行列秩的区分；列关系保持的证明与平面算例直接核验。
