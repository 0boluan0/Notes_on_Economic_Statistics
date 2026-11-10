---
aliases:
  - Ax as a linear combination of columns
  - 列组合解释
student_os: knowledge-atom
atom_id: LA-SYS-001
atom_set: linear-systems-four-subspaces
atom_type: identity
status: source-checked
related:
  - "[[列空间]]"
  - "[[表示矩阵的列]]"
  - "[[张成]]"
leads_to:
  - "[[方程组解分类]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

<!-- bilingual-en:start -->
# Ax as a combination of columns
<!-- bilingual-en:end -->
*Ax 的列组合解释*

<!-- bilingual-en:start -->
Multiplying a matrix by a vector can be read as building one output from several available columns. Each entry of the input tells us how much of one whole column to use; we scale those columns and add them. This is a [[张成|linear combination]], using ordinary scalar multiplication and vector addition.
<!-- bilingual-en:end -->

矩阵乘向量，可以理解为用几根现成的列拼出一个输出。输入中的每个数告诉我们取相应整列的多少倍；把各列数乘后相加，就是一个[[张成|线性组合]]，只使用通常的数乘与向量加法。

<!-- bilingual-en:start -->
For example, take a matrix with three rows and two columns, and an input with two entries:
<!-- bilingual-en:end -->

例如，取一个三行两列的矩阵，以及一个含两个分量的输入：

$$A=\begin{pmatrix}1&2\\0&1\\1&-1\end{pmatrix},\qquad
x=\begin{pmatrix}3\\-1\end{pmatrix}.$$

<!-- bilingual-en:start -->
The first input entry asks for three copies of the first column. The second asks for minus one copy of the second column, meaning subtract that column:
<!-- bilingual-en:end -->

第一个输入分量要求取第一列的三倍，第二个要求取第二列的负一倍，也就是减去第二列：

$$Ax=3\begin{pmatrix}1\\0\\1\end{pmatrix}
-\begin{pmatrix}2\\1\\-1\end{pmatrix}
=\begin{pmatrix}1\\-1\\4\end{pmatrix}.$$

<!-- bilingual-en:start -->
There are two adjustable coefficients because there are two columns. The result has three entries because each column has three entries. Coefficients need not be positive or sum to one: this is a linear combination, not necessarily an average.
<!-- bilingual-en:end -->

有两列，所以需要两个可调系数；每列有三个分量，所以结果也有三个分量。系数不必为正，也不必加起来等于一：这里是线性组合，不一定是平均。

<!-- bilingual-en:start -->
## One coefficient for each whole column
<!-- bilingual-en:end -->
*每个系数乘一整列*

<!-- bilingual-en:start -->
Let $A=[a_1\ \cdots\ a_n]\in\mathbb F^{m\times n}$, where $m,n$ are positive integers and $\mathbb F$ is the chosen scalar field, usually $\mathbb R$ or $\mathbb C$. The vector $a_j$ is column $j$, containing $m$ entries. Write $x=(x_1,\ldots,x_n)^T$; the superscript $T$ means [[矩阵转置|transpose]], turning the listed row into a column. Then
<!-- bilingual-en:end -->

设 $A=[a_1\ \cdots\ a_n]\in\mathbb F^{m\times n}$，其中 $m,n$ 是正整数，$\mathbb F$ 为所选标量域，通常是 $\mathbb R$ 或 $\mathbb C$。$a_j$ 表示第 $j$ 列，包含 $m$ 个分量。写成 $x=(x_1,\ldots,x_n)^T$，上标 $T$ 表示[[矩阵转置|转置]]，把所列的一行变成一列。于是

> [!important] The column-combination identity · 列组合恒等式
> $$\boxed{Ax=x_1a_1+\cdots+x_na_n=\sum_{j=1}^{n}x_ja_j.}$$
>
> <!-- bilingual-en:start -->
> The index $j$ selects a column and its coefficient. Every term on the right is an $m$-entry vector, so they can be added. No independence, square shape or invertibility assumption is required.
> <!-- bilingual-en:end -->
>
> 指标 $j$ 选择一列及其系数；右边每一项都是含 $m$ 个分量的向量，可以相加。不要求各列无关，不要求方阵，也不要求可逆。

<!-- bilingual-en:start -->
To prove the identity from [[矩阵乘法|matrix multiplication]], call the entry in row $i$, column $j$ of $A$ by $a_{ij}$. The $i$th entry of $Ax$ is $\sum_{j=1}^n a_{ij}x_j$. The $i$th entry of $\sum_jx_ja_j$ is $\sum_jx_ja_{ij}$, the same scalar. This holds for every row $i=1,\ldots,m$, so the two vectors are equal. The row calculation and column calculation are two ways of collecting the same terms.
<!-- bilingual-en:end -->

从[[矩阵乘法]]定义即可证明：把 $A$ 第 $i$ 行、第 $j$ 列的元素记作 $a_{ij}$。$Ax$ 的第 $i$ 个分量为 $\sum_{j=1}^n a_{ij}x_j$，而 $\sum_jx_ja_j$ 的第 $i$ 个分量为 $\sum_jx_ja_{ij}$，二者相同。对每行 $i=1,\ldots,m$ 都成立，所以两个向量相等。按行算和按列算，只是对同一批项采取不同的组织方式。

<!-- bilingual-en:start -->
The identity also works over the complex numbers, without conjugating either factor. For instance, with $i^2=-1$, the row matrix $A=(1\ \ i)$ and input $x=(i,1)^T$ give $Ax=i+i=2i$. Ordinary matrix multiplication here is different from forming a [[内积|Hermitian inner product]].
<!-- bilingual-en:end -->

这个恒等式在复数域也成立，不需要对任何因子取共轭。例如 $i^2=-1$ 时，行矩阵 $A=(1\ \ i)$ 与输入 $x=(i,1)^T$ 给出 $Ax=i+i=2i$。这里做的是通常矩阵乘法，与形成 [[内积|Hermitian 内积]]不同。

<!-- bilingual-en:start -->
## From a given input to all possible outputs
<!-- bilingual-en:end -->
*从一个输入走到全部可能输出*

<!-- bilingual-en:start -->
Fix $A$ and now allow every coefficient $x_j$ to vary. The outputs form the [[列空间|column space]], which is the [[张成|span]] of the columns:
<!-- bilingual-en:end -->

固定 $A$，让所有系数 $x_j$ 自由变化，所得输出组成[[列空间]]，也就是各列的[[张成]]：

$$C(A)=\{Ax:x\in\mathbb F^n\}
=\left\{\sum_{j=1}^{n}x_ja_j:x_1,\ldots,x_n\in\mathbb F\right\}.$$

<!-- bilingual-en:start -->
Thus solving $Ax=b$ asks whether the target $b$ can be assembled from these columns. In the opening example, every output is $(s+2t,t,s-t)^T$. The target $(1,-1,4)^T$ is attainable with $(s,t)=(3,-1)$. The target $(0,0,1)^T$ is not: its second entry forces $t=0$, its first then forces $s=0$, and its third would have to be $0$. Choosing different coefficients cannot reach a target outside $C(A)$.
<!-- bilingual-en:end -->

所以，解 $Ax=b$ 就是在问目标 $b$ 能否由这些列拼出来。开头例子的每个输出都是 $(s+2t,t,s-t)^T$。目标 $(1,-1,4)^T$ 可由 $(s,t)=(3,-1)$ 得到；目标 $(0,0,1)^T$ 却不行：第二分量迫使 $t=0$，第一分量再迫使 $s=0$，此时第三分量只能是 $0$。改变系数也到不了 $C(A)$ 之外的目标。

<!-- bilingual-en:start -->
## Keep the input and output roles separate
<!-- bilingual-en:end -->
## 边界

<!-- bilingual-en:start -->
The input $x$ belongs to $\mathbb F^n$, while $Ax$ and the columns belong to $\mathbb F^m$. The coefficient list is itself an input vector; it is not generally the output's coordinate list. Even when $m=n$, equality of dimensions does not identify their roles. The identity matrix happens to leave coordinates unchanged, but a general matrix does not.
<!-- bilingual-en:end -->

输入 $x$ 属于 $\mathbb F^n$，而 $Ax$ 和各列属于 $\mathbb F^m$。系数列表本身也是输入向量，但通常不是输出向量的坐标列表。即使 $m=n$，维数相同也不等于角色相同。单位矩阵恰好保持坐标不变，一般矩阵却不会。

<!-- bilingual-en:start -->
If $e_j$ is the standard input vector with entry $1$ at position $j$ and $0$ elsewhere, the identity gives $Ae_j=a_j$: activating only input coordinate $j$ selects column $j$. This is the standard-coordinate version of [[表示矩阵的列|the column rule for a linear map]]. The columns need not themselves be a basis; whether a reachable output has one or several coefficient representations is a separate [[方程组解分类|question about the solution set]].
<!-- bilingual-en:end -->

若 $e_j$ 是第 $j$ 个分量为 $1$、其他分量为 $0$ 的标准输入向量，恒等式给出 $Ae_j=a_j$：只开启第 $j$ 个输入坐标，就选中第 $j$ 列。这是[[表示矩阵的列|线性映射列规则]]的标准坐标版本。各列本身不必构成一组基；一个可达输出有一套还是多套系数表示，属于另一项[[方程组解分类|解集问题]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.1sum.pdf#page=2|MIT 18.06SC Session 1.1, pp.2–3]]: column and row readings of $Ax$, and coefficients as weights on complete columns.
- [Axler, *Linear Algebra Done Right*, fourth edition, 3.49–3.50, PDF pp.89–90](https://linear.axler.net/LADR4e.pdf#page=90): the identity over the scalar field and its entrywise proof.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.6sum.pdf#page=1|MIT Session 1.6, pp.1–2]]: all attainable outputs and the solvability interpretation. Both displayed products and the unreachable-target calculation are checked directly.
<!-- bilingual-en:end -->

- MIT Session 1.1，第2–3页：核对 $Ax$ 的列、行两种读法，以及用系数乘整列。
- Axler 3.49–3.50，PDF89–90：核对所选标量域上的恒等式及逐分量证明。
- MIT Session 1.6，第1–2页：核对全部可达输出与可解性解释；正文两个乘积和不可达目标均直接计算。
