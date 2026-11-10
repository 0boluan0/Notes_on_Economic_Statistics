---
aliases:
  - Reading a row-space basis from RREF
student_os: knowledge-atom
atom_id: LA-SYS-026
atom_set: linear-systems-four-subspaces
atom_type: method
status: source-checked
requires:
  - "[[行最简形]]"
  - "[[基]]"
  - "[[行空间]]"
related:
  - "[[RREF读取子空间基]]"
  - "[[四个基本子空间]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

<!-- bilingual-en:start -->
# A row-space basis from RREF
<!-- bilingual-en:end -->
*从行最简形读取行空间基*

<!-- bilingual-en:start -->
Row reduction removes redundant equations while keeping every coefficient row obtainable from the original rows. After reaching reduced row echelon form, the remaining nonzero rows give a compact, independent description of that same row space. Here we keep the reduced rows themselves.
<!-- bilingual-en:end -->

行化简去除冗余方程，同时保持由原系数行能够组合出的全部行。得到行最简形后，剩下的非零行就以无关向量描述了同一个行空间。这里直接保留化简后的行。

<!-- bilingual-en:start -->
## A dependent row disappears
<!-- bilingual-en:end -->
*冗余行消去后留下什么*

<!-- bilingual-en:start -->
The [[行空间|row space]] is the set of all linear combinations of a matrix's rows. A [[基|basis]] must span this entire set and have no redundant vector. Start with
<!-- bilingual-en:end -->

[[行空间]]包含矩阵各行的全部线性组合。一组[[基]]要张成这个集合，而且不能包含冗余向量。从下面的矩阵出发：

$$
A=\begin{pmatrix}1&2&0\\2&4&0\\0&0&1\end{pmatrix}
\quad\longrightarrow\quad
\begin{pmatrix}1&2&0\\0&0&0\\0&0&1\end{pmatrix}
\quad\longrightarrow\quad
R=\begin{pmatrix}1&2&0\\0&0&1\\0&0&0\end{pmatrix}.
$$

<!-- bilingual-en:start -->
First subtract twice row $1$ from row $2$; then swap rows $2$ and $3$ to put the zero row last. The result is the [[行最简形|reduced row echelon form]], or RREF. Its nonzero rows are
<!-- bilingual-en:end -->

先用第二行减去第一行的两倍，再交换第二、三行，把零行放到末尾。结果是[[行最简形]]，英文缩写为 RREF。它的非零行为

$$r_1=(1,2,0),\qquad r_2=(0,0,1).$$

<!-- bilingual-en:start -->
Their combinations are $(s,2s,t)$ for arbitrary real $s,t$, exactly the combinations of all original rows. They are [[线性无关|independent]]: if $(s,2s,t)=0$, the first and third coordinates force $s=t=0$. Thus $\{r_1,r_2\}$ is a basis of the original row space, a plane with two independent directions.
<!-- bilingual-en:end -->

它们的组合为 $(s,2s,t)$，$s,t$ 可取任意实数，这恰好也是原矩阵全部行的组合。它们[[线性无关]]：若 $(s,2s,t)=0$，第一、三坐标就迫使 $s=t=0$。因此 $\{r_1,r_2\}$ 是原行空间的一组基，这个平面有两个独立方向。

> [!warning] Reduced row numbers do not select original rows · 化简后的行号不能直接用于选原行
> <!-- bilingual-en:start -->
> The nonzero rows of $R$ occupy positions $1$ and $2$, but the first two rows of $A$ are dependent and miss the direction $(0,0,1)$. Row operations mix and exchange rows. Use the actual nonzero rows of $R$; their positions are not a selection rule for rows of $A$.
> <!-- bilingual-en:end -->
>
> $R$ 的非零行位于第 $1$、$2$ 行，但 $A$ 的前两行相关，还缺少 $(0,0,1)$ 方向。行操作会混合、交换各行。应取 $R$ 的实际非零行，不能把其行号当作选择 $A$ 原行的规则。

<!-- bilingual-en:start -->
## Why spanning and independence both survive
<!-- bilingual-en:end -->
*为什么既能张成又没有冗余*

<!-- bilingual-en:start -->
Let $A\in\mathbb F^{m\times n}$ with positive integers $m,n$ and a field $\mathbb F$, such as $\mathbb R$ or $\mathbb C$. Exact elementary row operations give $R=EA$ with invertible $E$. Every row of $R$ is a combination of rows of $A$. Conversely, $A=E^{-1}R$ expresses every original row using rows of $R$. These two inclusions prove equality of their row spaces.
<!-- bilingual-en:end -->

设 $A\in\mathbb F^{m\times n}$，$m,n$ 为正整数，$\mathbb F$ 是一个域，例如 $\mathbb R$ 或 $\mathbb C$。精确初等行操作给出 $R=EA$，其中 $E$ 可逆。$R$ 的每行由 $A$ 的各行组合得到；反过来，$A=E^{-1}R$ 又把每条原行表示为 $R$ 的各行组合。两次包含证明了行空间相等。

<!-- bilingual-en:start -->
Removing zero rows changes no combinations. The remaining rows are also independent. Each has a leading $1$, called a [[主元|pivot]], and the RREF rule makes all other entries in that pivot column zero. In a zero combination of the nonzero rows, looking at the pivot column belonging to row $i$ therefore isolates its coefficient $c_i$. It must be zero. Repeating for every pivot column makes every coefficient zero.
<!-- bilingual-en:end -->

去掉零行不会改变能得到的组合。剩下的行也无关：每行的首个非零元素为 $1$，称为[[主元]]，而 RREF 要求该主元所在列的其他元素全为零。若这些非零行的某个组合为零，查看第 $i$ 行对应的主元列，就单独读出系数 $c_i$，所以它必须为零。逐一查看主元列，便得到全部系数为零。

> [!important] Use all the nonzero reduced rows · 取全部非零化简行
> <!-- bilingual-en:start -->
> The nonzero rows of $R$ span the same row space as $A$ and are independent. Hence they form a basis. Each row has $n$ coordinates; the number of selected rows counts the dimension, not the length of each row.
> <!-- bilingual-en:end -->
>
> $R$ 的全部非零行既张成与 $A$ 相同的行空间，又线性无关，因此构成一组基。每行有 $n$ 个坐标；取出的行数表示空间维数，不是每行的坐标数。

<!-- bilingual-en:start -->
## Complex entries and zero rows
<!-- bilingual-en:end -->
*复数与零行*

<!-- bilingual-en:start -->
For complex matrices, the algebraic row-space rule still uses the rows of $R$ without conjugation. If the requested space is instead the geometric row space $C(A^*)$ for standard Hermitian orthogonality, take the **conjugate transpose** of each selected row and regard it as a column. Here $^*$ means [[矩阵转置|transpose followed by entrywise complex conjugation]].
<!-- bilingual-en:end -->

对复矩阵，代数行空间仍直接使用 $R$ 的行，不取共轭。若要求的是标准 Hermitian 正交关系中的几何行空间 $C(A^*)$，则把每条所选行取**共轭转置**，写成列。这里 $^*$ 表示[[矩阵转置|先转置，再逐元素取复共轭]]。

<!-- bilingual-en:start -->
For example, $A=(1\ \ i)$, where $i^2=-1$, is already in RREF. Its algebraic row basis is $\{(1,i)\}$; a column basis of $C(A^*)$ is $\{(1,-i)^T\}$. These different coordinate lists serve different conventions. More generally, since $R^*=A^*E^*$ and $A^*=R^*(E^{-1})^*$, the two geometric column spaces also agree. Conjugating and transposing an independent row list preserves independence: conjugate the coefficients in a proposed zero relation to recover the original row relation.
<!-- bilingual-en:end -->

例如 $A=(1\ \ i)$，其中 $i^2=-1$，它已经是 RREF。其代数行空间基为 $\{(1,i)\}$，而 $C(A^*)$ 的列向量基为 $\{(1,-i)^T\}$。不同的坐标列表对应不同约定。一般地，$R^*=A^*E^*$ 与 $A^*=R^*(E^{-1})^*$ 说明这两个几何列空间也相同。无关行组取共轭转置后仍无关：把假设的零组合系数取共轭，就还原为原行组的零组合。

<!-- bilingual-en:start -->
If $A=0$, then $R=0$ and the selected list is empty. It is a basis of $\{0\}$, which contains the zero vector even though its basis has no members. A zero row cannot be included in an independent list. Finally, [[RREF读取列空间基|a column-space basis uses a different rule]]: use pivot-column positions to select original columns, because row operations need not preserve the column space.
<!-- bilingual-en:end -->

若 $A=0$，则 $R=0$，选出的列表为空。空列表是 $\{0\}$ 的一组基；空间包含零向量，而基没有任何成员。零行不能放入无关组。最后，[[RREF读取列空间基|列空间的选基规则不同]]：根据主元列号选择原列，因为行操作未必保持列空间。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf#page=2|MIT 18.06SC four-subspace summary, p.2]]: reversibility proves equal row spans, and the nonzero reduced rows form a basis. The proof above makes the pivot-column independence step explicit.
- [Axler, *Linear Algebra Done Right*, 3.52–3.57, PDF pp.91–92; 7.6–7.9, PDF pp.245–246](https://linear.axler.net/LADR4e.pdf#page=91): algebraic row spans over real or complex scalars, ordinary transpose, and the separate Hermitian-adjoint convention. The displayed row operations, counterexample to original-row selection and complex basis are checked directly.
<!-- bilingual-en:end -->

- MIT 四子空间摘要第2页：核对可逆性保证行张成相同，以及化简后的非零行给出基；正文把主元列保证无关的步骤写全。
- Axler 3.52–3.57，PDF91–92页；7.6–7.9，PDF245–246页：核对实数或复数域上的代数行张成、普通转置及独立的 Hermitian 伴随约定；行操作算例、不能按行号选原行的反例与复数基直接核验。
