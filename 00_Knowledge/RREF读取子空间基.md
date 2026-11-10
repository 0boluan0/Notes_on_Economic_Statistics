---
aliases:
  - Choosing subspace bases from RREF
  - Four asymmetric routes from RREF to fundamental-subspace bases
  - 从 RREF 读四个基本子空间
student_os: knowledge-atom
atom_id: LA-SYS-014
atom_set: linear-systems-four-subspaces
atom_type: method-distinction
status: source-checked
requires:
  - "[[行最简形]]"
  - "[[基]]"
  - "[[行操作与四个子空间]]"
  - "[[主元与自由变量]]"
  - "[[零空间]]"
  - "[[四个基本子空间]]"
leads_to:
  - "[[RREF读取列空间基]]"
  - "[[RREF读取行空间基]]"
  - "[[RREF读取零空间基]]"
  - "[[消元追踪左零空间基]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

<!-- bilingual-en:start -->
# Choosing subspace bases from RREF
<!-- bilingual-en:end -->
*从行最简形选择子空间基的读取路线*

<!-- bilingual-en:start -->
A row-reduced matrix contains useful information about all four fundamental subspaces, but the information takes different forms. Sometimes the reduced vectors themselves are the answer; sometimes they only identify positions in the original matrix; sometimes an equation or a record of row operations is needed. First name the space being requested, then decide **which vectors to keep and from where**.
<!-- bilingual-en:end -->

行化简后的矩阵包含四个基本子空间的信息，但信息的形式不同：有时直接取化简后的向量，有时只用它定位原矩阵中的向量，有时需要解方程或保留行操作记录。先说清题目要哪个空间，再决定**从哪里取哪些向量**。

<!-- bilingual-en:start -->
A [[基|basis]] must both span the requested space and be independent. Merely finding the right number of vectors, or finding vectors of the right length, does not establish either requirement. Here the [[四个基本子空间|four spaces]] are for a real $m\times n$ matrix $A$: column and left-null vectors have $m$ coordinates, while row and nullspace vectors have $n$. Write $C(B)$ for the column span of a matrix $B$, and $N(B)=\{v:Bv=0\}$ for its nullspace.
<!-- bilingual-en:end -->

一组[[基]]必须既张成所求空间，又线性无关。只找到正确数量或正确长度的向量，并不能证明这两个要求。这里先讨论实 $m\times n$ 矩阵 $A$ 的[[四个基本子空间]]：列空间和左零空间中的向量有 $m$ 个坐标，行空间和零空间中的向量有 $n$ 个坐标。用 $C(B)$ 表示矩阵 $B$ 的列张成，用 $N(B)=\{v:Bv=0\}$ 表示其零空间。

<!-- bilingual-en:start -->
## Choose the evidence before copying vectors
<!-- bilingual-en:end -->
## 使用顺序

<!-- bilingual-en:start -->
Keep the original $A$, and use exact reversible row operations to obtain its [[行最简形|reduced row echelon form]], abbreviated RREF, $R=EA$. The square matrix $E$ records the operations and is invertible. Let $r$ be the number of [[主元|pivots]], equal to the [[矩阵秩|rank]]. A pivot is the leading $1$ in a nonzero row of $R$, alone in its column. The following table selects the appropriate route; each linked method supplies its detailed construction and proof.
<!-- bilingual-en:end -->

保留原矩阵 $A$，通过精确的可逆行操作得到其[[行最简形]]，英文缩写为 RREF，记作 $R=EA$。方阵 $E$ 记录操作且可逆。令 $r$ 为[[主元]]数，也就是[[矩阵秩|秩]]。主元是 $R$ 非零行开头的 $1$，且其所在列的其他元素为零。下面选择对应路线；各链接方法分别给出详细构造与证明。

<!-- bilingual-en:start -->
| Requested original space | What to use | Why this is the right evidence |
|---|---|---|
| $C(A)\subseteq\mathbb R^m$ | [[RREF读取列空间基\|Pivot-column positions in $R$, actual columns from $A$]] | The relations among columns survive; their entries can change. |
| $C(A^T)\subseteq\mathbb R^n$ | [[RREF读取行空间基\|All nonzero rows of $R$, transposed if written as columns]] | Reversible row combinations preserve the row span. |
| $N(A)\subseteq\mathbb R^n$ | [[RREF读取零空间基\|Solve $Rx=0$ and activate the free variables one at a time]] | The homogeneous solution set is unchanged. |
| $N(A^T)\subseteq\mathbb R^m$ | [[消元追踪左零空间基\|Transpose the rows of $E$ corresponding to zero rows of $R$]]; alternatively solve $A^Ty=0$ | The record supplies cancellation weights for the original rows. |
<!-- bilingual-en:end -->

| 所求原空间 | 使用什么 | 为什么这样取证据 |
|---|---|---|
| $C(A)\subseteq\mathbb R^m$ | [[RREF读取列空间基\|由 $R$ 找主元列号，返回 $A$ 取实际列]] | 各列之间的关系保持，列向量本身可能改变。 |
| $C(A^T)\subseteq\mathbb R^n$ | [[RREF读取行空间基\|取 $R$ 全部非零行，按列写时转置]] | 可逆行组合保持行张成。 |
| $N(A)\subseteq\mathbb R^n$ | [[RREF读取零空间基\|解 $Rx=0$，依次激活各自由变量]] | 齐次解集没有改变。 |
| $N(A^T)\subseteq\mathbb R^m$ | [[消元追踪左零空间基\|取 $R$ 零行对应的 $E$ 行并转置]]；也可直接解 $A^Ty=0$ | 记录给出原有各行的抵消权重。 |

<!-- bilingual-en:start -->
The superscript $T$ means [[矩阵转置|transpose]]. These choices follow from the [[行操作与四个子空间|effect of row operations]]: $N(R)=N(A)$ and the row spans agree, but $C(R)=E\,C(A)$ and $N(R^T)=E^{-T}N(A^T)$. Thus equal dimensions do not justify copying all four bases from $R$.
<!-- bilingual-en:end -->

上标 $T$ 表示[[矩阵转置|转置]]。这些选择来自[[行操作与四个子空间|行操作的作用]]：$N(R)=N(A)$，行张成也相同；但 $C(R)=E\,C(A)$、$N(R^T)=E^{-T}N(A^T)$。因此维数保持，并不允许从 $R$ 直接照抄全部四组基。

<!-- bilingual-en:start -->
## One matrix, four different selections
<!-- bilingual-en:end -->
*一个矩阵，四种不同选取*

<!-- bilingual-en:start -->
Subtract the first two rows from the third:
<!-- bilingual-en:end -->

用第三行减去前两行：

$$A=\begin{pmatrix}1&0&1&0\\0&1&1&0\\1&1&2&0\end{pmatrix},\qquad
E=\begin{pmatrix}1&0&0\\0&1&0\\-1&-1&1\end{pmatrix},\qquad
R=EA=\begin{pmatrix}1&0&1&0\\0&1&1&0\\0&0&0&0\end{pmatrix}.$$

<!-- bilingual-en:start -->
**Column space:** the pivot positions are 1 and 2, so keep the original columns $(1,0,1)^T$ and $(0,1,1)^T$. They are independent, and the third original column is their sum while the fourth is zero. Their span is all of $C(A)$. The reduced columns $(1,0,0)^T,(0,1,0)^T$ instead span a different plane.
<!-- bilingual-en:end -->

**列空间：**主元列号为 $1,2$，所以保留原列 $(1,0,1)^T$、$(0,1,1)^T$。两列无关，原第三列是它们之和，第四列为零，因此它们张成整个 $C(A)$。化简列 $(1,0,0)^T,(0,1,0)^T$ 张成的是另一个平面。

<!-- bilingual-en:start -->
**Row space:** take the two nonzero reduced rows $(1,0,1,0)$ and $(0,1,1,0)$. Their first two coordinates establish independence; every original row is their combination. These are four-coordinate coefficient lists, unlike the three-coordinate column-space vectors.
<!-- bilingual-en:end -->

**行空间：**取化简后的两条非零行 $(1,0,1,0)$、$(0,1,1,0)$。前两个坐标说明它们无关，全部原行又都是它们的组合。它们是四坐标的系数列表，不是列空间中的三坐标向量。

<!-- bilingual-en:start -->
**Nullspace:** $Rx=0$ gives $x_1=-x_3$ and $x_2=-x_3$. Let the two free variables be $x_3=s,x_4=t$. Then
<!-- bilingual-en:end -->

**零空间：**$Rx=0$ 给出 $x_1=-x_3$、$x_2=-x_3$。把两个自由变量记为 $x_3=s,x_4=t$，则

$$x=s(-1,-1,1,0)^T+t(0,0,0,1)^T,\qquad s,t\in\mathbb R.$$

<!-- bilingual-en:start -->
The two coefficient vectors are a basis: every homogeneous solution has this form, and their third and fourth coordinates force a zero combination to have zero coefficients.
<!-- bilingual-en:end -->

这两个系数向量是一组基：每个齐次解都有这种形式，而第三、四坐标保证零组合只能使用全零系数。

<!-- bilingual-en:start -->
**Left nullspace:** row 3 of $R$ is zero, so use row 3 of $E$, giving the column $(-1,-1,1)^T$. Its weighted row combination is $-A_{1:}-A_{2:}+A_{3:}=0$, where $A_{i:}$ denotes row $i$. Conversely, $A^Ty=0$ requires $y_1+y_3=y_2+y_3=0$, so every left-null vector is a multiple of this one. The vector $(0,0,1)^T$ obtained from $N(R^T)$ is not an original cancellation weight: its transpose selects the nonzero third row of $A$.
<!-- bilingual-en:end -->

**左零空间：**$R$ 的第三行为零，因此使用 $E$ 的第三行，转置后得到 $(-1,-1,1)^T$。它的加权行组合为 $-A_{1:}-A_{2:}+A_{3:}=0$，其中 $A_{i:}$ 表示第 $i$ 行。反过来，$A^Ty=0$ 要求 $y_1+y_3=y_2+y_3=0$，所以每个左零向量都是它的倍数。从 $N(R^T)$ 取得的 $(0,0,1)^T$ 并不是原行的抵消权重：它的转置会选中 $A$ 非零的第三行。

<!-- bilingual-en:start -->
## What must be retained during elimination
<!-- bilingual-en:end -->
*消元时需要保留什么*

<!-- bilingual-en:start -->
If only $R$ survives, the original row space and nullspace can still be recovered, but the original column and left-null spaces generally cannot. For instance, both $A_1=(1,1)^T$ and $A_2=(1,0)^T$ have RREF $(1,0)^T$, yet their column spaces are different lines and their left nullspaces are respectively the lines through $(1,-1)^T$ and $(0,1)^T$. The same reduced matrix cannot identify which original pair was present.
<!-- bilingual-en:end -->

若只剩下 $R$，仍能恢复原行空间和原零空间，但通常不能恢复原列空间与原左零空间。例如 $A_1=(1,1)^T$、$A_2=(1,0)^T$ 的 RREF 都是 $(1,0)^T$，列空间却是不同直线，左零空间分别是 $(1,-1)^T$、$(0,1)^T$ 所在直线。同一个化简结果无法说明原来是哪一对空间。

<!-- bilingual-en:start -->
Keeping $A$ is enough to solve $A^Ty=0$ separately. To reuse the same elimination, append the [[单位矩阵|identity]] and record $[A\mid I_m]\to[R\mid E]$; both $I_m$ and $E$ are $m\times m$. In fact, keeping $R$ and $E$ also recovers $A=E^{-1}R$. Recording is therefore a choice about which evidence remains available, not an additional assumption on $A$.
<!-- bilingual-en:end -->

保留 $A$ 后，可以另外解 $A^Ty=0$。若要复用同一次消元，则附上[[单位矩阵]]并记录 $[A\mid I_m]\to[R\mid E]$，其中 $I_m$、$E$ 都是 $m\times m$。实际上，保留 $R$ 与 $E$ 也能通过 $A=E^{-1}R$ 恢复 $A$。是否记录，是关于保留什么证据的选择，不是对 $A$ 增加新假设。

<!-- bilingual-en:start -->
For a zero-dimensional space, use an empty basis, not a zero vector as a basis member. If $A=0$, both column and row spaces are $\{0\}$, while $N(A)=\mathbb R^n$ and $N(A^T)=\mathbb R^m$. More generally the four basis lengths are $r,r,n-r,m-r$. These counts check an answer but do not replace membership, spanning and independence checks. Exact RREF decisions also differ from [[数值秩|tolerance-based numerical-rank]] decisions.
<!-- bilingual-en:end -->

空间维数为零时，使用空基，不能把零向量当成基成员。若 $A=0$，列空间和行空间均为 $\{0\}$，而 $N(A)=\mathbb R^n$、$N(A^T)=\mathbb R^m$。一般四组基的长度为 $r,r,n-r,m-r$。这些计数可用来检查答案，却不能替代成员归属、张成与无关性的核验。精确 RREF 的判断也不同于[[数值秩|按容差决定的数值秩]]。

<!-- bilingual-en:start -->
For complex matrices, state the convention before selecting vectors. The ordinary algebraic row space uses the nonzero reduced rows with no conjugation. For the Hermitian geometric space $C(A^*)$, use the [[矩阵转置|conjugate transpose]] $^*$ to turn those rows into columns. Likewise, a recorded row $w^T$ satisfying $w^TA=0$ gives $w\in N(A^T)$; the corresponding vector in $N(A^*)$ is $\overline w$, because $(\overline w)^*=w^T$. Column and right-nullspace selection otherwise use the same complex-field algebra.
<!-- bilingual-en:end -->

对复矩阵，应先声明约定再选向量。普通代数行空间直接使用非零化简行，不取共轭；若求 Hermitian 几何空间 $C(A^*)$，则用[[矩阵转置|共轭转置]] $^*$ 把这些行写成列。同样，记录行 $w^T$ 满足 $w^TA=0$ 时，给出 $w\in N(A^T)$；对应的 $N(A^*)$ 向量为 $\overline w$，因为 $(\overline w)^*=w^T$。列空间与右零空间的选取则仍使用同样的复数域代数。

> [!question]- Does the reduced left kernel answer the original question? · 化简后的左核回答原问题了吗？
> <!-- bilingual-en:start -->
> Why not simply use a basis of $N(R^T)$ for $N(A^T)$? **Answer:** it constrains transformed outputs. With the record $E$, multiply it by $E^T$ to recover original constraints; without that record, solve the retained original transpose system.
> <!-- bilingual-en:end -->
>
> 为什么不能直接把 $N(R^T)$ 的基当作 $N(A^T)$ 的基？**答案：**它约束的是变换后的输出。有记录 $E$ 时，左乘 $E^T$ 恢复原约束；没有记录时，应解保留下来的原转置系统。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf#page=1|MIT 18.06SC Session 1.7, pp.1–2]]: pivot/free coordinates and the construction of special homogeneous solutions.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf#page=2|MIT Session 1.9, pp.2–3]]: original pivot columns and nullspace dimension. The second displayed null vector on p.3 has a final-coordinate sign error; the examples here are checked directly.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf#page=2|MIT Session 1.10, pp.1–3]]: the different basis sources, preserved row span and tracking of left-null weights; both tracking blocks have $m\times m$ size.
- [Axler, *Linear Algebra Done Right*, 3.52–3.57 and 7.6–7.9, PDF pp.91–92 and 245–246](https://linear.axler.net/LADR4e.pdf#page=91): row/column spans and the transpose versus adjoint distinction. The common example and the information-loss comparison are verified by direct multiplication and solving the displayed equations.
<!-- bilingual-en:end -->

- MIT Session 1.7，第1–2页：核对主元、自由坐标与特殊齐次解的构造。
- MIT Session 1.9，第2–3页：核对原主元列和零空间维数；第3页第二个零空间向量的末坐标符号误印，正文例子均直接核算。
- MIT Session 1.10，第1–3页：核对不同基的来源、行张成保持和左零权重追踪；两个记录块均为 $m\times m$。
- Axler 3.52–3.57及7.6–7.9，PDF91–92、245–246：核对行列张成与普通转置、伴随的区别；共用例子和信息丢失对照通过直接相乘、求解所列方程核验。
