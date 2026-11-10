---
aliases:
- Reading a nullspace basis from RREF
- Special solutions of the nullspace
student_os: knowledge-atom
atom_id: LA-SYS-027
atom_set: linear-systems-four-subspaces
atom_type: method
status: source-checked
requires:
- '[[行最简形]]'
- '[[基]]'
- '[[主元与自由变量]]'
- '[[自由变量]]'
- '[[零空间]]'
related:
- '[[RREF读取子空间基]]'
- '[[秩零度定理]]'
part_of:
- '[[线性方程组与四个基本子空间.canvas]]'
---

<!-- bilingual-en:start -->
# Reading a nullspace basis from RREF
<!-- bilingual-en:end -->
*从行最简形读取零空间基*

<!-- bilingual-en:start -->
A [[零空间|nullspace]] contains every input vector that a matrix sends to zero. To describe all those inputs, we want a [[基|basis]]: independent directions whose linear combinations give every solution. Row reduction makes the choices visible. The [[自由变量|free variables]] can be chosen freely; the remaining coordinates must then follow the equations. Choosing one free coordinate at a time produces a basis.
<!-- bilingual-en:end -->

[[零空间]]包含矩阵送到零的全部输入向量。要描述这些输入，我们希望找一组[[基]]：方向彼此独立，而且线性组合覆盖全部解。消元把可以选择的部分显露出来：[[自由变量]]可以自由取值，其余坐标随后必须满足方程。逐个选择自由坐标，就能构造这组基。

<!-- bilingual-en:start -->
## Let the free coordinates describe the solution
<!-- bilingual-en:end -->
*让自由坐标描述整个解*

<!-- bilingual-en:start -->
Consider the following matrix and its [[行最简形|reduced row echelon form]], abbreviated RREF. In this form each nonzero row has a leading $1$, called a pivot, and the other entries in that pivot's column are zero.
<!-- bilingual-en:end -->

考虑下列矩阵及其[[行最简形]]，英文缩写为 RREF。在这种形式中，每个非零行以一个主元 $1$ 开头，该主元所在列的其余元素均为零。

$$
A=\begin{pmatrix}1&2&3&1\\1&1&2&1\\1&2&3&1\end{pmatrix},
\qquad
R=\begin{pmatrix}1&0&1&1\\0&1&1&0\\0&0&0&0\end{pmatrix}.
$$

<!-- bilingual-en:start -->
The equation $Rx=0$ says $x_1+x_3+x_4=0$ and $x_2+x_3=0$. Columns $1,2$ are pivot columns, so $x_1,x_2$ are determined by the [[主元与自由变量|free choices]] $x_3=s$ and $x_4=t$. Here $s,t$ are arbitrary scalars. Writing vectors as columns, with $T$ denoting [[矩阵转置|transpose]], gives
<!-- bilingual-en:end -->

方程 $Rx=0$ 表示 $x_1+x_3+x_4=0$、$x_2+x_3=0$。第 $1,2$ 列是主元列，所以 $x_1,x_2$ 由[[主元与自由变量|自由选择]]的 $x_3=s$、$x_4=t$ 决定；$s,t$ 为任意标量。将向量写成列向量，上标 $T$ 表示[[矩阵转置|转置]]，得到

$$
x=(-s-t,-s,s,t)^T
=s\underbrace{(-1,-1,1,0)^T}_{v_1}
+t\underbrace{(-1,0,0,1)^T}_{v_2}.
$$

<!-- bilingual-en:start -->
Setting $(s,t)=(1,0)$ gives $v_1$; setting $(s,t)=(0,1)$ gives $v_2$. Each is a complete four-coordinate solution, including the pivot coordinates. Direct multiplication verifies $Av_1=Av_2=0$. Every solution is their displayed combination, and the third and fourth coordinates show that $c_1v_1+c_2v_2=0$ forces $c_1=c_2=0$. Thus these two vectors form a basis, not just two examples of solutions.
<!-- bilingual-en:end -->

令 $(s,t)=(1,0)$ 得到 $v_1$，令 $(s,t)=(0,1)$ 得到 $v_2$。每个向量都是含主元坐标在内的完整四坐标解。直接相乘可核对 $Av_1=Av_2=0$。全部解都由上述组合给出，而第三、第四坐标说明 $c_1v_1+c_2v_2=0$ 必须有 $c_1=c_2=0$。因此这两个向量构成基，而不只是解的两个例子。

<!-- bilingual-en:start -->
## Why solving the reduced system is enough
<!-- bilingual-en:end -->
*为什么只需求化简后的齐次系统*

<!-- bilingual-en:start -->
Let $A$ be an $m\times n$ matrix over $\mathbb R$ or $\mathbb C$, where $m,n$ are positive integers. Every elementary row operation is reversible. Their product is an invertible $m\times m$ matrix $E$, so row reduction gives $R=EA$; see [[初等矩阵|elementary matrices]]. Consequently, writing $N(A)$ for the nullspace of $A$,
<!-- bilingual-en:end -->

设 $A$ 是实数域或复数域上的 $m\times n$ 矩阵，$m,n$ 为正整数。每一步初等行操作都可逆，合起来得到可逆的 $m\times m$ 矩阵 $E$，使得 $R=EA$；参见[[初等矩阵]]。用 $N(A)$ 表示 $A$ 的零空间，于是

$$
Ax=0\iff EAx=0\iff Rx=0,\qquad N(A)=N(R).
$$

<!-- bilingual-en:start -->
The reverse implication uses $E^{-1}$; no inverse of $A$ is required. This is why the method works for rectangular and rank-deficient matrices. Over $\mathbb C$ the same equations and free-coordinate construction apply, with complex coefficients; finding $N(A)$ still means solving $Ax=0$.
<!-- bilingual-en:end -->

反向推导使用 $E^{-1}$，不要求 $A$ 可逆。因此矩形矩阵和秩亏矩阵同样适用。复数域上的方程与自由坐标构造完全相同，只是系数可以是复数；求 $N(A)$ 仍然是解 $Ax=0$。

> [!important] Construct the entire basis · 构造完整的一组基
> <!-- bilingual-en:start -->
> If $R$ has $r$ pivots, its [[矩阵秩|rank]] is $r$. Let $p_1,\ldots,p_r$ be the pivot-column indices and $f_1,\ldots,f_k$ the free-column indices, where $k=n-r$. In pivot row $i$, the equation is
> <!-- bilingual-en:end -->
>
> 若 $R$ 有 $r$ 个主元，其[[矩阵秩|秩]]为 $r$。用 $p_1,\ldots,p_r$ 表示主元列编号，用 $f_1,\ldots,f_k$ 表示自由列编号，其中 $k=n-r$。第 $i$ 个主元行的方程为
>
> $$x_{p_i}=-\sum_{j=1}^{k}R_{i f_j}x_{f_j}.$$
>
> <!-- bilingual-en:start -->
> Here $R_{i f_j}$ is the entry in row $i$, free column $f_j$. For each $j$, set $x_{f_j}=1$, set every other free coordinate to $0$, and compute all pivot coordinates from these equations. Call the resulting full vector $v_j$. The list $v_1,\ldots,v_k$ is a basis of $N(A)$.
> <!-- bilingual-en:end -->
>
> $R_{i f_j}$ 是第 $i$ 行、第 $f_j$ 列的元素；第 $f_j$ 列是自由列。对每个 $j$，令 $x_{f_j}=1$、其余自由坐标均为 $0$，再用方程计算全部主元坐标，所得完整向量记为 $v_j$。向量组 $v_1,\ldots,v_k$ 就是 $N(A)$ 的一组基。

<!-- bilingual-en:start -->
For **spanning**, take any solution and write its free coordinates as $t_1,\ldots,t_k$. The vector $\sum_jt_jv_j$ has exactly those free coordinates and the uniquely determined pivot coordinates, so it equals that solution. For **independence**, look at free coordinate $f_j$ in an equation $\sum_jc_jv_j=0$: it reads $c_j=0$. Doing this for every free coordinate proves independence. These two arguments establish a basis directly; the count $k=n-r$ agrees with the [[秩零度定理|rank–nullity theorem]].
<!-- bilingual-en:end -->

证明**张成**时，取任意解，将其自由坐标记为 $t_1,\ldots,t_k$。向量 $\sum_jt_jv_j$ 恰好具有这些自由坐标，以及随之唯一确定的主元坐标，所以它就是原解。证明**线性无关**时，在 $\sum_jc_jv_j=0$ 中观察第 $f_j$ 个自由坐标，便得到 $c_j=0$；逐个观察所有自由坐标即可。两个论证直接证明了基的两个要求，向量个数 $k=n-r$ 也与[[秩零度定理]]一致。

<!-- bilingual-en:start -->
## Keep the original coordinate order
<!-- bilingual-en:end -->
*保留原来的坐标次序*

<!-- bilingual-en:start -->
Pivot columns need not come first: columns $1,3$ might be pivot columns while $2,4$ are free. The construction above puts every coordinate back in its original position. The compact [[零空间矩阵|nullspace-matrix formula]] $\binom{-F}{I_k}$ assumes the variables have first been ordered with pivot variables before free variables; $F$ is then the free-column block in the pivot equations and $I_k$ is the [[单位矩阵|identity matrix]]. If columns were reordered to obtain that form, undo that coordinate permutation when reporting a basis for the original $A$.
<!-- bilingual-en:end -->

主元列不一定排在最前面，例如第 $1,3$ 列可能是主元列，第 $2,4$ 列是自由列。上面的构造始终将每个坐标放回原位置。简写的[[零空间矩阵]]公式 $\binom{-F}{I_k}$ 预先假定变量已按“主元变量在前、自由变量在后”排序；此时 $F$ 是主元方程中的自由列块，$I_k$ 为[[单位矩阵]]。若为了得到这种形式交换过列，给出原矩阵 $A$ 的零空间基时必须还原坐标次序。

<!-- bilingual-en:start -->
If there are no free variables, the only solution is zero and the basis is the **empty list**, not a list containing the zero vector. If $A=0$, every coordinate is free and the method returns the usual coordinate basis. When free directions exist, the special-solution basis is a convenient choice rather than a unique one: any invertible mixing of its vectors gives another basis of the same space. These statements concern exact algebraic row reduction; deciding whether small floating-point entries count as zero is a separate [[数值秩|numerical-rank decision]].
<!-- bilingual-en:end -->

如果没有自由变量，唯一解是零，基是**空向量组**，不是只含零向量的向量组。如果 $A=0$，每个坐标都自由，这种方法返回通常的坐标基。存在自由方向时，特殊解基是一种方便的选择，并非唯一选择；将这些向量作任意可逆的线性组合，仍得到同一空间的一组基。这里讨论的是精确代数消元；浮点数中的小量是否视为零，属于另一个[[数值秩]]决策。

> [!question]- Read a basis from an existing parametrisation · 从已有参数式读取基
> <!-- bilingual-en:start -->
> If all solutions are $x=s(1,0,-2)^T+t(0,1,3)^T$, which special solutions result? Taking $(s,t)=(1,0)$ and $(0,1)$ gives $(1,0,-2)^T$ and $(0,1,3)^T$. Their first two coordinates also verify independence.
> <!-- bilingual-en:end -->
>
> 若全部解写成 $x=s(1,0,-2)^T+t(0,1,3)^T$，特殊解是什么？分别取 $(s,t)=(1,0)$ 和 $(0,1)$，得到 $(1,0,-2)^T$ 与 $(0,1,3)^T$；前两个坐标也能直接核对线性无关。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf#page=1|MIT 18.06SC, solving the homogeneous system, pp. 1–2]]: row reduction, free variables, special solutions, and the column-order requirement for the block formula.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf#page=1|MIT 18.06SC, independence, basis, and dimension, pp. 1–3]]: the two requirements for a basis and the worked matrix. The vectors above are checked by multiplying by the original matrix; the spanning and independence arguments are given explicitly.
<!-- bilingual-en:end -->

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf#page=1|MIT 18.06SC 齐次方程讲义，第 1–2 页]]：核对消元、自由变量、特殊解及块公式所需的列次序。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf#page=1|MIT 18.06SC 线性无关、基与维数讲义，第 1–3 页]]：核对基的两个要求及算例矩阵；上文向量均与原矩阵直接相乘核验，并完整给出张成与独立的证明。
