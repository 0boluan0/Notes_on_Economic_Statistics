---
aliases:
  - Weighted graph Laplacian
  - 加权图拉普拉斯矩阵
  - Graph Laplacian
student_os: knowledge-atom
atom_id: MCS-GRAPH-022
atom_type: definition
status: source-checked
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[关联矩阵势差映射]]"
---

<!-- bilingual-en:start -->
# Weighted graph Laplacian
<!-- bilingual-en:end -->
*加权图拉普拉斯矩阵*

<!-- bilingual-en:start -->
The **weighted graph Laplacian** measures how a node value differs from values at its neighbors, giving stronger connections more weight. It combines three simple operations: compute differences along edges, multiply each difference by its edge weight, and combine the results at nodes. In a resistor network it relates node voltages to the external currents needed to maintain them.
<!-- bilingual-en:end -->

**加权图拉普拉斯矩阵**衡量节点值与相邻节点值的差，并让连接更强的边贡献更大。它把三个简单操作合在一起：先求边上的差，再乘各边权重，最后汇总回节点。在电阻网络里，它把节点电压与维持这些电压所需的外部电流联系起来。

<!-- bilingual-en:start -->
## From edge differences back to nodes
<!-- bilingual-en:end -->
*从边差值回到节点量*

<!-- bilingual-en:start -->
Take a finite undirected graph with $n\ge1$ nodes and $m\ge0$ edges, allowing parallel edges but initially no self-loops. Use the real [[关联矩阵势差映射|incidence matrix]] $A\in\mathbb R^{m\times n}$: an edge $i\to j$ has $-1$ at its tail and $+1$ at its head, so $(Ax)_e=x_j-x_i$. Assign each edge a finite nonnegative weight $c_e$. In the resistor interpretation this weight is its conductance, current per unit voltage difference. Define the [[对角矩阵|diagonal matrix]] $C=\operatorname{diag}(c_1,\ldots,c_m)$, with zero off-diagonal entries, and let $T$ denote [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

取有限无向图，有 $n\ge1$ 个节点、$m\ge0$ 条边；允许平行边，先不含自环。使用实[[关联矩阵势差映射|关联矩阵]] $A\in\mathbb R^{m\times n}$：边 $i\to j$ 的起点列为 $-1$、终点列为 $+1$，所以 $(Ax)_e=x_j-x_i$。为每条边赋有限非负权重 $c_e$；在电阻解释中，它是电导，即单位电压差对应的电流。定义[[对角矩阵]] $C=\operatorname{diag}(c_1,\ldots,c_m)$，非对角元均为零；$T$ 表示[[矩阵转置|转置]]。

$$\boxed{L_G=A^TCA\in\mathbb R^{n\times n}.}$$

<!-- bilingual-en:start -->
We also write $L=L_G$. The dimensions explain the composition: $x$ has $n$ node coordinates, $Ax$ and $CAx$ have $m$ edge coordinates, and $A^TCAx$ has $n$ node coordinates again. With no edges, the sum of edge contributions is empty and $L$ is the zero $n\times n$ matrix. This is the unnormalized, nonnegative-weight Laplacian; other normalized or directed versions require their own definitions.
<!-- bilingual-en:end -->

也简写为 $L=L_G$。尺寸说明了这次复合：$x$ 有 $n$ 个节点坐标，$Ax$ 与 $CAx$ 有 $m$ 个边坐标，$A^TCAx$ 又回到 $n$ 个节点坐标。无边时，边贡献的和为空，$L$ 是 $n\times n$ 零矩阵。这里定义的是非负权重的非归一化拉普拉斯矩阵；归一化版本或有向图版本须另行定义。

> [!example] Two weighted edges · 两条带权边
> <!-- bilingual-en:start -->
> Use the path $1\to2\to3$, with conductances $2$ and $1$:
> <!-- bilingual-en:end -->
>
> 对路径 $1\to2\to3$，取两条边的电导分别为 $2$、$1$：
>
> $$A=\begin{pmatrix}-1&1&0\\0&-1&1\end{pmatrix},\quad
> C=\begin{pmatrix}2&0\\0&1\end{pmatrix},\quad
> L=\begin{pmatrix}2&-2&0\\-2&3&-1\\0&-1&1\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> For $x=(3,1,0)^T$, the differences are $Ax=(-2,-1)^T$, their weighted values are $CAx=(-4,-1)^T$, and $Lx=(4,-3,-1)^T$. The node entries sum to zero. Below we interpret these signs as external injection and withdrawal.
> <!-- bilingual-en:end -->
>
> 当 $x=(3,1,0)^T$ 时，差值为 $Ax=(-2,-1)^T$，加权后为 $CAx=(-4,-1)^T$，最后 $Lx=(4,-3,-1)^T$。节点结果之和为零；下面会把这些正负号解释为外部注入与抽出。

<!-- bilingual-en:start -->
For a loopless graph, each incident edge contributes $+c_e$ to a diagonal entry; an edge between distinct nodes $i,j$ contributes $-c_e$ to $L_{ij}$. Consequently
<!-- bilingual-en:end -->

对无自环图，每条相接边给对角元贡献 $+c_e$；连接不同节点 $i,j$ 的边给 $L_{ij}$ 贡献 $-c_e$。因此

$$L_{ii}=\sum_{e\text{ incident to }i}c_e,\qquad
L_{ij}=-\sum_{e\text{ joining }i,j}c_e\quad(i\ne j).$$

<!-- bilingual-en:start -->
Parallel edges contribute separately. Equivalently, $(Lx)_i$ sums $c_e(x_i-x_j)$ over edges from node $i$ to a neighbor $j$, regardless of their reference arrows. Each row sums to zero. The diagonal is weighted degree and the off-diagonal entries are negative couplings, unlike an [[邻接表与邻接矩阵|adjacency matrix]], which records the connections themselves.
<!-- bilingual-en:end -->

平行边各自贡献。等价地，$(Lx)_i$ 对节点 $i$ 到邻点 $j$ 的各条边求和 $c_e(x_i-x_j)$，不受参考箭头影响。每行之和为零；对角元是加权度，非对角元为负耦合，与记录连接本身的[[邻接表与邻接矩阵|邻接矩阵]]不同。

<!-- bilingual-en:start -->
## Symmetry, nonnegative energy and orientation
<!-- bilingual-en:end -->
## 对称、半正定与方向无关

<!-- bilingual-en:start -->
Because $C^T=C$, transposition gives $L^T=A^TC^TA=L$. For every real node vector $x$, writing $\Delta x_e=(Ax)_e$,
<!-- bilingual-en:end -->

因为 $C^T=C$，转置后有 $L^T=A^TC^TA=L$。对每个实节点向量 $x$，记 $\Delta x_e=(Ax)_e$，则

$$x^TLx=(Ax)^TC(Ax)=\sum_{e=1}^m c_e(\Delta x_e)^2\ge0.$$

<!-- bilingual-en:start -->
Symmetry together with this inequality is exactly [[半正定矩阵|positive semidefiniteness]]. The quadratic value measures the weighted disagreement across edges. The example gives $2\cdot(-2)^2+1\cdot(-1)^2=9$, also $x^TLx=3\cdot4+1\cdot(-3)=9$.
<!-- bilingual-en:end -->

对称性加上这个不等式，恰好就是[[半正定矩阵|半正定]]。该二次型值衡量边两端的加权差异。例中为 $2\cdot(-2)^2+1\cdot(-1)^2=9$，而 $x^TLx=3\cdot4+1\cdot(-3)=9$，结果相同。

<!-- bilingual-en:start -->
Reversing selected arrows replaces $A$ by $DA$, where the diagonal entries of $D$ are $\pm1$. Since $D^TCD=C$, the new product is $(DA)^TC(DA)=A^TCA$. Thus $L$ belongs to the weighted undirected graph, independently of the reference arrows. Negative edge weights are outside the assumptions: even one edge of weight $-1$ would give $x^TLx=-(x_1-x_2)^2$, which can be negative.
<!-- bilingual-en:end -->

反转若干参考箭头，相当于把 $A$ 换成 $DA$，其中 $D$ 的对角元为 $\pm1$。由于 $D^TCD=C$，新乘积满足 $(DA)^TC(DA)=A^TCA$。所以 $L$ 由带权无向图决定，不依赖参考箭头。负权不在本条假设内：即使只有一条权重为 $-1$ 的边，也会得到 $x^TLx=-(x_1-x_2)^2$，可能为负。

<!-- bilingual-en:start -->
## Which components determine the nullspace?
<!-- bilingual-en:end -->
*哪些连通分量决定零空间？*

<!-- bilingual-en:start -->
Keep all nodes but retain only edges with $c_e>0$. Call this the **positive-weight support graph**. Let its [[无向图连通分量|connected components]] be $S_1,\ldots,S_q$, including isolated nodes. Zero-weight edges contribute nothing to $L$ and therefore do not connect this graph.
<!-- bilingual-en:end -->

保留全部节点，但只保留 $c_e>0$ 的边，称所得图为**正权支撑图**。记它的[[无向图连通分量|连通分量]]为 $S_1,\ldots,S_q$，包括孤立节点。零权边对 $L$ 没有贡献，因此不在这张图里起连接作用。

<!-- bilingual-en:start -->
If $Lx=0$, its quadratic value is zero. Every summand is nonnegative, so every positive-weight edge has equal endpoint values. The path argument makes $x$ constant on each $S_k$. Conversely, such constants make every coordinate of $CAx$ zero, including those on zero-weight edges, hence $Lx=0$. This proves the exact [[零空间|nullspace]] description
<!-- bilingual-en:end -->

若 $Lx=0$，二次型值必为零。各项都非负，所以每条正权边两端的值相等；沿路径传递，$x$ 在每个 $S_k$ 上分别为常数。反过来，这样的常数赋值使 $CAx$ 的每个坐标都为零，包括零权边的坐标，因而 $Lx=0$。这就证明了准确的[[零空间]]描述：

$$N(L)=\{x:x\text{ is constant on each }S_k\},\qquad
\dim N(L)=q,\quad \operatorname{rank}L=n-q.$$

<!-- bilingual-en:start -->
The dimension is $q$ because each component's constant can be chosen independently; the rank follows by [[秩零度定理|rank–nullity]]. If the positive-weight support graph is connected, the nullspace consists precisely of multiples of $\mathbf1=(1,\ldots,1)^T$. Since $n\ge1$, this nonzero vector always makes $L$ singular, even when every edge weight is positive.
<!-- bilingual-en:end -->

维数为 $q$，因为每个分量上的常数可以独立选取；秩由[[秩零度定理]]得到。正权支撑图连通时，零空间恰为 $\mathbf1=(1,\ldots,1)^T$ 的全部倍数。由于 $n\ge1$，这个非零向量总在核中，所以即使所有边权都为正，$L$ 仍然奇异。

<!-- bilingual-en:start -->
## Conservation, compatibility and a reference potential
<!-- bilingual-en:end -->
## 守恒方程与可解性

<!-- bilingual-en:start -->
To interpret signs physically, let $y_e$ be positive current along the reference arrow $i\to j$. In a passive resistor, current follows decreasing voltage, so Ohm's law is $y_e=c_e(x_i-x_j)$, or $y=-CAx$. The quantity $(A^Ty)_i$ is network inflow minus outflow. If $f_i$ is external injection, positive for input, no accumulation means $A^Ty+f=0$. Combining these equations gives
<!-- bilingual-en:end -->

解释物理符号时，令 $y_e$ 表示沿参考箭头 $i\to j$ 为正的电流。无源电阻中的电流沿电势下降方向流动，因此欧姆定律为 $y_e=c_e(x_i-x_j)$，即 $y=-CAx$。$(A^Ty)_i$ 表示网络内部的流入减流出；若 $f_i$ 为外部注入，输入为正，那么无积累要求 $A^Ty+f=0$。合并得到

$$y=-CAx,\qquad f=-A^Ty,\qquad \boxed{Lx=f}.$$

<!-- bilingual-en:start -->
In the path example, $y=(4,1)^T$ and $f=(4,-3,-1)^T$: node $1$ receives four units externally, node $2$ removes three and node $3$ removes one. A convention using $y=CAx$ can describe the opposite signed edge response, but its source signs must change consistently. The abstract incidence map alone does not impose Ohm's law.
<!-- bilingual-en:end -->

路径例中，$y=(4,1)^T$、$f=(4,-3,-1)^T$：节点 $1$ 从外部接收四单位，节点 $2$ 抽出三单位，节点 $3$ 抽出一单位。也可以使用 $y=CAx$ 描述相反符号的边响应，但源项符号必须相应一致改变。抽象关联映射本身并没有附带欧姆定律。

<!-- bilingual-en:start -->
Because $L=L^T$, its [[左零空间相容性|left-nullspace solvability criterion]] tests $f$ against $N(L)$. A basis of that nullspace has one vector for each $S_k$, with entry $1$ on that component and $0$ elsewhere. The tests therefore give a necessary and sufficient condition:
<!-- bilingual-en:end -->

由于 $L=L^T$，它的[[左零空间相容性|左零空间可解性判据]]要求用 $N(L)$ 检验 $f$。该零空间的一组基为：每个 $S_k$ 对应一根向量，在该分量上取 $1$，其他位置取 $0$。于是得到必要且充分的条件：

$$Lx=f\text{ is solvable}\quad\Longleftrightarrow\quad
\sum_{i\in S_k}f_i=0\quad\text{for every }k=1,\ldots,q.$$

<!-- bilingual-en:start -->
Once this condition holds, any two solutions differ by one constant on each component. Set one chosen node in **each** component to zero, a procedure called grounding. Starting from any solution, subtract its value at the chosen node throughout that component; the positive-weight edge differences and $Lx$ stay unchanged. Two grounded solutions cannot differ by a nonzero componentwise constant, so the grounded solution is unique. Requiring a separate zero mean on each component is another unique choice. One global mean condition is insufficient when $q>1$.
<!-- bilingual-en:end -->

该条件成立后，任意两解之差是在每个分量上分别加一个常数。在**每个**分量中选一个节点，把它的值固定为零，这称为接地：从任意解出发，在整个分量上减去所选节点的值，正权边的差值与 $Lx$ 均不变。两个已接地的解不可能再相差非零的分量常数，因此解唯一。要求每个分量分别零均值，也能选出唯一解；若 $q>1$，只要求一个全局零均值还不够。

<!-- bilingual-en:start -->
Deleting the grounded coordinates and their matching equations gives an invertible reduced matrix whenever any coordinates remain. Indeed, extend a reduced vector by zeros at the grounded nodes. Its quadratic value can be zero only if it is constant on every positive-weight component, and its grounded zeros then force the entire vector to be zero. Thus the reduced matrix is [[正定矩阵|positive definite]]. For a compatible $f$, the omitted equation in each component follows from the zero row sums and that component's zero total injection. With incompatible $f$, solving only the retained equations does not solve the original full system.
<!-- bilingual-en:end -->

删除已接地坐标及对应方程后，只要还有剩余坐标，所得约化矩阵就可逆。理由是：把约化向量在接地点补零，其二次型若为零，就必须在每个正权分量上为常数；接地点的零又迫使全向量为零，因此约化矩阵[[正定矩阵|正定]]。对相容的 $f$，每个分量中被略去的方程由零行和与该分量净注入为零自动推出。如果 $f$ 不相容，只解保留的方程并没有解出原完整系统。

> [!warning] A drawn connection can carry zero weight · 画着连接线，也可能没有正权连接
> <!-- bilingual-en:start -->
> Set the second path weight to zero. Then
> <!-- bilingual-en:end -->
>
> 把路径第二条边的权重改为零，得到
>
> $$L=\begin{pmatrix}2&-2&0\\-2&2&0\\0&0&0\end{pmatrix},\qquad
> N(L)=\{(s,s,t)^T:s,t\in\mathbb R\}.$$
>
> <!-- bilingual-en:start -->
> The drawn path is connected, but the positive-weight components are $\{1,2\}$ and $\{3\}$. The vector $f=(1,0,-1)^T$ sums to zero globally yet is incompatible: each component has a nonzero total. The vector $(1,-1,0)^T$ is compatible, but grounding only node $3$ still leaves a common additive offset on nodes $1,2$ free. Both components need a reference value.
> <!-- bilingual-en:end -->
>
> 图上画出的路径仍连通，但正权分量为 $\{1,2\}$ 与 $\{3\}$。$f=(1,0,-1)^T$ 虽然总和为零，却不相容，因为各分量净注入不为零。$(1,-1,0)^T$ 则相容，但只把节点 $3$ 接地，仍不能确定同时加在节点 $1,2$ 上的常数；两个分量都要有参考值。

<!-- bilingual-en:start -->
If every weight is zero, all nodes are isolated in the positive-weight support graph: $L=0$, and compatibility requires $f=0$ at every node. If self-loops are included using zero incidence rows, their contribution to $A^TCA$ is zero for any weight. They must therefore be excluded from the diagonal weighted-degree sum above; adding their weight there would define a different matrix. Under the passive law, such a zero-difference loop also has zero current.
<!-- bilingual-en:end -->

若所有权重都为零，正权支撑图中每个节点都孤立：$L=0$，相容性要求每个节点的 $f$ 都为零。若按零关联行约定加入自环，无论权重是多少，它对 $A^TCA$ 的贡献均为零。因此，上面的对角加权度求和必须排除这些自环；若把其权重加进去，就定义了另一张矩阵。在无源电阻定律下，这种零势差自环的电流也为零。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S13_Lecture_Graphs_Networks_Incidence_Matrices.pdf#page=22|MIT 18.06SC graphs and networks transcript, pp.22–24]] and [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.12sum.pdf#page=3|graph summary, pp.3–4]]: edge differences, conductance, transpose conservation and the product $A^TCA$. The explicit current and external-injection signs here are derived from their stated meanings.
- [Spielman, *Spectral Graph Theory*, Lecture 2 (2015), §2.2, PDF pp.1–2](https://cs.yale.edu/homes/spielman/561/lect02-15.pdf#page=1): the weighted squared-difference form, individual edge contributions, diagonal degree and neighbor-difference action. Zero weights are handled by deleting their zero contributions while retaining every node.
- [[01_Ax = b and the Four Subspaces#Session 1.13 Graphs, networks, and incidence matrices|Permanent Session 1.13 note]]: the course's current/source convention, network example and grounding. The proofs above identify the positive-weight components and justify both directions of compatibility and unique grounding; the numerical examples are calculated directly.
<!-- bilingual-en:end -->

- MIT 图与网络转录第22–24页、图论摘要第3–4页：核对边差值、电导、转置守恒与 $A^TCA$ 的复合。本文电流和外部注入的正负号由明确含义推导。
- Spielman《Spectral Graph Theory》2015年Lecture 2，§2.2，PDF1–2页：核对加权差平方和、单条边贡献、对角度数与邻点差作用。零权通过删除零贡献处理，同时保留全部节点。
- 永久 Session 1.13 笔记：保留课程电流／源项约定、网络例子和接地背景。正文证明明确正权分量，并论证相容性的双向条件与接地唯一性；数值例子直接核算。
