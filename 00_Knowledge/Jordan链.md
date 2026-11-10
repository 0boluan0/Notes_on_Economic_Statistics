---
aliases:
- Jordan chain
- Jordan 链
student_os: knowledge-atom
atom_id: LA-EIG-012
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
requires:
- '[[广义特征向量]]'
related:
- '[[代数重数与几何重数]]'
- '[[Jordan块]]'
- '[[Jordan标准形]]'
leads_to:
- '[[Jordan链向量线性无关]]'
part_of:
- '[[特征值、对角化与线性动力系统.canvas]]'
---

<!-- bilingual-en:start -->
# Jordan chains
<!-- bilingual-en:end -->

*Jordan 链*

<!-- bilingual-en:start -->
A **Jordan chain** arranges vectors in a specific order: after subtracting a fixed eigenvalue's scaling from the matrix action, each vector moves to the previous one, and the first moves to zero. It turns several [[广义特征向量|generalized eigenvectors]] into one coordinated description of the matrix. The order of the list is part of the definition.
<!-- bilingual-en:end -->

**Jordan 链**按特定顺序排列向量：从矩阵作用中减去固定特征值的缩放后，每个向量都被送到前一个，最前面的向量则归零。它把若干[[广义特征向量]]组织成矩阵的一段完整作用，向量的排列顺序也是定义的一部分。

> [!example] Read the list from the eigenvector outward / 列表从普通特征向量向后排列
> <!-- bilingual-en:start -->
> Let $J=\begin{pmatrix}2&1\\0&2\end{pmatrix}$, $e_1=(1,0)^T$ and $e_2=(0,1)^T$. These are column vectors, using $T$ for [[矩阵转置|transpose]]. Set $N=J-2I_2$, where $I_2$ is the [[单位矩阵|identity]]. Then
> <!-- bilingual-en:end -->
>
> 令 $J=\begin{pmatrix}2&1\\0&2\end{pmatrix}$、$e_1=(1,0)^T$、$e_2=(0,1)^T$，其中 $T$ 表示[[矩阵转置|转置]]，把坐标写成列向量。令 $N=J-2I_2$，其中 $I_2$ 是[[单位矩阵]]，则
>
> $$Ne_1=0,\qquad Ne_2=e_1.$$
>
> <!-- bilingual-en:start -->
> The ordered list $(e_1,e_2)$ is a chain of length two. Applying $N$ runs in the opposite direction: $e_2\mapsto e_1\mapsto0$. Applying $J$ itself instead gives $Je_1=2e_1$ and $Je_2=2e_2+e_1$. The chain records the extra contribution $e_1$ as well as the scaling by $2$. Reversing the list does not satisfy the stated chain convention.
> <!-- bilingual-en:end -->
>
> 有序列表 $(e_1,e_2)$ 是长度为二的链。施加 $N$ 时，方向与列表相反：$e_2\mapsto e_1\mapsto0$。施加 $J$ 本身则得到 $Je_1=2e_1$、$Je_2=2e_2+e_1$；链既记录缩放 $2$，也记录额外贡献 $e_1$。倒过来排列列表，就不再满足这里的链约定。

<!-- bilingual-en:start -->
## One recurrence fixes every position
<!-- bilingual-en:end -->

*一组递推关系确定每个位置*

<!-- bilingual-en:start -->
Let $A$ be an $n\times n$ matrix with $n\ge1$ over $\mathbb F=\mathbb R$ or $\mathbb C$, let $\lambda\in\mathbb F$ be an eigenvalue, and put $N=A-\lambda I_n$. For a positive integer $r$, a Jordan chain for $\lambda$ is a list $v_1,\ldots,v_r\in\mathbb F^n$ satisfying
<!-- bilingual-en:end -->

设 $A$ 是 $\mathbb F=\mathbb R$ 或 $\mathbb C$ 上的 $n\times n$ 方阵，$n\ge1$，$\lambda\in\mathbb F$ 是一个特征值，令 $N=A-\lambda I_n$。给定正整数 $r$，对应于 $\lambda$ 的 Jordan 链是满足下式的有序向量列 $v_1,\ldots,v_r\in\mathbb F^n$：

$$v_1\ne0,\qquad Nv_1=0,\qquad Nv_{j+1}=v_j\quad(1\le j<r).$$

<!-- bilingual-en:start -->
The index $j$ records a position in the list, and $r$ is its length. Repeated application gives $N^{j-1}v_j=v_1\ne0$ and $N^jv_j=0$. Hence every vector is nonzero and $v_j$ has exact generalized-eigenvector order $j$. In particular $v_1$ is an ordinary [[特征对|eigenvector]], while later vectors are proper generalized eigenvectors. A length-one chain is allowed; it consists of one ordinary eigenvector.
<!-- bilingual-en:end -->

下标 $j$ 表示列表中的位置，$r$ 表示长度。重复作用得到 $N^{j-1}v_j=v_1\ne0$、$N^jv_j=0$，所以每个向量都非零，且 $v_j$ 的广义特征向量阶恰为 $j$。尤其 $v_1$ 是普通[[特征对|特征向量]]，后面的则是真广义特征向量。允许长度为一的链，即只含一个普通特征向量。

<!-- bilingual-en:start -->
The nonzero first vector is essential. If zeros were allowed at the beginning, a list could satisfy the displayed recurrence without having these orders or independent directions. The [[Jordan链向量线性无关|chain-independence theorem]] proves that a genuine length-$r$ chain supplies $r$ independent directions, so $r\le n$.
<!-- bilingual-en:end -->

第一个向量非零是关键。如果允许开头出现零向量，列表即使满足递推关系，也不再保证这些阶数和独立方向。[[Jordan链向量线性无关]]证明：真正的长度为 $r$ 的链提供 $r$ 个独立方向，因此 $r\le n$。

<!-- bilingual-en:start -->
## Build a chain from its last vector
<!-- bilingual-en:end -->

*从链尾构造整条链*

<!-- bilingual-en:start -->
Suppose $w$ has exact order $r$: $N^rw=0$ but $N^{r-1}w\ne0$. Compute its successive images and list them in reverse order:
<!-- bilingual-en:end -->

若 $w$ 的阶恰为 $r$，即 $N^rw=0$ 而 $N^{r-1}w\ne0$，计算它的连续像，再反向排列：

$$v_1=N^{r-1}w,\quad v_2=N^{r-2}w,\quad\ldots,\quad v_r=w.$$

<!-- bilingual-en:start -->
This automatically gives $Nv_{j+1}=v_j$ and a nonzero $v_1$. For order three, the list is $(N^2w,Nw,w)$; two applications to its last vector reach the first, and the third reaches zero. Using a power larger than the minimal order would put zeros at the front, producing an invalid chain of that claimed length.
<!-- bilingual-en:end -->

这样自动满足 $Nv_{j+1}=v_j$，且 $v_1$ 非零。阶为三时，列表是 $(N^2w,Nw,w)$；对末尾向量作用两次到达首向量，第三次才归零。如果使用大于最小阶的次数，就会在前面加入零向量，不能得到那个声称长度的有效链。

<!-- bilingual-en:start -->
Starting from the first vector instead means solving $Nv_2=v_1$, then $Nv_3=v_2$, and so on. An extension need not exist. For example, when $A=2I_n$, the adjusted matrix is $N=0$; it cannot map anything to a nonzero $v_1$. All chains then have length one.
<!-- bilingual-en:end -->

若从首向量开始，则要依次求解 $Nv_2=v_1$、$Nv_3=v_2$ 等方程；延长链未必可行。例如 $A=2I_n$ 时，调整后的矩阵 $N=0$ 不可能把任何向量送到非零 $v_1$，所以所有链都只能长一。

<!-- bilingual-en:start -->
## One chain describes one block
<!-- bilingual-en:end -->

*一条链描述一个块*

<!-- bilingual-en:start -->
The recurrence is equivalent to
<!-- bilingual-en:end -->

链递推等价于

$$Av_1=\lambda v_1,\qquad Av_{j+1}=\lambda v_{j+1}+v_j.$$

<!-- bilingual-en:start -->
The [[张成|span]] of the chain consists of all its linear combinations. These equations show that $A$ keeps that span inside itself, so it is an [[不变子空间|invariant subspace]]. Since the chain is independent, it is a [[基|basis]] of this subspace. Relative to the order $(v_1,\ldots,v_r)$, the matrix of the restricted action is the [[Jordan块|Jordan block]] $J_r(\lambda)$: $\lambda$ on the diagonal and ones directly above it. The above-diagonal position comes from the preceding vector $v_j$ appearing in the image of column $j+1$.
<!-- bilingual-en:end -->

链的[[张成|张成空间]]包含其全部线性组合。上述等式说明 $A$ 把这个空间留在自身内部，因此它是[[不变子空间]]。由于链线性无关，它又是这个子空间的一组[[基]]。按 $(v_1,\ldots,v_r)$ 的顺序记录受限作用，得到 [[Jordan块]] $J_r(\lambda)$：对角为 $\lambda$，紧邻上方为一。之所以是一写在上方，是因为第 $j+1$ 列的像含有前一个向量 $v_j$。

<!-- bilingual-en:start -->
In the opening example the chain spans all of $\mathbb F^2$, although there is only one ordinary eigenvector direction. A chain in a larger problem may span only part of the space. Arbitrarily collecting several chains does not guarantee that their vectors are jointly independent. A [[Jordan标准形|full Jordan basis]] requires an appropriately chosen collection; its existence over the chosen field has the [[Jordan标准形的域条件|characteristic-polynomial splitting condition]]. No such whole-matrix splitting assumption is needed merely to define one valid chain for an available eigenvalue.
<!-- bilingual-en:end -->

开头例子的链张成整个 $\mathbb F^2$，虽然普通特征向量只有一个方向。在更大的问题中，一条链可能只覆盖部分空间。随意收集几条链，也不保证全部向量联合线性无关。[[Jordan标准形|完整 Jordan 基]]需要恰当地选择整组链，其存在性还受到所用数域上的[[Jordan标准形的域条件|特征多项式分裂条件]]约束。不过，仅定义某个已有特征值的一条有效链，不要求整个矩阵的特征多项式都分裂。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, §8A exercise 2, PDF p. 320, and §8C, PDF pp. 335–338](https://linear.axler.net/LADR4e.pdf#page=320): successive powers of a vector, reversing their order into a chain basis, and its Jordan-block representation. The displayed recurrence, exact orders and failed-extension example are verified directly.
- [Colorado Math 6140, §12.3, PDF p. 1](https://math.colorado.edu/~mayr/teaching/math6140spring25/PID3.pdf): the splitting hypothesis for assembling a full Jordan basis over a field.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=3|MIT Session 3.4 summary, pp. 3–4]] and [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf#page=19|Lecture 28 transcript, pp. 19–20]]: course context for missing independent eigenvectors and Jordan blocks, rather than a full chain-construction proof.
- [[03_Positive Definite Matrices and Applications#3.4.4 Jordan链与Jordan标准形|Permanent course §3.4.4]]: the course's eigenvector-first chain convention.
<!-- bilingual-en:end -->

- Axler 第8A节习题2（PDF320页）及第8C节（PDF335–338页）：核对连续幂向量、反向排列的链基及 Jordan 块表示。递推、精确阶数和无法延长的例子直接核算。
- Colorado Math6140 第12.3节，PDF第1页：核对在所用数域中组成完整 Jordan 基的分裂假设。
- MIT Session3.4 摘要第3–4页和 Lecture28 讲稿第19–20页：保留独立特征向量不足与 Jordan 块的课程背景，不把它们当成完整链构造证明。
- 永久课程 §3.4.4：核对先列普通特征向量的链顺序。
