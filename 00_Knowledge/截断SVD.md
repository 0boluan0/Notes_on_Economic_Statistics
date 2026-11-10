---
aliases:
- 截断奇异值分解
- Rank-k truncated SVD
- Truncated SVD
student_os: knowledge-atom
atom_id: LA-SVD-019
atom_set: singular-value-decomposition-low-rank
atom_type: definition
status: source-checked
requires:
- '[[紧SVD]]'
- '[[奇异向量]]'
- '[[SVD秩一展开]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
leads_to:
- '[[截断SVD最佳低秩近似]]'
- '[[奇异值能量比]]'
related:
- '[[低秩近似]]'
- '[[Frobenius低秩近似唯一性]]'
- '[[谱范数低秩近似不唯一]]'
---

<!-- bilingual-en:start -->
# Truncated SVD
<!-- bilingual-en:end -->

*截断 SVD*

<!-- bilingual-en:start -->
A truncated SVD replaces a matrix by one that acts along fewer input–output directions. Start with the [[SVD秩一展开|SVD rank-one expansion]]: each piece reads one input coordinate, stretches it, and produces one output direction. Truncation keeps the pieces with the largest stretches and removes the rest. This constructs a [[低秩近似|low-rank approximation]]—a matrix allowed only a limited number of independent output directions.
<!-- bilingual-en:end -->

截断 SVD 用只沿较少输入—输出方向起作用的矩阵代替原矩阵。先看 [[SVD秩一展开]]：每一块读取一个输入坐标，作伸缩，再产生一个输出方向。截断保留伸缩量最大的几块，删除其余部分，从而构造一个[[低秩近似]]，即只允许有限个独立输出方向的近似矩阵。

> [!example] Keep two directions / 保留两个方向
> <!-- bilingual-en:start -->
> Let $A=\operatorname{diag}(5,2,1)$, a [[对角矩阵|diagonal matrix]]. It sends $(x_1,x_2,x_3)^T$ to $(5x_1,2x_2,x_3)^T$. Keeping two directions gives $A_2=\operatorname{diag}(5,2,0)$: the first two actions stay intact, while the third output is set to zero. Thus $A_2(1,1,1)^T=(5,2,0)^T$, compared with $(5,2,1)^T$ for $A$. The superscript $T$ denotes [[矩阵转置|transpose]].
> <!-- bilingual-en:end -->
>
> 取[[对角矩阵]] $A=\operatorname{diag}(5,2,1)$，它把 $(x_1,x_2,x_3)^T$ 送到 $(5x_1,2x_2,x_3)^T$。保留两个方向得到 $A_2=\operatorname{diag}(5,2,0)$：前两项作用不变，第三输出归零。因此输入 $(1,1,1)^T$ 的输出由 $(5,2,1)^T$ 变成 $(5,2,0)^T$。上标 $T$ 表示[[矩阵转置|转置]]。

<!-- bilingual-en:start -->
## Define the retained matrix
<!-- bilingual-en:end -->

*定义保留后的矩阵*

<!-- bilingual-en:start -->
Let $A$ be a real or complex $m\times n$ matrix, with $m,n\ge1$ and [[矩阵秩|rank]] $r>0$. The [[紧SVD|compact SVD]] retains all $r$ positive [[奇异值|singular values]], ordered as $\sigma_1\ge\cdots\ge\sigma_r>0$. Its paired [[奇异向量|singular vectors]] $v_i$ and $u_i$ are perpendicular unit input and output directions, respectively. With $^*$ denoting conjugate transpose, the expansion and its truncation are
<!-- bilingual-en:end -->

设 $A$ 为实或复 $m\times n$ 矩阵，$m,n\ge1$，[[矩阵秩|秩]]为 $r>0$。[[紧SVD]]保留全部 $r$ 个正[[奇异值]]，按 $\sigma_1\ge\cdots\ge\sigma_r>0$ 排序；配对的[[奇异向量]] $v_i,u_i$ 分别是相互垂直的单位输入、输出方向。以 $^*$ 表示共轭转置，原展开和截断定义为

$$
A=\sum_{i=1}^{r}\sigma_i u_iv_i^*,
\qquad A_k=\sum_{i=1}^{k}\sigma_i u_iv_i^*,
\qquad k\in\{0,1,\ldots,r\}.
$$

<!-- bilingual-en:start -->
The index $i$ labels a direction pair; $k$ is the number of positive pieces retained. Each [[秩一矩阵外积刻画|outer product]] $u_iv_i^*$ has the same $m\times n$ shape as $A$. For $k>0$, writing the retained columns as $U_k$ and $V_k$ gives $A_k=U_k\operatorname{diag}(\sigma_1,\ldots,\sigma_k)V_k^*$. This factorization can store the action through $k$ directions without storing every entry of the full matrix.
<!-- bilingual-en:end -->

下标 $i$ 标记方向对，$k$ 是保留的正分量个数。每个[[秩一矩阵外积刻画|外积]] $u_iv_i^*$ 都与 $A$ 一样为 $m\times n$。当 $k>0$ 时，把保留列排成 $U_k,V_k$，可写成 $A_k=U_k\operatorname{diag}(\sigma_1,\ldots,\sigma_k)V_k^*$；因此可通过这 $k$ 个方向保存矩阵作用，不必存储完整矩阵的每个元素。

<!-- bilingual-en:start -->
The retained directions satisfy $A_kv_i=\sigma_i u_i$ for $i\le k$. Every output is a combination of those $k$ independent $u_i$, and each can actually be produced, so $\operatorname{rank}(A_k)=k$. At $k=0$, the sum is empty and means the zero matrix. At $k=r$, every positive piece is retained and $A_r=A$. If $A=0$, use $r=k=0$. If the full list of $p=\min(m,n)$ singular values includes zeros, the same sum can also be used for $r<k\le p$: then $A_k=A$ and its rank stays $r$, not $k$. More generally, an at-most-$k$ rank budget with $k\ge r$ is already satisfied by $A$.
<!-- bilingual-en:end -->

保留方向满足 $A_kv_i=\sigma_i u_i$，$i\le k$。所有输出都由这 $k$ 个线性无关的 $u_i$ 组合，而且每个方向都确实能产生，所以 $\operatorname{rank}(A_k)=k$。$k=0$ 时空求和表示零矩阵；$k=r$ 时保留全部正分量，$A_r=A$。若 $A=0$，取 $r=k=0$。若完整的 $p=\min(m,n)$ 个奇异值列表包含零，也可把同一求和延伸到 $r<k\le p$；此时 $A_k=A$，秩仍是 $r$，不是 $k$。更一般地，若允许的秩预算为“至多 $k$”且 $k\ge r$，原矩阵 $A$ 已满足约束。

<!-- bilingual-en:start -->
## Removing directions is not deleting entries
<!-- bilingual-en:end -->

*删除方向，不是逐元素删除*

<!-- bilingual-en:start -->
In the diagonal example, truncation happens to erase a diagonal entry. In general the directions are tilted relative to the coordinate axes, so removing one piece can change many entries. For $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$, take $u_1=v_1=(1,1)^T/\sqrt2$ and $u_2=v_2=(1,-1)^T/\sqrt2$, with stretches 3 and 1. Then
<!-- bilingual-en:end -->

对角例子中，截断碰巧就是擦掉一个对角元素。一般情况下，方向相对于坐标轴是倾斜的，删除一块会改变许多元素。例如 $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$ 的方向为 $u_1=v_1=(1,1)^T/\sqrt2$、$u_2=v_2=(1,-1)^T/\sqrt2$，伸缩量分别为3、1，因此

$$
A_1=3u_1v_1^T=\frac32\begin{pmatrix}1&1\\1&1\end{pmatrix},
\qquad A-A_1=\frac12\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.
$$

<!-- bilingual-en:start -->
All four entries change. The kept action is the common-coordinate direction, while the discarded action distinguishes the two coordinates. Small singular values describe weak matrix action; their usefulness to a particular scientific task is a [[奇异值能量不等于任务信息|separate question]].
<!-- bilingual-en:end -->

四个元素都改变了。保留的是两个坐标共同变化的方向，删除的是区分两坐标的方向。小奇异值只表示矩阵在该方向作用较弱，对具体科学任务是否有用是[[奇异值能量不等于任务信息|另一个问题]]。

<!-- bilingual-en:start -->
## What is fixed when values tie?
<!-- bilingual-en:end -->

*重复值时，哪些部分已经确定*

<!-- bilingual-en:start -->
Choose one SVD before taking its first $k$ pieces. If $1\le k<r$ and $\sigma_k=\sigma_{k+1}$, the cutoff splits a repeated positive block. Its [[重奇异值下方向不唯一|paired bases may rotate]], changing which part is retained. For $A=\operatorname{diag}(3,3)$ and $k=1$, both $\operatorname{diag}(3,0)$ and $\operatorname{diag}(0,3)$ are valid truncations. If a whole repeated block is retained or removed, rotating its paired bases does not change that block's contribution.
<!-- bilingual-en:end -->

应先选定一组 SVD，再取它的前 $k$ 项。若 $1\le k<r$ 且 $\sigma_k=\sigma_{k+1}$，截断位置切穿重复正值块；该块的[[重奇异值下方向不唯一|配对基可以旋转]]，保留部分因而可以改变。例如 $A=\operatorname{diag}(3,3)$、$k=1$ 时，$\operatorname{diag}(3,0)$ 与 $\operatorname{diag}(0,3)$ 都是有效截断。如果整个重复块一同保留或删除，旋转该块配对基则不改变它的总贡献。

> [!important] Construction, optimality and inversion / 构造、最优性与反演
> <!-- bilingual-en:start -->
> This card defines $A_k$. The [[截断SVD最佳低秩近似|Eckart–Young–Mirsky theorem]] explains why it minimizes two common reconstruction errors. The [[谱范数低秩近似不唯一|spectral]] and [[Frobenius低秩近似唯一性|Frobenius]] minimizer sets have different uniqueness rules. [[截断SVD反演|TSVD inversion]] is another operation: it divides by retained positive singular values to estimate an input from an output.
> <!-- bilingual-en:end -->
>
> 本卡定义 $A_k$；[[截断SVD最佳低秩近似|Eckart–Young–Mirsky 定理]]解释它为何使两种常用重构误差最小。[[谱范数低秩近似不唯一|谱范数]]与 [[Frobenius低秩近似唯一性|Frobenius 范数]]的最优解集合有不同唯一性规则。[[截断SVD反演]]则是另一项操作：除以保留的正奇异值，根据输出估计输入。

## 来源与核验

<!-- bilingual-en:start -->
- [Murray et al., *Randomized Numerical Linear Algebra: A Perspective on the Field With an Eye to Software*, §4.1.1, PDF72–73 / printed65–66](https://www.netlib.org/utk/people/JackDongarra/PAPERS/rand-templates.pdf#page=72): truncated outer-product construction and its distinction from retaining all positive components. The definition here includes the explicit zero endpoint.
- [MIT18.065 Lecture7, official summary and problem set](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-7-eckart-young-the-closest-rank-k-matrix-to-a/): leading-component construction and concrete diagonal/tilted examples. The displayed actions, exact rank and tied-block example are checked directly.
<!-- bilingual-en:end -->

- Murray 等，第4.1.1节 PDF72–73 / 印刷65–66页：核对外积截断构造及与保留全部正分量的区别；本卡显式补齐零端点。
- MIT18.065 第7讲官方摘要和题目：核对前几项构造及对角／倾斜方向的具体例子；正文向量作用、精确秩与重复块例子直接计算。
