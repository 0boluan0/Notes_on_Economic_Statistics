---
aliases:
  - "Column-pivoted QR rank-revealing boundary"
  - "CPQR rank-revealing limitations"
  - "列主元 QR 的秩揭示边界"
student_os: knowledge-atom
atom_id: LA-PROJ-023
atom_set: orthogonal-projection-least-squares
atom_type: numerical-boundary
status: source-checked
requires:
  - "[[QR分解]]"
  - "[[数值秩]]"
leads_to:
  - "[[CPQR回代不保最小范数]]"
  - "[[秩亏最小二乘算法]]"
related:
  - "[[条件性与算法稳定性]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
  - "[[广义逆与最小范数解.canvas]]"
---

<!-- bilingual-en:start -->
# What CPQR reveals about rank
<!-- bilingual-en:end -->

*CPQR 的秩揭示边界*

<!-- bilingual-en:start -->
When several matrix columns nearly repeat the same information, we would like to select useful independent directions first. **Column-pivoted QR (CPQR)** changes the column order while constructing a [[QR分解|QR factorization]]. At each step, ordinary CPQR chooses a remaining column whose part outside the directions already selected has the largest length. This is a useful way to look for dependence. Its local choices do not, by themselves, provide every global guarantee of a strong rank-revealing factorization.
<!-- bilingual-en:end -->

当几个矩阵列近乎重复相同信息时，我们希望先选出有用的独立方向。**列主元 QR（CPQR）**在构造 [[QR分解]]的同时改变列顺序。普通 CPQR 每一步查看剩余列在已选方向之外的部分，优先选取这部分长度最大的列。这是寻找依赖的实用方法，但这些局部选择本身不提供强秩揭示分解的全部整体保证。

> [!example] New information left in a column / 一列还剩多少新方向
> <!-- bilingual-en:start -->
> Take $a_1=(1,0)^T$ and $a_2=(3/5,4/5)^T$, both of length 1. If the tie is resolved by choosing $a_1$ first, the part of $a_2$ along $a_1$ is $(3/5,0)^T$ and the part left over is $(0,4/5)^T$. Thus $A=[a_1\ a_2]$ already has $Q=I_2$ and
>
> $$R=\begin{pmatrix}1&3/5\\0&4/5\end{pmatrix}.$$
>
> The diagonal $1,4/5$ records successive remaining lengths. It is not the list of [[奇异值|singular values]], the matrix's stretch factors: those are $\sqrt{8/5}$ and $\sqrt{2/5}$, obtained from the eigenvalues $8/5,2/5$ of $A^TA$. This example distinguishes the two quantities; it does not say that CPQR made a bad rank decision here. $I_2$ is the [[单位矩阵|identity matrix]], and $T$ denotes [[矩阵转置|transpose]].
> <!-- bilingual-en:end -->
>
> 取两个长度均为1的列 $a_1=(1,0)^T$、$a_2=(3/5,4/5)^T$。并列时若先取 $a_1$，$a_2$ 沿它的部分为 $(3/5,0)^T$，剩余部分为 $(0,4/5)^T$，因此 $Q=I_2$、$R$ 如上。对角元 $1,4/5$ 记录逐步剩余的长度；[[奇异值]]却是矩阵的伸缩倍数，此处由 $A^TA$ 的特征值 $8/5,2/5$ 得到 $\sqrt{8/5},\sqrt{2/5}$。这个例子区分两种量，并没有说此处 CPQR 的秩判断失效。$I_2$ 是[[单位矩阵]]，$T$ 表示[[矩阵转置|转置]]。

<!-- bilingual-en:start -->
For a real $m\times n$ matrix with $m\ge n$, write

$$AP=Q\begin{pmatrix}R_{11}&R_{12}\\0&R_{22}\end{pmatrix}.$$

The [[置换矩阵|permutation matrix]] $P$ reorders the $n$ columns, while the $m\times m$ [[正交矩阵|orthogonal matrix]] $Q$ changes output coordinates without changing lengths. Choose a candidate rank $k$ with $1\le k<n$. The leading [[三角矩阵|upper-triangular]] block $R_{11}$ is $k\times k$; $R_{12}$ has $n-k$ columns and $R_{22}$ has $m-k$ rows and $n-k$ columns. If $R_{11}$ is invertible, the selected columns are independent. A small $R_{22}$ says that the other columns have little remaining component outside their span. Whether that remainder counts as zero is a [[数值秩|numerical-rank decision]] with a stated tolerance, rather than proof that it is exactly zero.
<!-- bilingual-en:end -->

对 $m\ge n$ 的实 $m\times n$ 矩阵，写出上式。[[置换矩阵]] $P$ 重排 $n$ 个列，$m\times m$ 的[[正交矩阵]] $Q$ 在保持长度的前提下改变输出坐标。选候选秩 $k$，其中 $1\le k<n$；前部[[三角矩阵|上三角]]块 $R_{11}$ 为 $k\times k$，$R_{12}$ 有 $n-k$ 列，尾块 $R_{22}$ 为 $(m-k)\times(n-k)$。若 $R_{11}$ 可逆，选中的列就独立。$R_{22}$ 很小说明其他列在这些列的张成空间之外剩余很少；是否把它视为零，是需声明容差的[[数值秩]]决策，不是它严格为零的证明。

<!-- bilingual-en:start -->
## What a strong rank-revealing guarantee adds
<!-- bilingual-en:end -->

## “强秩揭示”多要求了什么

<!-- bilingual-en:start -->
A useful leading block should preserve the matrix's important directions, not merely have visually descending diagonal entries. It must be compared with the singular values of the whole matrix. The matrix $R_{11}^{-1}R_{12}$ also matters: its columns are the coefficients used to express the projections of unselected columns in the selected-column basis. Very large coefficients can make that representation sensitive.
<!-- bilingual-en:end -->

有用的领先块应保留原矩阵的重要方向，而不只是对角元看起来递减。需要把它与整个矩阵的奇异值比较。$R_{11}^{-1}R_{12}$ 也重要：它的每一列都是将未选列的投影写成已选列组合时所需的系数；系数过大可能使这种表示敏感。

> [!note] A quantitative comparison / 用不等式说明保证
> <!-- bilingual-en:start -->
> For the real $m\ge n$ setting, the Gu–Eisenstat strong RRQR construction with parameter $f>1$ and a retained rank $1\le k\le\operatorname{rank}(A)$, $k<n$, supplies bounds including
>
> $$q=\sqrt{1+f^2k(n-k)},\qquad
> \sigma_{\min}(R_{11})\ge\frac{\sigma_k(A)}{q},\qquad
> \|R_{22}\|_2\le q\,\sigma_{k+1}(A),\qquad
> |(R_{11}^{-1}R_{12})_{ij}|\le f.$$
>
> Here singular values are ordered from largest to smallest; $\sigma_{\min}$ is the smallest one of $R_{11}$, and [[谱范数|$\|R_{22}\|_2$]] is the largest stretch of the trailing block. The first bound limits how much the selected basis can lose relative to the $k$th direction of $A$. The second controls the discarded part. The last bounds each dependence coefficient. These are relative guarantees: if $\sigma_k(A)$ itself is tiny relative to $\sigma_1(A)$, they do not promise an absolutely well-conditioned retained problem.
> <!-- bilingual-en:end -->
>
> 在上述实数 $m\ge n$ 情形中，Gu–Eisenstat 强 RRQR 构造取参数 $f>1$，保留秩满足 $1\le k\le\operatorname{rank}(A)$ 且 $k<n$，可给出上列界。奇异值从大到小排列；$\sigma_{\min}$ 是 $R_{11}$ 的最小奇异值，[[谱范数|$\|R_{22}\|_2$]] 是尾块的最大伸缩倍数。第一项限制已选基相对于原矩阵第 $k$ 个方向损失多少，第二项控制舍去的部分，最后一项限制每个依赖系数。这些是相对保证：如果 $\sigma_k(A)$ 相对 $\sigma_1(A)$ 本来就很小，不能据此承诺保留问题绝对良态。

<!-- bilingual-en:start -->
Ordinary CPQR makes greedy column choices and may not satisfy such polynomial-size bounds in the worst case. Gu and Eisenstat exhibit a family in which that separation becomes arbitrarily severe as the dimension grows; their stronger algorithms use additional column exchanges to control it. A diagonal drop can therefore be useful evidence without being this theorem. The issue is the strength of the bound, not whether every CPQR result must fail.
<!-- bilingual-en:end -->

普通 CPQR 作逐步贪心列选择，最坏情况下可能不满足这种多项式量级的界。Gu 与 Eisenstat 给出一族矩阵，其差距随维数增长而变得严重，并以额外列交换控制它。因此，对角元突然变小可以提供有用证据，却不等于已经证明上述定理。区别在保证有多强，不在于每次 CPQR 都会失败。

<!-- bilingual-en:start -->
## Using the evidence in a solver
<!-- bilingual-en:end -->

## 实际使用边界

<!-- bilingual-en:start -->
LAPACK `DGELSY` uses CPQR and an estimate of the leading block's condition number to choose its effective rank using `RCOND`. This is a different test from directly counting singular values above an SVD threshold; the same parameter value need not give the same rank. Report the matrix scaling, tolerance, rank rule and selected rank. If an argument needs strong RRQR bounds, use a method with those bounds or verify the needed quantities, rather than inferring them from a diagonal plot.

Choosing the rank and choosing the shortest solution are also separate tasks. Even an exact rank decision does not make pivot-only back substitution return minimum norm; see [[CPQR回代不保最小范数|basic versus minimum-norm solutions]]. A [[秩亏最小二乘算法|complete orthogonal or SVD solve]] handles that extra requirement. A [[条件性与算法稳定性|stable algorithm]] still cannot remove sensitivity already present in the data problem.
<!-- bilingual-en:end -->

LAPACK `DGELSY` 使用 CPQR，并结合领先块条件数的估计与 `RCOND` 选择有效秩。这与直接计数超过 SVD 阈值的奇异值是不同判定，同一个参数值不保证给出相同秩。应报告矩阵缩放、容差、秩判定规则和选定秩。若论证依赖强 RRQR 的界，就要采用有该保证的方法，或核实所需量，不能仅从对角元图形推断。

选秩与选最短解还是两项任务。即使秩判断完全正确，仅主元回代也不保证最小范数，见[[CPQR回代不保最小范数|基本解与最小范数解]]；[[秩亏最小二乘算法|完整正交或 SVD 求解]]处理后一个要求。[[条件性与算法稳定性|稳定算法]]仍不能消除原数据问题已有的敏感性。

## 来源与核验

<!-- bilingual-en:start -->
- [Gu and Eisenstat (1996), *Efficient Algorithms for Computing a Strong Rank-Revealing QR Factorization*, §§1.1–1.2 and §2 Example 1, PDF pp.1–4](https://math.berkeley.edu/~mgu/MA273/Strong_RRQR.pdf) ([DOI](https://doi.org/10.1137/0917055)): strong-factorization inequalities, column-coupling control, ordinary CPQR's worst-case limitation and extra exchanges. The two-column arithmetic is computed directly.
- [LAPACK DGELSY, Purpose and RCOND](https://www.netlib.org/lapack/explore-html/d6/d4b/dgelsy_8f_source.html): the leading-block condition estimate and effective-rank rule. [DGELSD, RCOND and RANK](https://www.netlib.org/lapack/explore-html/d9/d67/group__gelsd_ga0bee7e1b9e7e43f59ecf2419b2759c42.html) gives the distinct SVD-threshold rule.
<!-- bilingual-en:end -->

- Gu 与 Eisenstat 原论文第1.1–1.2节、第2节例1核对强分解不等式、列耦合约束、普通 CPQR 的最坏情况边界与额外列交换；两列算例由直接计算核验。
- LAPACK DGELSY 的 Purpose、RCOND 段核对领先块条件估计与有效秩规则；DGELSD 的 RCOND、RANK 段核对另一种 SVD 阈值规则。
