---
student_os: knowledge-atom
atom_id: LA-SYS-065
aliases:
  - Span of reduced row echelon matrices
status: source-checked
---

<!-- bilingual-en:start -->
# What do all square RREF matrices span?
<!-- bilingual-en:end -->
*行最简形的集合与它张成的空间*

<!-- bilingual-en:start -->
Here we treat each entire matrix as one vector: add matrices entry by entry and multiply every entry by a scalar, as in a [[矩阵空间|matrix space]]. Suppose our building blocks are **all** real $n\times n$ matrices in [[行最简形|reduced row echelon form (RREF)]], where $n$ is a positive integer. Which matrices can we build by taking finite linear combinations of these blocks? This is a question about their [[张成|span]].
<!-- bilingual-en:end -->

这里把每个完整矩阵当作一个向量：对应元素相加，每个元素乘同一个标量，这就是[[矩阵空间]]中的运算。设现成的构件是**全部**实 $n\times n$ [[行最简形|行最简形矩阵（RREF）]]，其中 $n$ 是正整数。把这些构件取有限项线性组合，能够拼出哪些矩阵？这问的是它们的[[张成]]。

<!-- bilingual-en:start -->
The answer is exactly the space of [[三角矩阵|upper triangular matrices]]: entries below the main diagonal must be zero, while every entry on or above it can be chosen freely. If $\mathcal R_n$ denotes the set of those RREF matrices and $U_n$ the upper triangular space, the claim is
<!-- bilingual-en:end -->

答案恰好是[[三角矩阵|上三角矩阵]]组成的空间：主对角线下方全为零，对角线及其上方的元素可以任意选择。用 $\mathcal R_n$ 表示这些 RREF 矩阵的集合，用 $U_n$ 表示上三角空间，结论是

$$\operatorname{span}_{\mathbb R}(\mathcal R_n)=U_n
=\{X=(x_{ij})\in\mathbb R^{n\times n}:x_{ij}=0\text{ whenever }i>j\}.$$

> [!warning] The matrices and their combinations are different collections · 原矩阵集合与线性组合的集合不同
> <!-- bilingual-en:start -->
> The [[单位矩阵|identity matrix]] $I_n$ is in RREF, but $2I_n$ is not: its first nonzero entry in each row is $2$, rather than $1$. Nevertheless, $2I_n$ is a linear combination of RREF matrices. Thus $\mathcal R_n$ itself is not a vector space, even though its span is. We are combining whole matrices of the same size, not finding the [[行空间|row space]] of one fixed matrix.
> <!-- bilingual-en:end -->
>
> [[单位矩阵]] $I_n$ 是 RREF，$2I_n$ 却不是，因为每行首个非零项为 $2$，不是 $1$。但 $2I_n$ 仍是 RREF 矩阵的线性组合。因此，$\mathcal R_n$ 本身不是向量空间，它的张成却是。这里组合的是同尺寸的完整矩阵，不是在求某一个固定矩阵的[[行空间]]。

<!-- bilingual-en:start -->
## Every square RREF matrix is upper triangular
<!-- bilingual-en:end -->
*先证明张成空间不会超出去*

<!-- bilingual-en:start -->
Recall the relevant RREF conditions: zero rows come last, and the first nonzero entry of each nonzero row is a leading $1$, whose position moves strictly right as we go down. If row $i$ is nonzero, all preceding $i-1$ rows are nonzero too. Write $j_i$ for its leading column. Since $1\le j_1<j_2<\cdots<j_i$, we have $j_i\ge i$.
<!-- bilingual-en:end -->

先回顾需要用到的 RREF 条件：零行在最后；每个非零行的首个非零项为领先的 $1$，其位置从上到下严格右移。若第 $i$ 行非零，它之前的 $i-1$ 行也都非零。把该行领先项的列号记作 $j_i$，由 $1\le j_1<j_2<\cdots<j_i$ 得到 $j_i\ge i$。

<!-- bilingual-en:start -->
A position below the diagonal in row $i$ has column index $j<i\le j_i$, so it is to the left of the first nonzero entry and must be zero. A zero row already has zeros in every position. Thus every square RREF matrix is upper triangular. Adding or scaling such matrices preserves all zeros below the diagonal, proving $\operatorname{span}_{\mathbb R}(\mathcal R_n)\subseteq U_n$.
<!-- bilingual-en:end -->

第 $i$ 行中位于对角线下方的位置满足 $j<i\le j_i$，在该行首个非零项左边，所以必为零。零行则所有位置本来就是零。因此每个方形 RREF 都是上三角矩阵；相加或数乘也不会破坏下方这些零，故 $\operatorname{span}_{\mathbb R}(\mathcal R_n)\subseteq U_n$。

<!-- bilingual-en:start -->
## Every upper triangular direction can be generated
<!-- bilingual-en:end -->
*再证明所有上三角矩阵都能得到*

<!-- bilingual-en:start -->
Let $E_{ij}$ be the $n\times n$ matrix with one $1$ in row $i$, column $j$, and zeros elsewhere. It changes only one position when used in a linear combination. Every upper triangular matrix therefore has the expansion
<!-- bilingual-en:end -->

令 $E_{ij}$ 是一个 $n\times n$ 矩阵，只在第 $i$ 行、第 $j$ 列为 $1$，其余为零。在线性组合中，它只控制一个位置。因此，每个上三角矩阵都可以展开为

$$X=\sum_{1\le i\le j\le n}x_{ij}E_{ij}.$$

<!-- bilingual-en:start -->
It is enough to generate each of these $E_{ij}$. Define $P_i=E_{11}+\cdots+E_{ii}$ for $1\le i\le n$, and let $P_0$ be the zero matrix. The first $i$ diagonal entries of $P_i$ are $1$, and all other entries are zero, so $P_i$ is in RREF. For $j>i$, the matrix $P_i+E_{ij}$ is also in RREF: the added entry lies to the right of the leading $1$ in row $i$, in a column with no leading entry. The first $i$ leading columns remain unchanged, and the remaining rows stay zero.
<!-- bilingual-en:end -->

只要能生成每个这样的 $E_{ij}$ 就足够。对 $1\le i\le n$，定义 $P_i=E_{11}+\cdots+E_{ii}$，并令 $P_0$ 为零矩阵。$P_i$ 的前 $i$ 个对角元为 $1$，其他位置全为零，所以是 RREF。当 $j>i$ 时，$P_i+E_{ij}$ 也是 RREF：新增项在第 $i$ 行领先的 $1$ 右边，所在列没有领先项；前 $i$ 个领先列不变，剩余各行仍为零。

$$E_{ii}=P_i-P_{i-1},\qquad
E_{ij}=(P_i+E_{ij})-P_i\quad(i<j).$$

<!-- bilingual-en:start -->
These differences of RREF matrices produce every diagonal and above-diagonal coordinate direction. Hence $U_n\subseteq\operatorname{span}_{\mathbb R}(\mathcal R_n)$, completing the two-inclusion proof. The allowed $E_{ij}$ are also independent: an entry of their linear combination reads off its own coefficient. They form a [[基|basis]], with $n+(n-1)+\cdots+1=n(n+1)/2$ members.
<!-- bilingual-en:end -->

这些 RREF 矩阵之差生成了每一个对角方向与对角线上方的坐标方向，因此 $U_n\subseteq\operatorname{span}_{\mathbb R}(\mathcal R_n)$，两个方向的包含共同证明相等。允许的 $E_{ij}$ 还线性无关，因为组合结果中的每个对应位置都直接读出自己的系数。所以它们构成一组[[基]]，共有 $n+(n-1)+\cdots+1=n(n+1)/2$ 个成员。

<!-- bilingual-en:start -->
## The 3 by 3 exam case
<!-- bilingual-en:end -->
*三阶情形的六维空间*

<!-- bilingual-en:start -->
For $n=3$, the result is the six-dimensional space
<!-- bilingual-en:end -->

当 $n=3$ 时，得到六维空间

$$U_3=\left\{\begin{pmatrix}a&b&c\\0&d&e\\0&0&f\end{pmatrix}:a,b,c,d,e,f\in\mathbb R\right\}.$$

<!-- bilingual-en:start -->
A basis consisting of RREF matrices themselves is the following list. The notation $P_2$ means $E_{11}+E_{22}$, and $I_3$ has three diagonal ones:
<!-- bilingual-en:end -->

下面这一组基的成员本身全是 RREF。这里 $P_2=E_{11}+E_{22}$，$I_3$ 的三个对角元均为一：

$$\begin{aligned}
G_1&=E_{11},&G_2&=E_{11}+E_{12},&G_3&=E_{11}+E_{13},\\
G_4&=P_2,&G_5&=P_2+E_{23},&G_6&=I_3.
\end{aligned}$$

<!-- bilingual-en:start -->
Indeed, $E_{11}=G_1$, $E_{12}=G_2-G_1$, $E_{13}=G_3-G_1$, $E_{22}=G_4-G_1$, $E_{23}=G_5-G_4$ and $E_{33}=G_6-G_4$. Thus these six matrices span the six-dimensional space and are a basis.
<!-- bilingual-en:end -->

因为 $E_{11}=G_1$、$E_{12}=G_2-G_1$、$E_{13}=G_3-G_1$、$E_{22}=G_4-G_1$、$E_{23}=G_5-G_4$、$E_{33}=G_6-G_4$，这六个矩阵张成整个六维空间，因而是一组基。

> [!example] Building a matrix that is not in RREF · 拼出一个并非 RREF 的矩阵
> $$\begin{pmatrix}2&3&-1\\0&4&5\\0&0&6\end{pmatrix}
> =-4G_1+3G_2-G_3-7G_4+5G_5+6G_6.$$
>
> <!-- bilingual-en:start -->
> The above-diagonal entries are $3,-1,5$, read from the coefficients of $G_2,G_3,G_5$. On the diagonal, the sums are $-4+3-1-7+5+6=2$, $-7+5+6=4$ and $6$. All entries below it remain zero. The result is in the span even though its leading entries are not ones.
> <!-- bilingual-en:end -->
>
> 对角线上方的元素为 $3,-1,5$，分别从 $G_2,G_3,G_5$ 的系数读出。对角元则依次为 $-4+3-1-7+5+6=2$、$-7+5+6=4$ 和 $6$；下方仍全为零。结果的领先项不是一，但完全属于张成空间。

<!-- bilingual-en:start -->
The smallest allowed size also fits: the real $1\times1$ RREF matrices are $[0]$ and $[1]$, and their span consists of every $[a]$ with $a\in\mathbb R$. Its dimension is $1=1(1+1)/2$.
<!-- bilingual-en:end -->

最小允许尺寸也符合结论：实一阶 RREF 只有 $[0]$、$[1]$，它们张成全部 $[a]$，其中 $a\in\mathbb R$；维数为 $1=1(1+1)/2$。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex1s.pdf#page=5|MIT 18.06SC Unit 1 Exam Solutions, Question 4(c), PDF p.5 / printed p.6]]: the three-by-three span is the upper triangular space. The two-inclusion proof above establishes the result for every positive integer $n$; the displayed six generators and numerical combination are verified directly.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.11sum.pdf#page=1|MIT 18.06SC, Matrix spaces; rank 1; small world graphs, p.1]]: whole matrices as vectors and the six standard upper triangular basis matrices.
- [Margalit and Rabinoff, Interactive Linear Algebra, §1.2.2](https://textbooks.math.gatech.edu/ila/row-reduction.html): the precise RREF conditions used to check $P_i$, $P_i+E_{ij}$ and the leading-column argument.
<!-- bilingual-en:end -->

- MIT Unit 1 Exam 官方解答第4(c)题，PDF第5页、印刷第6页：给出三阶 RREF 张成上三角空间；上面的双向包含证明建立任意正整数 $n$ 的结论，六个生成矩阵及数值组合另作直接核验。
- MIT《Matrix spaces; rank 1; small world graphs》第1页：核对完整矩阵作为向量，以及六个标准上三角基矩阵。
- Margalit–Rabinoff《Interactive Linear Algebra》§1.2.2：核对用于检查 $P_i$、$P_i+E_{ij}$ 及领先列论证的准确 RREF 条件。
