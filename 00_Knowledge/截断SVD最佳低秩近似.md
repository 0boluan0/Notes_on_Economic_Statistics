---
aliases:
- 截断 SVD 最佳低秩近似
- Best low-rank approximation by truncated SVD
- Eckart–Young–Mirsky theorem
student_os: knowledge-atom
atom_id: LA-SVD-003
atom_set: singular-value-decomposition-low-rank
atom_type: theorem
status: source-checked
requires:
- '[[低秩近似]]'
- '[[截断SVD]]'
- '[[SVD秩一展开]]'
- '[[谱范数]]'
- '[[谱范数与最大奇异值]]'
- '[[Frobenius范数]]'
- '[[Frobenius范数与奇异值]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
leads_to:
- '[[Frobenius低秩近似唯一性]]'
- '[[谱范数低秩近似不唯一]]'
related:
- '[[奇异值能量比]]'
- '[[截断SVD与岭正则化]]'
---

<!-- bilingual-en:start -->
# Best low-rank approximation by truncated SVD
<!-- bilingual-en:end -->

*截断 SVD 最佳低秩近似*

<!-- bilingual-en:start -->
Suppose a matrix must be represented using at most $k$ independent output directions. Which information should be kept to make the replacement as close as possible to the original? The Eckart–Young–Mirsky theorem says that keeping the largest [[奇异值|singular stretches]] gives a best answer for two common error measures. It also tells us exactly how much error remains.
<!-- bilingual-en:end -->

如果必须用至多 $k$ 个独立输出方向表示一个矩阵，应该保留哪些信息，才能让替代矩阵尽可能接近原矩阵？Eckart–Young–Mirsky 定理说明：保留最大的[[奇异值|奇异伸缩量]]，能在两种常用误差度量下得到最佳答案，并能精确读出剩余误差。

<!-- bilingual-en:start -->
## Specify what an error measures
<!-- bilingual-en:end -->

*先说明怎样计量误差*

<!-- bilingual-en:start -->
For matrices $A,B$ of the same size, $E=A-B$ is the error matrix: on input $x$, the output discrepancy is $Ex$. Its [[谱范数|spectral norm]] $\|E\|_2$ is the largest output error over all unit inputs. Its [[Frobenius范数|Frobenius norm]] $\|E\|_F$ is the square root of the sum of squared entry magnitudes. The first measures the worst direction; the second accumulates error throughout the matrix. A [[低秩近似|low-rank problem]] also fixes the allowed rank, rather than minimizing over arbitrary replacements.
<!-- bilingual-en:end -->

同尺寸矩阵 $A,B$ 的差 $E=A-B$ 是误差矩阵：输入为 $x$ 时，输出差就是 $Ex$。[[谱范数]] $\|E\|_2$ 是全部单位输入中的最大输出误差；[[Frobenius范数]] $\|E\|_F$ 是全部元素模平方之和再开平方。前者看最坏方向，后者累计整张矩阵的误差。[[低秩近似]]还必须固定允许的秩，而不是任意选择替代矩阵。

> [!example] One retained direction / 只保留一个方向
> <!-- bilingual-en:start -->
> For the [[对角矩阵|diagonal matrix]] $A=\operatorname{diag}(5,2,1)$, the main-diagonal entries are $5,2,1$ and all other entries are zero. Its [[截断SVD|rank-one truncation]] is $A_1=\operatorname{diag}(5,0,0)$. The residual is $\operatorname{diag}(0,2,1)$, with spectral norm 2 and Frobenius norm $\sqrt5$. Keeping only the middle direction instead gives $\operatorname{diag}(0,2,0)$, whose errors are 5 and $\sqrt{26}$. The theorem goes further: no tilted rank-one candidate can beat the first pair of error values either.
> <!-- bilingual-en:end -->
>
> 对[[对角矩阵]] $A=\operatorname{diag}(5,2,1)$，主对角元素为 $5,2,1$，其余元素为零。它的[[截断SVD|秩一截断]]为 $A_1=\operatorname{diag}(5,0,0)$。残差是 $\operatorname{diag}(0,2,1)$，谱范数为2，Frobenius 范数为 $\sqrt5$。若只保留中间方向，得到 $\operatorname{diag}(0,2,0)$，两种误差变成5和 $\sqrt{26}$。定理还保证：即使选择倾斜方向的秩一矩阵，也不能把前一组误差变得更小。

<!-- bilingual-en:start -->
## The minimum errors are the discarded tail
<!-- bilingual-en:end -->

*最小误差来自被舍弃的尾部*

<!-- bilingual-en:start -->
Let $A\in\mathbb F^{m\times n}$, where $m,n\ge1$ and $\mathbb F$ is $\mathbb R$ or $\mathbb C$. Write $r=\operatorname{rank}(A)$ and order its positive singular values as $\sigma_1\ge\cdots\ge\sigma_r>0$. The [[紧SVD|compact SVD]] gives paired unit directions $u_i,v_i$, with $^*$ denoting conjugate transpose. For an integer $0\le k<r$, set $A_k=\sum_{i=1}^k\sigma_i u_iv_i^*$, using the zero matrix when $k=0$. Then
<!-- bilingual-en:end -->

设 $A\in\mathbb F^{m\times n}$，$m,n\ge1$，$\mathbb F$ 为 $\mathbb R$ 或 $\mathbb C$。记 $r=\operatorname{rank}(A)$，正奇异值按 $\sigma_1\ge\cdots\ge\sigma_r>0$ 排序。[[紧SVD]]提供配对单位方向 $u_i,v_i$，$^*$ 表示共轭转置。对整数 $0\le k<r$，令 $A_k=\sum_{i=1}^k\sigma_i u_iv_i^*$，其中 $k=0$ 时取零矩阵。则

$$
\min_{\operatorname{rank}(B)\le k}\|A-B\|_2
=\|A-A_k\|_2=\sigma_{k+1},
\qquad
\min_{\operatorname{rank}(B)\le k}\|A-B\|_F
=\|A-A_k\|_F=\left(\sum_{i=k+1}^{r}\sigma_i^2\right)^{1/2}.
$$

<!-- bilingual-en:start -->
Both minima range over all real or complex matrices $B$ of the same size as $A$. “At most $k$” permits lower rank too. For $1\le k<r$, no gap between $\sigma_k$ and $\sigma_{k+1}$ is required to attain these values. If $k\ge r$, $B=A$ is allowed and both minima are zero; zero error forces $B=A$. Thus the formula does not need an undefined next singular value beyond the list. For the zero matrix, this zero-error case applies for every nonnegative rank budget.
<!-- bilingual-en:end -->

两个最小值都在与 $A$ 同尺寸、同实／复数域的全部矩阵 $B$ 中求取。“至多 $k$”也允许更低的秩。当 $1\le k<r$ 时，达到这些误差不要求 $\sigma_k$ 与 $\sigma_{k+1}$ 之间存在严格间隔。若 $k\ge r$，$B=A$ 已允许，两种最小误差均为零，且零误差强制 $B=A$；无需引用列表之外未定义的“下一奇异值”。零矩阵对每个非负秩预算都属于这一零误差情形。

<!-- bilingual-en:start -->
## Why the truncation attains the bound
<!-- bilingual-en:end -->

*为什么截断能达到这个界*

<!-- bilingual-en:start -->
The residual is precisely the [[SVD秩一展开|sum of the discarded perpendicular pieces]], $\sum_{i=k+1}^r\sigma_i u_iv_i^*$. Its largest stretch is $\sigma_{k+1}$ by the [[谱范数与最大奇异值|spectral-norm formula]], and its squared Frobenius norm is the tail sum by [[Frobenius范数与奇异值|the Frobenius formula]]. This proves that the displayed errors are achievable. To prove optimality, we must also rule out every other candidate.
<!-- bilingual-en:end -->

残差恰好是[[SVD秩一展开|被舍弃的正交小块之和]] $\sum_{i=k+1}^r\sigma_i u_iv_i^*$。由[[谱范数与最大奇异值|谱范数公式]]，其最大伸缩为 $\sigma_{k+1}$；由[[Frobenius范数与奇异值|Frobenius 公式]]，其范数平方为尾部平方和。这证明显示误差能够达到；还需排除其他候选能够更好的可能，才能证明最优。

<!-- bilingual-en:start -->
For the spectral bound, restrict any rank-at-most-$k$ candidate $B$ to the $(k+1)$-dimensional space spanned by $v_1,\ldots,v_{k+1}$. Its output still has dimension at most $k$, so [[秩零度定理|rank–nullity]] supplies a nonzero input it sends to zero. Normalize that input to $x=\sum_{i=1}^{k+1}c_i v_i$ with $\sum_i|c_i|^2=1$ and $Bx=0$. Perpendicular output directions give
<!-- bilingual-en:end -->

证明谱范数下界时，把任意秩至多 $k$ 的候选 $B$ 限制在 $v_1,\ldots,v_{k+1}$ 张成的 $k+1$ 维空间上。输出维数仍至多为 $k$，所以[[秩零度定理]]保证存在一个被它送到零的非零输入。把该输入归一化为 $x=\sum_{i=1}^{k+1}c_i v_i$，满足 $\sum_i|c_i|^2=1$、$Bx=0$。输出方向相互垂直，故

$$
\|A-B\|_2^2\ge\|(A-B)x\|_2^2
=\|Ax\|_2^2
=\sum_{i=1}^{k+1}\sigma_i^2|c_i|^2
\ge\sigma_{k+1}^2.
$$

<!-- bilingual-en:start -->
A single direction proves the spectral bound, but not the whole Frobenius tail sum. For that, let $P$ be [[正交投影|orthogonal projection]] onto the [[列空间|column space]] of $B$, of dimension $d\le k$. Each column of $A$ is at least as close to its projection as to the corresponding column of $B$. Summing these squared distances gives
<!-- bilingual-en:end -->

一个方向足以证明谱范数下界，却不足以证明整个 Frobenius 尾部平方和。为此，令 $P$ 为向 $B$ 的[[列空间]]作[[正交投影]]的映射，该空间维数为 $d\le k$。$A$ 的每一列到其投影的距离，不大于到 $B$ 对应列的距离；把这些距离平方相加，得到

$$
\|A-B\|_F^2\ge\|A-PA\|_F^2
=\|A\|_F^2-\|PA\|_F^2.
$$

<!-- bilingual-en:start -->
Complete the left singular directions to an [[标准正交组|orthonormal basis]] $u_1,\ldots,u_m$, and put $w_i=\|Pu_i\|_2^2$. These numbers lie between 0 and 1 and sum to $d$: in an orthonormal basis of the $d$-dimensional projection space, each unit basis vector contributes one to the sum. The SVD then gives $\|PA\|_F^2=\sum_{i=1}^r\sigma_i^2w_i$. There is at most $k$ units of weight to allocate, no direction can receive more than one, and the coefficients $\sigma_i^2$ are decreasing. Hence the largest possible captured total is $\sum_{i=1}^k\sigma_i^2$. Subtracting from the full total leaves at least $\sum_{i=k+1}^r\sigma_i^2$. This proves the Frobenius lower bound, including complex matrices because squared magnitudes and conjugate inner products are used throughout.
<!-- bilingual-en:end -->

把左奇异方向补成[[标准正交组|标准正交基]] $u_1,\ldots,u_m$，令 $w_i=\|Pu_i\|_2^2$。这些数都在0与1之间，总和为 $d$：在投影空间的 $d$ 个标准正交基方向上展开，每个单位基方向对这个总和贡献1。SVD 给出 $\|PA\|_F^2=\sum_{i=1}^r\sigma_i^2w_i$。可分配的权重总共至多 $k$，每个方向至多分到1，而系数 $\sigma_i^2$ 已从大到小排列，所以最多能保留 $\sum_{i=1}^k\sigma_i^2$。从完整总量中扣除它，至少剩下 $\sum_{i=k+1}^r\sigma_i^2$，从而证明 Frobenius 下界。全程使用模平方与共轭内积，复矩阵同样适用。

<!-- bilingual-en:start -->
## Optimal error does not identify a unique matrix
<!-- bilingual-en:end -->

*最小误差未必确定唯一矩阵*

<!-- bilingual-en:start -->
For singular values $9,4,4,1$ and $k=2$, the best errors are $4$ and $\sqrt{17}$. Any valid truncation reaches them even though the cutoff splits the repeated value 4. The [[Frobenius低秩近似唯一性|Frobenius uniqueness criterion]] concerns that boundary gap; [[谱范数低秩近似不唯一|spectral minimizers remain nonunique]] whenever $1\le k<r$, even with a strict gap. The theorem optimizes these matrix errors, not prediction, classification or causal usefulness. Those are [[奇异值能量不等于任务信息|different objectives]].
<!-- bilingual-en:end -->

奇异值为 $9,4,4,1$、$k=2$ 时，最佳误差为4和 $\sqrt{17}$。尽管截断切穿重复值4，任何有效截断都达到这两个值。[[Frobenius低秩近似唯一性|Frobenius 唯一性判据]]关心截断处间隔；只要 $1\le k<r$，[[谱范数低秩近似不唯一|谱范数最优解仍不唯一]]，即使间隔严格为正。定理优化的是这些矩阵误差，并非预测、分类或因果用途；后者属于[[奇异值能量不等于任务信息|不同目标]]。

<!-- visual-lab:links:start -->
## 交互理解

- [[07_Programme/可视化实验室/实验/svd|奇异值分解与低秩重构]]
<!-- visual-lab:links:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Murray et al., *Randomized Numerical Linear Algebra: A Perspective on the Field With an Eye to Software*, §4.1.1, equations4.4–4.6, PDF72–73 / printed65–66](https://www.netlib.org/utk/people/JackDongarra/PAPERS/rand-templates.pdf#page=72): truncation, optimality and both residual norms. This card explicitly states the at-most-rank constraint and endpoint cases.
- [EPFL, *Best low-rank approximation*, PDF20 and24–26](https://sma.epfl.ch/~anchpcommon/lecture1.pdf#page=24): the two norm objectives, optimality and distinct uniqueness behavior. The lower bounds and numerical examples above are derived directly, with squared singular values retained in squared-length calculations.
- [MIT18.065 Lecture7, official summary and problem set](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-7-eckart-young-the-closest-rank-k-matrix-to-a/): best-rank-$k$ interpretation and small matrix exercises.
- Historical reference: Eckart and Young (1936), [*The Approximation of One Matrix by Another of Lower Rank*](https://doi.org/10.1007/BF02288367). The original full text was not available in this verification; the opened sources above support the claims.
<!-- bilingual-en:end -->

- Murray 等，第4.1.1节式4.4–4.6，PDF72–73 / 印刷65–66页：核对截断、最优性和两种残差范数；本卡显式说明“秩至多”约束及端点。
- EPFL 讲义 PDF20、24–26页：核对两种范数目标、最优性及不同唯一性行为。正文下界和数值例子直接推导，长度平方计算保留奇异值的平方。
- MIT18.065 第7讲官方摘要和题目：核对最佳秩 $k$ 的解释与小矩阵练习。
- 保留 Eckart 与 Young（1936）的原始文献入口作为历史参考；本轮未取得其全文，主张由上述已打开来源支持。
