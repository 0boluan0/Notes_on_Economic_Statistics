---
aliases:
  - Heine-Borel theorem
  - Heine–Borel theorem
  - Heine-Borel 定理的边界
student_os: knowledge-atom
atom_id: TOPO-MET-013
atom_set: topology-foundations
atom_type: theorem-boundary
status: source-checked
requires:
  - "[[紧致性]]"
  - "[[闭集]]"
part_of:
  - "[[拓扑基础：开集、闭集与连续.canvas]]"
related:
  - "[[极值定理]]"
---

<!-- bilingual-en:start -->
# The Heine–Borel theorem and its scope
<!-- bilingual-en:end -->

*Heine–Borel 定理的适用边界*

<!-- bilingual-en:start -->
A closed interval such as $[-2,3]$ cannot escape to infinity and includes its endpoints. In finite-dimensional Euclidean space, these two features have a precise payoff: **closed and bounded is equivalent to compact**. The space is part of the theorem. In a different metric space, even a closed, bounded set can fail to be compact.
<!-- bilingual-en:end -->

闭区间 $[-2,3]$ 不会延伸到无穷远，也包含自己的端点。在有限维欧氏空间中，这两种特征有一个精确结论：**闭且有界等价于紧致**。所处空间是定理条件的一部分。换成别的度量空间，即使闭且有界，仍可能不紧致。

<!-- bilingual-en:start -->
Let $n\ge1$ be a finite integer and use the usual [[欧氏范数|Euclidean norm]] $\|x\|_2=(\sum_{j=1}^n x_j^2)^{1/2}$ on $\mathbb R^n$, where $x_j$ is the $j$th coordinate. A subset $K$ is [[闭集|closed in $\mathbb R^n$]] when its complement is open. It is bounded when $\|x\|_2\le R$ for all $x\in K$ and some finite number $R\ge0$. This is equivalent to containment in a ball about any suitably chosen centre.
<!-- bilingual-en:end -->

设 $n\ge1$ 为有限整数，在 $\mathbb R^n$ 上使用通常的[[欧氏范数]] $\|x\|_2=(\sum_{j=1}^n x_j^2)^{1/2}$，其中 $x_j$ 是第 $j$ 个坐标。子集 $K$ 在 $\mathbb R^n$ 中为[[闭集]]，指其补集开；有界指存在有限 $R\ge0$，使所有 $x\in K$ 都满足 $\|x\|_2\le R$。这与能够被某个中心的一个有限半径球容纳等价。

> [!important] Finite-dimensional Euclidean space · 有限维欧氏空间
> <!-- bilingual-en:start -->
> For $K\subseteq\mathbb R^n$ with its usual topology,
> $$K\text{ is compact}\quad\Longleftrightarrow\quad K\text{ is closed in }\mathbb R^n\text{ and bounded}.$$
> Here [[紧致性|compactness]] retains its open-cover meaning. The empty set is included. In an arbitrary metric space, compactness still implies closedness and boundedness, but the converse needs additional assumptions.
> <!-- bilingual-en:end -->
>
> 对通常拓扑的 $K\subseteq\mathbb R^n$，上式成立；[[紧致性]]仍按每个开覆盖都有有限子覆盖定义，空集也包括在内。在任意度量空间中，紧致仍推出闭且有界，但反方向需要额外条件。

<!-- bilingual-en:start -->
## Why compact metric sets are closed and bounded
<!-- bilingual-en:end -->

*紧致度量集合为什么闭且有界*

<!-- bilingual-en:start -->
Take a nonempty compact subset $K$ of a metric space $(X,d)$ and choose $x_0\in K$. The balls $B_m(x_0)$, with positive integer radii $m$, cover $K$: each point has a finite distance from $x_0$. A finite subcover has a largest radius, and that one ball already contains $K$. Hence $K$ is bounded.
<!-- bilingual-en:end -->

取度量空间 $(X,d)$ 中非空紧致子集 $K$，选 $x_0\in K$。以正整数 $m$ 为半径的球 $B_m(x_0)$ 覆盖 $K$，因为每个点与 $x_0$ 的距离都有限。有限子覆盖有最大半径，而这一个最大的球就包含了 $K$，所以 $K$ 有界。

<!-- bilingual-en:start -->
To show closedness, fix any point $z\in X\setminus K$. For each $x\in K$, put $r_x=d(x,z)/3>0$. The balls $B_{r_x}(x)$ cover $K$; select finitely many, centred at $x_1,\ldots,x_m$. Set $\delta=\min_j r_{x_j}>0$. If a point $y$ lay in both $B_\delta(z)$ and $K$, some selected ball would contain it, and the triangle inequality would give
$$3r_{x_j}=d(z,x_j)\le d(z,y)+d(y,x_j)<\delta+r_{x_j}\le2r_{x_j},$$
a contradiction. Thus every point outside $K$ has a ball disjoint from $K$, so the complement is open. The empty set is already closed and bounded.
<!-- bilingual-en:end -->

要证明闭性，固定任意 $z\in X\setminus K$。对每个 $x\in K$，令 $r_x=d(x,z)/3>0$。这些球 $B_{r_x}(x)$ 覆盖 $K$，选出有限多个，中心记为 $x_1,\ldots,x_m$，再令 $\delta=\min_j r_{x_j}>0$。若某个 $y$ 同时在 $B_\delta(z)$ 和 $K$ 中，它就落在某个所选球内；三角不等式便给出上式的矛盾。因此每个 $K$ 外的点都有一个不碰到 $K$ 的球，补集开。空集本身已经闭且有界。

<!-- bilingual-en:start -->
## Why the converse works in finitely many real coordinates
<!-- bilingual-en:end -->

*反方向为什么在有限个实坐标中成立*

<!-- bilingual-en:start -->
Assume $K\subseteq\mathbb R^n$ is closed and bounded, and take any sequence in $K$. It lies in a box $[-M,M]^n$ for some $M>0$. Bisect each coordinate interval, making $2^n$ smaller closed boxes. At least one box contains terms at infinitely many indices. Keep such a box, bisect it again, and repeat. This produces nested boxes, each containing infinitely many sequence terms, whose side lengths after $k$ steps are $2M/2^k$.
<!-- bilingual-en:end -->

假设 $K\subseteq\mathbb R^n$ 闭且有界，任取其中一条序列。存在 $M>0$ 使它位于盒子 $[-M,M]^n$ 内。把每个坐标区间二等分，得到 $2^n$ 个较小的闭盒子，其中至少一个包含无穷多个索引对应的序列项。保留这样的盒子，再二等分，依此继续。得到一列嵌套盒子，每个都含无穷多项，第 $k$ 次分割后的边长为 $2M/2^k$。

<!-- bilingual-en:start -->
In each coordinate, the nested closed intervals have a common point: their left endpoints increase and are bounded above, so their [[上确界|least upper bound]] lies in every interval. This is the [[实数的确界性质|completeness property of the real numbers]] used here. The shrinking lengths make that point unique. The $n$ coordinate points form a vector $x\in\mathbb R^n$. Choose a sequence term from the $k$th box at an index larger than the previously chosen index; infinitely many available indices make this possible. Both that term and $x$ lie in the same box, so
<!-- bilingual-en:end -->

对每个坐标，嵌套闭区间都有公共点：左端点递增且有上界，其[[上确界|最小上界]]就在每个区间中；这里使用了[[实数的确界性质|实数的完备性质]]。区间长度趋于零又保证公共点唯一。把 $n$ 个坐标点合起来得到 $x\in\mathbb R^n$。从第 $k$ 个盒子中选一项，索引比上一次所选的大；盒中无穷多个可用索引保证这能做到。所选项和 $x$ 都在同一个盒子内，因此

$$\|x_{n_k}-x\|_2\le\frac{2M\sqrt n}{2^k}\longrightarrow0.$$

<!-- bilingual-en:start -->
Because $K$ is closed, the [[闭集的序列判别|limit remains in $K$]]. We have found a suitable subsequence for every sequence in $K$, so [[紧致性的序列判别|the metric sequence criterion]] proves compactness. The finite number of coordinates and the completeness of real coordinate intervals are substantive parts of this argument.
<!-- bilingual-en:end -->

由于 $K$ 闭，[[闭集的序列判别|极限仍属于 $K$]]。我们为 $K$ 中每条序列都找到了合适子列，故由[[紧致性的序列判别]]得到紧致性。坐标数有限、实坐标区间具有完备性质，都是这个论证中的实质条件。

<!-- bilingual-en:start -->
## Three ways the reverse implication can fail
<!-- bilingual-en:end -->

*反方向可能失败的三种情形*

<!-- bilingual-en:start -->
First, change the ambient space to $X=(0,1)$ with its usual inherited distance. The whole space $X$ is closed in itself and bounded. Yet the relative open sets $(1/m,1)$, for integers $m\ge2$, cover it without a finite subcover, as shown in [[紧致性|the compactness definition]]. Closedness is relative: it does not add the missing real endpoint $0$ to $X$.
<!-- bilingual-en:end -->

第一，把环境换成继承通常距离的 $X=(0,1)$。整个 $X$ 在自身中闭且有界，但相对开集 $(1/m,1)$（整数 $m\ge2$）构成没有有限子覆盖的开覆盖，详见[[紧致性]]。闭性是相对的，它不会把缺失的实数端点 $0$ 添入 $X$。

<!-- bilingual-en:start -->
Second, give the positive integers the discrete metric, with distance $1$ between any two different points. The whole space is closed and bounded, yet the singleton balls of radius $1/2$ form an open cover with no finite subcover. This space is not [[全有界性|totally bounded]]: boundedness places all points in one large ball, while total boundedness requires a finite ball cover at every smaller precision. The two conditions are different.
<!-- bilingual-en:end -->

第二，给正整数集合配上离散度量，不同点之间距离都为 $1$。整个空间闭且有界，但半径 $1/2$ 的单点球构成没有有限子覆盖的开覆盖。这个空间并不[[全有界性|全有界]]：有界只让所有点落入一个大球，全有界却要求每种更细精度下都有有限球覆盖。两者不是同一条件。

<!-- bilingual-en:start -->
Third, consider $\ell^2$, the space of infinite real sequences $x=(x_1,x_2,\ldots)$ with $\sum_{j=1}^{\infty}x_j^2<\infty$, using the norm $\|x\|=(\sum_jx_j^2)^{1/2}$. Its closed unit ball $K=\{x:\|x\|\le1\}$ is bounded and closed. Closedness follows from $|\|x\|-\|y\||\le\|x-y\|$, which makes the norm continuous. Let $e_j$ have a $1$ in coordinate $j$ and zeros elsewhere. All $e_j$ lie in $K$, but for $j\ne k$,
<!-- bilingual-en:end -->

第三，考虑 $\ell^2$：由满足 $\sum_{j=1}^{\infty}x_j^2<\infty$ 的无限实数序列 $x=(x_1,x_2,\ldots)$ 组成，范数为 $\|x\|=(\sum_jx_j^2)^{1/2}$。其闭单位球 $K=\{x:\|x\|\le1\}$ 有界且闭；闭性可由不等式 $|\|x\|-\|y\||\le\|x-y\|$ 保证的范数连续性得到。令 $e_j$ 在第 $j$ 个坐标为 $1$，其余为 $0$。全部 $e_j$ 都在 $K$ 内，但 $j\ne k$ 时，

$$\|e_j-e_k\|^2=1+1=2,\qquad\|e_j-e_k\|=\sqrt2.$$

<!-- bilingual-en:start -->
No subsequence can converge, since two sufficiently late terms would then be arbitrarily close by the triangle inequality. Hence this closed, bounded unit ball is not compact. The same obstruction works with any infinite [[标准正交组|orthonormal sequence]] in an infinite-dimensional Hilbert space, a complete [[内积|inner-product]] space: unit vectors that are mutually orthogonal have the same separation $\sqrt2$.
<!-- bilingual-en:end -->

任何子列都不能收敛，因为若收敛，三角不等式会使其中足够靠后的两项任意接近。因此，这个闭且有界的单位球并不紧致。同样的障碍适用于无限维 Hilbert 空间中的任意无限[[标准正交组|标准正交序列]]：Hilbert 空间是完备的[[内积]]空间，彼此正交的单位向量之间也有同样的 $\sqrt2$ 间距。

> [!tip] Before claiming an optimum exists · 声称最优解存在之前
> <!-- bilingual-en:start -->
> For a **nonempty** closed and bounded feasible set in finite-dimensional $\mathbb R^n$, Heine–Borel gives compactness. A **continuous real-valued** objective then attains a maximum and a minimum by [[极值定理|the extreme value theorem]]. Outside this setting, establish compactness by a valid argument first. Closedness and boundedness alone do not supply the missing hypothesis.
> <!-- bilingual-en:end -->
>
> 对有限维 $\mathbb R^n$ 中**非空**、闭且有界的可行域，Heine–Borel 定理给出紧致性；**连续实值**目标函数再由[[极值定理]]取得最大值与最小值。离开这个场景时，应先用有效论证建立紧致性，不能仅靠闭且有界补上缺失条件。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.101, Review of Topology, §1.6, PDF pp.7–8](https://ocw.mit.edu/courses/18-101-analysis-ii-fall-2005/babd982be745679b6d691f78b1c18f53_lectures.pdf#page=7), and [Oxford, Part A Topology, Corollary 2.14, Proposition 2.17 and Theorem 2.24, PDF pp.21–24](https://www.maths.ox.ac.uk/system/files/attachments/Part%20A%20Topology.pdf#page=21): metric compactness implies closedness and boundedness; Heine–Borel applies in finite-dimensional Euclidean space. The nested-box argument above spells out the subsequence construction.
- [MIT 18.102, full lecture notes, Example 193, PDF p.97](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/8fb8d5c170f1613151aca71de21027bc_MIT18_102s21_full_lec.pdf#page=97): noncompactness of the infinite-dimensional closed unit ball. The distance is computed explicitly above as $\sqrt2$.
- [[01_Math/08_EC400_MathsCamp/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf#page=90|LSE EC400 Revision Mathematics Notes, §8.8, PDF p.90 / printed p.85]]: the course's boundedness and optimization setting. The application here explicitly includes nonemptiness and continuity.
<!-- bilingual-en:end -->

- MIT 18.101《Review of Topology》第1.6节，PDF7–8页，以及 Oxford《Part A Topology》推论2.14、命题2.17与定理2.24，PDF21–24页：度量紧致推出闭且有界，以及有限维欧氏空间的 Heine–Borel 定理；正文展开了嵌套盒子选择子列的步骤。
- MIT 18.102 合订讲义，例193，PDF97页：无限维闭单位球非紧；正文直接算出距离为 $\sqrt2$。
- LSE EC400《Revision Mathematics Notes》第8.8节，PDF90页／印刷85页：课程中的有界性与优化语境；本文明确补齐非空与连续条件。
