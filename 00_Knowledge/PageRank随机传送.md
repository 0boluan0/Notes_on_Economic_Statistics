---
aliases:
- PageRank teleportation
- PageRank damping
- PageRank 随机传送
student_os: knowledge-atom
atom_id: PROB-DTMC-022
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[图上随机游走]]'
- '[[悬空节点转移]]'
- '[[Markov矩阵左右约定]]'
related:
- '[[Markov稳态分布]]'
leads_to:
- '[[PageRank阻尼收敛]]'
- '[[全支持传送收敛]]'
- '[[PageRank模型依赖]]'
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# PageRank teleportation
<!-- bilingual-en:end -->

*PageRank 随机传送*

<!-- bilingual-en:start -->
Imagine browsing a finite collection of web pages. Sometimes you follow a link on the current page; sometimes you start again at a page drawn from a chosen distribution. **PageRank teleportation** combines these two moves into one transition rule. The restart can cross a missing link or leave a part of the graph that ordinary link-following would never escape.
<!-- bilingual-en:end -->

想象在有限个网页之间浏览：有时沿当前页面的链接走，有时按指定概率重新选一个页面开始。**PageRank 随机传送**把这两种动作合成一步转移。重新选择的页面不必与当前页面有链接，因此可以跨过缺失的连接，也可以离开只跟随原链接时出不去的区域。

<!-- bilingual-en:start -->
## Specify the link walk before mixing
<!-- bilingual-en:end -->

*先把链接游走定义完整，再混合*

<!-- bilingual-en:start -->
Number the pages $1,\ldots,n$, with integer $n\ge1$. A [[图上随机游走|random walk on a graph]] turns outgoing links, possibly with weights, into probabilities. Use the [[Markov矩阵左右约定|row convention]]: $P_{ij}$ is the probability of moving from page $i$ to page $j$. The repaired matrix $P$ must be [[Markov矩阵|row-stochastic]]: $P_{ij}\ge0$ and $\sum_jP_{ij}=1$ for every $i$.
<!-- bilingual-en:end -->

把页面编号为 $1,\ldots,n$，整数 $n\ge1$。[[图上随机游走]]把出链及其可能的权重转换为概率。本卡采用[[Markov矩阵左右约定|行约定]]：$P_{ij}$ 表示从页面 $i$ 到页面 $j$ 的概率。修补后的 $P$ 必须是[[Markov矩阵|行随机矩阵]]：$P_{ij}\ge0$，且每行都满足 $\sum_jP_{ij}=1$。

<!-- bilingual-en:start -->
A page without an outgoing link needs a [[悬空节点转移|dangling-node rule]] first. For example, replace its empty row by a chosen probability row. Leaving that row zero does not define a transition: there would be no destination when the link-following branch is selected. The dangling rule and the teleportation distribution may be chosen together, but they are separate parts of the model.
<!-- bilingual-en:end -->

没有出链的页面必须先指定[[悬空节点转移|悬空节点规则]]，例如用一个选定的概率行替代空行。直接保留零行不能定义合法转移，因为选择“跟随链接”这一分支时将没有下一站。悬空规则与传送分布可以配套选择，但在模型里是两个不同的部分。

<!-- bilingual-en:start -->
Let $v=(v_1,\ldots,v_n)^T$ be the teleportation probability **column**, where $v_j\ge0$ and $\sum_jv_j=1$. The superscript $T$ means [[矩阵转置|transpose]], so $v^T$ is the corresponding row. Fix $0\le\alpha<1$: follow $P$ with probability $\alpha$, and draw a destination from $v$ with probability $1-\alpha$, regardless of the current page. Thus
<!-- bilingual-en:end -->

令 $v=(v_1,\ldots,v_n)^T$ 为传送概率**列**，其中 $v_j\ge0$ 且 $\sum_jv_j=1$。上标 $T$ 表示[[矩阵转置|转置]]，所以 $v^T$ 是对应的概率行。固定 $0\le\alpha<1$：以概率 $\alpha$ 按 $P$ 走，以概率 $1-\alpha$ 按 $v$ 重新选目的地；后一种选择不依赖当前页面。因此

$$
G_{ij}=\alpha P_{ij}+(1-\alpha)v_j,\qquad
G=\alpha P+(1-\alpha)\mathbf1v^T,
$$

<!-- bilingual-en:start -->
Here $\mathbf1$ is the column of $n$ ones, so every row of $\mathbf1v^T$ equals $v^T$. The two contributions add because they are alternative ways to reach $j$. Each row of $G$ sums to $\alpha+(1-\alpha)=1$. Uniform teleportation sets $v_j=1/n$; personalized teleportation can assign different probabilities, including zeros. A teleport can return to the current page.
<!-- bilingual-en:end -->

这里 $\mathbf1$ 是由 $n$ 个一组成的列，因此 $\mathbf1v^T$ 的每行都等于 $v^T$。两个贡献对应到达 $j$ 的两种互斥方式，所以相加。$G$ 的每行和为 $\alpha+(1-\alpha)=1$。均匀传送取 $v_j=1/n$；个性化传送可以给不同页面不同概率，也允许零概率。传送也可能选回当前页面。

> [!example] Two linked pages · 两个互相链接的页面
> <!-- bilingual-en:start -->
> Let $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$, $\alpha=1/2$, and $v=(3/4,1/4)^T$. Then
>
> $$G=\begin{pmatrix}3/8&5/8\\7/8&1/8\end{pmatrix}.$$
>
> Starting at page 1, the chance of reaching page 2 is $\tfrac12\cdot1+\tfrac12\cdot\tfrac14=5/8$. A probability row $x_0=(1,0)$ therefore becomes $x_1=x_0G=(3/8,5/8)$. The restart has softened the original compulsory alternation.
> <!-- bilingual-en:end -->
>
> 取 $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$、$\alpha=1/2$、$v=(3/4,1/4)^T$，就得到上面的 $G$。从页面 1 出发，到页面 2 的概率是 $\tfrac12\cdot1+\tfrac12\cdot\tfrac14=5/8$。所以初始概率行 $x_0=(1,0)$ 变成 $x_1=x_0G=(3/8,5/8)$。原来必然交替的运动，被重新选择页面的机制打破了。

<!-- bilingual-en:start -->
## The score belongs to the modified walk
<!-- bilingual-en:end -->

*分数属于修改后的游走*

<!-- bilingual-en:start -->
Repeating this fixed rule gives $x_{k+1}=x_kG$. The PageRank score row is the [[Markov稳态分布|stationary probability distribution]] $\pi$: it has nonnegative entries summing to one and satisfies $\pi G=\pi$. In the example, $\pi=(7/12,5/12)$; substituting it into the equation verifies that the mass distribution stays unchanged. [[PageRank阻尼收敛|Damping convergence]] proves uniqueness and convergence from every initial probability row, even when $v$ has zeros.
<!-- bilingual-en:end -->

反复使用这个固定规则，得到 $x_{k+1}=x_kG$。PageRank 分数行就是[[Markov稳态分布|稳态概率分布]] $\pi$：各分量非负、总和为一，并满足 $\pi G=\pi$。本例中 $\pi=(7/12,5/12)$，代入即可检查概率分配在下一步保持不变。[[PageRank阻尼收敛]]证明它唯一，并且从任何初始概率行都会收敛，即使 $v$ 中有零分量。

> [!warning] Keep the parameter and orientation explicit · 参数含义与方向要明确
> <!-- bilingual-en:start -->
> Here $\alpha$ is the **link-following** probability, not the restart probability. With probability columns, the same process uses $G^T=\alpha P^T+(1-\alpha)v\mathbf1^T$ and $x_{k+1}^T=G^Tx_k^T$. At $\alpha=0$, every next distribution is $v^T$; at $\alpha=1$, teleportation disappears and the general convergence guarantee no longer applies. Also, mixing a zero dangling row with $v^T$ gives row sum $1-\alpha$, not one when $\alpha>0$.
> <!-- bilingual-en:end -->
>
> 本卡的 $\alpha$ 是**跟随链接**的概率，不是重新开始的概率。若改用概率列，同一过程须写成 $G^T=\alpha P^T+(1-\alpha)v\mathbf1^T$、$x_{k+1}^T=G^Tx_k^T$。$\alpha=0$ 时，每次下一步分布都是 $v^T$；$\alpha=1$ 时没有传送，一般收敛保证不再适用。此外，把悬空零行直接与 $v^T$ 混合，只会得到行和 $1-\alpha$；当 $\alpha>0$ 时它仍不是一。

<!-- bilingual-en:start -->
Full support, meaning every $v_j>0$, gives [[全支持传送收敛|additional positivity and communication properties]]. The numerical scores still depend on the graph and the chosen rules; [[PageRank模型依赖|model dependence]] explains that interpretation with explicit comparisons.
<!-- bilingual-en:end -->

全支持指每个 $v_j>0$，它提供[[全支持传送收敛|额外的正性与互通性质]]。数值分数仍然依赖图和具体规则；[[PageRank模型依赖]]用明确的比较说明应如何解释它们。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Gleich, *PageRank Beyond the Web*, §2, Definition 2.1 and equations (2.1)–(2.3), PDF pp.3–4 / printed pp.323–324](https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/Gleich-2015-PageRank.pdf#page=3): the model, normalized nonnegative teleportation vector and stationary equation. The paper uses column-stochastic matrices; every formula here is transposed consistently.
- [Weber, Cambridge *Markov Chains*, §8.3, Example 8.6, PDF p.36 / printed p.32](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=36): the row convention, repair of dangling pages, uniform restarts and $\alpha\in[0,1)$. The two-page numerical example is computed directly.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Pagerank.pdf#page=1|MIT 6.042J PageRank slides, PDF pp.1–3]]: the random-surfer interpretation and the distinction between this model and a complete search-ranking system.
<!-- bilingual-en:end -->

- Gleich《PageRank Beyond the Web》§2、定义 2.1 与式 (2.1)–(2.3)，PDF3–4页／印刷323–324页：支持模型、非负且归一化的传送向量与稳态方程；原文用列随机矩阵，本文一致地转置为行约定。
- Weber §8.3、例 8.6，PDF36页／印刷32页：支持行约定、悬空修补、均匀重新开始及 $\alpha\in[0,1)$；两页面数例直接计算。
- MIT 6.042J PageRank slides，PDF1–3页：支持随机浏览者解释，以及该模型与完整搜索排序系统的区别。
