---
aliases:
- PageRank model dependence
- PageRank 排名对象边界
student_os: knowledge-atom
atom_id: PROB-DTMC-042
atom_set: discrete-time-markov-chains
atom_type: interpretation-boundary
status: source-checked
requires:
- '[[PageRank随机传送]]'
related:
- '[[悬空节点转移]]'
- '[[全支持传送收敛]]'
leads_to: []
part_of:
- '[[离散时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# What PageRank scores depend on
<!-- bilingual-en:end -->

*PageRank 的模型依赖*

<!-- bilingual-en:start -->
A PageRank score answers a precise question: how much probability does a page receive in the long run under a specified random-surfer rule? Changing that rule can change the answer, even if each version has a unique, well-behaved limit. **Model dependence** is the need to interpret the score together with the graph, transition weights, restart distribution and dangling-page treatment that produced it.
<!-- bilingual-en:end -->

PageRank 分数回答一个精确问题：在指定的随机浏览规则下，一个页面长期获得多少概率？改变规则就可能改变答案，即使每个版本都有唯一且收敛良好的极限。**模型依赖**就是：解释分数时，必须同时知道产生它的图、转移权重、传送分布与悬空页面处理。

<!-- bilingual-en:start -->
In the [[PageRank随机传送|row-based PageRank model]], $P$ is the repaired link-transition matrix, $v$ is the teleportation probability column, and $\alpha$ is the chance of following $P$. For $0\le\alpha<1$,

$$G=\alpha P+(1-\alpha)\mathbf1v^T,\qquad \pi G=\pi.$$

Here $T$ means [[矩阵转置|transpose]], $\mathbf1$ is the column of ones, and $\pi$ is a [[Markov稳态分布|stationary probability row]]. [[PageRank阻尼收敛|Damping convergence]] makes $\pi$ unique for these fixed inputs. It does not select the inputs for us.
<!-- bilingual-en:end -->

按[[PageRank随机传送|行约定的 PageRank 模型]]，$P$ 是修补后的链接转移矩阵，$v$ 是传送概率列，$\alpha$ 是跟随 $P$ 的概率。$0\le\alpha<1$ 时模型与固定点方程如上。$T$ 表示[[矩阵转置|转置]]，$\mathbf1$ 是全一列，$\pi$ 是[[Markov稳态分布|稳态概率行]]。[[PageRank阻尼收敛]]保证对这组固定输入，$\pi$ 唯一；它不会替我们决定输入应当怎样选择。

<!-- bilingual-en:start -->
## The same links can yield opposite rankings
<!-- bilingual-en:end -->

*相同链接，也能得到相反排名*

> [!example] Change the restart preference · 改变重新开始时的偏好
> <!-- bilingual-en:start -->
> Use two pages linked only to each other, $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$, and $\alpha=1/2$. With $v=(3/4,1/4)^T$, the stationary row is $(7/12,5/12)$, so page 1 ranks above page 2. With $v=(1/4,3/4)^T$, it is $(5/12,7/12)$, reversing the ranking. Both restart distributions have full support; both walks converge uniquely.
> <!-- bilingual-en:end -->
>
> 两个页面只互相链接，取 $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$、$\alpha=1/2$。若 $v=(3/4,1/4)^T$，稳态为 $(7/12,5/12)$，页面 1 排在页面 2 前面。若换成 $v=(1/4,3/4)^T$，稳态为 $(5/12,7/12)$，名次反转。两种传送分布都全支持，两种游走也都收敛到唯一稳态。

<!-- bilingual-en:start -->
The calculation can be checked without an algorithm. Write $p=\pi_1$, so $\pi_2=1-p$. In this two-page graph, the stationary equation is $p=\alpha(1-p)+(1-\alpha)v_1$, hence

$$p=\frac{\alpha+(1-\alpha)v_1}{1+\alpha}.$$

Keeping $v_1=3/4$ and changing $\alpha$ from $1/2$ to $3/4$ changes $p$ from $7/12$ to $15/28$. The score changes, but the ordering does not reverse. Model changes can alter scores or rankings; neither change is automatic in every example.
<!-- bilingual-en:end -->

不用算法也能核算。写 $p=\pi_1$，于是 $\pi_2=1-p$。两页面图的稳态方程为 $p=\alpha(1-p)+(1-\alpha)v_1$，解得上式。保持 $v_1=3/4$，把 $\alpha$ 从 $1/2$ 改为 $3/4$，$p$ 会从 $7/12$ 变为 $15/28$。分数改变了，但名次没有反转。模型变化可能改变分数或排序，并不是每个例子都会发生这些变化。

<!-- bilingual-en:start -->
## A missing outgoing link requires a substantive choice
<!-- bilingual-en:end -->

*缺失出链时，修补规则会影响问题*

<!-- bilingual-en:start -->
Suppose page $a$ links to $b$, while $b$ has no outgoing link. Keep $\alpha=1/2$ and uniform teleportation. One legal [[悬空节点转移|dangling repair]] sends $b$ to $a$ with certainty, giving the swap matrix and stationary row $(1/2,1/2)$. Another keeps the walker at $b$ on the link-following branch:

$$P=\begin{pmatrix}0&1\\0&1\end{pmatrix},\qquad
G=\begin{pmatrix}1/4&3/4\\1/4&3/4\end{pmatrix},\qquad
\pi=(1/4,3/4).$$

The observed link $a\to b$ is unchanged. The different continuation rule at $b$ changes the long-run answer.
<!-- bilingual-en:end -->

设页面 $a$ 链接到 $b$，而 $b$ 没有出链。保持 $\alpha=1/2$ 与均匀传送。一种合法[[悬空节点转移|悬空修补]]让 $b$ 必定去 $a$，得到交换矩阵，稳态为 $(1/2,1/2)$。另一种在“跟随链接”分支让浏览者留在 $b$，便得到上面的 $P$、$G$ 与 $\pi=(1/4,3/4)$。已经观察到的链接 $a\to b$ 没变；改变的是到达 $b$ 后如何继续，这足以改变长期答案。

<!-- bilingual-en:start -->
A uniform repair is another option, and using the same distribution for repair and teleportation is common but not compulsory. Likewise, a [[图上随机游走|weighted graph walk]] assigns different transition probabilities to strong and weak links. These choices define $P$ before the restart mixture is formed. If a graph has no dangling pages, changing only a dangling-page policy has no effect on its matrix.
<!-- bilingual-en:end -->

均匀修补也是一种选择；让修补与传送使用同一分布很常见，但不是必须。类似地，[[图上随机游走|加权图游走]]会给强弱不同的链接不同转移概率。这些选择先定义 $P$，然后才与传送分支混合。如果图里没有悬空页面，仅改变悬空处理政策就不会影响矩阵。

<!-- bilingual-en:start -->
## Interpret a result together with its generating rule
<!-- bilingual-en:end -->

*把结果与产生它的规则一起解释*

<!-- bilingual-en:start -->
To compare PageRank outputs, identify the page set and directed links, how link weights are normalized, the dangling rule, the meaning and value of $\alpha$, and the distribution $v$. A uniform $v$ asks for a broad centrality score; a $v$ concentrated around selected topics or pages asks a more localized question. Neither is a parameter-free definition of importance.
<!-- bilingual-en:end -->

比较 PageRank 输出时，要明确页面集合与有向链接、链接权重如何归一化、悬空规则、$\alpha$ 的含义与数值，以及传送分布 $v$。均匀 $v$ 对应较整体的中心性问题；集中在指定主题或页面附近的 $v$ 对应较局部的问题。两者都不是不依赖参数的“重要性”定义。

> [!warning] A unique limit is not an external validation · 极限唯一不等于外部价值已被证实
> <!-- bilingual-en:start -->
> A high score means high stationary mass in the specified modified walk. It does not by itself establish truthfulness, usefulness for a particular query, or a page's objective value. Conversely, changing an input need not change the result: if every row of $P$ equals $v^T$, then $G=P$ for every $\alpha$. Mathematical convergence concerns a fixed model; assessing whether that model answers the intended question is a separate judgement.
> <!-- bilingual-en:end -->
>
> 高分表示在指定的修改后游走中，稳态质量高。它本身不能证明内容真实、对特定查询有用，或网页具有某种客观价值。反过来，改变输入也未必改变结果：若 $P$ 每行都等于 $v^T$，则任意 $\alpha$ 下都有 $G=P$。数学收敛针对固定模型；这个模型是否回答了我们真正关心的问题，是另一项判断。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Gleich, *PageRank Beyond the Web*, §2 and §3.1, PDF pp.3–4 and 8–9 / printed pp.323–324 and 328–329](https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/Gleich-2015-PageRank.pdf#page=3): centrality versus localized choices of $v$, weighted walks, and distinct dangling-node treatments. The examples above use an explicitly row-stochastic construction and are solved directly.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Pagerank.pdf#page=1|MIT 6.042J PageRank slides, PDF pp.1–3]]: the model's ranking interpretation, personalized restarts and the distinction from a complete search-ranking system.
- [Weber, Cambridge *Markov Chains*, Example 8.6, PDF p.36 / printed p.32](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=36): stationary probabilities in the random-surfer model and the distinction between original PageRank and a broader search algorithm.
<!-- bilingual-en:end -->

- Gleich §2、§3.1，PDF3–4及8–9页／印刷323–324及328–329页：支持整体中心性与局部化 $v$ 的区别、加权游走和不同悬空处理。正文例子明确采用行随机构造，并直接求解。
- MIT 6.042J PageRank slides，PDF1–3页：支持模型的排名解释、个性化重新开始，以及与完整搜索排序系统的区别。
- Weber 例 8.6，PDF36页／印刷32页：支持随机浏览者模型的稳态概率解释，以及原始 PageRank 与更广义搜索算法的区别。
