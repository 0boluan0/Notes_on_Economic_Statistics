---
aliases:
  - "PageRank 得分属于修改后随机游走并依赖链接权重、传送分布与悬空修补"
  - PageRank is model-dependent
  - PageRank personalization boundary
  - PageRank 排名对象边界
student_os: knowledge-atom
atom_id: PROB-DTMC-042
atom_set: discrete-time-markov-chains
atom_type: interpretation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[PageRank随机传送]]"
related:
  - "[[悬空节点转移]]"
  - "[[全支持传送收敛]]"
leads_to: []
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# PageRank 得分属于修改后随机游走并依赖链接权重、传送分布与悬空修补
<!-- bilingual-en:start -->
*A PageRank score belongs to the modified random walk and depends on link weight, teleportation, and dangling-node repair*
<!-- bilingual-en:end -->

> [!summary] 唯一解不等于唯一客观价值
> PageRank 向量是
> $$P_\alpha=\alpha P+(1-\alpha)\mathbf1v^T$$
> 的平稳分布，而不是未修改链接图自身携带的唯一数值。改变 $\alpha$、传送分布 $v$ 或 [[悬空节点转移|悬空修补]]，都会改变转移矩阵，因而可能改变排名。
> <!-- bilingual-en:start -->
> PageRank is the stationary distribution of a specified modified walk; changing its modeling choices can change the resulting ranking.
> <!-- bilingual-en:end -->

$\alpha$ 控制链接结构相对于传送机制的权重；$v$ 可以均匀，也可以偏向某个主题或用户兴趣；悬空节点可以自环、均匀回流或采用其他分布。[[全支持传送收敛]] 保证给定模型下的唯一性与收敛，却不会消除这些模型选择。

因此正确解释是“在这套浏览者行为假设下的长期访问权重”，而不是“页面脱离模型后唯一、客观的价值”。

> [!question]- 自检
> 两套 PageRank 都满足唯一收敛条件，但使用不同的个性化向量 $v$。它们是否必须给出同一排名？
>
> **答案：** 不必。唯一性只针对各自固定的转移矩阵；不同 $v$ 定义了不同随机游走。

## 来源与核验

- [Page, Brin, Motwani, and Winograd, The PageRank Citation Ranking](http://ilpubs.stanford.edu:8090/422/)：核对 random-surfer、damping 与排名模型的原始语境。
- [Stanford Introduction to Information Retrieval, Topic-specific PageRank](https://nlp.stanford.edu/IR-book/html/htmledition/topic-specific-pagerank-1.html)：核对传送分布可用于个性化或主题偏置，因而排名依赖模型选择。
