---
aliases:
  - "PageRank 随机传送把链接转移与传送分布混合成新的转移矩阵"
  - PageRank teleportation
  - PageRank damping
  - PageRank 随机传送
student_os: knowledge-atom
atom_id: PROB-DTMC-022
atom_set: discrete-time-markov-chains
atom_type: model-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[图上随机游走]]"
  - "[[悬空节点转移]]"
  - "[[Markov矩阵左右约定]]"
related:
  - "[[Markov稳态分布]]"
leads_to:
  - "[[全支持传送收敛]]"
  - "[[PageRank模型依赖]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# PageRank 随机传送把链接转移与传送分布混合成新的转移矩阵
<!-- bilingual-en:start -->
*PageRank teleportation mixes link-following transitions with a teleportation distribution to form a new transition matrix*
<!-- bilingual-en:end -->

> [!summary] 链接行走与随机传送的混合模型
> 先把悬空节点修成合法行随机矩阵 $P$。在行向量约定下，PageRank 随机传送定义
> $$P_\alpha=\alpha P+(1-\alpha)\mathbf1v^T,$$
> 其中 $0<\alpha<1$，$v$ 是概率向量。这里 $\alpha$ 明确表示**沿链接转移的权重**，$1-\alpha$ 表示按 $v$ 随机传送的权重；$\mathbf1v^T$ 的每一行都等于 $v^T$。
> <!-- bilingual-en:start -->
> PageRank combines a repaired link-following matrix with a teleportation distribution; here $\alpha$ is the weight on following links.
> <!-- bilingual-en:end -->

每一步先以概率 $\alpha$ 按 $P$ 走链接，以概率 $1-\alpha$ 忽略当前页面并从 $v$ 抽取下一页面。若有 $n$ 个页面且 $v_j=1/n$，传送部分就是每行均匀的矩阵。这个定义本身不包含唯一性或收敛结论；所需条件见 [[全支持传送收敛]]。

> [!question]- 自检
> 在上述约定中，$\alpha$ 越接近 1，随机浏览者更常沿链接走，还是更常随机传送？
>
> **答案：** 更常沿链接走；随机传送的权重是 $1-\alpha$。

## 来源与核验

- [Page, Brin, Motwani, and Winograd, The PageRank Citation Ranking](http://ilpubs.stanford.edu:8090/422/)：核对 random-surfer、链接转移与 damping 的原始模型语境。
- [Cambridge Markov Chains notes, Example 8.6](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf#page=31)：核对 $\alpha\hat P+(1-\alpha)/n$ 的传送混合形式与权重约定。
