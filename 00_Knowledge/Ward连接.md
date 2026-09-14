---
aliases:
  - "Ward 连接把两簇的合并代价定义为总组内平方和的增量，每步合并增量最小的一对"
  - "Ward linkage defines merge cost by the increase in total within-cluster sum of squares"
student_os: knowledge-atom
atom_id: STAT-CL-008
atom_set: cluster-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[凝聚层次聚类]]"
related:
  - "[[聚类尺度]]"
  - "[[单连接]]"
  - "[[完全连接]]"
  - "[[平均连接]]"
leads_to:
  - "[[Ward平方欧氏边界]]"
  - "[[Ward高度口径]]"
  - "[[树状图切割]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# Ward 连接把两簇的合并代价定义为总组内平方和的增量，每步合并增量最小的一对
<!-- bilingual-en:start -->
*Ward linkage defines merge cost by the increase in total within-cluster sum of squares and merges the pair with the smallest increase at each step*
<!-- bilingual-en:end -->

> [!summary] 比较的是合并带来的 WSS 增量
> Ward 法不用最近、最远或平均点对相异度直接定义合并。它在每一步计算合并两簇后总 within-cluster sum of squares（WSS，组内平方和）会增加多少，再选增量最小的一对。
> <!-- bilingual-en:start -->
> Ward's method compares the increase in total within-cluster sum of squares caused by each possible merge and chooses the smallest increase.
> <!-- bilingual-en:end -->

## 精确合并增量

对 Euclidean 向量簇 $C$，定义

$$
W(C)=\sum_{x_i\in C}\lVert x_i-\bar x_C\rVert_2^2.
$$

若要合并非空簇 $A$ 和 $B$，总 WSS 的增量为

$$
\begin{aligned}
\Delta(A,B)
&=W(A\cup B)-W(A)-W(B)\\
&=\frac{|A||B|}{|A|+|B|}
\left\lVert \bar x_A-\bar x_B\right\rVert_2^2.
\end{aligned}
$$

Ward 每步合并当前 $\Delta(A,B)$ 最小的一对簇。因子 $|A||B|/(|A|+|B|)$ 说明这不只是质心距离：簇大小也进入准则。

## 优化对象是当前一步，不是最终全局分区

Ward 倾向产生紧凑、近球形的簇，但它仍是逐步贪心且不可逆的层次法。“当前合并的 WSS 增量最小”不等于“对每个最终簇数都找到全局 WSS 最小分区”。该准则的几何适用条件见 [[Ward平方欧氏边界]]，树上纵轴的实现口径见 [[Ward高度口径]]。

> [!question]- 自检
> 两对候选簇的质心距离相同，但第一对簇都很小，第二对簇都很大。Ward 合并代价是否必然相同？
>
> **答案：** 不必然。$\Delta(A,B)=\frac{|A||B|}{|A|+|B|}\lVert\bar x_A-\bar x_B\rVert^2$，除了质心距离，还有簇大小因子。簇大小不同可以使合并代价不同。

## 来源与核验

- Ward (1963), [*Hierarchical Grouping to Optimize an Objective Function*](https://doi.org/10.1080/01621459.1963.10500845)：核对通过逐步合并优化组内变差准则的原始定义。
- Murtagh and Legendre (2014), [*Ward's Hierarchical Agglomerative Clustering Method: Which Algorithms Implement Ward's Criterion?*](https://doi.org/10.1007/s00357-014-9161-z)：核对精确 WSS 增量公式和逐步准则。
- [[Ward平方欧氏边界]]：单独承载 squared-Euclidean 几何条件。
- [[Ward高度口径]]：单独承载 method、输入与 dendrogram height 的实现口径。

> [!success] 审核状态
> 本卡已通过独立内容审核，状态为 `source-checked`；掌握状态仍为 `unassessed`。
