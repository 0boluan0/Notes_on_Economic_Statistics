---
aliases:
  - "PCA 稳定性是主成分结果在合理样本扰动或新样本下保持可比结构的程度"
  - PCA stability
  - 主成分稳定性
student_os: knowledge-atom
atom_id: STAT-PCA-018
atom_set: principal-component-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[主成分分析.canvas|主成分分析]]"
requires:
  - "[[主成分谱结构]]"
related:
  - "[[PCA协方差敏感性]]"
  - "[[主成分符号任意]]"
  - "[[重根下主成分不唯一]]"
  - "[[PCA成分数]]"
  - "[[PCA验证与泄漏]]"
---

# PCA 稳定性是主成分结果在合理样本扰动或新样本下保持可比结构的程度
<!-- bilingual-en:start -->
*PCA stability is the extent to which principal-component results retain comparable structure under reasonable sample perturbations or new samples*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> PCA 稳定性不是一个脱离对象的单一数字。它先要说明比较的是特征值与解释方差、单条主成分方向、前 $k$ 维子空间、观测得分，还是低秩重构。然后通过 bootstrap、重复抽样、时间切分或独立新样本重新拟合 PCA，检查该对象是否保持可比。
> <!-- bilingual-en:start -->
> PCA stability must name the object being compared—such as eigenvalues, individual directions, a retained subspace, scores, or reconstruction—and then examine it across refitted samples or perturbations.
> <!-- bilingual-en:end -->

直接逐元素比较载荷往往会误判。首先，同一条主成分轴可以整体反号，比较前要先做符号对齐。其次，当相邻特征值接近或重合时，单个方向可能在共同子空间内旋转；此时应比较子空间投影、主夹角或重构，而不是要求每一列载荷原样不动。

例如，两个领先特征值几乎相等时，不同 bootstrap 样本中 PC1 与 PC2 可能交换或旋转，但它们张成的二维子空间仍很稳定。这时正确结论是“二维表示稳定，单条轴不稳定”，而不是笼统地说“PCA 稳定”或“PCA 不稳定”。

稳定性检查回答的是样本扰动下的可重现性；在预测管线中如何隔离验证数据，由[[PCA验证与泄漏]]单独处理。

> [!question]- 自检
> 两次抽样中 PC1 的载荷不同，是否已经证明保留的 PCA 表示不稳定？
>
> **答案：** 不足以。先排除整体反号，再检查是否存在接近重根导致的子空间内旋转；研究目标若是前 $k$ 维表示，应直接比较该子空间或重构。

## 来源与核验

- A. Fisher, B. Caffo, B. Schwartz and V. Zipunnikov, [Fast, Exact Bootstrap Principal Component Analysis for $p>1$ Million](https://pmc.ncbi.nlm.nih.gov/articles/PMC5014451/)：核对通过 bootstrap 估计主成分方向、得分与特征值变异性的方法，以及重采样解中的旋转问题。
- [[主成分符号任意]] 与 [[重根下主成分不唯一]]：核对逐列比较前必须处理的两类表示不唯一。
