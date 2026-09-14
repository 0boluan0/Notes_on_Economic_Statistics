---
aliases:
  - "主成分数量应由目标与稳定性共同选择而不能由单一累计阈值机械决定"
  - choosing the number of principal components
  - PCA 成分数选择
student_os: knowledge-atom
atom_id: STAT-PCA-006
atom_set: principal-component-analysis
atom_type: decision-rule
status: source-checked
mastery_state: unassessed
part_of:
  - "[[主成分分析.canvas|主成分分析]]"
requires:
  - "[[解释方差比]]"
related:
  - "[[PCA验证与泄漏]]"
  - "[[PCA稳定性]]"
  - "[[重根下主成分不唯一]]"
  - "[[解释方差不等于任务信息]]"
---

# 主成分数量应由目标与稳定性共同选择而不能由单一累计阈值机械决定
<!-- bilingual-en:start -->
*The number of principal components should reflect purpose and stability rather than a single mechanical cumulative threshold*
<!-- bilingual-en:end -->

> [!summary] 原子决策
> 累计解释率、scree plot 的拐点、平行分析、重采样稳定性与下游验证回答不同问题。应先写明 PCA 的用途，再选择能支持该用途的证据；“达到 80% 或 90% 就停止”不是普遍定理。
> <!-- bilingual-en:start -->
> Cumulative variance, scree elbows, parallel analysis, resampling stability, and downstream validation provide different evidence. No universal 80% or 90% cutoff determines the correct dimension.
> <!-- bilingual-en:end -->

若目的是可视化，$k=2$ 或 $3$ 可能由展示限制决定；若目的是重构，可用允许的重构误差；若 PCA 是预测管线的一步，应在训练折内部拟合 PCA，并按验证性能选择 $k$；若要解释成分，则还应检查载荷和子空间在重采样或新样本中是否稳定。

scree plot 的“肘部”可能模糊，累计解释率会偏爱高方差噪声，平行分析依赖参照随机模型。它们都能提供线索，但没有任何单一图形替代目标定义与外部检查。

> [!question]- 自检
> 为什么“所有数据上先选出解释率 90% 的 $k$，再交叉验证模型”可能偏乐观？
>
> **答案：** 因为验证折已参与选择 PCA 维数；应在每个训练折内拟合中心、尺度、方向并选择 $k$。

## 来源与核验

- [Penn State STAT 505, Lesson 11](https://online.stat.psu.edu/stat505/Lesson11)：核对累计解释率与 scree plot 是成分数选择依据而非唯一规则。
- J. L. Horn, [A rationale and test for the number of factors in factor analysis](https://doi.org/10.1007/BF02289447)：核对平行分析以随机参照特征值判断应保留维数的原始方法；它提供一种参照证据，不是无条件真值。
- [scikit-learn, Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：核对所有数据驱动的预处理和模型选择都必须隔离验证数据。
- [[01_Math/04_多元统计分析/08_主成分分析principal component.md#1.6. 主成分数量选择|本地课程页]]：核对课程要求及经验阈值边界。
