---
aliases:
  - "PCA 的中心尺度与方向必须只在训练数据上拟合并对新数据原样应用"
  - PCA validation and leakage
  - PCA 训练测试隔离
student_os: knowledge-atom
atom_id: STAT-PCA-012
atom_set: principal-component-analysis
atom_type: validation-rule
status: source-checked
mastery_state: unassessed
part_of:
  - "[[主成分分析.canvas|主成分分析]]"
requires:
  - "[[验证管线隔离]]"
  - "[[PCA尺度选择]]"
  - "[[PCA成分数]]"
related:
  - "[[PCA协方差敏感性]]"
  - "[[PCA稳定性]]"
---

# PCA 的中心尺度与方向必须只在训练数据上拟合并对新数据原样应用
<!-- bilingual-en:start -->
*PCA centring, scaling, and directions must be fitted only on training data and then applied unchanged to new data*
<!-- bilingual-en:end -->

> [!summary] 原子验证规则
> 这是 [[验证管线隔离]] 在 PCA 中的专门应用：在预测、交叉验证或外部验证中，均值、由训练数据估计的标准差、主成分方向，以及任何由数据选择的成分数，都属于训练管线。每个训练折内估计或选择这些量，再用同一组参数变换验证折；不能先在全数据上做 PCA 再切分。若尺度规则或 $k$ 在看数据前已经固定，则无需重新“选择”，但其数值变换仍只由训练数据拟合。
> <!-- bilingual-en:start -->
> This is the PCA-specific application of [[验证管线隔离|validation-pipeline isolation]]. Means, data-estimated scales and directions, and any data-selected component count belong to the training pipeline. Fit or select them inside each training split, then apply the same transform to validation or test observations; a component count fixed a priori need not be re-selected.
> <!-- bilingual-en:end -->

若把验证数据用于中心化、数据驱动地选 $k$ 或估计载荷，验证数据的信息已经进入模型管线，得到的误差或准确率会偏乐观。部署到新观测时也必须使用训练期 $\bar x$、训练期标准差和 $V_k$：
$$
z_{new}=(x_{new}-\bar x_{train})^TV_{k,train}.
$$
该式写的是未缩放的协方差 PCA；若训练时做了标准化，则应使用
$$
z_{new}=\left((x_{new}-\bar x_{train})\oslash s_{train}\right)^TV_{k,train},
$$
其中 $\oslash$ 表示逐元素相除。不能用新批次自己的均值或标准差重新定义坐标系。

> [!question]- 自检
> 为什么测试集也要用训练集均值中心化，而不是用测试集自己的均值？
>
> **答案：** 测试集必须模拟真正未知的新数据；用其自身均值重新拟合变换会改变坐标系，并让测试信息泄漏进模型。

## 来源与核验

- [scikit-learn, Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)：核对预处理只能在训练数据上 `fit`，PCA 明确属于可能泄漏的变换。
- [scikit-learn, PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)：核对在 `fit` 学习方向后对新数据使用同一投影。
