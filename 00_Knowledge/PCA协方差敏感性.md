---
aliases:
  - "经典 PCA 继承样本协方差对异常值和重尾分布的敏感性"
  - PCA sensitivity to outliers
  - 鲁棒 PCA 边界
student_os: knowledge-atom
atom_id: STAT-PCA-011
atom_set: principal-component-analysis
atom_type: robustness-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[主成分分析.canvas|主成分分析]]"
requires:
  - "[[样本协方差矩阵]]"
related:
  - "[[异常观测处理原则]]"
  - "[[主成分不等于潜变量]]"
  - "[[PCA稳定性]]"
---

# 经典 PCA 继承样本协方差对异常值和重尾分布的敏感性
<!-- bilingual-en:start -->
*Classical PCA inherits the sensitivity of sample covariance to outliers and heavy tails*
<!-- bilingual-en:end -->

> [!summary] 原子稳健边界
> 经典 PCA 从平方偏差构成的样本协方差矩阵提取方向。离中心很远的少数观测会以距离平方进入协方差，可能显著旋转特征向量并放大相应特征值。
> <!-- bilingual-en:start -->
> Classical PCA is based on squared deviations in the sample covariance matrix. A small number of distant observations can rotate components and inflate eigenvalues.
> <!-- bilingual-en:end -->

异常观测不一定是错误：它可能是录入问题、另一总体、真实极端状态或研究最关心的风险。应先检查原始尺度、散点投影、得分距离、重采样稳定性和数据来源，再决定修正、分层分析或采用稳健协方差/稳健 PCA；不能为了让图“更整齐”自动删除。

重尾分布下，即使没有孤立坏点，样本协方差也可能高度不稳定。报告 PCA 时应说明样本范围、预处理和敏感性检查，避免把某次样本的载荷表当作固定结构。

> [!question]- 自检
> 删除一个高杠杆观测后 PC1 大幅旋转，至少说明了什么？
>
> **答案：** 说明当前 PCA 方向对该观测敏感；它不单独证明该点错误，也不证明删除后的方向就是真实机制。

## 来源与核验

- [[样本协方差矩阵]]：核对样本协方差由中心化外积平均构成。
- [[异常观测处理原则]]：连接异常观测的来源核实、模型诊断与敏感性报告。
- I. T. Jolliffe and J. Cadima, [Principal component analysis: a review and recent developments, §3(c)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/#s3c)：核对少数离群观测可对经典 PCA 产生不成比例影响，以及稳健替代方法的背景。
