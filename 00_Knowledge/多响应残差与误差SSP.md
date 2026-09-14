---
aliases:
  - "多响应残差矩阵与误差 SSP 具有不同维度和用途"
  - The residual matrix and error SSP have different dimensions and roles
  - residual matrix versus error SSP
student_os: knowledge-atom
atom_id: STAT-MVR-005
atom_set: multivariate-linear-regression
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[多元线性回归.canvas|多元线性回归]]"
requires:
  - "[[共享设计的逐响应OLS]]"
leads_to:
  - "[[误差SSP抽样律]]"
  - "[[误差SSP秩边界]]"
related:
  - "[[多响应一般线性假设]]"
---

# 多响应残差矩阵与误差 SSP 具有不同维度和用途
<!-- bilingual-en:start -->
*The multivariate residual matrix and error SSP have different dimensions and roles*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> 拟合后的残差矩阵是
> $$R=Y-X\hat B\in\mathbb R^{n\times q},$$
> 它保留每个观测、每个响应的残差。误差平方和—交叉乘积矩阵是
> $$E_{SSP}=R^TR\in\mathbb R^{q\times q},$$
> 对角元是各响应残差平方和，非对角元是响应对的残差交叉乘积。二者不能都简写为 $E$ 后混用。
> <!-- bilingual-en:start -->
> $R=Y-X\hat B$ is an $n\times q$ matrix of case-by-response residuals, whereas $E_{SSP}=R^TR$ is a $q\times q$ residual sums-of-squares-and-products matrix.
> <!-- bilingual-en:end -->

$R$ 用于定位观测层异常、拟合形状和依赖；$E_{SSP}$ 压缩掉观测身份，用于协方差估计以及与假设 SSP 矩阵 $H$ 比较。仅保留 $E_{SSP}$ 不能恢复哪一行产生了异常。

这两个定义只依赖拟合结果，不要求误差服从 Gaussian 分布。只有加入独立 Gaussian 行等抽样条件后，才可进一步讨论 Wishart 分布、无偏分母和极大似然分母；这些结论由 [[误差SSP抽样律]] 单独承担。

> [!question]- 自检
> 两个响应、80 个观测时，$R$ 与 $E_{SSP}$ 各多大？
>
> **答案：** $R$ 是 $80\times2$，$E_{SSP}$ 是 $2\times2$。

## 来源与核验

- [R `SSD` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/SSD.html)：核对 multivariate linear model 的 residual sums of squares and products matrix、残差自由度与协方差估计对象。
- [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08)：核对误差 SSP 的对角元、交叉乘积项和 error degrees of freedom，以及它与假设 SSP 的分工。
- [[误差SSP抽样律]]：承接独立 Gaussian 行下的 Wishart 分布、协方差估计分母和奇异边界；本卡只定义并区分两个残差对象。
- [R `summary.manova` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.manova.html)：核对 MANOVA 输出使用 SSP matrices 并单独报告 residual degrees of freedom。
<!-- bilingual-en:start -->
- R's official `SSD` and `summary.manova` documentation and Penn State STAT 505 support the distinction between the residual matrix and the residual SSP object.
<!-- bilingual-en:end -->
