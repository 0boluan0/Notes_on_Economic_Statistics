---
aliases:
  - "PCA 压缩总变异而因子分析用潜变量模型解释共同协方差"
  - PCA versus factor analysis
  - 主成分分析与因子分析
student_os: knowledge-atom
atom_id: STAT-PCA-009
atom_set: principal-component-analysis
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[主成分分析.canvas|主成分分析]]"
requires:
  - "[[主成分分析]]"
contrasts_with:
  - "[[公共因子模型]]"
implies:
  - "[[主成分不等于潜变量]]"
related:
  - "[[概率PCA]]"
---

# PCA 压缩总变异而因子分析用潜变量模型解释共同协方差
<!-- bilingual-en:start -->
*PCA compresses total variation, whereas factor analysis uses a latent-variable model to explain common covariance*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> 经典 PCA 把观测变量旋转成确定的正交线性组合，并按总方差排序；因子分析假定观测由较少潜在因子与变量特有误差生成，例如 $X=\mu+\Lambda F+\varepsilon$。二者都出现“载荷”，但估计目标与误差结构不同。
> <!-- bilingual-en:start -->
> PCA forms deterministic orthogonal combinations ordered by total variance. Factor analysis posits latent factors plus variable-specific errors to model common covariance.
> <!-- bilingual-en:end -->

若目标是压缩、正交坐标、可视化或预测前处理，PCA 直接回答问题。若研究问题主张一组潜在构念生成变量间共同相关，并且独特方差与测量误差需要分开，才进入因子模型；此时还要说明因子数、识别、旋转和拟合检验。

普通 PCA 本身没有“潜变量真实存在”的生成假设；它是对中心化数据矩阵或协方差矩阵做特征分解/SVD 的代数方法。若需要一个具有明确分布假设的 PCA 生成模型，由[[概率PCA]]单独定义；它不是普通 PCA 的隐含前提。

> [!question]- 自检
> 多个考试科目在 PC1 上系数都很大，能否直接命名为“总体能力因子”？
>
> **答案：** 不能直接。PCA 只说明这些科目沿一个高方差方向共同变化；潜在能力解释需要理论、因子模型和外部验证。

## 来源与核验

- [scikit-learn, Factor Analysis](https://scikit-learn.org/stable/modules/decomposition.html#factor-analysis)：核对因子分析允许逐变量噪声方差、载荷不必正交及其与概率 PCA 的区别。
- [Penn State STAT 505, Lesson 11](https://online.stat.psu.edu/stat505/Lesson11)：核对 PCA 的确定线性组合与总变异口径。
- [[因子分析.canvas|因子分析]]：进入公共因子模型、提取、旋转、验证与解释边界的完整主题图。
