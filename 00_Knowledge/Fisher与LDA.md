---
aliases:
  - "两类共享协方差时 Fisher 与 LDA 的判别方向成比例，但相同方向不等于相同的完整分类规则"
  - "Fisher 判别最大化投影后的类间分离相对类内变异；两类共享协方差时方向与 LDA 成比例但阈值仍由先验和损失决定"
  - Fisher discrimination versus LDA
student_os: knowledge-atom
atom_id: STAT-DA-015
atom_set: discriminant-analysis
atom_type: method-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[Fisher判别]]"
  - "[[LDA共享协方差]]"
  - "[[先验与分类风险]]"
part_of:
  - "[[判别分析.canvas|判别分析]]"
---

# 两类共享协方差时 Fisher 与 LDA 的判别方向成比例，但相同方向不等于相同的完整分类规则
<!-- bilingual-en:start -->
*With two classes and shared covariance, Fisher discrimination and LDA have proportional discriminant directions, but the same direction does not make them the same complete classification rule*
<!-- bilingual-en:end -->

> [!summary] 怎样辨析
> 两类 Fisher 准则给出 $a\propto S_W^{-1}(\bar x_1-\bar x_2)$；共享协方差 LDA 的边界法向量为 $\Sigma^{-1}(\mu_1-\mu_2)$。当 $S_W$ 估计同一个 $\Sigma$ 时，两者沿同一轴区分类别。Fisher 只由分离准则确定投影方向；LDA 还用 Gaussian 模型、先验和损失形成完整决策规则。
>
> <!-- bilingual-en:start -->
> Fisher's two-class direction and the normal vector of a shared-covariance LDA boundary are proportional when the within-class scatter estimates the same covariance. Fisher defines a projection criterion; LDA combines a Gaussian model with priors and losses to define a complete decision rule.
> <!-- bilingual-en:end -->

两类 Fisher 方向满足

$$
a_F\propto S_W^{-1}(\bar x_1-\bar x_2).
$$

在类条件 Gaussian 分布共享 $\Sigma$ 时，LDA 的两类得分差关于 $x$ 的系数为

$$
a_L=\Sigma^{-1}(\mu_1-\mu_2).
$$

因此，用 pooled within-class covariance 估计 $\Sigma$ 时，样本 Fisher 轴与 LDA 边界法向量只差一个非零比例常数。比例和整体反号不改变同一条投影轴。

## 方向相同仍不等于规则相同

Fisher 准则没有为投影轴自动指定分类切点。LDA 的截距来自类均值、先验以及误判损失；只有在共享协方差、先验相等且两种误判成本相等时，两类投影均值的中点才与相应 Bayes 阈值一致。改变先验或成本会沿同一法向量平移边界，却不改变 Fisher 轴本身。

两者的解释资格也不同。Fisher 方向可在没有 Gaussian 假设时由均值与类内散布定义；LDA 的 posterior 概率解释则依赖类条件 Gaussian 模型及其参数估计。轴相同不能证明两套概率、阈值或校准相同。

> [!question]- 自检
> 两类 Fisher 与 LDA 画出同一条投影轴，是否可以不说明先验和误判成本，直接断言它们会把所有新观测分到同一类？
>
> **答案：** 不可以。同一投影轴只确定排序方向；实际切点仍受 LDA 的先验和损失影响，Fisher 准则本身没有给出这个切点。

## 来源与核验

- [Penn State STAT 505, Lesson 10](https://online.stat.psu.edu/stat505/Lesson10)：核对两类 Fisher 判别向量、共享组内协方差与分类切点。
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed., §4.3：核对 Fisher 准则与 LDA 判别子空间的联系。
- [[先验与分类风险]]：给出先验和误判成本怎样移动分类阈值；本卡只拥有“方向相同不等于完整规则相同”的辨析。
