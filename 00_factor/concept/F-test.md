---
aliases:
- F检验
- F
- F-test
tags:
- 统计学
- 假设检验
- concept
---
F检验（F-test）是用于比较两个方差是否相等的统计检验方法，在计量经济学中常用于多参数联合显著性检验。

>[!note] 定义
>
> F统计量定义为两个独立方差估计量的比值：
>
> $F = \frac{s_1^2 / \sigma_1^2}{s_2^2 / \sigma_2^2}$
>
> 在原假设H₀: σ₁² = σ₂²下：
>
> $F = \frac{s_1^2}{s_2^2} \sim F(n_1-1, n_2-1)$
>
## F分布

F分布是两个独立卡方分布除以自由度之比：

$F = \frac{\chi_1^2 / d_1}{\chi_2^2 / d_2} \sim F(d_1, d_2)$

其中d₁和d₂是自由度。

## 回归中的F检验

### 1. 整体显著性检验

**原假设H₀**：β₁ = β₂ = ... = β_k = 0
**备择假设H₁**：至少有一个β_i ≠ 0

F统计量：

$F = \frac{(R^2 / k)}{(1 - R^2) / (n - k - 1)} \sim F(k, n-k-1)$

其中：
- R²：判定系数
- k：解释变量个数
- n：样本量

### 2. 参数约束检验

比较受限模型和非受限模型：

$F = \frac{(SSR_R - SSR_U) / r}{SSR_U / (n - k)} \sim F(r, n-k)$

其中：
- $SSR_R$：受限模型的残差平方和
- $SSR_U$：非受限模型的残差平方和
- r：约束条件个数

### 3. 模型比较

比较两个嵌套模型：

$F = \frac{(RSS_1 - RSS_2) / (k_2 - k_1)}{RSS_2 / (n - k_2)} \sim F(k_2-k_1, n-k_2)$

其中RSS₁ > RSS₂（模型2包含更多变量）。

## F检验步骤

1. 提出原假设和备择假设
2. 计算F统计量
3. 确定自由度
4. 计算p值：$p = P(F > F_obs)$
5. 比较p值与显著性水平α

## 判断标准

- 若p < α，拒绝H₀
- 若p ≥ α，无法拒绝H₀

## 与t检验的关系

在简单线性回归中：

$F = t^2$

即F检验等价于t检验的平方。

## 常见应用

1. **ANOVA（方差分析）**：比较多组均值是否相等
2. **回归整体显著性**：检验模型是否有解释力
3. **参数约束检验**：检验多个参数是否同时为0
4. **模型选择**：比较嵌套模型
5. **同方差检验**：检验误差项方差是否相等

>[!example] 示例：邹检验
>
> 检验两个时期或两个样本的回归系数是否相同。
>
> $F = \frac{[SSR - (SSR_1 + SSR_2)] / k}{(SSR_1 + SSR_2) / (n_1 + n_2 - 2k)}$
>
> 相关链接: [[t Test|t检验]], [[Chi-square Test|卡方检验]], [[R-squared|判定系数]]
>
## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
