---
aliases:
- F统计量
- F检验
- F
- F-test
tags:
- system
- 计量经济学
---
# F检验

## 诊断目的

检验回归模型的整体显著性，判断所有解释变量联合起来是否对因变量有显著影响。

## 计算方法

### 整体显著性检验

$F = \frac{\text{模型均方}}{\text{残差均方}} = \frac{ESS/k}{RSS/(n-k-1)} \sim F(k, n-k-1)$

其中：
- ESS：回归平方和
- RSS：残差平方和
- k：解释变量个数
- n：样本容量

### 嵌套模型比较

$F = \frac{(RSS_R - RSS_U)/(k_U - k_R)}{RSS_U/(n-k_U-1)} \sim F(k_U-k_R, n-k_U-1)$

其中：
- $RSS_R$：受限模型的残差平方和
- $RSS_U$：无约束模型的残差平方和
- $k_R$：受限模型参数个数
- $k_U$：无约束模型参数个数

## 判断标准

### 与R²的关系

$F = \frac{R^2/k}{(1-R^2)/(n-k-1)}$

| F值 | p值范围 | 判断 |
|------|----------|------|
| F < $F_critical$ | p > 0.05 | 不拒绝H0，模型整体不显著 |
| F > $F_critical$ | p ≤ 0.05 | 拒绝H0，模型整体显著 |

### 警告信号

| 情况 | 诊断 | 含义 |
|------|------|------|
| F显著但所有t都不显著 | F值高，t值低 | 存在严重多重共线性 |
| F不显著但R²很高 | 大样本、低R² | 模型解释力不足 |
| F值极大 | R²接近1 | 可能过拟合或数据问题 |

## 常见问题与对策

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| F检验显著但单个t不显著 | 多重共线性、设定错误 | 检查VIF、重新考虑变量选择 |
| F检验不显著 | 遗漏重要变量、样本太小 | 增加相关变量、扩大样本 |
| 比较F值时自由度不同 | 模型复杂度不同 | 使用信息准则（AIC/BIC）比较 |

## 相关概念
[[t Test|t检验]]
[[R-squared|判定系数]]
[[Multicollinearity|多重共线性]]
[[00_factor/system/Variance Inflation Factor|方差膨胀因子]]

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
