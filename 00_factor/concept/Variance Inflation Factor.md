---
aliases:
- 方差膨胀因子
- VIF
- Variance Inflation Factor
tags:
- 计量经济学
- 统计学
- concept
---
方差膨胀因子（Variance Inflation Factor, VIF）是衡量回归模型中多重共线性严重程度的指标。

## 定义

对于解释变量$x_i$，其VIF定义为：

$VIF_i = \frac{1}{1 - R_i^2}$

其中$R_i$²是$x_i$对其他所有解释变量回归的判定系数（R²）。

## 含义

$VIF_i$衡量由于多重共线性导致$x_i$系数估计量的方差膨胀的程度。

- **$R_i$² = 0**：$x_i与其他变量完全无关，VIF = 1（无膨胀）$
- **$R_i$² → 1**：$x_i$可由其他变量完全线性表示，VIF → ∞（完全多重共线性）

## 判断标准

| VIF值 | 多重共线性程度 |
|--------|----------------|
| VIF < 5 | 不严重 |
| 5 ≤ VIF < 10 | 中度 |
| VIF ≥ 10 | 严重 |

## 与标准误的关系

考虑多重共线性后，$x_i$系数估计量的标准误为：

$\text{SE}(\hat{\beta}_i) = \frac{\sigma_{\varepsilon}}{\sqrt{SST_i}} \cdot \sqrt{VIF_i}$

其中：
- σ_ε：误差项标准差
- $SST_i$：$x_i$的离差平方和
- √$VIF_i$：多重共线性导致标准误放大的倍数

## 计算步骤

1. 对每个$x_i$，用其他解释变量回归：$x_i = \gamma_0 + \sum_{j \neq i} \gamma_j x_j + u_i$
2. 计算该回归的R²（记为$R_i$²）
3. $计算VIF_i = 1/(1-R_i²)$

## 性质

1. **VIF ≥ 1**：因为0 ≤ $R_i$² < 1
2. **对变量敏感**：不同变量VIF可能不同
3. **与模型整体相关**：平均VIF反映整体多重共线性水平

## 应用

1. **诊断多重共线性**：识别哪些变量存在多重共线性
2. **变量选择**：VIF高的变量可能需要删除
3. **模型改进**：通过降维或变换降低VIF

## 扩展：广义VIF

对于多元线性回归，可以定义广义VIF矩阵：

$\text{VIF} = \text{diag}[(R(X) R(X)']^{-1}$

其中R(X)是X的相关矩阵。

## 与其他诊断方法比较

| 方法 | 特点 |
|------|------|
| 相关系数 | 两两相关，简单直观 |
| 辅助回归+R² | 综合考虑所有变量 |
| VIF | 标准化指标，便于判断 |
| 条件指数 | 考虑所有变量相关 |

相关链接: [[Multicollinearity|多重共线性]], [[Condition Index|条件指数]]

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
