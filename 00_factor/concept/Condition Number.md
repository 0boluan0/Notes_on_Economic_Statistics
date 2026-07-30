---
aliases:
- 条件指数
- Condition Number
tags:
- 计量经济学
- 线性代数
- concept
---

# Condition Number

条件指数（Condition Number）是衡量矩阵数值稳定性的指标，用于检测回归模型中多重共线性的严重程度。

>[!note] 定义
>
> 对于矩阵X（n×k），条件指数κ定义为：
>
> $\kappa = \frac{\lambda_{max}}{\lambda_{min}}$
>
> 其中：
> - λ_max：X'X的最大特征值
> - λ_min：X'X的最小特征值
>
## 含义

条件指数衡量矩阵X'X的条件数，反映回归问题的数值稳定性。

- **κ = 1**：矩阵是正交的，无多重共线性
- **κ → ∞**：矩阵接近奇异，存在严重多重共线性

## 计算步骤

1. 构造矩阵X'X
2. 计算X'X的特征值：λ_1, λ_2, ..., λ_k
3. 找出最大特征值λ_max和最小特征值λ_min
4. 计算条件指数：κ = λ_max / λ_min

## 判断标准

| κ值 | 多重共线性程度 |
|------|----------------|
| κ < 100 | 不严重 |
| 100 ≤ κ < 1000 | 中度 |
| κ ≥ 1000 | 严重 |

## 几何解释

条件指数是X矩阵列空间"扁化"程度的度量：

- κ小：列空间"胖"，接近球形
- κ大：列空间"扁"，接近扁平超平面

## 性质

1. **κ ≥ 1**：因为λ_max ≥ λ_min
2. **对标准化敏感**：变量标准化会影响κ值
3. **与VIF的关系**：κ²接近平均VIF

## 应用

1. **诊断多重共线性**：条件指数高表示存在多重共
2. **模型选择**：比较不同模型的条件指数
3. **数值稳定性评估**：判断模型估计是否稳定

>[!example] 示例
>
> 考虑两个高度相关的变量：
>
> $$
> X =
> \begin{pmatrix}
>  1 & 1 \\ 1 & 1.1 \\ 1 & 0.9
> \end{pmatrix}
> $$
>
> $$
> X'X =
> \begin{pmatrix}
>  3 & 3 \\ 3 & 3.02
> \end{pmatrix}
> $$
>
> 计算特征值并求条件指数，会发现κ很大。
>
## 与其他诊断方法比较

| 方法 | 特点 |
|------|------|
| VIF | 针对每个变量，更具体 |
| 相关系数 | 两两相关，简单 |
| 条件指数 | 整体指标，数学性质好 |

## 扩展：条件数

对于一般矩阵A，条件数定义为：

$\kappa(A) = \|A\| \cdot \|A^{-1}\|$

其中 $\| \cdot \|$ 是矩阵范数。

相关链接：[[Multicollinearity|多重共线性]]、[[Variance Inflation Factor|方差膨胀因子]]。

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Multicollinearity]]。

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
