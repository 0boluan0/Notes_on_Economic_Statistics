---
aliases:
- 协积
- 协整
- Cointegration
tags:
- 时间序列
- 计量经济学
- concept
---
协整（Cointegration）是指多个非平稳的线性组合可以形成平稳序列的现象，反映了变量之间的长期均衡关系。

>[!note] 定义
>
> 对于n×1维的I(1)向量$y_t$，如果存在一个n×1维的非零向量β，使得：
>
> $\beta^T y_t = u_t$
>
> 其中$u_t$是平稳过程I(0)，则称$y_t$是协整的，β称为协整向量。
>
## 经济含义

协整表示变量之间存在长期均衡关系。虽然单个变量可能是非平稳的，但它们的线性组合保持平稳，意味着变量不会长期偏离彼此。

## Engle-Granger两步法

### 第一步：长期关系估计

1. 估计静态回归：$y_t = \alpha + \beta x_t + u_t$
2. 检验残差$u_t$的单位根：若$u_t$平稳，则存在协整关系

### 第二步：误差修正模型（ECM）

$\Delta y_t = \alpha + \gamma u_{t-1} + \sum_{i=1}^{p-1} \phi_i \Delta y_{t-i} + \sum_{j=1}^{q-1} \theta_j \Delta x_{t-j} + \varepsilon_t$

其中γ是调整系数，γ < 0表示存在误差修正机制。

## Johansen检验

Johansen检验是基于向量自回归（VAR）框架的系统协整检验方法。

### 检验步骤

1. 设定VAR模型
2. 确定协整秩r（协整关系的数量）
3. 使用迹检验（Trace test）和最大特征根检验（Max eigenvalue test）

### 迹检验统计量

$LR_{trace}(r) = -T \sum_{i=r+1}^{n} \ln(1 - \lambda_i)$

其中λ_i是特征值，T是样本量。

## 误差修正模型（ECM）

ECM表示变量向长期均衡关系的调整速度：

$\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i} + \varepsilon_t$

其中Π = αβ'，α是调整系数矩阵，β是协整向量矩阵。

## 应用

1. **均衡关系检验**：检验经济理论中的长期关系
2. **长期和短期分析**：分离长期均衡和短期动态
3. **政策分析**：分析政策冲击的长期和短期效应

相关链接: [[Error Correction Model|误差修正模型]], [[Unit Root Test|单位根检验]], [[VAR Model|VAR]]

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
