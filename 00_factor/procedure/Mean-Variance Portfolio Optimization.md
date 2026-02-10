---
aliases:
- 均值-方差组合优化步骤
- 均值-方差组合优化
- Mean-Variance Portfolio Optimization
- Mean
tags:
- procedure
- 06_证券投资学
---
# 均值-方差组合优化

## 适用场景

在给定预期收益或风险约束下，寻找最优投资组合权重，用于投资组合构建和资产配置。

## 所需数据/条件

- $N 个资产的预期收益率向量 \mu = (\mu_1, \dots, \mu_N)'$
- $N \times N$ 协方差矩阵 $\Sigma$
- 无风险利率 $R_f$（如包含无风险资产）

## 计算步骤

### 步骤 1：准备输入数据

收集各资产的：
- 预期收益率（历史均值或预测值）
- 收益率协方差矩阵
- 投资约束（如权重和为1、非负约束等）

**注意点**：确保协方差矩阵正定，否则需调整。

### 步骤 2：定义最小方差组合问题

仅含风险资产时：
$ \min_w w' \Sigma w $
$ \text{s.t. } w' \mathbf{1} = 1, \quad w \ge 0 $

包含无风险资产时：
$ \min_w w' \Sigma w $
$ \text{.s.t. } w' \mathbf{1} = 1 $
其中权重向量包含无风险资产权重。

**注意点**：这是全局最小方差组合，风险最低。

### 步骤 3：求解最小方差组合

使用拉格矩阵日法：
$ L = w' \Sigma w - \lambda (w' \mathbf{1} - 1) - \gamma' w $

一阶条件：
$ \Sigma w - \lambda \mathbf{1} = 0 $

**封闭解**：
$ w^* = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}' \Sigma^{-1} \mathbf{1}} $

**注意点**：若有非负约束，需用二次规划求解。

### 步骤 4：计算有效前沿

对一系列目标收益率 \mu_p，求解：

$ \min_w w' \Sigma w $
$\text{s.t. } w' \mu = \mu_p, \quad w' \mathbf{1} = 1$

得到有效前沿上的组合 $($w_p$, \sigma_p, \mu_p)$。

**注意点**：$\sigma_p = \sqrt{w_p' \Sigma w_p} 为组合标准差。$

### 步骤 5：绘制有效前沿

将 $(\sigma_p, \mu_p)$ 在风险-收益平面描点：
- 有效前沿为凸向外的曲线
- 左上方为可行集，曲线为最优边界

**注意点**：有效前沿上的组合不可通过分散进一步改善。

### 步骤 6：寻找切点组合（CAL分离定理）

给定无风险利率 $R_f$，有效前沿与资本市场线（CAL）相切的组合为最优：

CAL方程：
$E[R_p] = R_f + \frac{E[R_m] - R_f}{\sigma_m^2} \sigma_p$

求切点使斜率相等，得到最优组合权重。

**注意点**：分离定理说明所有投资者持有相同的风险资产组合。

### 步骤 7：计算组合统计量

对任意组合 $w$：
- 预期收益：$\mu_p = w' \mu$
- 方差：$\sigma_p^2 = w' \Sigma w$
- 标准差：$\sigma_p = \sqrt{\sigma_p^2}$

**注意点**：这些统计量用于比较不同组合。

## 关键公式

**组合预期收益**：
$\mu_p = \sum_{i=1}^N w_i \mu_i$

**组合方差**：
$\sigma_p^2 = \sum_{i=1}^N \sum_{j=1}^N w_i w_j \sigma_{ij}$

**全局最小方差组合权重**：
$ w_{\text{min}} = \frac{A \mathbf{1}}{\mathbf{1}' A \mathbf{1}}, \quad A = \Sigma^{-1} $

**有效前沿关系**：
$\sigma_p^2 = \frac{a \mu_p^2 + b \mu_p + c}{d \mu_p^2 + e \mu_p + f}$

**资本市场线**：
$\mu_p = R_f + \frac{\mu_m - R_f}{\sigma_m^2} \sigma_p$

## 常见问题

1. **协方差矩阵奇异**：存在完全共线性，无法求逆，需剔除资产。
2. **非负约束**：数学解析解可能含负权重，需用数值优化。
3. **输入估计误差**：预期收益和协方差估计不准确影响结果。
4. **交易成本**：理论最优组合未考虑交易成本和流动性。

## 相关概念
[[Efficient Frontier|有效前沿]]
[[Separation Theorem (Finance)|分离定理]]
[[Sharpe Ratio|夏普比率]]
[[CAPM]]

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
