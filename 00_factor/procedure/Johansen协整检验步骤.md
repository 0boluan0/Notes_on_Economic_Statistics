---
aliases:
  - Johansen Cointegration Test Steps
  - Johansen协整检验步骤
tags:
  - procedure
  - 01_Econometrics
  - 06_时间序列分析
---

# Johansen协整检验步骤

## 适用场景

检验多个非平稳时间序列之间是否存在长期均衡关系（协整关系），以及确定协整关系的数量和形式。

## 所需数据/条件

- $m$ 个时间序列 $\mathbf{y}_t = (y_{1t}, y_{2t}, \dots, y_{mt})'$，样本量 $t=1,\dots,T$
- 所有序列均为 $I(1)$ 过程（需先通过单位根检验确认）
- 滞后阶数 $p$（需预先确定）

## 计算步骤

### 步骤 1：确定VAR模型滞后阶数

使用信息准则确定最优滞后阶数 $p$：

$$ \text{AIC}(p) = T \ln(\hat{\sigma}_p^2) + 2(m^2 p + m) $$
$$ \text{SBC}(p) = T \ln(\hat{\sigma}_p^2) + (m^2 p + m) \ln T $$

选择使信息准则最小的 $p$ 值。

**注意点**：协整检验对滞后阶数选择敏感，通常使用最大10阶。

### 步骤 2：构建VAR(p)模型

$$ \Delta \mathbf{y}_t = \Pi \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta \mathbf{y}_{t-i} + \mathbf{\epsilon}_t $$

其中 $\Pi$ 是 $m \times m$ 系数矩阵，$\mathbf{\epsilon}_t$ 为白噪声向量。

**注意点**：此形式称为误差修正模型（ECM）表示。

### 步骤 3：估计VAR模型参数

使用最大似然法或OLS估计 $\Pi$ 和 $\Gamma_i$ 矩阵。

**注意点**：Johansen方法使用完全信息最大似然估计。

### 步骤 4：计算协整矩阵的特征值

对矩阵 $\Pi$ 进行特征值分解：
$$ \text{det}(\Pi - \lambda I) = 0 $$

得到 $m$ 个特征值 $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_m$。

**注意点**：协整关系的数量 $r$ 等于非零特征值的个数。

### 步骤 5：计算Johansen检验统计量

**迹检验（Trace Test）**：
检验 $H_0: \text{rank}(\Pi) \le r$ 对 $H_1: \text{rank}(\Pi) > r$

$$ \lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{m} \ln(1 - \hat{\lambda}_i) $$

**最大特征值检验（Max-Eigen Test）**：
检验 $H_0: \text{rank}(\Pi) = r$ 对 $H_1: \text{rank}(\Pi) = r+1$

$$ \lambda_{\text{max}}(r) = -T \ln(1 - \hat{\lambda}_{r+1}) $$

**注意点**：两种检验从 $r=0$ 开始依次检验。

### 步骤 6：确定协整关系数量

**检验流程**：
1. 从 $r=0$ 开始检验
2. 若拒绝原假设，则至少存在 $r+1$ 个协整关系
3. 递增 $r$ 继续检验，直到无法拒绝原假设
4. 最终协整关系数量 $\hat{r}$ 为无法拒绝的最大 $r$ 值

**注意点**：迹检验更严格，通常采用迹检验结论。

### 步骤 7：估计协整向量（若 $\hat{r} > 0$）

将 $\Pi$ 矩阵分解为：
$$ \Pi = \alpha \beta' $$

其中 $\beta$ 为 $m \times \hat{r}$ 协整向量矩阵，$\alpha$ 为调整速度矩阵。

**注意点**：协整关系不是唯一确定，需要进行标准化。

### 步骤 8：构建协整方程

对于每个协整关系 $j=1,\dots,\hat{r}$：
$$ \beta_{1j} y_{1t} + \beta_{2j} y_{2t} + \cdots + \beta_{mj} y_{mt} = \text{长期均衡项} $$

或写成误差修正项：
$$ \text{ECT}_{j,t-1} = \beta_j' \mathbf{y}_{t-1} $$

**注意点**：ECT项反映系统向均衡调整的力度。

## 关键公式

**VAR(p)误差修正表示**：
$$ \Delta \mathbf{y}_t = \Pi \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta \mathbf{y}_{t-i} + \mathbf{\epsilon}_t $$

**迹检验统计量**：
$$ \lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{m} \ln(1 - \hat{\lambda}_i) \xrightarrow{d} \chi^2(m-r) $$

**最大特征值检验统计量**：
$$ \lambda_{\text{max}}(r) = -T \ln(1 - \hat{\lambda}_{r+1}) \xrightarrow{d} \chi^2(1) $$

## 常见问题

1. **样本量不足**：Johansen检验需要较大样本量，通常建议 $T > 50m$。
2. **滞后阶数敏感**：不同 $p$ 可能得到不同的协整关系数量。
3. **协整向量标准化**：协整向量的符号和大小需要经济意义解释。
4. **结构断点**：样本期间存在结构变化时，检验结论可能不可靠。

## 相关概念
[[协整]]
[[误差纠正模型]]
[[单位根]]
[[ADF检验步骤]]
