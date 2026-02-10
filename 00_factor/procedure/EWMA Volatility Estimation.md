---
aliases:
- EWMA波动率估计
- EWMA
- EWMA Volatility Estimation
tags:
- procedure
- 07_金融机构与风险管理
---
# EWMA波动率估计

## 适用场景

当需要波动率快速响应新信息时使用，特别适合日常风险监控。RiskMetrics系统使用此方法。

## 所需数据/条件

- $历史收益率序列 {\epsilon_t}_{t=1}^{T}$
- 衰减因子 $\lambda$（RiskMetrics建议日频数据用0.94）

## 计算步骤

### 步骤 1：初始化波动率估计

使用初始波动率估计：
$$
\sigma_0^2 = \frac{1}{T_0} \sum_{t=1}^{T_0} \epsilon_t^2
$$

其中 $T_0$ 为初始窗口长度（如60天）。

**注意点**：初始估计用于启动递归。

### 步骤 2：设定衰减因子

RiskMetrics标准值：
- 日频数据：$\lambda = 0.94$
- 月频数据：$\lambda = 0.97$

**注意点**：$\lambda$ 越大，历史信息衰减越慢。

### 步骤 3：递归更新波动率

对每个新观测 \epsilon_t，更新：

$$
\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) \epsilon_t^2
$$

**注意点**：这是指数加权移动平均的递归形式。

### 步骤 4：计算波动率

$$
\sigma_t = \sqrt{\sigma_t^2}
$$

**注意点**：波动率为方差的平方根。

### 步骤 5：展开形式（理解权重）

$$
\sigma_t^2 = \lambda^{T-1} \sigma_0^2 + (1-\lambda) \sum_{i=1}^{T-1} \lambda^{i-1} \epsilon_{T-i}^2
$$

**注意点**：近期数据权重为 $(1-\lambda)$，远期数据权重按 $\lambda^{i-1}$ 衰减。

### 步骤 6：计算长期平均波动率

EWMA的长期平均波动率：
$$
\sigma_{\infty}^2 = \frac{1-\lambda}{1-\lambda} E[\epsilon_t^2]
$$

**注意点**：EWMA本身无均值回归（若 $\lambda$ 接近1，可视为常数均值）。

## 关键公式

**EWMA递归公式**：
$$
\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) \epsilon_{t-1}^2
$$

**展开形式**：
$$
\sigma_t^2 = (1-\lambda) \sum_{i=0}^{\infty} \lambda^{i} \epsilon_{t-1-i}^2
$$

**多日波动率缩放**：
$$
\sigma_{N\text{天}}^2 \approx N \times \sigma_{\text{天}}^2
$$

**注意点**：假设收益率独立同分布。

## 权重分配

**历史数据权重序列**：
$$
w_0 = 1-\lambda,\; w_1 = (1-\lambda)\lambda,\; w_2 = (1-\lambda)\lambda^2, \dots
$$

**半衰期**：
$$
\text{Half-life} = \frac{\ln(0.5)}{\ln(\lambda)} \approx \frac{-0.6931}{\ln(\lambda)}
$$

**注意点**：$\lambda = 0.94$ 时，半衰期约 11 天。

## 与GARCH比较

| 特性 | EWMA | GARCH(1,1) |
|------|------|-------------|
| 公式 | $\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)\epsilon_{t-1}^2$ | $\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$ |
| 均值回归 | 通常无 | 有，收敛到 $\omega/(1-\alpha-\beta)$ |
| 参数数量 | 1 | 3 |
| 适用场景 | 日常监控、快速响应 | 建模波动、均值回归 |

**注意点**：GARCH 更灵活但更复杂，EWMA 更简单实用。

## 常见问题

1. **衰减因子选择**：$\lambda$ 过大反应慢，过小不稳定。
2. **异常值影响**：极端事件会显著提高后续波动率估计。
3. **初始值敏感**：递归更新对初始值敏感。
4. **均值漂移**：EWMA不捕捉均值变化，需单独建模。
5. **数据缺口**：缺失值需要插值处理。

## 相关概念
[[Historical Volatility|历史波动率]]
[[GARCH]]
[[Volatility Clustering|波动率聚集]]

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
