---
aliases:
- 随机游走
- Random Walk
tags:
- 时间序列
- 统计学
- concept
---
随机游走（Random Walk）是指下一期值等于当期值加上一个随机冲击的时间序列模型。

## 基本形式

**简单随机游走：**

$y_t = y_{t-1} + \varepsilon_t$

其中ε_t是白噪声：ε_t ~ WN(0, σ²)

## 带漂移的随机游走

$y_t = \mu + y_{t-1} + \varepsilon_t$

其中μ是漂移项。

## 性质

1. **非平稳**：随机游走是非平稳过程（存在单位根）
2. **方差随时间增长**：$Var(y_t) = tσ²，随时间线性增长$
3. **无均值回归**：对历史冲击具有永久记忆
4. **不可预测**：$E(y_{t+1} | F_t) = y_t（或y_t + μ）$

## 累积和表示

$y_t = y_0 + \sum_{i=1}^{t} \varepsilon_i$

或带漂移：
$y_t = y_0 + \mu t + \sum_{i=1}^{t} \varepsilon_i$

## 几何布朗运动

连续时间的随机游走模型：

$dy_t = \mu dt + \sigma dW_t$

其中$W_t$是布朗运动（维纳过程）。

## 应用

1. **资产价格建模**：股票价格常被视为随机游走（或几何布朗运动）
2. **假设检验**：有效市场假说认为资产价格遵循随机游走
3. **单位根检验**：检验序列是否为随机游走

## 检验方法

- **单位根检验**：DF检验、ADF检验
- **游程检验**：检验价格变化的方向随机性

## 相关模型

1. **AR(1)过程**：$y_t = φy_{t-1} + ε_t，当φ=1时为随机游走$
2. **带趋势的随机游走**：$y_t = α + βt + y_{t-1} + ε_t$

相关链接: [[Unit Root Test|单位根检验]], [[Stationarity|平稳性]], [[Efficient Market Hypothesis|有效市场假说]]

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
