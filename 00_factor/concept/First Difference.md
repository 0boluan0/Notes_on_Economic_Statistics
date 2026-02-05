---
aliases:
- 一阶差分
- First Difference
- First
tags:
- 时间序列
- 数据变换
- concept
---
一阶差分（First Difference）是将时间序列相邻两期数值相减的变换，用于消除时间序列的趋势和单位根。

## 定义

对于时间序列{$y_t$}，其一阶差分定义为：

$\Delta y_t = y_t - y_{t-1}$

## 运算子表示

$\Delta = 1 - L$

其中L是滞后算子：$Ly_t = y_{t-1}。$

## 性质

1. **消除线性趋势**：如果$y_t$有线性趋势，Δ$y_t$是平稳的
2. **降低单整阶数**：I(d)序列的一阶差分是I(d-1)
3. **减少观测值数**：差分后损失一个观测值

## 多阶差分

二阶差分：
$\Delta^2 y_t = \Delta(\Delta y_t) = y_t - 2y_{t-1} + y_{t-2}$

m阶差分：
$\Delta^m y_t = \sum_{j=0}^{m} (-1)^j \binom{m}{j} y_{t-j}$

## 应用

### 1. 单位根检验

在ADF检验中使用差分来检验平稳性：

$\Delta y_t = \alpha + \beta t + \rho y_{t-1} + \sum_{i=1}^{p} \gamma_i \Delta y_{t-i} + \varepsilon_t$

### 2. 平稳化

对非平稳序列进行差分使其平稳。

### 3. 经济增长率

$\text{增长率} = \frac{\Delta y_t}{y_{t-1}} \approx \ln(y_t) - \ln(y_{t-1}) = \Delta \ln(y_t)$

## 与其他变换比较

| 变换 | 公式 | 效果 |
|------|------|------|
$| 一阶差分 | Δy_t = y_t - y_{t-1} | 消除线性趋势 |$
| 对数差分 | Δln($y_t$) | 近似增长率 |
$| 季节差分 | Δ_d y_t = y_t - y_{t-d} | 消除周期性 |$

## 注意事项

1. 差分损失长期信息
2. 差分改变经济含义（从水平值变为变化率）
3. 过度差分可能导致过度差分（过度平稳）

相关链接: [[Unit Root Test|单位根检验]], [[Stationarity|平稳性]], [[Random Walk|随机游走]]
