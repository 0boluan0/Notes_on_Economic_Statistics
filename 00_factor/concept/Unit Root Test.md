---
aliases:
- Unit Root Test
- unit root test
- 单位根检验
- 单位根
- 平稳性检验
tags:
- concept
- 时间序列
- 计量经济学
---
# Unit Root Test

## 先记一句话

单位根检验就是：**判断序列是不是像随机游走一样非平稳**。

它最常见的原假设是：

> 序列存在单位根，非平稳。

## 它是什么

如果 AR(1)
$$
y_t=\rho y_{t-1}+\varepsilon_t
$$
中
$$
\rho=1,
$$
那么序列有单位根。

等价地，写成差分形式：
$$
\Delta y_t=\gamma y_{t-1}+\varepsilon_t,
$$
其中
$$
\gamma=\rho-1.
$$

检验单位根就是检验
$$
H_0:\gamma=0.
$$

## 它解决什么判断

单位根检验决定后续建模分流：

- 拒绝单位根：可考虑平稳模型，如 [[ARMA]]。
- 不能拒绝单位根：考虑 [[First Difference]]、[[ARIMA]]。
- 多个变量都是 $I(1)$：继续检查 [[Cointegration]]。

## 常见检验

- [[Augmented Dickey-Fuller Test]]：通过加入滞后差分项处理自相关。
- [[Phillips-Perron Test]]：用非参数修正处理自相关和异方差。
- KPSS：原假设相反，通常是平稳。

对比诊断见 [[Stationarity Tests Comparison]]。

## 常见误区

- ADF 的原假设是“有单位根”，不是“平稳”。
- ADF 统计量不服从普通 t 分布。
- 是否加截距、趋势项会影响结论。
- 不能把“不拒绝单位根”写成“证明有单位根”；只能说证据不足以拒绝。

## 来自课程位置

- [[03_平稳时间序列模型#1.1.3 ARIMA过程|时间序列 03：单位根与 ARIMA]]
- [[07_协整和误差修正模型#3.1 EG两步法|时间序列 07：协整前的单位根预检]]

## 关联卡片

- [[Augmented Dickey-Fuller Test]]
- [[Phillips-Perron Test]]
- [[ADF Test Steps]]
- [[Stationarity Tests Comparison]]
- [[Random Walk]]
- [[First Difference]]
- [[Cointegration]]

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
