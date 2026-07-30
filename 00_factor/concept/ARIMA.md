---
aliases:
- ARIMA
- ARIMA Model
- Autoregressive Integrated Moving Average Model
- ARIMA模型
- 自回归积分移动平均模型
tags:
- concept
- 时间序列
---

# ARIMA

## 先记一句话

ARIMA 就是：**先把非平稳序列差分到平稳，再对差分后的序列建 ARMA**。

ARIMA$(p,d,q)$ 的意思是：

- $p$：AR 阶数；
- $d$：差分阶数；
- $q$：MA 阶数。

## 它是什么

如果
$$
\Delta^d y_t
$$
可以用 ARMA$(p,q)$ 描述，那么
$$
y_t\sim ARIMA(p,d,q).
$$

其中
$$
\Delta y_t=y_t-y_{t-1}.
$$

## 它解决什么判断

ARIMA 用于有单位根、但差分后可以平稳的序列。

题目信号：

- 原序列非平稳；
- ADF/PP 不能拒绝单位根；
- 一阶差分后变平稳；
- 需要对单变量序列建模和预测。

## 一个最小例子

随机游走
$$
y_t=y_{t-1}+\varepsilon_t
$$
本身非平稳。

一阶差分后：
$$
\Delta y_t=\varepsilon_t.
$$

所以它可以看成 ARIMA$(0,1,0)$。

## 常见误区

- ARIMA 不是“比 ARMA 更高级”，而是处理非平稳单变量序列的一种方式。
- 差分会丢掉水平上的长期关系；如果多个 $I(1)$ 变量之间有长期均衡，应看 [[Cointegration]] 和 [[Error Correction Model]]。
- 不要过度差分；过度差分会制造不必要的 MA 结构。

## 来自课程位置

- [[03_平稳时间序列模型#1.1.3 ARIMA过程|时间序列 03：ARIMA 过程]]
- [[07_协整和误差修正模型#2.1 协整的定义|时间序列 07：差分与协整的分流]]

## 关联卡片

- [[ARMA]]
- [[First Difference]]
- [[Unit Root Test]]
- [[Random Walk]]
- [[Cointegration]]
- [[Spurious Regression]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Cointegration]]、[[Error Correction Model]]、[[03_平稳时间序列模型]]、[[07_协整和误差修正模型]]、[[ARMA]]、[[First Difference]]、[[Unit Root Test]]、[[Random Walk]]。

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
