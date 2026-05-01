---
aliases:
- Partial Autocorrelation Function
- PACF
- partial autocorrelation
- 偏自相关函数
tags:
- concept
- 时间序列
---
# Partial Autocorrelation Function

## 先记一句话

PACF 就是：**控制中间滞后以后，$y_t$ 和 $y_{t-k}$ 之间还剩多少直接相关**。

它和 ACF 的区别是：

- [[Autocorrelation Function]] 看总相关；
- PACF 看直接相关。

## 它是什么

滞后 $k$ 的 PACF 通常记作
$$
\phi_{kk}.
$$

它可以理解为回归
$$
y_t=\phi_{k1}y_{t-1}+\cdots+\phi_{kk}y_{t-k}+e_t
$$
中最后一个系数 $\phi_{kk}$。

## 它解决什么判断

PACF 主要帮助识别 AR 阶数。

| 模型 | PACF 图像 |
| --- | --- |
| AR(p) | p 阶后截尾 |
| MA(q) | 拖尾 |
| ARMA(p,q) | 拖尾 |

所以看到 PACF 在某个阶数后明显不再显著，先怀疑 AR 模型。

## 一个最小例子

AR(1)
$$
y_t=a_0+a_1y_{t-1}+\varepsilon_t
$$
只有一阶直接影响。

所以理论 PACF 在 1 阶后截尾。

高阶滞后的 ACF 可能仍然不为 0，但那是通过 $y_{t-1}$ 传递出来的间接影响。

## 常见误区

- PACF 不是 ACF 的“修正版”，它回答的是不同问题。
- 样本 PACF 不会完美截尾，要结合显著性界限和信息准则。
- 纯 MA 模型的 PACF 通常拖尾，不要用 PACF 直接读 q。

## 来自课程位置

- [[03_平稳时间序列模型#0.回忆用|时间序列 03：PACF 回忆索引]]
- [[03_平稳时间序列模型#3. ACF|时间序列 03：ACF/PACF 识别]]

## 关联卡片

- [[Autocorrelation Function]]
- [[ARMA]]
- [[Autoregressive Model]]
- [[Yule-Walker equations]]
- [[ARMA Model Identification Steps]]

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
