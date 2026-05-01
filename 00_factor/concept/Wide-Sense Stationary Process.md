---
aliases:
- Wide-Sense Stationary Process
- Weakly Stationary Process
- Covariance Stationary Process
- WSS
- 宽平稳过程
- 弱平稳过程
- 协方差平稳过程
tags:
- concept
- 时间序列
- 随机过程
---
# Wide-Sense Stationary Process

## 先记一句话

宽平稳就是：**只要求均值、方差和自协方差结构不随时间漂移的平稳性**。

时间序列课里说“平稳”，很多时候默认就是这个。

## 它是什么

过程 $\{y_t\}$ 宽平稳，如果：

$$
E(y_t)=\mu,
$$

$$
\operatorname{Var}(y_t)=\sigma^2,
$$

并且
$$
\operatorname{Cov}(y_t,y_{t-s})=\gamma_s
$$
只依赖滞后 $s$，不依赖具体时间 $t$。

## 一个最小例子

白噪声是宽平稳。

AR(1)
$$
y_t=a_0+a_1y_{t-1}+\varepsilon_t
$$
在
$$
|a_1|<1
$$
时是宽平稳。

## 它在题里负责什么

- 判断 [[ARMA]] 是否可直接建模。
- 让 [[Autocorrelation Function]] 只写成 $\rho_k$，而不是依赖 $t$ 的相关结构。
- 支撑 Yule-Walker 方程和 Box-Jenkins 识别。

## 和严平稳的关系

宽平稳只看二阶矩。

严平稳看整个分布。

所以：

- 严平稳 + 二阶矩存在，可以推出宽平稳；
- 宽平稳一般推不出严平稳；
- Gaussian 情形下，两者等价。

## 常见误区

- 宽平稳不是“图像水平”这么粗略，而是三个数学条件。
- 自协方差只依赖滞后，不是说自协方差必须为 0。
- ARCH/GARCH 残差可以条件方差变化，但仍可能在无条件意义下有稳定二阶矩，要分清条件与无条件。

## 来自课程位置

- [[03_平稳时间序列模型#2.1 平稳性定义|时间序列 03：弱平稳定义]]

## 关联卡片

- [[Stationarity]]
- [[Strict-Sense Stationary Process]]
- [[Autocorrelation Function]]
- [[AR(1) stationarity]]
- [[ARMA]]

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
