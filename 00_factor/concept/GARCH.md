---
aliases:
- GARCH
- GARCH Model
- Generalized Autoregressive Conditional Heteroskedasticity
- 广义自回归条件异方差
tags:
  - concept
  - 时间序列
  - 波动建模
---
# GARCH

## 先记一句话

GARCH 就是：**用过去冲击平方和过去方差一起解释当前条件方差**。

它是 ARCH 的更节省参数版本。

## 它是什么

GARCH(1,1)：
$$
\varepsilon_t=\nu_t\sqrt{h_t},
$$
$$
h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1}.
$$

其中：

- $\alpha$：新冲击对波动的影响；
- $\beta$：过去波动的持续性；
- $\alpha+\beta$：波动持久性。

## 它解决什么判断

GARCH 用来刻画波动聚集和持久波动。

如果 ARCH 需要很多滞后阶数才能拟合，GARCH(1,1) 常常能用更少参数完成同样事情。

## 长期方差

若
$$
\alpha+\beta<1,
$$
则无条件方差存在：
$$
E(\varepsilon_t^2)=\frac{\omega}{1-\alpha-\beta}.
$$

若 $\alpha+\beta$ 接近 1，说明冲击影响很持久。

## 常见误区

- GARCH 不是均值模型；通常先建 ARMA 等均值模型，再对残差建 GARCH。
- $\alpha+\beta<1$ 是平稳性/长期方差存在的重要条件。
- $\alpha+\beta=1$ 进入 [[IGARCH]] 语境，长期方差不再均值回复。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#2.3 GARCH|时间序列 04：GARCH 模型]]

## 关联卡片

- [[ARCH]]
- [[IGARCH]]
- [[TARCH]]
- [[EGARCH]]
- [[GARCH Model Estimation Steps]]
- [[Conditional Heteroskedasticity]]


## 最小例子

把 **GARCH** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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
