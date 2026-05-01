---
aliases:
- EWMA
- Exponentially Weighted Moving Average
- 指数加权移动平均
- 指数加权移动平均模型
tags:
- concept
- 时间序列
- 波动建模
---
# EWMA

## 先记一句话

EWMA 就是：**给最近的平方收益更高权重，快速更新波动率估计**。

它比等权历史窗口更敏感。

## 它是什么

EWMA 方差递推：
$$
h_t=\lambda h_{t-1}+(1-\lambda)u_{t-1}^2,
$$
其中
$$
0<\lambda<1.
$$

$\lambda$ 越大，历史记忆越长；$\lambda$ 越小，对最新冲击反应越快。

## 一个最小直觉

RiskMetrics 日频常用
$$
\lambda=0.94.
$$

这意味着昨天的平方收益权重是 $0.06$，更早数据的权重按 $0.94$ 递减。

## 它在题里负责什么

- 日常风险监控中快速更新波动率。
- 作为 [[GARCH]] 的简化近似。
- 在 VaR 参数法里给出动态波动率输入。

具体计算见 [[EWMA Volatility Estimation]]。

## 和 GARCH 的关系

EWMA 类似没有常数项、且
$$
\alpha+\beta=1
$$
的 GARCH(1,1)。

因此它没有固定长期方差均值，波动率冲击不会像标准 GARCH 那样均值回复。

## 常见误区

- EWMA 不是普通移动平均；权重是指数衰减。
- $\lambda$ 大不是“更好”，只是更慢、更平滑。
- EWMA 简单，但不能显式估计长期方差回归水平。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#3.1 IGARCH|时间序列 04：EWMA / IGARCH 思路]]
- [[10_波动率|金融风险管理 10：波动率估计]]

## 关联卡片

- [[EWMA Volatility Estimation]]
- [[Historical Volatility]]
- [[GARCH]]
- [[IGARCH]]
- [[Volatility Clustering]]

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
