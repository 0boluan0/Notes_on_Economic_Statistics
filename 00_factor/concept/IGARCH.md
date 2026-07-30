---
aliases:
- IGARCH
- Integrated GARCH
- 积整GARCH
tags:
  - concept
  - 时间序列
  - 波动建模
---
# IGARCH

## 先记一句话

IGARCH 就是：**波动率方程里冲击影响像单位根一样不衰减的 GARCH**。

以 GARCH(1,1) 看：
$$
\alpha+\beta=1.
$$

## 它是什么

GARCH(1,1)：
$$
h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1}.
$$

若
$$
\alpha+\beta=1,
$$
则波动冲击具有极强持久性，长期无条件方差不再回到固定水平。

## 它解决什么判断

当估计出的 GARCH 模型 $\alpha+\beta$ 非常接近 1，说明波动高度持久，可能要考虑 IGARCH 或 EWMA 近似。

## 和 EWMA 的关系

[[EWMA]] 可以看成一种特殊的 IGARCH 风格递推：
$$
h_t=\lambda h_{t-1}+(1-\lambda)u_{t-1}^2.
$$

它也没有显式长期均值回归项。

## 常见误区

- IGARCH 不是“更稳”的 GARCH；它意味着冲击极持久。
- $\alpha+\beta=1$ 时，标准长期方差公式不再适用。
- 接近 1 和等于 1 在解释上不同，估计时要看统计检验和样本稳定性。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#3.1 IGARCH|时间序列 04：IGARCH]]

## 关联卡片

- [[GARCH]]
- [[EWMA]]
- [[Volatility Clustering]]


## 最小例子

把 **IGARCH** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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
