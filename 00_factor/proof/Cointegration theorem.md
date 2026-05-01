---
aliases:
- Cointegration theorem
- Granger representation theorem
- Engle-Granger representation theorem
- 协整定理
- Granger 表示定理
tags:
- proof
- 时间序列
---
# Cointegration theorem

## 假设

设 $x_t$ 是 $n$ 维 $I(1)$ 向量，且可由 VAR$(p)$ 描述：
$$
x_t=A_1x_{t-1}+\cdots+A_px_{t-p}+\varepsilon_t.
$$

目标是说明：

> 协整关系和误差修正模型是同一结构的两种写法。

## 推导链

把 VAR 改写成差分形式：
$$
\Delta x_t=\Pi x_{t-1}+\sum_{i=1}^{p-1}\Gamma_i\Delta x_{t-i}+\varepsilon_t.
$$

关键矩阵是 $\Pi$。

若
$$
\operatorname{rank}(\Pi)=r,\qquad 0<r<n,
$$
则可分解为
$$
\Pi=\alpha\beta^T.
$$

于是
$$
\Delta x_t=\alpha(\beta^Tx_{t-1})+\sum_{i=1}^{p-1}\Gamma_i\Delta x_{t-i}+\varepsilon_t.
$$

其中：

- $\beta^Tx_{t-1}$ 是长期均衡误差；
- $\alpha$ 是偏离均衡后的调整速度。

## 为什么 $\beta$ 是协整向量

因为 $x_t$ 是 $I(1)$，所以 $\Delta x_t$ 是 $I(0)$。

右侧的差分滞后项也是 $I(0)$。

要让等式两边同阶，长期项
$$
\beta^Tx_{t-1}
$$
也必须是 $I(0)$。

这正是协整定义。

## 三种秩情形

| $\operatorname{rank}(\Pi)$ | 含义 |
| --- | --- |
| $0$ | 无协整，模型退化为纯差分 |
| $0<r<n$ | 有 $r$ 个协整关系 |
| $n$ | 变量本身平稳，不是典型 $I(1)$ 协整问题 |

## 结论

若 $I(1)$ 变量存在协整关系，就可以写成误差修正形式。

反过来，如果存在有效 ECM/VECM 形式，其中误差修正项平稳，也说明变量之间存在协整。

## 来自课程位置

- [[07_协整和误差修正模型#2.3 协整与误差修正模型|时间序列 07：Granger 表示定理]]

## 关联卡片

- [[Cointegration]]
- [[Error Correction Model]]
- [[Johansen Cointegration Test]]
- [[VAR Model]]
- [[Matrix Rank]]

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
