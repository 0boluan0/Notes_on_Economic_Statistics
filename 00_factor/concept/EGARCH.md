---
aliases:
- EGARCH
- Exponential GARCH
- 指数GARCH
tags:
- concept
- 时间序列
- 波动建模
---
# EGARCH

## 先记一句话

EGARCH 就是：**用对数方差建模，并把冲击方向和冲击大小分开处理的 GARCH 扩展**。

它天然保证预测方差为正。

## 它是什么

EGARCH(1,1) 的典型形式是：
$$
\ln h_t
=\alpha_0
+\alpha_1\frac{\varepsilon_{t-1}}{\sqrt{h_{t-1}}}
+\lambda\left|\frac{\varepsilon_{t-1}}{\sqrt{h_{t-1}}}\right|
+\beta\ln h_{t-1}.
$$

其中标准化残差
$$
z_{t-1}=\frac{\varepsilon_{t-1}}{\sqrt{h_{t-1}}}
$$
同时保留方向和大小。

## 它解决什么判断

EGARCH 用于：

- 方差必须保持正；
- 负冲击和正冲击影响不对称；
- 标准 GARCH 参数非负约束太笨重。

## 和 TARCH 的区别

- [[TARCH]] 用指示变量区分负冲击；
- EGARCH 用标准化残差和绝对值项分解方向与幅度；
- EGARCH 建模 $\ln h_t$，所以不需要直接约束 $h_t>0$。

## 常见误区

- EGARCH 的核心不是“指数分布”，而是 exponential/log variance structure。
- 对数方差为负没有问题，方差本身仍为正。
- 非对称性要看方向项的系数。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#3.4 非对称模型:TARCH,EGARCH|时间序列 04：EGARCH]]

## 关联卡片

- [[GARCH]]
- [[TARCH]]
- [[Conditional Heteroskedasticity]]
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
