---
aliases:
- TARCH
- Threshold GARCH
- TGARCH
- 门限GARCH
tags:
- concept
- 时间序列
- 波动建模
---
# TARCH

## 先记一句话

TARCH 就是：**让负冲击和正冲击对波动率产生不同影响的 GARCH 扩展**。

它用来刻画 leverage effect。

## 它是什么

TARCH(1,1) 常写成：
$$
h_t=\alpha_0+\alpha_1\varepsilon_{t-1}^2
+\lambda d_{t-1}\varepsilon_{t-1}^2
+\beta h_{t-1}.
$$

其中
$$
d_{t-1}=
\begin{cases}
1,&\varepsilon_{t-1}<0,\\
0,&\varepsilon_{t-1}\geq0.
\end{cases}
$$

如果 $\lambda>0$，负冲击会比正冲击带来更高的未来波动。

## 它解决什么判断

当你发现坏消息比好消息更能提高波动率时，标准 GARCH 的对称结构不够，需要 TARCH 或 EGARCH。

## 常见误区

- TARCH 的门限不是看收益率大小，而是看残差符号。
- TARCH 建的是方差方程，不是均值方程。
- 非对称效应要看 $\lambda$ 的符号和显著性。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#3.4 非对称模型:TARCH,EGARCH|时间序列 04：TARCH 与 leverage effect]]

## 关联卡片

- [[GARCH]]
- [[EGARCH]]
- [[Volatility Clustering]]
- [[Conditional Heteroskedasticity]]

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
