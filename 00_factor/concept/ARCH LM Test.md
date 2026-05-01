---
aliases:
- ARCH LM Test
- ARCH-LM Test
- Engle ARCH LM Test
- ARCH-LM检验
tags:
- concept
- 时间序列
- 波动建模
---
# ARCH LM Test

## 先记一句话

ARCH-LM 检验就是：**检验残差平方能否被自己的滞后解释，从而判断是否存在 ARCH 效应**。

它检验的是条件方差结构，不是均值自相关。

## 它是什么

对均值模型残差 $\hat\varepsilon_t$，做辅助回归：
$$
\hat\varepsilon_t^2
=\alpha_0+\alpha_1\hat\varepsilon_{t-1}^2+\cdots+\alpha_q\hat\varepsilon_{t-q}^2+u_t.
$$

原假设：
$$
H_0:\alpha_1=\cdots=\alpha_q=0.
$$

若拒绝，说明存在 ARCH 效应。

常用统计量：
$$
TR^2\sim\chi^2(q).
$$

## 它解决什么判断

ARCH-LM 回答：

> 均值模型残差的波动是否仍然有可预测结构？

若有，应考虑 [[ARCH]] / [[GARCH]]。

## 常见误区

- 先拟合均值模型，再对残差做 ARCH-LM。
- 检验对象是残差平方，不是残差本身。
- ARCH-LM 显著不告诉你一定用哪个 GARCH 扩展，只告诉你方差动态不可忽略。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#2.4 侦测ARCH/GARCH效应|时间序列 04：侦测 ARCH/GARCH 效应]]

## 关联卡片

- [[Conditional Heteroskedasticity]]
- [[Volatility Clustering]]
- [[ARCH]]
- [[GARCH]]
- [[McLeod-Li Test]]

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
