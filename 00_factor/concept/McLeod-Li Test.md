---
aliases:
- McLeod-Li Test
- McLeod-Li Portmanteau Test
- McLeod-Li检验
tags:
  - concept
  - 时间序列
  - 波动建模
---
# McLeod-Li Test

## 先记一句话

McLeod-Li 检验就是：**对残差平方做 Ljung-Box 型检验，判断是否存在 ARCH/GARCH 效应**。

它和 ARCH-LM 都是在看波动是否有记忆。

## 它是什么

先得到均值模型残差 $\hat\varepsilon_t$。

再看平方残差：
$$
\hat\varepsilon_t^2.
$$

如果平方残差有显著自相关，说明条件方差不是常数。

## 它解决什么判断

McLeod-Li 回答：

> 残差平方序列是否还像白噪声？

如果不是，就考虑 [[ARCH]] / [[GARCH]]。

## 和 ARCH-LM 的分工

- [[ARCH LM Test]]：做残差平方滞后回归，LM 检验。
- McLeod-Li：对残差平方的自相关做 portmanteau 检验。

两者都是波动建模前的诊断工具。

## 常见误区

- 不要对原残差做 McLeod-Li；重点是平方残差。
- 显著结果说明波动结构存在，不等于均值模型一定错。
- 若残差本身也有自相关，先修均值模型。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#2.4 侦测ARCH/GARCH效应|时间序列 04：侦测 ARCH/GARCH 效应]]

## 关联卡片

- [[ARCH LM Test]]
- [[Ljung-Box Test]]
- [[Conditional Heteroskedasticity]]
- [[GARCH]]


## 最小例子

把 **McLeod-Li Test** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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
