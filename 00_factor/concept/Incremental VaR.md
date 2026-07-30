---
aliases:
- Incremental VaR
- incremental Value at Risk
- IVaR
- 递增VaR
tags:
  - concept
  - 金融风险
  - VaR
---
# Incremental VaR

## 先记一句话

递增 VaR 就是：**加入或移除一笔具体交易后，组合 VaR 改变了多少**。

它看的是有限变动，不是微小边际变化。

## 它是什么

若原组合 VaR 是 $VaR(P)$，加入交易 $x$ 后是 $VaR(P+x)$，则
$$
Incremental\ VaR=VaR(P+x)-VaR(P).
$$

也可以用来评估删除某个头寸后的 VaR 变化。

## 它解决什么判断

Incremental VaR 用来回答：

> 这笔新交易会让组合风险增加还是减少？增加多少？

它非常适合交易审批和限额管理。

## 和边际 VaR 的关系

小头寸时：
$$
Incremental\ VaR \approx Marginal\ VaR\times \Delta w.
$$

但头寸较大或组合非线性时，必须重新计算组合 VaR。

## 常见误区

- Incremental VaR 不是分摊总 VaR；分摊看 [[Component VaR]]。
- 不能只看该交易自身风险，要看它与原组合的相关性。
- 新交易可能自身有风险，但因对冲原组合而降低总 VaR。

## 来自课程位置

- [[12_VAR风险|金融风险管理 12：VaR 分解]]

## 关联卡片

- [[VaR]]
- [[Marginal VaR]]
- [[Component VaR]]
- [[Position]]


## 最小例子

把 **Incremental VaR** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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
