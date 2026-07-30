---
aliases:
- Hansen J Test
- Hansen's J Test
- J检验
- GMM J检验
- 过度识别检验
tags:
- system
- econometrics
- gmm
---
# Hansen J Test

## 诊断目标

检验 GMM 或 IV 模型中的过度识别约束是否整体可信。

## 什么时候用

工具变量数量多于内生解释变量数量时，模型过度识别，可以检验额外矩条件是否与残差相容。

## 原假设

所有工具变量整体外生，矩条件成立。

## 统计量

常见形式：

$$
J=n\cdot g(\hat\theta)'Wg(\hat\theta)\sim\chi^2(m-k)
$$

其中 $m$ 是矩条件个数，$k$ 是参数个数。

## 易混点

- 不拒绝原假设不等于工具变量一定正确。
- J 检验不能检验恰好识别模型。
- 这个 J Test 不同于非嵌套模型的 [[Davidson-MacKinnon J Test]]。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[GMM]]
- [[Instrumental Variable]]
- [[2SLS]]
- [[Dynamic Panel Data Model]]
