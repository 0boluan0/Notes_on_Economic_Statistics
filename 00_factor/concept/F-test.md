---
aliases:
- F-test
- F检验
- F Test
tags:
- concept
- statistics
- econometrics
---

# F-test

## 先记一句话

F-test 用来检验多个线性限制是否同时成立。

## 它是什么

回归里最常见的是联合显著性或嵌套模型限制检验。受限模型和非受限模型比较：

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/(n-k)}
$$

其中 $q$ 是限制个数。

## 解决什么判断

它回答：“一组变量或一组限制整体是否有统计意义？”

## 最小例子

检验三个行业虚拟变量是否整体显著：$H_0:\beta_1=\beta_2=\beta_3=0$。这应使用 F-test，而不是只看三个 t 检验。

## 易混点

- 单个限制时，$F=t^2$。
- F-test 依赖标准误和协方差估计；异方差或自相关下要用稳健版本。
- F 显著但多个 t 不显著，常见原因是 [[Multicollinearity]]。

## 来自课程位置

- [[03_多元线性回归]]

## 关联卡片

- [[t Test]]
- [[R-squared]]
- [[Multicollinearity]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Multicollinearity]]、[[03_多元线性回归]]、[[t Test]]、[[R-squared]]。
