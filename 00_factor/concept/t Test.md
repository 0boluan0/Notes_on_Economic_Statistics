---
aliases:
- t Test
- t-test
- Student's t-test
- t检验
tags:
- concept
- statistics
- econometrics
---
# t Test

## 先记一句话

t Test 用估计值除以标准误，检验一个参数或均值是否显著偏离假设值。

## 它是什么

回归系数检验常写为：

$$
t=\frac{\hat\beta_j-\beta_{j,0}}{SE(\hat\beta_j)}
$$

常见原假设是 $H_0:\beta_j=0$。

## 解决什么判断

它回答：“单个系数是否显著不同于某个假设值？”

## 最小例子

若教育年限系数为 0.08，标准误为 0.02，则 $t=4$，通常拒绝系数为 0 的假设。

## 易混点

- t 显著不等于因果成立。
- 标准误错，t 检验就错；异方差看 [[White Robust Standard Errors]]，自相关看 [[Newey-West]]。
- 多个系数联合检验用 [[F-test]]。

## 来自课程位置

- [[03_多元线性回归]]

## 关联卡片

- [[F-test]]
- [[R-squared]]
- [[OLS Estimation Steps]]
