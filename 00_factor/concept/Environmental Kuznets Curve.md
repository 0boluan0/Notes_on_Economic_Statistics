---
aliases:
- Environmental Kuznets Curve
- EKC
- 环境库兹涅茨曲线
tags:
- concept
- econometrics
- environmental-economics
---
# Environmental Kuznets Curve

## 先记一句话

环境库兹涅茨曲线描述环境污染可能随收入先上升、后下降的倒 U 型关系。

## 它是什么

常见二次项模型：

$$
WG_i=\beta_0+\beta_1PGDP_i+\beta_2PGDP_i^2+u_i
$$

若 $\beta_1>0,\beta_2<0$，则是倒 U 型。

## 解决什么判断

它回答：“经济发展和环境压力之间是否存在转折点？”

## 最小例子

收入较低阶段工业扩张增加污染；收入较高后技术升级和环保需求增强，污染下降。

## 易混点

- 倒 U 型需要同时看一次项和二次项的符号。
- 转折点为 $-\beta_1/(2\beta_2)$。
- EKC 是经验关系，不是自动成立的经济定律。

## 来自课程位置

- [[03_多元线性回归]]

## 关联卡片

- [[Multiple Linear Regression]]
- [[Model Misspecification]]
- [[Omitted Variable Bias]]
