---
aliases:
- White Test
- White Heteroskedasticity Test
- 怀特检验
tags:
- concept
- econometrics
---
# White Test

## 先记一句话

White Test 用残差平方的辅助回归检验异方差，不需要预先指定方差函数形式。

## 它是什么

先做原模型 OLS，得到残差 $\hat u_i$。再把 $\hat u_i^2$ 对解释变量、平方项和交叉项做辅助回归。若辅助回归解释力显著，说明方差可能随变量变化。

检验统计量：

$$
LM=nR^2_{aux}\sim \chi^2(q)
$$

其中 $q$ 是辅助回归中除常数外的解释变量个数。

## 解决什么判断

它回答：“是否存在不需要具体形式假设的一般性异方差证据？”

## 最小例子

对工资回归残差平方，辅助回归包含教育、经验、教育平方、经验平方和交叉项，若 $nR^2$ 显著，拒绝同方差。

## 易混点

- White Test 是检验；[[White Robust Standard Errors]] 是修正标准误。
- 变量很多时，完整 White 检验会吃掉大量自由度。
- 拒绝同方差后，不代表系数估计一定有偏；重点是推断和效率。

## 来自课程位置

- [[07_异方差]]

## 关联卡片

- [[Heteroskedasticity]]
- [[White Test Steps]]
- [[Heteroscedasticity Diagnosis]]
- [[White Robust Standard Errors]]
