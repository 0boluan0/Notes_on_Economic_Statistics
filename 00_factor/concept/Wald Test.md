---
aliases:
- Wald Test
- Wald检验
- Wald 约束检验
tags:
- concept
- econometrics
- statistics
---
# Wald Test

## 先记一句话

Wald Test 用无约束估计量检查参数是否离原假设约束足够远。

## 它是什么

若约束为：

$$
R\beta=r
$$

Wald 统计量为：

$$
W=(R\hat\beta-r)'
\left[R\widehat{\operatorname{Var}}(\hat\beta)R'\right]^{-1}
(R\hat\beta-r)
$$

大样本下服从 $\chi^2(q)$。

## 解决什么判断

它回答：“用无约束模型估计出来的参数，是否显著违反原假设约束？”

## 最小例子

检验 $\beta_1+\beta_2=1$，直接把无约束估计量代入 $R\hat\beta-r$。

## 易混点

- Wald Test 主要依赖无约束模型。
- 小样本下结果可能对参数化方式敏感。
- [[Likelihood Ratio Test]] 看似然下降，[[Lagrange Multiplier Test]] 看受约束模型是否仍有改进空间。

## 来自课程位置

- [[05_多元回归模型的矩阵表达]]

## 关联卡片

- [[Likelihood Ratio Test]]
- [[Lagrange Multiplier Test]]
- [[F-test]]
