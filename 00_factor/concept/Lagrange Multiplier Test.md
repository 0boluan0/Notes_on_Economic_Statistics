---
aliases:
- Lagrange Multiplier Test
- LM Test
- LM检验
- 拉格朗日乘数检验
tags:
  - concept
  - econometrics
  - statistics
---
# Lagrange Multiplier Test

## 先记一句话

LM Test 用受约束模型的残差判断“放松约束后是否还能显著改进”。

## 它是什么

常见辅助回归形式下：

$$
LM=nR^2\sim\chi^2(q)
$$

其中 $q$ 是被检验的限制个数。

## 解决什么判断

它回答：“在原假设模型下，残差是否还含有系统性信息？”

## 最小例子

怀疑遗漏 $X^2$ 时，先估计原模型，再看残差能否被 $X^2$ 解释。若能，说明原模型可能设定不足。

## 易混点

- LM Test 通常只需要估计受约束模型。
- White 检验、BG 检验、ARCH-LM 都属于 LM 思路的具体应用。
- LM 显著不直接告诉你正确模型是什么，只说明原模型有问题。

## 来自课程位置

- [[04_模型设定]]
- [[05_多元回归模型的矩阵表达]]

## 关联卡片

- [[Ramsey RESET Test]]
- [[Breusch-Godfrey Test]]
- [[White Test]]
- [[ARCH LM Test]]
