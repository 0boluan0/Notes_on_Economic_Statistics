---
aliases:
- Davidson-MacKinnon J Test
- J Test
- 非嵌套模型 J 检验
- 戴维森-麦金农 J 检验
tags:
- system
- econometrics
---
# Davidson-MacKinnon J Test

## 诊断目标

在两个非嵌套模型之间判断另一个模型是否提供了当前模型没有解释的信息。

## 什么时候用

两个模型不能通过加减一组变量变成彼此的受约束版本，因此普通 F 检验不适用。

## 检验做法

设模型 A 为 $Y=X\beta+u$，模型 B 为 $Y=Z\gamma+v$。

1. 先估计模型 B，得到 $\hat Y_B$。
2. 在模型 A 中加入 $\hat Y_B$：

   $$
   Y=X\beta+\lambda\hat Y_B+e
   $$

3. 检验 $H_0:\lambda=0$。

## 易混点

- 这个 J Test 是非嵌套模型检验，不是 GMM 里的 [[Hansen J Test]]。
- 两个方向都应检验：A 对 B、B 对 A。
- 结果可能出现两个模型都被拒绝，说明都不够好。

## 来自课程位置

- [[04_模型设定]]

## 关联卡片

- [[Model Misspecification]]
- [[Ramsey RESET Test]]
- [[F-test]]
