---
aliases:
- Fisher Discriminant Procedure
- Fisher Linear Discriminant Steps
- Fisher 判别步骤
tags:
- procedure
- multivariate statistics
type: procedure
---
# Fisher Discriminant Procedure

## 输入

- 两类样本 $\pi_1,\pi_2$。
- 两组样本均值 $\bar x_1,\bar x_2$。
- pooled covariance $S_{\text{pooled}}$。
- 新观测 $x_0$。

## 输出

- $x_0$ 的分类结果。

## Step 1. 计算判别方向

$$
\hat a=S_{\text{pooled}}^{-1}(\bar x_1-\bar x_2).
$$

## Step 2. 计算投影分数

$$
y_0=\hat a'x_0.
$$

两组中心投影为
$$
\hat a'\bar x_1,\qquad \hat a'\bar x_2.
$$

## Step 3. 设置阈值

相等先验和相等成本下，阈值为
$$
c=\frac12\hat a'(\bar x_1+\bar x_2).
$$

## Step 4. 分类

若 $\hat a'\bar x_1>\hat a'\bar x_2$，则
$$
y_0\geq c
$$
时分到 $\pi_1$，否则分到 $\pi_2$。

## 检查点

- 两组协方差近似相等时 Fisher 线性判别更合适。
- 若先验概率或误分类成本不相等，阈值需要调整。

## 常见错误

- 只算出 $\hat a$，但忘记把新样本和组均值都投影。
- 不检查两个组中心投影的大小方向。

## 来自课程位置

- [[11_分类与判别Discrimination and Classifications#1.4. Fisher 判别方法|第11章 4 Fisher 判别方法]]

## 关联卡片

- [[Fisher Linear Discriminant]]
- [[Expected Cost of Misclassification]]
- [[Classification Rule Selection]]
