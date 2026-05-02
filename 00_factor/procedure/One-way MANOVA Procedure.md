---
aliases:
- One-way MANOVA Procedure
- One-Way MANOVA Steps
- 单因子 MANOVA 步骤
tags:
- procedure
- multivariate statistics
---
# One-way MANOVA Procedure

## 输入

- $g$ 个组，每组为 $p$ 维观测。
- 显著性水平 $\alpha$。

## 输出

- 对 $H_0:\mu_1=\cdots=\mu_g$ 的检验结论。
- Wilks Lambda 或 MANOVA 表。

## Step 1. 写出模型和假设

$$
X_{ij}=\mu+\tau_i+e_{ij}.
$$

检验：
$$
H_0:\tau_1=\cdots=\tau_g=0.
$$

## Step 2. 计算 SSP 分解

$$
T=H+E.
$$

- $H$：组间 SSP。
- $E$：组内误差 SSP。

## Step 3. 计算 Wilks Lambda

$$
\Lambda^*=\frac{|E|}{|E+H|}.
$$

## Step 4. 作出判断

根据课程给定的近似分布、查表或软件输出判断是否拒绝 $H_0$。

## 检查点

- 每组观测应相互独立。
- 变量维度不能高到让 $E$ 奇异。
- 显著后还要解释哪些变量或线性组合造成差异。

## 常见错误

- 只看每个变量的单独 ANOVA，不看联合结构。
- 把 Wilks Lambda 越大误认为越显著。

## 来自课程位置

- [[06_比较多个均值向量comparisons of multivariate mean vectors#1.5. 多个总体均值向量比较：单因子 MANOVA|第6章 4 单因子 MANOVA]]

## 关联卡片

- [[MANOVA]]
- [[Wilks Lambda]]
- [[SSP Matrix]]
