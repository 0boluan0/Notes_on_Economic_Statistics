---
aliases:
- LSDV Estimation
- Least Squares Dummy Variable Estimation
- LSDV
- 固定效应 LSDV 估计
tags:
- procedure
- econometrics
- panel-data
---
# LSDV Estimation

## 这张卡什么时候用

面板数据中需要用虚拟变量控制个体固定效应或时间固定效应时使用。

## 输入

- 面板数据 $y_{it},x_{it}$。
- 个体编号 $i$。
- 时间编号 $t$。
- 是否需要个体固定效应、时间固定效应或双向固定效应。

## 输出

- 固定效应模型中的 $\hat\beta$。
- 个体或时间虚拟变量系数。
- 组内解释力和回归诊断。

## Step 1：写出固定效应模型

$$
y_{it}=\alpha+\beta x_{it}+\mu_i+\lambda_t+u_{it}
$$

## Step 2：加入虚拟变量

有截距时，$N$ 个个体只放 $N-1$ 个个体虚拟变量，避免完全多重共线性。

## Step 3：一起做 OLS

把解释变量和虚拟变量同时放入回归，OLS 得到 $\hat\beta$。

## Step 4：解释 $\beta$

$\beta$ 来自控制个体固定差异后的组内变化。

## 检查点

- 不要同时放截距和全部类别虚拟变量。
- LSDV 与组内变换给出的核心斜率一致。
- 动态面板中直接用 LSDV 会有偏，见 [[Dynamic Panel Data Model]]。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[Fixed Effects Model]]
- [[Panel Data Model]]
- [[Hausman Test]]
