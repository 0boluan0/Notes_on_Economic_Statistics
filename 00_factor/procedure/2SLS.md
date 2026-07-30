---
aliases:
- 2SLS
- Two-Stage Least Squares
- 两阶段最小二乘法
tags:
- procedure
- econometrics
type: procedure
---
# 2SLS

## 这张卡什么时候用

当模型有内生解释变量，并且有满足条件的 [[Instrumental Variable|工具变量]] 时，用 2SLS 估计因果参数。

## 输入

- 因变量 $y$。
- 内生解释变量 $X_{endog}$。
- 外生控制变量 $W$。
- 工具变量 $Z$。

## 输出

- 2SLS 系数估计。
- 第一阶段强度诊断。
- 内生性和过度识别检验结果。

## Step 1：第一阶段

用工具变量和所有外生控制解释内生变量：

$$
X_{endog}=Z\Pi+W\Gamma+v
$$

保存预测值 $\hat X_{endog}$，并检查第一阶段 F 统计量。

## Step 2：第二阶段

用 $\hat X_{endog}$ 和外生控制估计结果方程：

$$
y=\beta \hat X_{endog}+W\delta+u
$$

实际软件会用正确的 2SLS 方差公式，不要手动把第一阶段预测值当普通变量后报告普通 OLS 标准误。

## Step 3：做诊断

- 弱工具：第一阶段 F。
- 内生性：[[Hausman Test]] / Durbin-Wu-Hausman。
- 过度识别：Sargan/Hansen。

## 常见错误

- 第二阶段手动 OLS 后直接用普通标准误。
- 工具变量只满足相关性，不满足外生性。
- 工具变量数量很多但很弱。

## 来自课程位置

- [[09_联立方程模型(内生性)]]

## 关联卡片

- [[Instrumental Variable]]
- [[Endogeneity Diagnosis]]
- [[GMM]]
- [[Hausman Test]]
