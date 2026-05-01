---
aliases:
- Monte Carlo Simulation VaR
- 蒙特卡罗模拟法VaR
- 蒙特卡罗VaR计算
tags:
- procedure
- risk-management
---
# Monte Carlo Simulation VaR

## 这张卡什么时候用

当组合含非线性产品、路径依赖产品，或题目要求模拟风险因子并重新定价组合时，用这张卡。方法概念见 [[Monte Carlo Simulation Method]]。

## 输入

- 风险因子模型：价格、利率、汇率、波动率等。
- 参数：波动率、相关矩阵、漂移或情景假设。
- 当前组合和定价函数。
- 模拟次数 $M$。
- [[Confidence Level]] $\alpha$ 与 [[Holding Period]] $h$。

## 输出

- 模拟损益分布。
- $\operatorname{VaR}_{\alpha,h}$。
- 可选：[[ES]] 与情景解释。

## Step 1：设定风险因子模型

明确哪些变量会动，以及怎么动。例如股票收益可用正态或厚尾分布，利率可用期限结构模型，波动率可接 [[GARCH]] 或 [[EWMA]]。

## Step 2：生成相关随机冲击

如果风险因子相关，先分解相关矩阵：

$$
\Sigma=LL^\top,\qquad \epsilon=Lz
$$

其中 $z$ 是独立标准随机数。

## Step 3：模拟期末风险因子

对每条路径 $m=1,\dots,M$，得到持有期末风险因子：

$$
X_h^{(m)}
$$

路径依赖产品需要模拟中间路径；普通欧式产品通常只需期末状态。

## Step 4：重新定价组合

用模拟出的风险因子重新定价：

$$
\Delta P^{(m)}=P(X_h^{(m)})-P_0
$$

这是蒙特卡罗 VaR 和线性参数法的关键区别。

## Step 5：排序并取分位数

把 $\Delta P^{(m)}$ 从小到大排序，令：

$$
k=\lceil M(1-\alpha)\rceil
$$

则：

$$
\operatorname{VaR}_{\alpha}=-\Delta P_{(k)}
$$

## 检查点

- 模拟次数越大，随机误差越小，但模型风险不会自动消失。
- 相关矩阵必须可用；若不正定，先做数据和矩阵诊断。
- 定价模型错，会直接污染 VaR。
- 高置信度尾部需要很多路径，否则尾部分位数稀疏。

## 常见错误

- 只模拟收益率，不重新定价非线性产品。
- 忽略相关性，把多资产组合当成独立资产。
- 把蒙特卡罗当作“无假设方法”；它的假设藏在风险因子模型里。

## 来自课程位置

- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[VaR]]
- [[Monte Carlo Simulation Method]]
- [[VaR Method Selection]]
- [[Delta-Gamma Approximation]]
- [[Historical Simulation VaR]]
