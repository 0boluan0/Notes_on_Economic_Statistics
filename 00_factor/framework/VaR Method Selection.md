---
aliases:
- VaR方法选择
- Value at Risk method selection
- VaR计算方法选择
tags:
- framework
- risk-management
---
# VaR Method Selection

## 先记一句话

VaR 方法选择不是在“哪种公式更高级”之间选，而是在组合结构、分布假设、尾部目标和数据可用性之间取舍。

## 什么时候用

当题目问“用参数法、历史模拟法、蒙特卡罗法还是 EVT”时，先用这张卡判断方向，再去对应 procedure 算数字。

## 题型识别

| 题目特征 | 优先方法 | 原因 |
| --- | --- | --- |
| 线性组合、给均值方差协方差、假设正态 | [[VaR Parametric Method]] | 可以直接用分位数和组合波动率 |
| 给历史收益或损益样本，让取分位数 | [[Historical Simulation VaR]] | 题目已经给出经验分布 |
| 含期权、路径依赖产品、需要重估组合价值 | [[Monte Carlo Simulation VaR]] | 需要模拟风险因子并重新定价 |
| 高置信度尾部、样本分位数不够稳定 | [[EVT VaR Calculation]] | 需要对尾部做外推 |
| 监管或模型验证语境 | [[Backtesting]] | 重点不在算 VaR，而在检验例外次数和聚束 |

## 为什么这样看

VaR 是损失分布的分位数。不同方法本质上只是在构造损失分布：

- 参数法：用分布假设和协方差矩阵构造。
- 历史模拟：用历史情景构造。
- 蒙特卡罗：用随机模型和重定价构造。
- EVT：只对尾部另建模型。

## 边界条件

- 组合明显非线性时，线性参数法只能当近似，必须警惕 [[Delta-Gamma Approximation]] 的适用范围。
- 历史模拟依赖观察窗口，窗口太短会导致 [[VaR Standard Error|分位数标准误]] 很大。
- 蒙特卡罗看起来灵活，但模型错了，模拟越多也只是更精确地错。
- EVT 只适合尾部；阈值以下的普通分位数不要硬套。

## 失败模式

- 只写“99% VaR”但不写 [[Holding Period|持有期]]。
- 把 [[Absolute VaR]] 和 [[Relative VaR]] 混用。
- 用日 VaR 直接乘天数，而不是在独立同分布近似下用平方根时间规则。
- 只算 VaR 不做 [[Backtesting|回测]]，无法知道模型是否系统性低估风险。

## 来自课程位置

- [[12_VAR风险]]
- [[14_VaR参数法和模拟法]]

## 关联卡片

- [[VaR]]
- [[Variance-Covariance Method]]
- [[Historical Simulation Method]]
- [[Monte Carlo Simulation Method]]
- [[EVT]]
- [[Backtesting]]
