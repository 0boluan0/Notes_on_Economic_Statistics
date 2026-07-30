---
aliases:
  - "Volatility, Correlation and Copulas"
  - "Copula"
  - "相关性与Copula"
status: source-checked
---

# 波动率、相关性与 Copula

> [!summary] 快速恢复
> **它解决什么：** 描述单个风险因子的时变波动、多资产共同变化及尾部依赖，而不把线性相关当完整联合分布。
> **具体锚点：** 两资产平时相关很低，危机时却同时大跌；一个常数相关系数会低估联合尾部损失。
> **核心难点：** 不相关不等于独立；Pearson 相关对线性和异常值敏感，Copula 分离边际与依赖但仍有模型选择风险。
> **为什么重要：** 组合 VaR、信用组合、对冲和压力测试的结果常被依赖假设主导。
> **继续：** 时间波动见 [[条件异方差：ARCH 与 GARCH]]；尾部损失见 [[历史模拟、蒙特卡罗与极值理论]]。

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
> - [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
> - Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。

## 波动率估计

历史波动常由收益平方平均年化，滚动窗口给等权。隐含波动由期权价格反推出风险中性定价参数，不等于未来实现波动的无偏预测。高频实现波动需处理微观结构噪声。

## EWMA 与 GARCH

EWMA 递推 $\sigma_t^2=\lambda\sigma_{t-1}^2+(1-\lambda)r_{t-1}^2$，权重指数衰减且无固定长期均值。GARCH 加常数和更一般持久性。模型选择看标准化残差、预测和结构变化。

## 协方差与相关矩阵

协方差含尺度，相关无量纲。有效协方差矩阵必须半正定，否则组合方差可为负；逐对估计和舍入可能破坏这一结构，需要整体估计或修正。

## 独立与不相关

独立推出零协方差（矩存在），反向一般不成立。非线性关系、共同波动或尾依赖可在 Pearson 相关为零时存在。rank correlation 更关注单调关系，但仍不完整描述联合尾部。

## Copula

Sklar 定理把联合分布写成边际 CDF 与 copula 的组合。Gaussian copula 没有非零尾依赖（相关小于 1），t-copula 可有对称尾依赖；Archimedean 家族可表现不对称。边际和 copula 都需估计。

## 压力与模型风险

危机相关上升可能来自共同波动和选择性观测。应比较多种 copula/动态相关、做尾部联合超越检验并设置相关破裂情景，而非把单一拟合当真联合分布。

## 最小自检

### 相关系数为 0 为什么仍可能同时极端下跌？

> [!answer]- 答案
> 线性平均共同变化可为零，但非线性或尾部依赖仍存在。
### Copula 分离了哪两个建模部分？

> [!answer]- 答案
> 各变量边际分布与它们的依赖结构；两部分都可能错设。
### 协方差矩阵为什么必须半正定？

> [!answer]- 答案
> 任意组合权重 w 的方差 $w^T\Sigma w$ 不能为负。

## 来源与核验

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
- Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
