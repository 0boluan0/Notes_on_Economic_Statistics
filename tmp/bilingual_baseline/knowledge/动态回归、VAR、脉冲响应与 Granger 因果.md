---
aliases:
  - "Dynamic Regression and VAR"
  - "VAR Model"
  - "Impulse Response"
  - "Granger Causality"
status: source-checked
---

# 动态回归、VAR、脉冲响应与 Granger 因果

> [!summary] 快速恢复
> **它解决什么：** 描述多个时间序列彼此滞后影响，并把一次冲击如何在系统中传播分解出来。
> **具体锚点：** 利率、通胀和产出相互影响；单方程很难预先指定谁完全外生，VAR 让每个变量都由系统过去共同解释。
> **核心难点：** reduced-form 残差通常同期相关，未经识别的“冲击”没有结构经济含义；Granger 因果只是增量预测关系。
> **为什么重要：** 它连接政策动态、预测、干预分析和结构识别。
> **继续：** 先确保平稳/协整处理，再解释 IRF；长期关系见 [[协整与误差修正模型]]。

> [!source] 本节依据
> - [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
> - Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。

## 动态回归与干预

动态回归可包含解释变量的当前/滞后项、因变量滞后和 ARMA 误差。加入因变量滞后会改变系数为短期效应，长期效应需汇总动态乘数。干预变量要区分脉冲、阶跃和渐进影响，并处理同期反向因果。

## VAR

VAR(p) 写作 $y_t=c+A_1y_{t-1}+\cdots+A_py_{t-p}+u_t$。每个方程可用 OLS 估计，但系统稳定性由 companion matrix 特征值决定。阶数选择结合信息准则、残差和经济用途，参数随维度和阶数快速增长。

## 脉冲响应

MA 表示把当前和过去创新映射到未来 y。若 reduced-form 创新同期相关，要通过 Cholesky 排序、短期/长期限制、符号限制或外部工具识别结构冲击。不同识别产生不同 IRF，必须报告假设。

## 预测误差方差分解

FEVD 把 h 步预测误差方差归因于各结构冲击，依赖同一识别和尺度。它不是现实世界因果贡献的无条件真值。

## Granger 因果

若在控制系统过去后，x 的滞后提高 y 的预测，称 x Granger-causes y。检验是对 x 滞后系数的联合限制。它不排除遗漏共同驱动、同期因果或纯信息领先，因此不能简写为结构因果。

## 非平稳与协整

对有单位根且协整的变量直接差分 VAR 会丢失长期关系，水平 VAR 的常规推断又可能失效；VECM 显式结合误差修正与短期动态。

## 最小自检

### 为什么 reduced-form VAR 残差不能直接叫经济结构冲击？

> [!answer]- 答案
> 不同方程残差可同期相关，只表示无法由过去预测的组合；需要额外识别限制分解为有含义的正交冲击。
### Granger 因果能否证明政策变量造成结果变化？

> [!answer]- 答案
> 不能。它只说明滞后信息提高预测，仍可能由遗漏变量、预期或共同冲击产生。
### IRF 为什么要报告变量排序或识别方法？

> [!answer]- 答案
> 冲击正交化不是数据唯一决定的；不同限制会改变冲击定义和传播路径。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
- Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
