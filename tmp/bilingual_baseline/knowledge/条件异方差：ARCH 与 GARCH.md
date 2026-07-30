---
aliases:
  - "ARCH and GARCH"
  - "Volatility Models"
  - "条件异方差模型"
status: source-checked
---

# 条件异方差：ARCH 与 GARCH

> [!summary] 快速恢复
> **它解决什么：** 当收益均值难预测但大波动成群出现时，直接建模条件方差如何随过去信息变化。
> **具体锚点：** 金融收益可能几乎无自相关，但平方收益显著相关；这说明方向不可预测不等于风险恒定。
> **核心难点：** 条件方差必须非负且参数要满足稳定条件；标准化残差而非原残差用于检验剩余波动结构。
> **为什么重要：** 风险预测、VaR、期权与资产配置依赖随时间变化的波动率。
> **继续：** 先建好均值方程，再做 ARCH-LM；风险度量见 [[波动率、相关性与 Copula]] 和 [[VaR、ES、回测与压力测试]]。

> [!source] 本节依据
> - [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
> - Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。

## 条件方差与波动聚集

写 $y_t=\mu_t+\varepsilon_t$，$\varepsilon_t=\sigma_t z_t$，其中 $E(z_t\mid\mathcal F_{t-1})=0$、$Var(z_t\mid\mathcal F_{t-1})=1$。$\sigma_t^2$ 是已知过去后的风险，不是无条件样本方差。波动聚集表现为 $|\varepsilon_t|$ 或 $\varepsilon_t^2$ 的持续性。

## ARCH 与 GARCH

ARCH(q)：$\sigma_t^2=omega+\sum_{i=1}^q\alpha_i\varepsilon_{t-i}^2$。GARCH(1,1) 再加 $\beta\sigma_{t-1}^2$，用较少参数捕捉持久性。常用充分约束为 $\omega>0,\alpha_i,\beta_j\ge0$；GARCH(1,1) 有有限无条件方差通常需 $\alpha+\beta<1$。

## 估计与创新分布

常用极大似然或准 MLE。金融收益厚尾时，Gaussian likelihood 的方差动态仍可作 QMLE，但区间和尾部风险常需 Student-t 等分布并做稳健推断。分布选择不应替代残差诊断。

## ARCH-LM 与标准化残差

先估计均值方程，检验残差平方对其滞后是否有解释力。拟合后检查 $z_t=\epsilon_t/\hat\sigma_t$ 及 $z_t^2$ 的相关、尾部和偏态；只有原残差去相关而平方仍相关，说明波动模型不足。

## 多步波动预测

GARCH 预测递归地向长期方差回归；持久度越高，衰减越慢。接近 IGARCH 时长期方差估计极敏感，结构突变也可能伪装成高持久性。

## 扩展与边界

EGARCH、GJR-GARCH 可表示正负冲击不对称。模型描述条件二阶矩，不等于解释波动的经济原因。

## 最小自检

### 收益无自相关，为什么仍可能需要 GARCH？

> [!answer]- 答案
> 均值可近似不可预测，但平方或绝对收益可能相关，说明条件方差随时间变化。
### GARCH(1,1) 中 $\alpha+\beta$ 接近 1 表示什么？

> [!answer]- 答案
> 波动冲击衰减很慢、条件方差高度持久，长期方差估计也更敏感。
### 拟合后为什么检查标准化残差平方？

> [!answer]- 答案
> 它剔除了模型预测的时变尺度；若仍有相关，说明条件方差动态没有被充分吸收。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
- Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
