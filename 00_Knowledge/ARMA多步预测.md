---
aliases:
  - "ARMA 多步预测将未来创新置零但二阶白噪声本身只保证最佳线性预测"
  - Recursive ARMA forecasting
  - Multi-step ARMA forecast
  - ARMA forecast function
  - ARMA 递归预测
student_os: knowledge-atom
atom_id: TS-ARMA-021
atom_set: arma-modeling
atom_type: forecasting-rule
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARMA(p,q)模型]]"
  - "[[创新]]"
  - "[[白噪声二阶定义]]"
  - "[[ARMA无限AR表示]]"
related:
  - "[[AR(1)多步预测]]"
  - "[[ARMA预测区间]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# ARMA 多步预测将未来创新置零但二阶白噪声本身只保证最佳线性预测
<!-- bilingual-en:start -->
*Multi-step ARMA forecasting sets future innovations to zero, but second-order white noise alone guarantees only the best linear predictor*
<!-- bilingual-en:end -->

> [!summary] 原子规则
> 把 ARMA 方程向前递推时：
> - 已观测的 $y_t,y_{t-1},\ldots$ 原样使用；
> - 当期及过去不可见创新用 fitted innovations 估计；
> - 所有日期晚于 $t$ 的创新在线性投影中置为 0。
> 若 $\{\varepsilon_t\}$ 相对 $\mathcal F_{t-1}$ 是鞅差序列（MDS），即 $E(\varepsilon_t\mid\mathcal F_{t-1})=0$，这套递推才给出平方损失下的条件均值 $E(y_{t+h}\mid\mathcal F_t)$。若只有二阶白噪声，它保证正交性，因而给出基于过去观测的最佳**线性**预测；它未必等于完整条件期望。
> <!-- bilingual-en:start -->
> Recursive ARMA forecasting uses observed past values, fitted past innovations, and zero linear projections for innovations dated after the forecast origin. If the innovations are an MDS relative to the forecasting information, those future innovations also have zero conditional mean, so the recursion gives $E(y_{t+h}\mid\mathcal F_t)$ and is optimal under squared loss. Second-order white noise alone gives orthogonality and hence the best linear predictor; it need not give the full conditional expectation.
> <!-- bilingual-en:end -->

一步预测可直接含最近的 MA fitted innovation；两步以后，新的未来创新还没发生，不能填入事后真实值。以
$$y_t=c+\phi_1y_{t-1}+\phi_2y_{t-2}+\varepsilon_t+\theta\varepsilon_{t-1}$$
为例，在时点 $t$ 用 fitted $\hat\varepsilon_t$ 得
$$\hat y_{t+1|t}=c+\phi_1y_t+\phi_2y_{t-1}+\theta\hat\varepsilon_t,$$
$$\hat y_{t+2|t}=c(1+\phi_1)+(\phi_1^2+\phi_2)y_t+\phi_1\phi_2y_{t-1}+\phi_1\theta\hat\varepsilon_t.$$
第二式把 $\hat y_{t+1|t}$ 代回 AR 部分，并把 $\varepsilon_{t+1},\varepsilon_{t+2}$ 的线性预测置零；它没有使用尚未发生的未来创新。更一般地，MA 的近期冲击可通过预测状态间接影响更长 horizon。

这条规则默认参数与模型形式给定。实际使用估计参数和估计 residual 时，点预测仍可按同一递推计算，但其采样不确定性不会自动进入简单的已知参数公式。
<!-- bilingual-en:start -->
For ARMA(2,1), substituting the one-step forecast into the AR recursion gives $\hat y_{t+2|t}=c(1+\phi_1)+(\phi_1^2+\phi_2)y_t+\phi_1\phi_2y_{t-1}+\phi_1\theta\hat\varepsilon_t$. Future realised innovations must never be inserted. This is a known-parameter recursion using a fitted past innovation; estimation uncertainty needs separate treatment.
<!-- bilingual-en:end -->

> [!question]- 自检
> 只知道 $\varepsilon_t$ 是二阶白噪声，能否把置零递推无条件地称为 $E(y_{t+h}\mid\mathcal F_t)$？
>
> **答案：** 不能。二阶白噪声只排除线性相关，因此支持最佳线性预测。还需 MDS、独立或 joint Gaussian 等能推出未来创新条件均值为 0 的结构，才能称为完整条件期望。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=124|课程讲义 pp. 124–127]]：核对 AR(1) 与 ARMA($2,1$) 的逐步条件期望递推。
- [Hyndman & Athanasopoulos, FPP3 Chapter 9](https://otexts.com/fpp3/arima.html)：核对 ARIMA/ARMA forecast 的递归条件均值解释。
- [[白噪声二阶定义]]：复用“white noise 不自动推出 MDS”的边界。
