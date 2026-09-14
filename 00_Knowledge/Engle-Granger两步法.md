---
aliases:
  - "Engle-Granger 两步法先估计长期残差再检验其单位根"
  - Engle-Granger two-step procedure
  - EG 两步法
student_os: knowledge-atom
atom_id: TS-CI-012
atom_set: cointegration-error-correction
atom_type: procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[协整]]"
related:
  - "[[EG残差检验]]"
  - "[[EG多变量局限]]"
  - "[[ECM长短期结构]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# Engle-Granger 两步法先估计长期残差再检验其单位根
<!-- bilingual-en:start -->
*The Engle-Granger two-step procedure estimates a long-run residual before testing it for a unit root*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Engle–Granger 程序先用水平回归估计候选协整向量，再检验估计残差是否为 $I(0)$；只有残差平稳，才把它作为误差修正项进入第二阶段动态模型。

一个可审计的顺序是：

1. 先检查各变量的整合阶数、样本区间和确定性项，避免把 $I(0)$、$I(1)$、$I(2)$ 混成同一个问题；
2. 依据理论选择归一化并估计长期方程，保存 $\hat e_t$；
3. 对 $\hat e_t$ 做残差型单位根检验，使用与估计协整回归相匹配的规格和临界值；
4. 若拒绝无协整，再估计含 $\hat e_{t-1}$ 与充分差分滞后的 ECM，检查调整方向、残差与参数稳定性。

“两步”在应用文献中有两种数法，必须说清。“残差型两步检验”常指长期回归加残差单位根检验；Engle–Granger 原文的“两步估计量”则指先估计协整向量，再把它作为误差修正项估计 ECM。残差检验决定第二种做法是否有协整基础，但它不是 ECM 估计本身。

第三步不是“普通 ADF 再做一次”，残差检验与第四步的 ECM 估计也不能互相替代。残差平稳支持一条统计长期关系；ECM 才描述短期动态及谁承担调整。

> [!question]- 自检
> 长期水平回归的 $R^2$ 很高，能否跳过残差单位根检验直接建 ECM？
>
> **答案：** 不能。高拟合可能来自伪回归；必须证明估计残差的整合阶数下降。

## 来源与核验

- [Engle & Granger (1987)](https://doi.org/10.2307/1913236)：核对两步估计与检验程序。
- [[02_Economy/01_Econometrics/12_非平稳时间序列.md]]：对照课程的应用顺序。
