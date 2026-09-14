---
aliases:
  - "当前市场风险框架以 FRTB 的压力 ES、流动性期限和 NMRF 为核心"
  - FRTB stressed expected shortfall liquidity horizons and NMRF
  - 当前 Basel 市场风险框架
student_os: knowledge-atom
atom_id: MB-BAS-014
atom_set: basel-capital-liquidity-regulation
atom_type: regulatory-method
status: source-checked
mastery_state: unassessed
requires:
  - "[[交易簿与银行簿边界]]"
  - "[[ES定义]]"
related:
  - "[[FRTB 交易台资格]]"
  - "[[压力测试方法]]"
  - "[[VaR-ES联合识别]]"
  - "[[压力VaR口径边界]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# 当前市场风险框架以 FRTB 的压力 ES、流动性期限和 NMRF 为核心
*The current market-risk framework centres on FRTB stressed ES, liquidity horizons and non-modellable risk factors*

> [!summary] 从旧 VaR 到更完整的尾部与流动性处理
> 当前 Basel 市场风险框架（FRTB）同时提供标准法和经批准的内部模型法。内部模型资本以压力期、97.5% 单尾 expected shortfall 为核心，并按风险因子的流动性期限调整；缺乏足够真实价格观测的非可模型化风险因子（NMRF）另做压力情景资本计量。

ES 平均固定概率质量的最坏尾部损失；只有当分位点累计概率恰好满足 $F_L(q_\alpha)=\alpha$ 时，才可简写为严格超过 VaR 后的平均损失。它比只报告分位点的 VaR 更直接反映尾部严重度。但监管 ES 不是把所有头寸统一假定可在十天内退出：风险因子被分配不同流动性期限，资本要反映较慢退出或对冲的风险。可模型化资格也不是“模型能拟合”即可，而取决于框架规定的真实价格观测与频率。

旧课程中的 99%、10 日 VaR 与乘数仍是理解 Basel 2.5 或历史 IMA 的材料；把它不加日期写成当前 FRTB 全部市场风险资本公式是错误的。当前标准法的 sensitivities-based method、default risk charge 与 residual risk add-on 也不能被 ES 一项替代。

## 边界

- FRTB 不是“用 ES 完全替代所有市场风险组件”；标准法与 IMA 都包含多个模块。
- NMRF 表示监管可模型化标准不足，不等于该因子经济上不可建模或一定更危险；其压力情景资本计量是 FRTB 的专门监管方法，不等于一般的全行压力测试。
- 实施日期和本地修改随法域变化。

> [!question]- 自检
> 两个头寸有相同十日波动率，为什么 FRTB 资本仍可能不同？
>
> **答案：** 它们的尾部、压力期表现、风险因子流动性期限和可模型化资格可能不同。

## 来源与核验

- [Basel Committee, Minimum capital requirements for market risk](https://www.bis.org/bcbs/publ/d457.htm)：核对 FRTB 标准法、IMA、压力 ES 与流动性期限架构。
- [Basel Framework, MAR33](https://www.bis.org/basel_framework/chapter/MAR/33.htm)：核对 ES、流动性期限和 NMRF 计量。
- [[VaR定义]] 与 [[ES定义]]：复用统计定义，不把监管参数当一般统计定义。
- 口径核验日：2026-08-29。
