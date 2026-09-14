---
aliases:
  - "Granger 表示定理在适当条件下连接协整与误差修正表示"
  - Granger representation theorem
  - 协整表示定理
student_os: knowledge-atom
atom_id: TS-CI-005
atom_set: cointegration-error-correction
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[协整秩与共同趋势]]"
related:
  - "[[ECM长短期结构]]"
  - "[[VAR到VECM重参数化]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# Granger 表示定理在适当条件下连接协整与误差修正表示
<!-- bilingual-en:start -->
*The Granger representation theorem links cointegration and error-correction representations under suitable conditions*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 对满足适当正则条件的 $I(1)$ 向量过程，协整、含约化秩长期项的误差修正表示，以及有限个共同随机趋势的表示，是同一长期结构的相容刻画。

这不是一句无条件的“只要残差看起来平稳就必有任意形式的 ECM”。经典结果需要排除爆炸根和退化情形，并要求过程具有合适的线性表示；VECM 的滞后结构、创新条件和确定性项也要与数据生成过程相容。定理告诉我们长期约束不能从短期动态中消失：若水平之间确有协整，纯差分 VAR 漏掉误差修正项；若没有协整，则不应凭经济故事硬塞一个平稳均衡误差。

定理连接的是统计表示，不替代经济识别。$\beta$ 给稳定组合，$\alpha$ 给哪些方程响应它；为什么存在这条关系、冲击是否结构性，仍是另一个问题。

> [!question]- 自检
> 已确认一个标准 $I(1)$ 系统协整，却只估计所有变量的一阶差分 VAR，主要遗漏了什么？
>
> **答案：** 上一期长期偏离对本期变化的误差修正通道。

## 来源与核验

- [Engle & Granger (1987)](https://doi.org/10.2307/1913236)：核对表示定理连接的移动平均、自回归与误差修正表示。
- [Johansen (1988)](https://doi.org/10.1016/0165-1889(88)90041-3)：核对高维约化秩误差修正框架。
