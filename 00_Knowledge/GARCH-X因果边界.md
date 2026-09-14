---
aliases:
  - "方差方程外生变量只能描述条件二阶矩而不能单独识别因果"
  - GARCH-X causal boundary
  - Exogenous variables in variance equations
  - 事件哑变量与条件方差
student_os: knowledge-atom
atom_id: TS-VOL-024
atom_set: conditional-volatility
atom_type: causal-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH(p,q)模型]]"
related:
  - "[[ARCH-M风险溢价边界]]"
  - "[[方差断点伪GARCH持久性]]"
  - "[[概率依赖与因果]]"
  - "[[因果研究设计检查]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# 方差方程外生变量只能描述条件二阶矩而不能单独识别因果
<!-- bilingual-en:start -->
*Exogenous variables in a variance equation describe conditional second moments but do not identify causality by themselves*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 可扩展方差方程为
> $$h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1}+\gamma x_t.$$
> $\gamma$ 描述在所给信息集、均值与方差规格下，$x_t$ 与条件方差水平的关联。即使显著，也不能单独证明 $x_t$ 导致了波动变化。

若 $h_t=\operatorname{Var}(\varepsilon_t\mid\mathcal F_{t-1})$，右侧的 $x_t$ 必须在 $t-1$ 已知，或明确把条件信息集扩展到包含它。事后才知道的同期变量不能偷偷进入真实的一步预测；突发事件当日 dummy 可以用于事后关联描述，但不能声称它在事件发生前就可预测当日方差。滞后变量、预定日历变量和事后事件编码的时间含义应分别报告。

事件 dummy 尤其需要明确编码：单日 pulse、事件后永久 step、窗口或逐日 effects 回答不同问题。以 9·11 为例，市场停市、同期宏观新闻、行业构成、预期反应和其他断点都可能与 dummy 重合；一个“事件后等于一”的 step 还把所有后来制度变化归到同一系数。

因果解释需要可辩护的反事实与识别设计，例如对照市场/资产、事件窗口、预趋势与同期冲击分析，而不是只靠 GARCH-X。线性加法 $\gamma x_t$ 还要连同 $x_t$ 的支持集保证所有允许历史下 $h_t>0$；必要时使用 log link 或受约束的函数形式。

> [!question]- 自检
> 事件后 dummy 的 $\hat\gamma>0$ 且 p 值很小，最强可以直接说什么？
>
> **答案：** 在当前模型与样本中，事件后时期和更高条件方差存在显著关联；除非另有识别设计，不能说事件本身造成了该上升。

## 来源与核验

- [Engle (1982)](https://doi.org/10.2307/1912773)：核对条件方差相对于既定过去信息集定义，支持方差回归量的信息时序边界。
- [Engle & Ng (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x)：核对可预测波动模型与遗漏 news-impact 结构的诊断边界。
- [Lamoureux & Lastrapes (1990)](https://doi.org/10.1080/07350015.1990.10509794)：核对遗漏方差结构变化如何污染持久性解释。
- [[01_Math/06_时间序列分析/lecture.pdf#page=177|课程讲义 p. 177]]：课程以 9·11 step dummy 作例；此卡保留统计规格并纠正其因果越界。
