---
aliases:
  - "DID 的处理前结果不能已被政策预期改变"
  - DID no-anticipation assumption
  - 双重差分无预期假设
  - Limited treatment anticipation
student_os: knowledge-atom
atom_id: ECON-DID-005
atom_type: assumption
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[双重差分法]]"
leads_to:
  - "[[2×2 DID 识别 ATT]]"
  - "[[事件研究系数语境]]"
---

# DID 的处理前结果不能已被政策预期改变

<!-- bilingual-en:start -->
*Pretreatment outcomes in DID must not already be changed by anticipation of the policy*
<!-- bilingual-en:end -->

> [!summary] 核心假设
> “处理前”必须真的是尚未受处理影响的时期。若个人、企业或地方政府在政策正式生效前已经因公告、预期或准备行为改变结果，最后一个日历上的 pre-period 就不能直接充当未处理基准。
>
> <!-- bilingual-en:start -->
> A “pre-period” must genuinely be unaffected by treatment. If people, firms, or local governments change outcomes before formal implementation because of announcements, expectations, or preparation, the last calendar pre-period cannot be used uncritically as an untreated baseline.
> <!-- bilingual-en:end -->

无预期的强形式要求：对在 $g$ 期开始处理的单位，所有 $t<g$ 都有 $Y_t(g)=Y_t(0)$。Callaway 与 Sant’Anna 的识别只需要相应的条件均值相等，即给定处理前特征 $X$ 和首次处理组后，$E[Y_t(g)\mid X,G=g]=E[Y_t(0)\mid X,G=g]$。若允许提前 $\delta$ 期反应，则只能把 $t<g-\delta$ 当作未受影响时期；识别时的基准期也要相应向前移。
<!-- bilingual-en:start -->
A strong no-anticipation condition requires $Y_t(g)=Y_t(0)$ for every $t<g$ among units first treated in period $g$. Callaway and Sant’Anna's identification results require only the corresponding conditional means to be equal: given pretreatment $X$ and treatment cohort, $E[Y_t(g)\mid X,G=g]=E[Y_t(0)\mid X,G=g]$. A limited-anticipation condition permits responses up to $\delta$ periods early. Only periods with $t<g-\delta$ are then unaffected, so the reference period used for identification must move further back.
<!-- bilingual-en:end -->

## 为什么它和“前趋势”不是一回事

处理前出现变化可能有两种完全不同的含义。一种是对照组本来就不能代表处理组的未处理趋势；另一种是政策消息已经产生了真实的提前效应。前者威胁平行趋势，后者改变处理开始时间。把二者都笼统叫作“pretrend”会让修正方向出错。
<!-- bilingual-en:start -->
A pretreatment movement can have two very different meanings. The comparison group may fail to represent the treated group's untreated trend, or the policy announcement may already have produced a genuine anticipatory effect. The first threatens parallel trends; the second changes when treatment effectively begins. Calling both simply a “pretrend” obscures the required remedy.
<!-- bilingual-en:end -->

例如，一项学费补贴在九月生效，但学校三月已收到确定通知并从四月开始扩招。若结果是招生人数，四月至八月不能再作为纯粹的处理前期。合理做法是根据制度事实界定预期窗口，而不是看到事件研究图后任意删除不方便的 lead。
<!-- bilingual-en:start -->
Suppose a tuition subsidy takes effect in September, but schools receive a definitive notice in March and begin expanding admissions in April. If enrolment is the outcome, April through August are not clean pretreatment periods. The anticipation window should be defined from institutional facts, not chosen after inspecting an event-study plot and discarding inconvenient leads.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 政策在 2027 年正式实施，但企业在 2026 年公告后已开始招聘。若结果是就业，为什么把 2026 年当最后一个未处理基准会有问题？
>
> **答案：** 因为公告已改变处理组的结果，2026 年不再观测 $Y(0)$。基准期需要移到预期反应之前，或显式建模预期窗口。

## 来源与核验

- Callaway & Sant’Anna (2021), [*Difference-in-Differences with Multiple Time Periods*](https://psantanna.com/files/Callaway_SantAnna_2020.pdf)，Assumption 3 与 Theorem 1：核验有限预期窗口 $\delta$、可用基准期和可识别 group-time ATT 的范围。
- Sun & Abraham (2021), [*Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects*](https://arxiv.org/abs/1804.05785)：交叉核验事件时间下预期效应与处理前系数的区分。
