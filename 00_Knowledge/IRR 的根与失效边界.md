---
aliases:
  - "IRR 是 NPV 方程的实数根而非常规现金流可有多根或无可用根"
  - "IRR roots and multiple IRRs"
  - "Internal rate of return boundary"
  - "内含报酬率的根与多根问题"
student_os: knowledge-atom
atom_id: CORP-CB-004
atom_set: capital-budgeting-investment-decisions
atom_type: model-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[净现值决策]]"
related:
  - "[[互斥项目的增量 NPV]]"
part_of:
  - "[[资本预算与投资决策.canvas]]"
---

# IRR 是 NPV 方程的实数根而非常规现金流可有多根或无可用根
<!-- bilingual-en:start -->
*IRR is a real root of the NPV equation, and non-conventional cash flows can have multiple roots or no usable root*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 内含报酬率（IRR）是满足 $NPV(r)=0$ 的贴现率，不是项目自动携带的唯一“真实回报率”。若 $C_0<0$、其后现金流都非负且至少一期为正，NPV 在 $r>-1$ 上严格下降：$NPV(0)>0$ 时恰有一个正根，等于零时唯一非负根是零，小于零时没有非负根。符号反复变化会破坏这个结构。
>
> <!-- bilingual-en:start -->
> IRR is a discount rate satisfying $NPV(r)=0$, not an automatically unique “true return.” If $C_0<0$ and all later cash flows are non-negative with at least one strictly positive, NPV is strictly decreasing for $r>-1$: $NPV(0)>0$ gives exactly one positive root, equality makes zero the only non-negative root, and $NPV(0)<0$ gives no non-negative root. Repeated sign changes can produce several roots or none.
> <!-- bilingual-en:end -->

现金流 $(-100,230,-132)$ 的方程

$$
-100+\frac{230}{1+r}-\frac{132}{(1+r)^2}=0
$$

同时有 $r=10\%$ 与 $r=20\%$ 两个根；仅说“IRR 高于资本成本就接受”无法确定在哪一段 NPV 为正。现金流 $(-100,50,-100)$ 则在 $r>-1$ 没有实根。贷款型现金流先流入后流出时，IRR 与门槛率的比较方向还可能反转。

> [!question]- 自检
> Excel 返回一个 IRR 数字，为什么还不能立即用它做决策？
>
> **答案：** 先检查现金流符号次数、搜索区间和是否还有其他根，再把 NPV 直接画成贴现率的函数；软件返回一个数不证明根唯一或比较方向正确。

## 来源与核验

- [MIT 15.401 Capital Budgeting, slides 25–31](https://ocw.mit.edu/courses/15-401-finance-theory-i-fall-2008/06b2ce70ad26ae3d3be8d94145495be6_MIT15_401F08_lec18.pdf)：核对 IRR 定义、贷款型现金流、无根、多根及规模/时序排序错误。
- [[02_Economy/05_财务管理/2023年注册会计师全国统一考试辅导教材---财务成本管理 (中国注册会计师协会) (Z-Library).pdf#page=147|CPA《财务成本管理》第五章第二节，第 139–141 页]]：核对 IRR 作为使 NPV 为零的折现率及其独立项目比较口径。
