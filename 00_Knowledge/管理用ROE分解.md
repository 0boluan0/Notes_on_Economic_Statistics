---
aliases:
  - 同口径ROE等于净经营资产净利率加经营融资利差乘净财务杠杆
  - Operating-financing decomposition of ROE
student_os: knowledge-atom
atom_id: FI-FS-066
atom_type: proposition
status: source-checked
part_of:
  - "[[财务报表、财务比率与财务预测.canvas]]"
---

# 同口径ROE等于净经营资产净利率加经营融资利差乘净财务杠杆
<!-- bilingual-en:start -->
On a consistent basis, ROE equals RNOA plus the operating-financing spread times net financial leverage.
<!-- bilingual-en:end -->

把[[管理用利润表拆分]]的 $NI=NOPAT-NFE$ 与[[管理用资产负债表恒等式]]的 $NOA=NFD+E$ 联立，可以区分经营本身的回报与融资结构对股东回报的影响。这里 $NFE$ 为税后净融资费用，所有利润与资本范围一致，存量也采用同一时点或同一种平均。
<!-- bilingual-en:start -->
Combine the [[管理用利润表拆分|income identity]] $NI=NOPAT-NFE$ with the [[管理用资产负债表恒等式|balance-sheet identity]] $NOA=NFD+E$ to separate operating return from financing's contribution to equity return. Here NFE is after-tax net financing expense, with consistent income, capital scope, and stock timing.
<!-- bilingual-en:end -->

在 $E,NOA,NFD$ 均非零时，令 $RNOA=NOPAT/NOA$，净借款成本率 $NBC=NFE/NFD$，[[净财务杠杆]] $NFL=NFD/E$，则：
<!-- bilingual-en:start -->
For nonzero equity, NOA, and NFD, define $RNOA=NOPAT/NOA$, net borrowing cost $NBC=NFE/NFD$, and [[净财务杠杆|net financial leverage]] $NFL=NFD/E$. Then:
<!-- bilingual-en:end -->

$$\begin{aligned}ROE&=\frac{NOPAT-NFE}{E}\\&=RNOA\left(1+\frac{NFD}{E}\right)-NBC\frac{NFD}{E}\\&=RNOA+(RNOA-NBC)NFL.\end{aligned}$$

ABC 的 $RNOA=206.72/1722\approx12.004646\%$，$NBC=70.72/762\approx9.280840\%$，$NFL=762/960=0.79375$。利差约 2.723806 个百分点，乘杠杆贡献约 2.162021 个百分点，合计 ROE 14.166667%。
<!-- bilingual-en:start -->
ABC has RNOA of approximately 12.004646%, NBC of 9.280840%, and NFL of 0.79375. The spread of 2.723806 percentage points contributes 2.162021 percentage points through leverage, producing ROE of 14.166667%.
<!-- bilingual-en:end -->

在正权益、正净负债且其余条件不变时，正利差对应正杠杆贡献，负利差则使 ROE 低于 RNOA，但不必已经出现净亏损；这不是提高借款一定增值的政策结论，见[[ROE与杠杆的解释边界]]。若 $NFD=0$，不要计算 $NBC$，直接用 $ROE=(NOPAT-NFE)/E$；净负债为零也不保证当期净融资费用为零。净金融资产头寸则须按投资收益解释，不能照搬“借款利率”的措辞。
<!-- bilingual-en:start -->
With positive equity and net debt, holding other conditions fixed, a positive spread contributes positively through leverage; a negative spread puts ROE below RNOA without necessarily producing a net loss. This is not a recommendation to borrow more; see [[ROE与杠杆的解释边界|the ROE boundary]]. When NFD is zero, NBC is undefined: use $(NOPAT-NFE)/E$ directly. Zero net debt does not guarantee zero period NFE. A net financial asset position requires an investment-income interpretation rather than an unqualified borrowing-rate label.
<!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/05_财务管理/2023年注册会计师全国统一考试辅导教材---财务成本管理 (中国注册会计师协会) (Z-Library).pdf#page=67|CPA2023，印刷59–60页（PDF67–68），表2-13和图2-2]]：目视核对公式与层次；按未提前舍入的数值重算；非零条件、净资产头寸和替代原式明确补足。
<!-- bilingual-en:start -->
CPA's formula and hierarchy were visually checked and recalculated without early rounding. Nonzero conditions, net asset positions, and the undivided alternative are explicit.
<!-- bilingual-en:end -->
