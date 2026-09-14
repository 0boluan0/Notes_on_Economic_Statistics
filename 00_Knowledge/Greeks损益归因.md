---
aliases:
  - "Greeks损益归因在统一头寸与现金流口径下比较实际价值变化和敏感度解释项并保留残差"
student_os: knowledge-atom
atom_id: FI-HEDGE-012
atom_type: method
status: source-checked
requires:
  - "[[多因子二阶损益近似]]"
  - "[[风险因子映射]]"
related:
  - "[[动态Delta对冲]]"
  - "[[风险模拟估值层]]"
  - "[[FRTB回测损益]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Greeks损益归因在统一头寸与现金流口径下比较实际价值变化和敏感度解释项并保留残差
<!-- bilingual-en:start -->
*Greek-based P&L attribution reconciles value changes with sensitivity contributions under common position and cash-flow conventions, retaining a residual*
<!-- bilingual-en:end -->

Greeks 损益归因是一种解释方法：用期初或另行约定的敏感度，把指定期间的组合价值变化分解为价格、波动率、时间等输入的贡献，并核对尚未解释的差额。它不是只把几个 Greek 相加，也不直接识别市场变化的经济原因。
<!-- bilingual-en:start -->
Greek-based P&L attribution explains a specified value change through price, volatility, time, and other input contributions, using initial or otherwise specified sensitivities, then reconciles the unexplained difference. It neither merely adds Greeks nor directly identifies the economic causes of market moves.
<!-- bilingual-en:end -->

先明确比较对象：固定期初持仓时，同一估值器下的价值变化为 $\Delta V_{full}=V(t_1,x_1)-V(t_0,x_0)$；若比较持有期损益，则另将期间合同现金流计入目标 $\Delta\Pi_{target}$。若要解释实际交易损益，新增／平仓交易、分红、融资、费用与估值调整也必须按同一币种和时点单独入账。不能把固定组合的 Taylor 解释额直接拿来覆盖一份交易后已改变持仓的账。
<!-- bilingual-en:start -->
First define the target. With initial holdings frozen, the same pricer gives the value change $\Delta V_{full}=V(t_1,x_1)-V(t_0,x_0)$. A holding-period P&L target $\Delta\Pi_{target}$ also includes contractual interim cash flows. Actual trading P&L requires separate entries for new and closed trades, dividends, funding, costs, and valuation adjustments in consistent currency and timing. A Taylor explanation for frozen holdings cannot directly cover a book whose positions changed through trading.
<!-- bilingual-en:end -->

然后依据 [[多因子二阶损益近似]] 计算已选择的一阶、二阶和时间贡献，并与同一损益口径下的现金账项目合为解释额 $A$，将不能对齐的差额明确记录。若目标仅为固定持仓的同刻价值变化，则 $\Delta\Pi_{target}=\Delta V_{full}$，不另加入现金项目：
<!-- bilingual-en:start -->
Next use [[多因子二阶损益近似|multi-factor approximation]] to compute selected first-order, second-order, and time contributions, combining them with cash-ledger entries on the same P&L basis into an explained amount $A$. Retain the reconciliation difference. For a same-time, frozen-position value-change target, $\Delta\Pi_{target}=\Delta V_{full}$ with no additional cash entries:
<!-- bilingual-en:end -->

$$\varepsilon=\Delta\Pi_{target}-A.$$

对于没有中间现金流的外币股票例子，完整变化为 500 美元，一阶解释 490，交叉项解释 10，余项为零。若只做一阶，10 就进入余项；这不证明模型有错，更不证明发生了一个新的风险因子。它首先反映了所选解释精度。
<!-- bilingual-en:start -->
In the foreign-equity example without interim cash flows, the full change is USD 500, first-order contributions are 490, the cross term is 10, and the residual is zero. Under first-order attribution alone, the 10 remains unexplained. That does not prove a model defect or a new risk factor; it initially reflects the chosen approximation order.
<!-- bilingual-en:end -->

分项归属也依赖方法。同一例子若依次重估，先改股价、再改汇率，分项是 240 与 260；反序则是汇率 250、股价 250。总额相同，交互项被分给后改变的因子。将交叉项单列可以明确这个相互作用，但不会把归因变成唯一的因果分解。
<!-- bilingual-en:start -->
Component allocation depends on the method. Repricing stock first and FX second attributes 240 and 260; reversing the order attributes 250 to FX and 250 to stock. The same total assigns the interaction to whichever factor changes second. Reporting it separately makes the interaction explicit without creating a unique causal decomposition.
<!-- bilingual-en:end -->

持续出现有结构的残差值得检查遗漏因子、报价时间错配、现金流漏记、模型或实现不一致、高阶项和行权边界，但残差本身不能唯一定位原因。完整重估也只是给定估值规则的结果。FRTB 用 HPL 与 RTPL 做的监管损益归因检验有自己的定义与用途，见 [[FRTB回测损益]]，不能与本卡的 Greek 解释表互换。
<!-- bilingual-en:start -->
Structured residuals warrant checking missing factors, asynchronous quotes, omitted cash flows, inconsistent models or implementations, higher-order terms, and exercise boundaries. The residual alone does not identify a unique cause, and full repricing is still conditional on valuation rules. FRTB's regulatory HPL–RTPL attribution test has separate definitions and purposes in [[FRTB回测损益|FRTB P&L conventions]]; it is not interchangeable with a Greek contribution table.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Carr、Wu（2020），印刷第 2272、2276–2278 页／PDF 第 2、6–8 页](https://engineering.nyu.edu/sites/default/files/2020-09/option-profit-carr-jofi-12894.pdf#page=2)：已重开损益归因的用途、指定风险结构、时间与市场因子的导数分解，目视式 (2)。外币股票及顺序归因数字是独立代数对照，不是该文实证结论。
- [Haugh，PDF 第 18–19 页](https://www.columbia.edu/~mh2078/QRM/DerivativesReview.pdf#page=18)：已重开并目视股票／现金账户和期权共同形成对冲损益的口径，支持实际动态交易必须额外记录现金账。
- [[FRTB回测损益]]：已重读 HPL、APL、RTPL 的监管职责，仅用于区分本卡的解释方法与监管检验。
<!-- bilingual-en:start -->
- Carr and Wu, printed pp. 2272 and 2276–2278 / PDF pp. 2 and 6–8, were reopened for the purpose, risk representation, and derivative decomposition of attribution; equation (2) was visually checked. The FX and sequential examples are independent algebra, not empirical findings from that paper.
- Haugh pp. 18–19 were reopened and visually checked for combined option, stock, and cash-account P&L, supporting explicit cash accounting for dynamic trades.
- [[FRTB回测损益|FRTB P&L conventions]] was reread only to distinguish the regulatory test from this explanatory method.
<!-- bilingual-en:end -->
