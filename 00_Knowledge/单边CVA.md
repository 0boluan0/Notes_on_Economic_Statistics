---
aliases:
  - "单边 CVA 是在本方视为无违约风险时，对手方违约造成的未覆盖正敞口损失之风险中性贴现期望；逐期 EE、边际违约概率与 LGD 的乘积只在适当分解条件下成立"
  - "Unilateral CVA as discounted counterparty-default loss"
student_os: knowledge-atom
atom_id: RM-CCR-003
atom_set: counterparty-credit-risk-and-valuation
atom_type: valuation-equation
status: source-checked
mastery_state: unassessed
requires:
  - "[[对手方敞口指标]]"
  - "[[净额与抵押品]]"
  - "[[违约强度模型]]"
related:
  - "[[违约损失口径]]"
  - "[[信用损失三参数]]"
  - "[[三类 CVA 口径]]"
leads_to:
  - "[[DVA与双边估值]]"
  - "[[错向风险]]"
part_of:
  - "[[对手方信用风险与估值调整.canvas|对手方信用风险与估值调整]]"
---

# 单边 CVA 是在本方视为无违约风险时，对手方违约造成的未覆盖正敞口损失之风险中性贴现期望；逐期 EE、边际违约概率与 LGD 的乘积只在适当分解条件下成立
<!-- bilingual-en:start -->
*Unilateral CVA is the risk-neutral discounted expectation of uncovered positive-exposure loss at counterparty default when the valuing party is treated as default-free; the bucketed EE-times-marginal-PD-times-LGD formula requires an appropriate factorisation*
<!-- bilingual-en:end -->

> [!summary] 先对“违约时究竟损失多少”联合取期望，再讨论能否拆成三个数
> 单边 CVA（unilateral CVA，UCVA）只让交易对手可能违约，把估值方本身视为无违约风险。它定价的是：交易对手在合约到期前违约时，依法可净额、可用抵押品和 close-out 后仍未覆盖的正敞口，有多少不能回收。风险中性贴现损失的联合期望是主式；逐期的贴现 EE、边际违约概率和 LGD 乘积，是在暴露与违约可适当分解、时间离散和回收口径已经固定之后的近似。

## 主式是违约事件上的联合期望

令 $B$ 为估值银行，$C$ 为交易对手，$\tau_C$ 为 $C$ 的违约时刻。令 $X_C(t)\ge 0$ 表示从银行视角、在 $C$ 于 $t$ 违约时按统一 close-out 规则得到的未覆盖正敞口；它应已经按 [[净额与抵押品|可执行净额集合、抵押品及保证金风险期]]处理，而不是逐笔交易正市值之和。令 $D(0,t)$ 为估值日到 $t$ 的贴现因子，$LGD_C(t)$ 为该 close-out 债权适用的损失率。单边 CVA 的基本形式为

$$
\boxed{
UCVA_0
=\mathbb E^{\mathbb Q}\!\left[
D(0,\tau_C)LGD_C(\tau_C)X_C(\tau_C)
\mathbf 1_{\{0<\tau_C\le T\}}
\right] }.
$$

这里的 $\mathbb Q$ 是与基础交易估值一致的风险中性定价测度。指标函数只在对手方于期限内违约的路径上取 1；$X_C(\tau_C)$ 又取决于同一路径上的市场状态、净额、抵押品和 close-out。因此主式天然保留了 [[错向风险|暴露与违约的依赖]]，也允许贴现、回收和暴露彼此相关。

若用违约计数过程 $H_C(t)=\mathbf 1_{\{\tau_C\le t\}}$，同一结构也可写成

$$
UCVA_0
=\mathbb E^{\mathbb Q}\!\left[
\int_0^T D(0,t)LGD_C(t)X_C(t)\,dH_C(t)
\right].
$$

$dH_C(t)$ 只在违约时跳一次；它不是把每天的 EE 连续相加。

## 逐期乘积何时成立

把期限分成 $I_i=(t_{i-1},t_i]$，定义风险中性**边际违约概率**

$$
q_i^{\mathbb Q}
=\mathbb Q(\tau_C\in I_i)
=S^{\mathbb Q}(t_{i-1})-S^{\mathbb Q}(t_i).
$$

若 [[违约强度模型|风险中性强度]] $\lambda_C^{\mathbb Q}(t)$ 为确定函数，则

$$
S^{\mathbb Q}(t)
=\exp\!\left(-\int_0^t\lambda_C^{\mathbb Q}(u)\,du\right).
$$

将联合期望按区间分组，可先写成不会假装独立的形式：

$$
UCVA_0
=\sum_i q_i^{\mathbb Q}
\mathbb E^{\mathbb Q}\!\left[
D(0,\tau_C)LGD_C(\tau_C)X_C(\tau_C)
\mid \tau_C\in I_i
\right].
$$

只有进一步采用期中点 $t_i^*$ 代表区间、$LGD_C$ 近似确定，并让违约事件与贴现暴露可分解时，才得到常见离散式

$$
\boxed{
UCVA_0
\approx LGD_C\sum_i q_i^{\mathbb Q}v_i,
\qquad
v_i=\mathbb E^{\mathbb Q}\!\left[D(0,t_i^*)X_C(t_i^*)\right] }.
$$

若 $D(0,t_i^*)$ 是确定的，才可再写成

$$
v_i=DF_i\,EE_i^{\mathbb Q},
\qquad
EE_i^{\mathbb Q}=\mathbb E^{\mathbb Q}[X_C(t_i^*)].
$$

这里的 $EE_i$ 是**某一未来时点**的平均正敞口，不是跨时间平均的 EPE，也不是高分位 PFE。若利率随机且与暴露相关，$\mathbb E[DX]$ 不能机械拆成 $\mathbb E[D]\mathbb E[X]$；若暴露与违约相关，则应保留违约条件暴露，而不能使用无条件 EE 乘边际违约概率。

## 恢复率必须与估值债权相匹配

在最简单的 recovery-of-claim 口径下可写 $LGD_C=1-R_C$，但 $R_C$ 必须说明是哪个 close-out 债权、哪个估值时点和哪种市场口径的回收。用于定价的市场共识预期损失率、CDS 合同结算损失率与 [[违约损失口径|Basel IRB 经济 LGD]] 服务不同目标，不能只因都写成 $1-R$ 就互换。

若抵押品已经从 $X_C$ 中扣除，同一抵押品的风险缓释不能又无条件放进较低 LGD；否则会重复计算。反过来，抵押品价值波动、法律不可执行、争议、haircut 与 close-out 延迟没有进入 $X_C$ 时，也不能靠一个静态回收率掩盖这些遗漏。

## 从市场路径到 CVA 的最小流程

1. 在与交易定价一致的 $\mathbb Q$ 测度下模拟利率、汇率、商品、股票或其他市场因子，并在未来网格重估交易。
2. 在每条路径、每个时点先按法律净额集合聚合市值，再应用抵押品、阈值、最低转移额与 margin period of risk，得到 $X_C(t)$。
3. 用联合违约—市场模型直接估计主式；只有在有依据时才改用无条件 EE 与边际违约概率的分解。
4. 使用与 close-out、币种和抵押协议一致的贴现与回收口径，最后跨路径求均值。
5. 分别检查网格、尾部路径、净额、抵押品和信用曲线的敏感度；路径生成可复用，不代表新增交易的 CVA 等于其单笔 CVA，因为净额后的正部是非线性的。

## 一个区分“联合期望”与“独立乘积”的算例

考虑一个单期近似，$DF=0.98$，$LGD=60\%$，该期边际风险中性违约概率为 $2\%$，无条件 EE 为 10。若暴露与违约可分解，

$$
UCVA_{\mathrm{ind}}
=0.98\times10\times0.02\times0.60
=0.1176.
$$

若模型在保持同一边际违约概率的同时给出“该期发生违约时”的条件平均敞口为 15，则应使用

$$
UCVA_{\mathrm{joint}}
=0.98\times15\times0.02\times0.60
=0.1764.
$$

差异来自违约条件暴露，不是把 $2\%$ 或 10 任意乘大。本例只说明两个公式为何不能混用；不同模型若同时改变边际分布、回收、净额或 close-out，不能仅凭“存在错向风险”断言总 CVA 必然高于另一个 independent 模型的结果。

> [!question]- 最小自检
> 两年累计风险中性违约概率为 3%，一年累计概率为 1%。第二年的边际违约概率在离散 CVA 中应取 3% 还是 2%？
>
> **答案：** 取 $3\%-1\%=2\%$。3% 是截至第二年末的累计概率；把它直接作为第二年概率会把第一年的违约再次计入。

## 边界

- 单边 CVA 把估值方视为无违约风险；一旦允许双方违约，就必须转入 [[DVA与双边估值|first-to-default 双边估值]]。
- $UCVA$ 是公允价值调整，不是预期会计信用损失、CCR 违约资本或 CVA 风险资本；三者的监管边界见 [[三类 CVA 口径]]。
- CVA 对信用利差、利率、波动率或汇率的 Greeks 只是揭示既有 CVA 对市场因子的敏感度，并不会“创造”新的风险，也不能把估值 CVA 自动改成 Basel CVA 资本。
- 期中点、确定贴现、确定 LGD、暴露—违约独立和粗时间网格都属于近似，应逐项说明；公式简短不等于条件可以省略。
- close-out 金额、担保品可执行性、回收延迟和 margin period of risk 会改变 $X_C(\tau_C)$；本卡不以单一 $\max(V-C,0)$ 代替完整合同建模。

## 来源与核验

- Basel Committee on Banking Supervision, [MAR50.2–50.3、50.31–50.32：regulatory CVA](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/50/inforce/2023-01-01/published/2020-07-08)：定位单边监管 CVA 将银行自身视为无违约、采用市场隐含 PD、市场共识 ELGD 与贴现未来敞口；监管口径不等同于全部会计 CVA。
- Basel Committee on Banking Supervision, [CRE50.26–50.33 与 CRE53.12–53.13](https://www.bis.org/basel_framework/chapter/CRE/53.htm?inforce=20230101&published=20200605&tldate=20020317)：定位 current exposure、EE、EPE 与 one-sided CVA 的术语边界，避免把时点 EE 与跨时点 EPE 混同。
- Brigo, Chourdakis & Bakkar, [*Counterparty Risk for Energy-Commodities Swaps*，Proposition 2.1](https://www.damianobrigo.it/commoditiescr_fs.pdf)：一手定价论文，定位风险中性贴现、违约指标与正 NPV 的联合 UCVA 公式，以及离散分解只是进一步近似。
- Brigo & Vrins, [*Disentangling Wrong-Way Risk: Pricing CVA via Change of Measures and Drift Adjustment*](https://arxiv.org/abs/1611.02877)：定位 CVA 依赖于暴露与对手信用风险的联合结构，不能默认使用独立暴露曲线。
- 本轮已通过独立的来源、定义、公式/算例与边界复核，因此状态为 `source-checked`；用户掌握度仍为 `unassessed`。用于实际机构估值时，仍须对账交易、市场与信用数据、净额和抵押品状态，并核对 close-out、回收、违约依赖和模型校准；`source-checked` 不代表模型批准或法律意见。
