---
aliases:
  - "SA-CCR 按法律净额集合计算 RC 与 PFE 并在当前 Basel 非模型标准法中取代 CEM"
  - SA-CCR replacement cost potential future exposure and netting sets
  - SA-CCR 与历史 CEM
student_os: knowledge-atom
atom_id: MB-BAS-016
atom_set: basel-capital-liquidity-regulation
atom_type: regulatory-method
status: source-checked
mastery_state: unassessed
requires:
  - "[[违约敞口口径]]"
  - "[[对手方敞口指标]]"
  - "[[净额与抵押品]]"
related:
  - "[[中央清算与多边净额]]"
  - "[[三类 CVA 口径]]"
leads_to:
  - "[[QCCP资本处理]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# SA-CCR 按法律净额集合计算 RC 与 PFE 并在当前 Basel 非模型标准法中取代 CEM
*SA-CCR calculates RC and PFE by legally enforceable netting set and replaces CEM in the current Basel non-model standardised approach*

> [!summary] 当前标准法敞口
> SA-CCR 对每个认可的法律净额集合计算违约暴露：$EAD=\alpha(RC+PFE)$，Basel 基线 $\alpha=1.4$。$RC$ 反映当前重置成本与抵押品状态，$PFE$ 由监管 add-on、净额和保证金期限等形成；不同净额集合不能凭经济相关性任意相抵。

法律净额是关键边界。ISDA Master Agreement 与 CSA 可以提供净额和抵押品机制，却不自动保证监管认可；银行还须有适用于相关交易、对手、破产与司法辖区的可执行法律意见。若净额不可执行，就不能为了降低 EAD 把正负市值放进同一集合。

历史当前敞口法（CEM）以净重置成本加名义额 add-on 估算 EAD，并曾用 NGR 调整净额 add-on。NGR 正确口径是净当前重置成本相对于总正当前重置成本，而不是“正市值之和除以绝对市值之和”。在当前 Basel 的非模型标准法中，SA-CCR 已取代 CEM；旧题仍可按 CEM 计算，但必须标为历史，现实合规还须核对该法域何时完成本地实施。

例：两笔相反方向互换若属于不同法律实体或不同不可交叉净额的协议，即使市场风险近似抵消，也必须分别计算对手方敞口。

## 边界

- RC 不是贷款账面余额，PFE 也不是未来最大损失的精确预测。
- 抵押品会改变 RC/PFE，却不能把结算、gap risk 和争议期风险降为零。
- SA-CCR 的净额集合与会计抵销展示并非同一问题。

> [!question]- 自检
> 两笔衍生品市值一正一负且金额相同，为什么 EAD 不一定为零？
>
> **答案：** 只有满足法律可执行净额条件才可在同一净额集合相抵，而且即便 RC 较低，PFE 仍覆盖未来市值变化。

## 来源与核验

- [Basel Framework, CRE52](https://www.bis.org/basel_framework/chapter/CRE/52.htm)：核对 SA-CCR 的 $RC$、$PFE$、$\alpha$、保证金与净额集合规则。
- [Basel Committee, The standardised approach for measuring counterparty credit risk exposures](https://www.bis.org/publ/bcbs279.htm)：核对 SA-CCR 取代 CEM/SM 的改革边界。
- [[中央清算与多边净额]]：复用中央清算怎样改变法律对手与净额边界；[[QCCP资本处理]] 再把 SA-CCR 的 EAD 分流到 CCP 交易敞口资本口径。
- 口径核验日：2026-08-29；监管认可仍以本地生效规则和法律意见为准。
