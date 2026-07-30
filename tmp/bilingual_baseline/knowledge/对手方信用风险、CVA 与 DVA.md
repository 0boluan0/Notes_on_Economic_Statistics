---
aliases:
  - "Counterparty Credit Risk"
  - "CVA"
  - "DVA"
  - "对手方信用风险"
status: source-checked
---

# 对手方信用风险、CVA 与 DVA

> [!summary] 快速恢复
> **它解决什么：** 衍生品未来市值会随市场变化，交易对手又可能在不利时违约；本主题把动态敞口与违约共同定价和管理。
> **具体锚点：** 利率互换今天价值为零不等于无信用敞口，未来利率变化可能使其对我方大幅为正，恰逢对手违约就有损失。
> **核心难点：** 敞口、PD、LGD 和市场因子可能相关；净额、抵押品和 margin period of risk 必须按法律集合建模。
> **为什么重要：** CVA 把对手信用成本纳入公允价值，资本和限额还需覆盖其波动与尾部。
> **继续：** 先建 exposure profile，再加入 default 与 recovery；监管口径见 [[巴塞尔资本监管与 OTC 清算]]。

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。

## 当前与潜在未来敞口

当前敞口是正的 replacement cost，负市值通常截为零信用敞口；PFE 是未来敞口分布高分位，EE 是期望正敞口，EPE 再按时间平均。指标服务限额、定价和资本的口径不同。

## 净额、抵押品与补救期

法律可执行的 netting set 内正负交易可抵消。variation margin 降低当前敞口，initial margin 覆盖违约到平仓期间风险；threshold、minimum transfer、频率和 dispute 会留下 gap。模拟必须在组合层应用协议。

## CVA

简化单边 CVA 是未来各期 discounted expected exposure × marginal default probability × LGD 的和。实践使用风险中性市场/信用输入用于公允价值，并考虑 wrong-way risk、净额和 collateral。

## DVA 与双边调整

DVA 反映自身信用恶化使本方负债公允价值下降的会计/定价调整，经济解释有争议且无法无摩擦兑现。双边 CVA/DVA 需一致处理谁先违约和 close-out。

## wrong-way risk

当对手更可能违约时我方敞口也更高，称 wrong-way risk；例如商品生产商在商品价格暴跌时既信用恶化又对某衍生品负担加重。独立假设会低估风险。

## Monte Carlo 与 Greeks

模拟市场路径、重估交易、应用 collateral/netting，再与违约模型整合得到 exposure 和 CVA。CVA Greeks 衡量利率、信用利差和波动变化，产生 CVA market risk 和 hedge basis。

## 限额与治理

同时管理当前/PFE、wrong-way、集中、评级触发和 collateral liquidity。模型校准、法律意见和数据质量与公式同等重要。

## 最小自检

### 互换初始价值为零为什么仍有对手风险？

> [!answer]- 答案
> 未来市场变化会使 replacement value 为正，而对手可能在那时违约；风险来自未来敞口分布。
### wrong-way risk 是什么？

> [!answer]- 答案
> 对手违约可能性升高的状态恰好也是我方对其敞口升高的状态。
### 净额与抵押品是否把敞口降到零？

> [!answer]- 答案
> 通常不会；估值变化、门槛、转移滞后、争议和补救期会留下 gap，且法律可执行性是前提。

## 来源与核验

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
