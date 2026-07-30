---
aliases:
  - "Basel Capital Regulation and OTC Clearing"
  - "Basel Accords"
  - "巴塞尔协议"
status: source-checked
---

# 巴塞尔资本监管与 OTC 清算

> [!summary] 快速恢复
> **它解决什么：** 规定银行用多少资本和流动性吸收信用、市场、操作和对手风险，并通过清算、保证金和披露降低系统外部性。
> **具体锚点：** 风险加权资本率高不等于现金很多；资本吸收损失，流动性资产应付现金流，二者是不同防线。
> **核心难点：** 监管比率依赖风险权重和模型；最低合规不是实际风险充分，对手风险也会迁移到 CCP。
> **为什么重要：** 2008 后改革不仅提高资本，还加入杠杆、流动性、缓冲、处置和衍生品基础设施。
> **继续：** 用当前 Basel Framework 核对口径，不背过时数字；信用/CVA 见对应知识文件。

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。

## 为什么监管资本

银行杠杆高、负债可赎回且失败有支付/信用外部性，存款保险又削弱市场纪律。资本要求让股东/债权先吸收损失，降低破产概率和公共救助预期，但过高或顺周期要求也可能压缩信贷。

## Basel I—III 的结构

Basel I 引入统一信用风险权重；Basel II 强化风险敏感、监督审查和市场纪律；危机后 Basel III 提高资本质量，加入资本缓冲、杠杆率、流动性标准、CVA 与交易账簿改革。最终口径应以 BIS 当前整合框架为准。

## 三类核心约束

风险加权资本率比较合格资本与 RWA；杠杆率对总敞口设后备；LCR/NSFR 分别关注短期压力流动性与稳定融资。一个机构可资本充足但流动性不足，反之亦然。

## 市场风险与 ES

FRTB 重划 trading/banking book 边界，标准法和内部模型法处理市场风险，并用 stressed expected shortfall 等捕捉尾部与流动性期限。模型可批准不等于风险可忽略，仍需 backtesting、P&L attribution 和治理。

## OTC、CCP 与保证金

标准化衍生品中央清算、交易报告和双边保证金减少不透明双边网络。variation margin 覆盖已发生市值变化，initial margin 覆盖违约到平仓的潜在变化。净额结算和抵押品降低敞口但产生流动性需求。

## CCP 风险与 waterfall

CCP 集中风险并用违约基金、成员出资和 recovery waterfall 分摊损失。它降低网络复杂度但可能成为系统关键节点；错误模型、集中头寸和同时追缴保证金可放大压力。

## 模型与监管套利

内部模型、风险权重和法律实体结构可被优化。output floor、披露和监督用于限制差异；评估应同时看比率、绝对敞口、情景和数据质量。

## 最小自检

### 资本充足和流动性充足为什么不是一回事？

> [!answer]- 答案
> 资本是损失吸收的净资产缓冲，流动性是按时履行现金义务的资产/融资能力。
### CCP 是否消除了对手风险？

> [!answer]- 答案
> 没有。它通过净额和保证金重组、集中风险，并引入对 CCP 模型、成员和流动性 waterfall 的依赖。
### 为什么还需要杠杆率而非只看 RWA 比率？

> [!answer]- 答案
> 风险权重可能低估、模型化或被套利，杠杆率提供不依赖风险权重的简单后备。

## 来源与核验

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- Basel Committee, [Minimum capital requirements for market risk](https://www.bis.org/bcbs/publ/d457.htm)：核验交易账簿与 expected shortfall 监管框架。
