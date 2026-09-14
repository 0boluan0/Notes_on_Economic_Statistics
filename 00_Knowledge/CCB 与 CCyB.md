---
aliases:
  - "资本留存缓冲与逆周期缓冲都用 CET1 但触发逻辑和计算对象不同"
  - Capital conservation buffer versus countercyclical capital buffer
  - CCB 与 CCyB
student_os: knowledge-atom
atom_id: MB-BAS-006
atom_set: basel-capital-liquidity-regulation
atom_type: regulatory-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[CET1、AT1 与 Tier 2]]"
  - "[[Basel 最低比率基线]]"
related:
  - "[[资本监管的收益与边界]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# 资本留存缓冲与逆周期缓冲都用 CET1 但触发逻辑和计算对象不同
*The capital conservation and countercyclical buffers both use CET1 but have different triggers and calculation bases*

> [!summary] 常态留存与时变系统风险
> 资本留存缓冲（CCB）在最低资本要求之上保留一层常态 CET1；银行落入缓冲区会受到分红、回购和奖金等分派约束。逆周期资本缓冲（CCyB）则由各法域随信贷周期和系统性风险启用，并按银行对不同法域的私人部门信用暴露加权形成银行特定比率。

CCB 的 Basel 基线为 RWA 的 2.5% CET1，但它是缓冲，不应被写成“CET1 最低线从 4.5% 永久改成 7%”。CCyB 的目的则是在风险积累期建立可释放的资本空间；法域将缓冲率设为零并不证明没有个体银行风险，也不取消其他资本要求。

例：银行一半相关私人信用暴露位于 CCyB 2% 的法域，另一半位于 0% 的法域，简化后的银行特定 CCyB 为 1%，而不是对全集团直接套 2%。真实计算须按框架定义的地理暴露权重。

## 边界

- CCB 与 CCyB 都要求以 CET1 满足，但设置主体、时变性和计算逻辑不同。
- 使用缓冲通常意味着分派受限，不等于银行已经违反最低资本率或立即资不抵债。
- 本地宏观审慎当局可采用不同范围、公告与过渡安排。

> [!question]- 自检
> 为什么“银行 CET1 低于 7%”不能直接改写成“违反 4.5% 最低要求”？
>
> **答案：** 7%把 4.5%最低线与2.5%资本留存缓冲相加；落入缓冲区和跌破最低线的法律与监督后果不同。

## 来源与核验

- [Basel Framework, RBC30](https://www.bis.org/basel_framework/chapter/RBC/30.htm)：核对 CCB、CCyB、分派约束与银行特定缓冲的计算。
- [Basel Committee, countercyclical capital buffer guidance](https://www.bis.org/publ/bcbs187.htm)：核对 CCyB 的政策目的与国家当局角色。
- 口径核验日：2026-08-29；实际缓冲率必须查当日法域公告。
