---
aliases:
  - "观察到价格高于边际成本并不否定 Bertrand 逻辑，而是要求检查产品差异、容量、成本、搜索与动态互动是否破坏了基准假设"
  - A price above marginal cost does not refute Bertrand logic but requires checking which benchmark assumptions fail
student_os: knowledge-atom
atom_id: GT-OLI-008
atom_set: oligopoly-competition
atom_type: assumption-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Bertrand边际成本均衡]]"
related:
  - "[[Hotelling价格竞争]]"
  - "[[Hotelling需求系统]]"
  - "[[Hotelling均衡]]"
  - "[[重复互动合作]]"
  - "[[永久触发合作条件]]"
  - "[[寡头模型选择]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# 观察到价格高于边际成本并不否定 Bertrand 逻辑，而是要求检查产品差异、容量、成本、搜索与动态互动是否破坏了基准假设
<!-- bilingual-en:start -->
*Observing price above marginal cost does not refute Bertrand logic; it calls for checking whether differentiation, capacity, cost, search, or dynamic interaction breaks the benchmark assumptions*
<!-- bilingual-en:end -->

> [!summary] 基准结果是一项条件命题
> $p=c$ 不是“价格竞争总会很激烈”的口号，而是同质产品、相同成本、消费者只看价格、低价企业能接走全部需求、企业一次性同时报价等条件共同推出的均衡。现实价格高于成本时，应定位哪项机制改变了降价收益。
>
> <!-- bilingual-en:start -->
> The result $p=c$ is a conditional equilibrium statement. It combines homogeneous goods, identical costs, price-only consumer choice, enough capacity for the low-price firm to serve all demand, and one-shot simultaneous pricing. A real price above cost asks which mechanism changes the return to undercutting.
> <!-- bilingual-en:end -->

## 五种常见机制怎样改变偏离

| 机制 | 基准中的关键一步为何失效 |
|---|---|
| 产品差异 | 略微降价不再取得全部需求；高价企业仍保留部分消费者 |
| 容量约束 | 低价企业不能供应整个市场，剩余需求仍会流向高价企业 |
| 成本差异 | “共同边际成本 $c$”不再存在；低成本企业与高成本企业的可盈利降价范围不同 |
| 搜索成本 | 消费者未必立即发现最低价，需求对相对价格的反应变缓 |
| 重复互动 | 今天的降价会改变未来行为；在 [[永久触发合作条件|特定条件]] 下，未来惩罚可抵消一次降价收益 |

<!-- bilingual-en:start -->
Differentiation leaves the high-price firm some demand; capacity prevents the low-price firm from serving everyone; cost asymmetry changes profitable undercutting ranges; search frictions slow consumers' discovery of the lowest price; and repeated interaction can attach a future punishment to a current price cut. Each mechanism changes a distinct step in the benchmark deviation proof.
<!-- bilingual-en:end -->

## 改变假设后要重新求解

不能看到差异化就直接写“所以 $p>c$”。必须给出消费者怎样在产品间选择，以及企业需求如何随双方价格变化。[[Hotelling价格竞争]]定义空间差异化定价模型，[[Hotelling需求系统]]再在端点、均匀消费者、线性运输成本、全覆盖和内点分割下给出需求机制；在进一步的相同成本和全局偏离条件下，[[Hotelling均衡]]才得到 $p=c+t$。其他差异化模型的加价公式可能不同。

同样，容量约束不等于随便给 Bertrand 公式加一个上限。容量若在定价前确定，它本身可能是第一阶段的战略选择；若定价和配给规则不同，均衡也会不同。基准失败告诉我们“需要更丰富的模型”，不会自动选定唯一替代模型。

<!-- bilingual-en:start -->
Changing an assumption requires a new solution. Differentiation needs an explicit consumer-choice rule; capacity may be a first-stage strategic commitment; search and dynamics need their own information and continuation structure. Failure of the benchmark identifies a missing mechanism but does not select a unique replacement model.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两家航空公司价格不同，且低价公司航班已经满座。看到高价仍有乘客，能否据此说 Bertrand 的降价逻辑是错的？
>
> **答案：** 不能。无容量约束是低价者夺走全部需求的关键条件；满座直接破坏了这一步。应建立含容量和配给规则的定价模型，而不是用该观察否定条件命题。

## 来源与核验

- MIT 15.010/15.011, [*The Basics of Game Theory*, p. 3](https://ocw.mit.edu/courses/15-010-economic-analysis-for-business-decisions-fall-2004/807ba86e100d349ef73c294b9e720931_the_bsc_game_thy.pdf)：核对同质品基准及产品差异会保留高于边际成本的价格。
- MIT 15.024, [*Applied Economics for Managers, Session 10*](https://www.ocw.mit.edu/courses/15-024-applied-economics-for-managers-summer-2004/0386fffd71e360dda606b76fd42f20b2_lec10.pdf)：直接核对 Bertrand 的产品差异与容量约束是基准边界。
- MIT 14.271, [*Static Competition and Models of Differentiation, Part 1*, p. 5](https://ocw.mit.edu/courses/14-271-industrial-organization-i-fall-2022/mit14_271_f22_lec5slides.pdf)：直接核对成本不对称会改变同成本 Bertrand 的均衡定价逻辑。
- MIT 14.12, [*Chapter 7: Application—Imperfect Competition*, §7.2.3](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/a870a72380a584e8d1ffd2b34fa24c9e_MIT14_12F12_chapter7.pdf)：直接核对搜索成本下消费者不再自动发现最低价，价格竞争结果随之改变。
- MIT 14.126, [*Repeated Games with Perfect Information*, slides 3–8](https://ocw.mit.edu/courses/14-126-game-theory-spring-2016/3fc58b6d07a73055a44dcbc5aaacc738_MIT14_126S16_Repeated.pdf)：直接核对重复互动把当前偏离与未来可信惩罚联结。
- MIT 14.271, [*Static Competition and Models of Differentiation, Part 1*](https://ocw.mit.edu/courses/14-271-industrial-organization-i-fall-2022/mit14_271_f22_lec5slides.pdf)：核对 Hotelling 差异化定价、正加价和战略互补。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#4. 伯特兰竞争|本地课程：同质品基准]]与[[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#5. Hotelling 模型|空间差异化]]：支持课程中从同质品分段需求到差异化连续需求的对照。
