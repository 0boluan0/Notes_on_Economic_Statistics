---
aliases:
  - "BNE 与 PBE 的比较必须先固定同一扩展式表示，前者检验开局逐类型最优，后者检验动态 assessment 的逐信息集序贯理性与信念一致性"
  - "Bayesian Nash equilibrium versus perfect Bayesian equilibrium"
  - "BNE and PBE"
student_os: knowledge-atom
atom_id: GT-PBE-004
atom_set: signaling-games-pbe
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[贝叶斯纳什均衡]]"
  - "[[完美贝叶斯均衡]]"
  - "[[扩展式博弈]]"
related:
  - "[[策略信念评估]]"
  - "[[序贯理性]]"
  - "[[信息集信念]]"
  - "[[信号博弈]]"
part_of:
  - "[[信号博弈与PBE.canvas]]"
---

# BNE 与 PBE 的比较必须先固定同一扩展式表示，前者检验开局逐类型最优，后者检验动态 assessment 的逐信息集序贯理性与信念一致性
<!-- bilingual-en:start -->
*Comparing BNE with PBE requires a common extensive-form representation: BNE checks type-by-type optimality from the start, whereas PBE checks a dynamic assessment for sequential rationality and belief consistency at every information set*
<!-- bilingual-en:end -->

> [!summary] 两个概念检查的对象不同
> BNE 以类型条件策略组合为对象，问每个类型在开局事中信息下是否愿意改变自己的完整计划。PBE 以策略—信念 assessment 为对象，进一步问游戏进行到每个信息集时，belief 是否符合所采用的一致性要求，以及续局是否仍然最优。
> <!-- bilingual-en:start -->
> BNE evaluates a profile of type-contingent strategies and asks whether each type wishes to change its complete plan at the interim starting point. PBE evaluates a strategy-belief assessment and additionally asks whether beliefs satisfy the chosen consistency condition and continuation play remains optimal at every information set.
> <!-- bilingual-en:end -->

## BNE 的开局事中检验
<!-- bilingual-en:start -->
*The interim-at-the-start BNE test*
<!-- bilingual-en:end -->

在一个[[贝叶斯博弈]]的策略式表示中，[[贝叶斯纳什均衡]]要求每名玩家的每个类型，在自己的事中 belief 下，对其他玩家的类型条件完整策略作最佳反应。偏离对象是一份可行的完整策略或当前静态行动；游戏树中后来观察到的具体历史和 posterior 不作为独立均衡对象报告。
<!-- bilingual-en:start -->
In the strategic representation of a Bayesian game, BNE requires every type of every player to best respond to the other players' complete type-contingent strategies under its interim belief. The deviation is a feasible complete strategy or current static action; beliefs formed after later observed histories are not reported as separate equilibrium objects.
<!-- bilingual-en:end -->

## PBE 的逐信息集检验
<!-- bilingual-en:start -->
*The information-set-by-information-set PBE test*
<!-- bilingual-en:end -->

在动态[[扩展式博弈]]中，[[完美贝叶斯均衡|PBE]]要求完整策略之外再给出每个信息集的[[信息集信念|belief]]。给定这些 beliefs，策略必须在每个信息集满足[[序贯理性]]；正概率信息集还须按所声明的 PBE 口径满足 Bayes consistency。路径外行动因此不能只靠开局期望收益中的零权重逃过检查。
<!-- bilingual-en:start -->
In a dynamic extensive-form game, PBE adds a belief at every information set to the complete strategy profile. Strategies must be sequentially rational at each information set, and beliefs at positive-probability information sets must satisfy the stated Bayesian consistency requirement. An off-path action therefore cannot escape scrutiny merely because it receives zero weight in ex-ante expected payoff.
<!-- bilingual-en:end -->

## 可比较时的限定关系
<!-- bilingual-en:start -->
*The qualified relationship on a common representation*
<!-- bilingual-en:end -->

只有把同一个动态贝叶斯博弈、同一先验、同一类型和同一完整策略空间同时写成策略式与扩展式时，才可比较两者。在标准两阶段信号博弈中，一份 PBE 的策略部分满足发送者逐类型不偏离和接收者在所有正概率观察后的最优性，因而满足相应策略式 BNE／Nash 条件；反向则还缺少 beliefs、路径上一致性以及路径外反应的可支持性。
<!-- bilingual-en:start -->
The concepts can be compared only when the same dynamic Bayesian game, prior, types, and complete strategy spaces are represented in strategic and extensive form. In a standard two-stage signaling game, the strategy component of a PBE satisfies the corresponding strategic-form BNE or Nash incentive conditions. The converse still lacks a belief system, on-path consistency, and supportability of off-path receiver responses.
<!-- bilingual-en:end -->

若改变博弈表示、时间结构、可观察信息或 PBE 的 consistency 强度，简单包含图就不再是同一个数学命题。一次性静态贝叶斯博弈没有后续信息集时，PBE 的额外动态内容也可能消失。比较应先写明共同模型，再陈述该模型内成立的关系。
<!-- bilingual-en:start -->
A simple inclusion diagram ceases to be a well-defined claim when the game representation, timing, observability, or consistency convention changes. In a one-shot Bayesian game with no later information sets, the additional dynamic content of PBE may disappear. State the common model first and only then state the relation valid within it.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一个信号博弈的完整策略组合是策略式 BNE，但某个路径外接收者行动不是任何 belief 下的最佳反应。它是否自动给出 PBE？
>
> **答案：** 不会。策略式 BNE 不要求为该零概率信息集提供可支持的 belief；PBE 的逐信息集序贯理性会排除这项反应。
> <!-- bilingual-en:start -->
> **Self-check.** A complete strategy profile is a strategic-form BNE of a signaling game, but an off-path receiver action is not optimal under any belief. Does the profile automatically extend to a PBE?
>
> **Answer:** No. Strategic-form BNE does not require a supporting belief at that zero-probability information set, whereas PBE imposes sequential rationality there.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [MIT OCW 14.126 Spring 2024, Lecture 3: *Signaling Games*, slides 4–8](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/resources/mit14_126_s24_lecture_3_signaling_pdf/)：核对信号博弈策略空间、发送者逐类型最优、路径上 receiver best response 与路径外条件。
- [MIT OCW 14.126 Spring 2024, Yildiz, *Game Theory Lecture Notes*, Chapter 3 §3.2 and Chapter 4 §4.1](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/resources/mit14_126_s24_yildiz-lecture-notes_pdf/)：分别核对 BNE 的逐类型事中定义与 assessment 的逐信息集定义。
- [[01_Math/03_game theory/07_子博弈不完全信息#4.3. 候选纳什均衡|本地课程：策略式候选还须继续接受 PBE 检查]]。
<!-- bilingual-en:start -->
- MIT Lecture 3, slides 4–8, supports the sender, receiver, on-path, and off-path conditions in a signaling game.
- Yildiz Chapters 3 and 4 distinguish interim BNE from information-set assessment and sequential rationality.
- The local course section shows why a strategic-form candidate still requires a PBE check.
<!-- bilingual-en:end -->
