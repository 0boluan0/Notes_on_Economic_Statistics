---
aliases:
  - "Weak PBE 一般不必是 SPE，因为其路径上一致性可能不约束零概率进入的真子博弈内部 beliefs"
  - "Weak PBE need not refine subgame-perfect equilibrium"
  - "Weak PBE versus SPE"
student_os: knowledge-atom
atom_id: GT-PBE-005
atom_set: signaling-games-pbe
atom_type: implication-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[完美贝叶斯均衡]]"
  - "[[子博弈完美均衡]]"
  - "[[路径上一致性]]"
related:
  - "[[PBE定义口径]]"
  - "[[序贯均衡]]"
  - "[[子博弈判定]]"
part_of:
  - "[[信号博弈与PBE.canvas]]"
---

# Weak PBE 一般不必是 SPE，因为其路径上一致性可能不约束零概率进入的真子博弈内部 beliefs
<!-- bilingual-en:start -->
*A weak PBE need not be subgame perfect because on-path consistency may leave beliefs inside a zero-probability proper subgame unrestricted*
<!-- bilingual-en:end -->

> [!summary] Weak PBE 与 SPE 没有无条件包含关系
> Weak PBE 只在从整棵树起点正概率到达的信息集上强制 Bayes 更新。若候选策略以零概率进入某个真子博弈，该子博弈内部的 beliefs 可能仍按 weak-PBE 口径自由选择；这些 beliefs 可以使续局逐信息集最优，却不能保证该续局策略是把子博弈本身当作起点时的 Nash 均衡。
> <!-- bilingual-en:start -->
> Weak PBE imposes Bayes' rule only at information sets reached with positive probability from the original root. If a proper subgame is entered with probability zero, beliefs inside it may remain unrestricted under weak PBE. Those beliefs can support information-set optimality without making the continuation strategy a Nash equilibrium when the subgame is treated as a game in its own right.
> <!-- bilingual-en:end -->

## 失效机制
<!-- bilingual-en:start -->
*Why refinement can fail*
<!-- bilingual-en:end -->

[[子博弈完美均衡|SPE]]要求策略限制在每个真子博弈上都是 Nash 均衡，这项要求与该子博弈在原候选路径上是否到达无关。相反，weak PBE 的 belief consistency 使用原策略从整棵树根部诱导的到达概率。一个整体概率为零的子博弈即使条件于“已经进入”后会产生正概率内部历史，weak PBE 也可能不要求相应 beliefs 由子博弈内策略重新 Bayes 化。
<!-- bilingual-en:start -->
SPE requires the restriction of the strategy profile to every proper subgame to be a Nash equilibrium, regardless of whether that subgame is reached by the candidate. Weak-PBE consistency instead uses probabilities induced from the original root. A subgame with zero overall entry probability may contain histories that would have positive conditional probability once entry is imposed, yet weak PBE need not recompute its beliefs from the continuation strategies.
<!-- bilingual-en:end -->

因此，按任意路径外 beliefs 得到的序贯最优行动，可能依赖一组与子博弈自身概率结构不相容的节点权重。把该子博弈单独拿出来检查时，某个玩家便可能有有利的完整策略偏离，导致 weak PBE 的策略部分不是 SPE。
<!-- bilingual-en:start -->
Sequentially optimal actions supported by arbitrary off-path beliefs may therefore rely on node weights inconsistent with the subgame's own probability structure. Once the subgame is checked as a separate game, a player may have a profitable complete-strategy deviation, so the weak-PBE strategy profile fails SPE.
<!-- bilingual-en:end -->

## 适用限定
<!-- bilingual-en:start -->
*Scope qualifications*
<!-- bilingual-en:end -->

这个边界针对一般不完美信息扩展式博弈。若所有信息集都是单点，belief 退化且逐节点序贯最优恢复熟悉的逆向归纳逻辑；若博弈除整棵树外没有真子博弈，SPE 与 Nash 的检查重合，此时谈“未精炼某个真子博弈”没有额外内容。标准两阶段信号博弈通常属于后者，但它仍需要 PBE 来约束观察后的 beliefs 与反应。
<!-- bilingual-en:start -->
This boundary concerns general imperfect-information extensive-form games. With singleton information sets, beliefs are degenerate and sequential optimality recovers the familiar backward-induction logic. If there is no proper subgame beyond the whole game, SPE adds no test beyond Nash, although PBE remains useful for beliefs and responses after observed signals.
<!-- bilingual-en:end -->

[[序贯均衡]]用同一完全混合策略序列在所有信息集生成 beliefs，并且其策略部分是 subgame perfect。这个更强结论来自 Kreps–Wilson consistency，不能归给 weak PBE 的路径上 Bayes 条件。
<!-- bilingual-en:start -->
Sequential equilibrium generates beliefs at all information sets from one sequence of completely mixed strategies, and its strategy component is subgame perfect. This stronger conclusion follows from Kreps–Wilson consistency, not from weak PBE's on-path Bayes condition.
<!-- bilingual-en:end -->

> [!question]- 自检
> 某真子博弈在候选策略下从整棵树根部到达概率为零。Weak PBE 已在其中每个信息集给出使行动最优的 belief，是否足以推出续局是该子博弈的 Nash 均衡？
>
> **答案：** 不足。还要检查这些 beliefs 是否与把子博弈作为起点时的策略诱导概率相容，以及完整策略偏离是否无利可图；weak PBE 的路径上一致性未必提供这些约束。
> <!-- bilingual-en:start -->
> **Self-check.** A proper subgame is reached with zero probability from the original root. Weak PBE supplies beliefs that make play optimal at each information set inside it. Does this prove that the continuation is a Nash equilibrium of the subgame?
>
> **Answer:** No. The beliefs must still be compatible with probabilities induced when the subgame is treated as the starting game, and complete-strategy deviations must be unprofitable. Weak-PBE on-path consistency need not impose those restrictions.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [MIT OCW 14.126 Spring 2024, Lecture 2: *Equilibrium Refinements*, slides 11–12](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/resources/mit14_126_s24_lecture_2_refinements_pdf/)：核对路径上一致性的最低要求，以及 weak PBE “does not always refine SPE”的明确边界。
- [[01_Math/03_game theory/07_子博弈不完全信息#3.2. 均衡关系|本地课程：SPE 与 belief-based 精炼的课程语境]]：核对无真子博弈时 SPE 的筛选边界。
<!-- bilingual-en:start -->
- MIT Lecture 2 defines weak PBE on slide 11 and states on slide 12 that it does not always refine SPE.
- The local course section anchors the distinction in games where subgame perfection has little additional force.
<!-- bilingual-en:end -->
