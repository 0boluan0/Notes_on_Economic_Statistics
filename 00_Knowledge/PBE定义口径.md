---
aliases:
  - "PBE 的定义在文献中并不统一，weak PBE、较强 PBE 与 sequential equilibrium 的一致性强度必须分开声明"
  - "Conventions for defining PBE"
  - "Weak PBE versus stronger PBE and sequential equilibrium"
student_os: knowledge-atom
atom_id: GT-PBE-002
atom_set: signaling-games-pbe
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[完美贝叶斯均衡]]"
  - "[[策略信念评估]]"
related:
  - "[[序贯均衡]]"
  - "[[路径外信念]]"
  - "[[弱PBE不必精炼SPE]]"
leads_to:
  - "[[信号博弈弱PBE检验]]"
part_of:
  - "[[信号博弈与PBE.canvas]]"
---

# PBE 的定义在文献中并不统一，weak PBE、较强 PBE 与 sequential equilibrium 的一致性强度必须分开声明
<!-- bilingual-en:start -->
*Definitions of PBE are not uniform across the literature, so weak PBE, stronger PBE, and sequential equilibrium must be distinguished by their consistency requirements*
<!-- bilingual-en:end -->

> [!summary] 名称相同不保证约束相同
> “PBE”可以指只要求路径上 Bayes 更新的 weak PBE，也可以指在特定动态结构中对路径外更新施加更多约束的较强定义。[[序贯均衡]]是另一个有明确定义的解概念，以 Kreps–Wilson consistency 统一约束全部信息集。应用结论只有在声明所用定义后才可解释。
> <!-- bilingual-en:start -->
> The label “PBE” may denote weak PBE, which imposes Bayes' rule only on path, or a stronger definition that restricts off-path updating in a specified class of dynamic games. Sequential equilibrium is a distinct solution concept with Kreps–Wilson consistency across all information sets. An application is interpretable only after its convention is stated.
> <!-- bilingual-en:end -->

## Weak PBE 的最低一致性要求
<!-- bilingual-en:start -->
*The minimal consistency requirement of weak PBE*
<!-- bilingual-en:end -->

Weak PBE 以[[策略信念评估|assessment]] $(\sigma,\mu)$ 为对象，要求策略在每个信息集上给定 belief 时满足[[序贯理性]]，并要求所有正概率信息集上的 belief 由策略与 Nature 概率经 Bayes 法则推出。零概率信息集上的 belief 仍须是合法分布并支持最优续局，但 weak-PBE 定义本身不规定它必须来自哪一种 tremble。
<!-- bilingual-en:start -->
Weak PBE is an assessment $(\sigma,\mu)$ that is sequentially rational at every information set and Bayes-consistent at every information set reached with positive probability. Beliefs at zero-probability information sets must still support optimal continuation play, but weak PBE does not require them to arise from a particular perturbation.
<!-- bilingual-en:end -->

## 较强 PBE 依赖所采用的模型域
<!-- bilingual-en:start -->
*Stronger PBE depends on its stated domain*
<!-- bilingual-en:end -->

Fudenberg–Tirole 一类较强 PBE 定义为路径外更新加入额外一致性要求，但其正式定义针对 multi-stage games with observed actions。它不是对所有扩展式博弈都无条件适用的统一替代品。不同论文若写“PBE”，可能采用不同的额外更新规则，因此必须读取其定义而不是只看缩写。
<!-- bilingual-en:start -->
Stronger notions such as the Fudenberg–Tirole PBE impose additional consistency on off-path updating, but their formal domain is the class of multistage games with observed actions. They are not a single universally defined replacement for weak PBE. The definition used by an application must therefore be read explicitly rather than inferred from the acronym.
<!-- bilingual-en:end -->

## Sequential equilibrium 的统一 tremble 要求
<!-- bilingual-en:start -->
*The common-tremble requirement of sequential equilibrium*
<!-- bilingual-en:end -->

[[序贯均衡]]同样要求序贯理性，但把 consistency 加强为：存在同一序列完全混合策略 $\sigma^m\to\sigma$，由每个 $\sigma^m$ 在所有信息集上经 Bayes 法则生成 $\mu^m$，并且 $\mu^m\to\mu$。因此不同信息集的 beliefs 不能分别从互不相容的 perturbations 中挑选。经典定义以有限行动集合为重要适用条件。
<!-- bilingual-en:start -->
Sequential equilibrium also requires sequential rationality, but strengthens consistency: one sequence of completely mixed strategies $\sigma^m\to\sigma$ must generate Bayes beliefs $\mu^m$ at every information set, with $\mu^m\to\mu$. Beliefs at different information sets therefore cannot be selected from mutually incompatible perturbations. Finite action sets are an important condition of the classical definition.
<!-- bilingual-en:end -->

## 应用中怎样声明口径
<!-- bilingual-en:start -->
*How to state the convention in an application*
<!-- bilingual-en:end -->

最低限度应写明三件事：候选对象是策略组合还是 assessment；路径外 beliefs 只需支持序贯最优，还是必须满足某种额外 consistency；结论所在的博弈类是否满足该定义的域条件。若使用标准两阶段信号博弈的简化 PBE 检验，也应明确它对应哪一种 consistency 强度。
<!-- bilingual-en:start -->
At minimum, state whether the candidate object is a strategy profile or an assessment, what consistency restriction applies to off-path beliefs, and whether the game lies in the domain of the chosen definition. A simplified PBE test for a standard two-stage signaling game should likewise identify the consistency strength it represents.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两篇论文都声称使用 PBE，但一篇允许各路径外信息集分别选择 belief，另一篇要求 beliefs 来自同一完全混合策略序列。能否把两篇论文的均衡集合直接视为同一对象？
>
> **答案：** 不能。两者使用了不同的 consistency 强度；应先固定定义和适用域，再比较均衡集合。
> <!-- bilingual-en:start -->
> **Self-check.** Two papers both use the label PBE. One selects beliefs separately at off-path information sets; the other requires a common sequence of completely mixed strategies. Can their equilibrium sets be treated as the same object?
>
> **Answer:** No. They impose different consistency requirements, so the definitions and domains must be aligned before the equilibrium sets are compared.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [MIT OCW 14.126 Spring 2024, Lecture 2: *Equilibrium Refinements*, slides 11–17](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/resources/mit14_126_s24_lecture_2_refinements_pdf/)：核对 weak PBE、较强 PBE 的定义差异、应用需声明口径，以及 sequential equilibrium 的 consistency 要求。
- [MIT OCW 14.126 Spring 2024, Yildiz, *Game Theory Lecture Notes*, Chapter 4 §4.1, Definitions 4.3–4.6](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/resources/mit14_126_s24_yildiz-lecture-notes_pdf/)：核对 assessment、序贯理性、Kreps–Wilson consistency 与序贯均衡。
- [[01_Math/03_game theory/07_子博弈不完全信息#3.1. 定义|本地课程：weak PBE 口径]]：核对本课程实际采用的最低一致性要求。
<!-- bilingual-en:start -->
- MIT Lecture 2, slides 11–17, distinguishes weak PBE, stronger PBE conventions, and sequential equilibrium, and asks applications to state their definition.
- Yildiz Chapter 4 §4.1 supplies the formal assessment, sequential-rationality, consistency, and sequential-equilibrium definitions.
- The local course section identifies the weak-PBE convention used in the course.
<!-- bilingual-en:end -->
