---
aliases:
  - "连续类型下，事中最优性可按几乎处处或逐类型口径定义，零测类型只有在后者下被约束"
  - "Almost-everywhere versus pointwise BNE with continuous types"
student_os: knowledge-atom
atom_id: GT-BNE-003
atom_set: bayesian-games
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[贝叶斯纳什均衡]]"
  - "[[零概率类型的条件信念]]"
related:
  - "[[零概率类型的BNE边界]]"
  - "[[贝叶斯策略]]"
part_of:
  - "[[贝叶斯博弈.canvas]]"
---

# 连续类型下，事中最优性可按几乎处处或逐类型口径定义，零测类型只有在后者下被约束
<!-- bilingual-en:start -->
*With continuous types, interim optimality may be defined almost everywhere or pointwise; null types are constrained only by the latter convention*
<!-- bilingual-en:end -->

连续类型模型通常让每个单独类型都具有零边际概率。因此，“每个类型都最优”究竟是逐点要求，还是只对几乎所有类型要求，不是无关紧要的措辞，而是 BNE 定义本身的一部分。
<!-- bilingual-en:start -->
In a continuous-type model, each individual type commonly has zero marginal probability. Whether “every type is optimal” is imposed pointwise or only almost everywhere is therefore part of the definition of BNE rather than harmless wording.
<!-- bilingual-en:end -->

## 几乎处处口径
<!-- bilingual-en:start -->
*Almost-Everywhere Convention*
<!-- bilingual-en:end -->

若条件信念来自 regular conditional distribution，它通常只在类型边际分布 almost everywhere 的意义下唯一。相应的 a.e. BNE 只要求最佳反应不等式对几乎所有 $t_i$ 成立；在零测类型集合上修改策略或 conditional-belief version，不改变这个均衡判断。
<!-- bilingual-en:start -->
A regular conditional distribution is generally unique only almost everywhere under the type marginal. An a.e. BNE therefore imposes the best-response inequality for almost every $t_i$; changing the strategy or the conditional-belief version on a null set does not change that equilibrium judgment.
<!-- bilingual-en:end -->

## 逐类型口径
<!-- bilingual-en:start -->
*Pointwise Convention*
<!-- bilingual-en:end -->

若模型把每个 $p_i(\cdot\mid t_i)$ 都直接列为原语，或明确固定了零测类型上的 version，就可以逐个 $t_i$ 要求最优。此时即使某类型在事前分布下概率为零，它的规定行动仍必须是该指定 belief 下的最佳反应。
<!-- bilingual-en:start -->
If the model supplies every $p_i(\cdot\mid t_i)$ as a primitive, or explicitly fixes a version on null types, it can impose optimality pointwise. A type with zero ex-ante probability must then still receive a best response under its specified belief.
<!-- bilingual-en:end -->

## 不能用 full support 消除差异
<!-- bilingual-en:start -->
*Topological Full Support Does Not Remove the Distinction*
<!-- bilingual-en:end -->

连续分布可以对每个非空开集赋正概率，同时对每个单点赋零概率。因此，拓扑意义的 full support 不会把 a.e. 与 pointwise 口径变成同一件事。阅读定理或计算均衡时，应先看作者把策略相等和最优性理解为逐点还是几乎处处。
<!-- bilingual-en:start -->
A continuous distribution may assign positive probability to every nonempty open set while assigning zero probability to every singleton. Topological full support therefore does not collapse the a.e. and pointwise conventions. Before applying a theorem or solving for equilibrium, check whether the author treats strategies and optimality pointwise or modulo null sets.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [Matthew O. Jackson, Tomas Rodriguez-Barraquer, and Xu Tan, “Epsilon-Equilibria of Perturbed Games”，§2，正文 p.5](https://web.stanford.edu/~jacksonm/emailgame.pdf)：展示连续类型模型选取任意 conditional-expectation version，并只要求对 $P$-almost all types 成立的 interim 最优性口径。
- [UC Berkeley Statistics, Will Fithian, “Measure Theory Basics: Conditional probability”](https://www.stat.berkeley.edu/~wfithian/courses/stat210a/measure-theory-basics.html)：核对零测事件上的普通条件概率失效及 conditional version 的歧义。
- [MIT OCW 14.126, *Game Theory Lecture Notes*, Chapter 3 §3.2，正文 pp.72–76](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/mit14_126_s24_yildiz-lecture-notes.pdf)：核对可测策略、逐类型 BNE 记号和零概率类型造成的事前／事中差异。
