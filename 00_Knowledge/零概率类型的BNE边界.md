---
aliases:
  - "给定相容的事前分布，逐类型 interim BNE 蕴含事前 Nash；有限类型下每个自身类型边际概率为正时二者等价"
  - "Interim BNE and ex-ante Nash at zero-probability types"
student_os: knowledge-atom
atom_id: GT-BNE-002
atom_set: bayesian-games
atom_type: equivalence-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[贝叶斯纳什均衡]]"
  - "[[零概率类型的条件信念]]"
related:
  - "[[纳什均衡]]"
  - "[[贝叶斯策略]]"
  - "[[贝叶斯混合与行为策略]]"
leads_to:
  - "[[连续类型BNE口径]]"
part_of:
  - "[[贝叶斯博弈.canvas]]"
---

# 给定相容的事前分布，逐类型 interim BNE 蕴含事前 Nash；有限类型下每个自身类型边际概率为正时二者等价
<!-- bilingual-en:start -->
*Given consistent ex-ante distributions, pointwise interim BNE implies ex-ante Nash; with finite types the two are equivalent when every own type has positive marginal probability*
<!-- bilingual-en:end -->

比较两种均衡前，必须先有与各玩家 interim beliefs 相容的 ex-ante 分布，从而能定义 ex-ante game $G(B)$。若只给了逐类型信念而没有自身类型的事前权重，$G(B)$ 并不由这些信念唯一确定。
<!-- bilingual-en:start -->
The comparison requires ex-ante distributions consistent with the players' interim beliefs so that the ex-ante game $G(B)$ is defined. Type-contingent beliefs alone do not determine ex-ante weights on a player's own types.
<!-- bilingual-en:end -->

## 有限类型的包含与等价
<!-- bilingual-en:start -->
*Inclusion and equivalence with finite types*
<!-- bilingual-en:end -->

逐类型最优会在对自身类型取期望后保持最优，因此

$$
BNE(B)\subseteq NE\bigl(G(B)\bigr).
$$

若对每名玩家和每个自身类型都有

$$
p_i(t_i)>0,
$$

那么任何类型上的严格有利偏离都会提高 ex-ante 收益，故反向包含也成立：

$$
BNE(B)=NE\bigl(G(B)\bigr).
$$

所需条件是每个自身类型的边际概率为正，不要求每个完整类型组合都有正概率。
<!-- bilingual-en:start -->
Type-by-type optimality remains optimal after averaging over own types, which gives the inclusion from pointwise BNE to ex-ante Nash. If every own type of every player has positive marginal probability, any profitable deviation at one type raises ex-ante utility, yielding equality. Positive probability is required for each own-type marginal, not for every complete type profile.
<!-- bilingual-en:end -->

若某类型的边际概率为零，改变该类型的行动不会改变 ex-ante 收益，所以 ex-ante Nash 可以在该类型规定非最优行动；若模型已为该类型给定条件信念，逐点 interim BNE 会排除这种规定，包含关系便可能严格。
<!-- bilingual-en:start -->
If an own type has zero marginal probability, changing its prescribed action leaves ex-ante utility unchanged, so an ex-ante Nash equilibrium may prescribe a suboptimal action there. A pointwise interim BNE rules out that prescription when the model supplies a conditional belief for the type, and the inclusion can then be strict.
<!-- bilingual-en:end -->

连续类型中每个单点通常都为零概率，不能直接搬用这里的有限型正概率条件；应另行选择[[连续类型BNE口径|几乎处处或逐类型的 BNE 口径]]。
<!-- bilingual-en:start -->
In a continuous-type model, every singleton commonly has probability zero, so the finite positive-probability condition cannot simply be carried over. The model must instead choose an [[连续类型BNE口径|almost-everywhere or pointwise BNE convention]].
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

[MIT OCW 14.126, Muhamet Yildiz, *Game Theory Lecture Notes*, Chapter 3 §3.2, Fact 3.1，正文 pp.73–74](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/mit14_126_s24_yildiz-lecture-notes.pdf)：正式给出 $BNE(B)\subseteq NE(G(B))$，以及每个自身类型边际概率为正时的等价。
<!-- bilingual-en:start -->
MIT formally states the inclusion and the equality condition based on positive marginal probability for every own type.
<!-- bilingual-en:end -->

[UC Berkeley, Federico Echenique, *Game Theory Lecture Notes*, Chapter 9 §9.1，正文 pp.81–82](https://eml.berkeley.edu/~fechenique/lecture_notes/game_theory.pdf)：核对 derived ex-ante normal form 与逐类型 interim 最优化之间的关系。
<!-- bilingual-en:start -->
The Berkeley notes connect the derived ex-ante normal form with type-by-type interim optimization.
<!-- bilingual-en:end -->
