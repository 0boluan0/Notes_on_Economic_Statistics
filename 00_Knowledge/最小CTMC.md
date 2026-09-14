---
aliases:
  - "最小 CTMC 在爆炸时进入墓地状态并不再返回原状态空间"
  - Minimal CTMC
  - Minimal process
  - 墓地状态构造
student_os: knowledge-atom
atom_id: PROB-CTMC-027
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC爆炸]]"
  - "[[CTMC转移半群]]"
leads_to:
  - "[[爆炸后延拓]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 最小 CTMC 在爆炸时进入墓地状态并不再返回原状态空间
<!-- bilingual-en:start -->
*A minimal CTMC is sent to a cemetery state at explosion and never returns to the original state space*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 设 $\zeta$ 是逐跳构造的爆炸时刻。在原状态空间 $S$ 外增加墓地状态 $\Delta$，并令
> $$
> X_t=\Delta,\qquad t\ge\zeta.
> $$
> 爆炸前按照给定生成率运行，爆炸后永久停在 $\Delta$；这个不为爆炸后另加“返回规则”的过程称为对应 $Q$ 的 **minimal CTMC**。
> <!-- bilingual-en:start -->
> The minimal process follows the prescribed jump rates up to explosion, is killed at the explosion time, and stays in an added cemetery state thereafter.
> <!-- bilingual-en:end -->

若只把转移半群写在原状态空间 $S$ 上，定义
$$
p_{ij}(t)=\Pr_i(X_t=j),\qquad i,j\in S,
$$
则
$$
\sum_{j\in S}p_{ij}(t)=\Pr_i(t<\zeta),
\qquad
1-\sum_{j\in S}p_{ij}(t)=\Pr_i(\zeta\le t).
$$
因此爆炸时它在 $S$ 上是 sub-Markov 半群；缺失质量是“截至时刻 $t$ 已爆炸”的概率。若把 $\Delta$ 纳入状态空间，全部概率质量仍为 1。

> [!question]- 自检
> minimal CTMC 在爆炸之后会自动从某个有限状态重新开始吗？
>
> **答案：** 不会。minimal 构造把过程送到墓地状态并永久留在那里；任何重新进入规则都是额外延拓。

## 来源与核验

- [Cambridge Applied Probability notes, §§1.7–1.9](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 minimal process、墓地状态与爆炸前存活概率。
- [James Norris, Markov Chains, §§2.7–2.9](https://www.statslab.cam.ac.uk/~jrn10/Markov/)：核对 minimal chain 在爆炸时被杀死的构造。
