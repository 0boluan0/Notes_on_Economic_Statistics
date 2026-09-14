---
aliases:
  - "不可约、非爆炸的可数 CTMC 存在平稳概率当且仅当正常返"
  - "可数不可约 CTMC 的平稳概率要求非爆炸与正常返"
  - Positive recurrence characterizes countable CTMC stationary existence
  - 正常返刻画可数 CTMC 平稳概率的存在
student_os: knowledge-atom
atom_id: PROB-CTMC-017
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC爆炸]]"
  - "[[常返与停留时间]]"
  - "[[CTMC平稳分布]]"
related:
  - "[[正常返与零常返]]"
  - "[[Kac回返公式]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
leads_to:
  - "[[可数CTMC平衡收敛]]"
  - "[[M-M-1稳态人数]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 不可约、非爆炸的可数 CTMC 存在平稳概率当且仅当正常返
<!-- bilingual-en:start -->
*An irreducible non-explosive countable CTMC has a stationary probability distribution if and only if it is positive recurrent*
<!-- bilingual-en:end -->

> [!summary] 无限状态下的存在—唯一条件
> 对不可约、非爆炸并由所给生成矩阵唯一确定转移半群的可数状态 CTMC，存在平稳概率分布当且仅当链正常返。存在时，平稳分布唯一。零常返只保证最终返回，不保证平均日历返回时间有限，因此不能产生可归一化的平稳概率；暂态链同样没有平稳概率。
> <!-- bilingual-en:start -->
> For an irreducible regular countable CTMC, positive recurrence is exactly the condition for existence of a stationary probability law; that law is then unique.
> <!-- bilingual-en:end -->

这里的 non-explosion/regularity 是定理前提，不是可从形式方程自动读出的结论。[[有限CTMC平稳生成矩阵判据|有限状态生成矩阵判据]]不能无条件外推到这里：即使找到非负且和为一的 $\pi$ 满足 $\pi Q=0$，仍要确认 minimal process 保守、生成矩阵没有留下爆炸后延拓的歧义，并以 $\pi P(t)=\pi$ 核对所讨论半群的平稳性。

这一结果判定平稳概率**是否存在以及是否唯一**。在相同条件下，转移概率收敛到它是进一步的结论，见 [[可数CTMC平衡收敛]]。

> [!example] M/M/1 的存在边界
> 无限等待空间的 M/M/1 系统内人数链在 $\lambda<\mu$ 时正常返，因而存在唯一几何平稳分布；$\lambda=\mu$ 时零常返，$\lambda>\mu$ 时暂态。三种情形的生成矩阵都逐行合法，但只有第一种产生平稳概率。

> [!question]- 自检
> 一条不可约、非爆炸的可数 CTMC 常返，是否必有平稳概率？
>
> **答案：** 不必。还要正常返，即平均日历返回时间有限；零常返没有可归一化的平稳概率。

## 来源与核验

- [Cambridge Applied Probability notes, §§2.4–2.5](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 invariant distribution、positive recurrence 与 non-explosion/regularity 条件。
- [James Norris, Markov Chains, §§3.5–3.6](https://www.statslab.cam.ac.uk/~jrn10/Markov/)：核对不可约可数 CTMC 的平稳存在与唯一性。
