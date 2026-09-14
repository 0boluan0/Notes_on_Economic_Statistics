---
aliases:
  - "对有限状态 CTMC，相对于同一分布的详细平衡与可逆等价且都推出平稳性"
  - CTMC detailed balance and reversibility
  - CTMC stationarity does not imply reversibility
  - 速率详细平衡与可逆
student_os: knowledge-atom
atom_id: PROB-CTMC-035
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC平稳分布]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
  - "[[CTMC详细平衡]]"
  - "[[CTMC可逆性]]"
leads_to:
  - "[[生灭链平稳递推]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 对有限状态 CTMC，相对于同一分布的详细平衡与可逆等价且都推出平稳性
<!-- bilingual-en:start -->
*For a finite-state CTMC, detailed balance and reversibility relative to the same distribution are equivalent and both imply stationarity*
<!-- bilingual-en:end -->

> [!summary] 等价与严格边界
> 对有限状态 CTMC 的生成矩阵 $Q$ 和概率分布 $\pi$，
> $$
> \pi_iq_{ij}=\pi_jq_{ji}\ \text{对所有 }i\ne j
> \iff
> \text{链相对于 }\pi\text{ 可逆}
> \implies
> \pi P(t)=\pi\ \text{对所有 }t\ge0.
> $$
> 最后一个箭头不能反向：平稳只要求每个状态的总流入与总流出相等，详细平衡与可逆还要求每一对状态的双向流逐对抵消。
> <!-- bilingual-en:start -->
> Rate detailed balance is equivalent to time reversibility relative to the same distribution, and either condition implies stationarity. Stationarity alone permits circulating probability flow.
> <!-- bilingual-en:end -->

详细平衡推出 $\pi Q=0$，因为对固定 $j$，逐对流量相加后总流入等于总流出；有限状态下再由[[有限CTMC平稳生成矩阵判据|生成矩阵判据]]得到平稳。平稳时间反演的生成率满足
$$
q^*_{ij}=\frac{\pi_jq_{ji}}{\pi_i}
$$
（在 $\pi_i>0$ 的支撑上），所以 $q^*_{ij}=q_{ij}$ 恰好等价于详细平衡。

可数状态空间可保留同一逻辑，但必须先固定保守、非爆炸、regular 的转移半群，并使相关无穷求和合法；形式上的 $\pi Q=0$ 不能替代这些过程条件。

> [!example] 平稳但不可逆
> 三状态 CTMC 只允许
> $$1\to2\to3\to1$$
> 且三条速率都为 1。均匀分布满足 $\pi Q=0$，所以由[[有限CTMC平稳生成矩阵判据|有限状态判据]]可知链平稳；但 $q_{12}=1$、$q_{21}=0$，详细平衡失败。平稳状态中仍有顺时针净流，因此链不可逆。

> [!question]- 自检
> 已验证 $\pi Q=0$。能否直接断言链相对于 $\pi$ 可逆？
>
> **答案：** 不能。还要逐对验证 $\pi_iq_{ij}=\pi_jq_{ji}$；global balance 不排除环流。

## 来源与核验

- [Cambridge Applied Probability notes, §2.6](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对时间反演生成矩阵、详细平衡、平稳与可逆之间的关系。
- [Ward Whitt, Continuous-Time Markov Chains, §§6, 10](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf)：核对 rate detailed balance、stationarity 与 reversibility。
