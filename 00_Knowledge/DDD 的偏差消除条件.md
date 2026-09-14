---
aliases:
  - "DDD 用第三个差分消除可比较的 DID 偏差"
  - Triple difference estimator
  - Difference-in-difference-in-differences
  - 三重差分
  - DDD
student_os: knowledge-atom
atom_id: ECON-DID-013
atom_type: identification
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[双重差分法]]"
---

# DDD 用第三个差分消除可比较的 DID 偏差

<!-- bilingual-en:start -->
*DDD uses a third difference to remove a comparable DID bias*
<!-- bilingual-en:end -->

> [!summary] 核心命题
> 三重差分（DDD）不是“多减一次所以更可靠”。它用另一个不应受到目标机制影响的类别，估计第一个 DID 中剩余的偏差；只有当两类的 DID 偏差在无处理时相同，第三个差分才把偏差消掉。
>
> <!-- bilingual-en:start -->
> Triple differences is not more credible merely because it subtracts once more. It uses an additional category, which should not be affected by the target mechanism, to estimate residual bias in the first DID. The third difference removes that bias only when the two categories would have the same DID bias without treatment.
> <!-- bilingual-en:end -->

设 $T$ 区分政策地区与对照地区，$Post$ 区分前后，$B$ 区分目标类别与辅助类别。DDD 可写成

$$
DDD=DID_{B=1}-DID_{B=0}.
$$

它可以由三个主效应、三个两两交互项和 $T\times Post\times B$ 的饱和回归计算；三重交互项等于八个单元格均值形成的第三个差分。和四格 DID 一样，回归等式只是计算，因果解释来自辅助类别能否正确刻画第一个 DID 的残余偏差。
<!-- bilingual-en:start -->
Let $T$ distinguish policy and comparison places, $Post$ distinguish periods, and $B$ distinguish the target and auxiliary categories. Then $DDD=DID_{B=1}-DID_{B=0}$. A saturated regression with all lower-order terms and the triple interaction $T\times Post\times B$ computes the same eight-cell contrast. As with ordinary DID, the regression identity performs the calculation; causal interpretation comes from whether the auxiliary category captures the residual bias in the first DID.
<!-- bilingual-en:end -->

## 一个结构化例子

某州的产假强制政策只直接覆盖特定人群。第一层 DID 比较该州目标人群与其他州目标人群的前后变化，但政策州可能同时经历一项影响当地所有劳动者的经济冲击。DDD 再计算不受产假政策直接覆盖的辅助人群的州际 DID，用它扣除这项共同的州—时间冲击。识别要求是：若没有政策，目标组相对辅助组的结果差在政策州与对照州会以相同方式变化。若同期冲击只影响目标组，辅助组就无法替它消偏。
<!-- bilingual-en:start -->
Suppose a state maternity mandate directly covers a particular population. The first DID compares that group's before–after change with the same group in other states, but the policy state may simultaneously experience an economic shock affecting all local workers. DDD computes the same interstate DID for an auxiliary population not directly covered by the mandate and subtracts it to remove that common state-by-time shock. Identification requires the target-versus-auxiliary outcome difference to evolve similarly across policy and comparison states without the mandate. A contemporaneous shock affecting only the target group cannot be removed by the auxiliary group.
<!-- bilingual-en:end -->

## 额外维度也可能带来新问题

辅助类别若被政策间接影响、构成变化不同、测量口径不同，或本来就承受不同的类别特定冲击，第三个差分会增加而不是消除偏差。DDD 也不要求两个组成 DID 各自都无偏；关键是它们的偏差相同。这一点比“需要两套平行趋势”更准确。
<!-- bilingual-en:start -->
If the auxiliary category is indirectly affected, changes composition differently, is measured differently, or experiences distinct category-specific shocks, the third difference can add rather than remove bias. DDD does not require each component DID to be unbiased separately; it requires their biases to be equal. This is more precise than saying that two independent parallel-trends assumptions are needed.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么 DDD 中第二个 DID 本身可以有偏，而最终第三个差分仍可能无偏？
>
> **答案：** 因为 DDD 减的是两个 DID。若它们在无处理时含有相同偏差，第三个差分会把共同偏差消掉；要求的是偏差相同，而不是两个 DID 分别为零偏。

## 来源与核验

- Olden & Møen (2022), [*The Triple Difference Estimator*](https://academic.oup.com/ectj/article/25/3/531/6545797)：核验 DDD 作为两个 DID 之差、八格回归表达及其“偏差相同”的单一平行趋势识别条件。
