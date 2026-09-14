---
aliases:
  - "KL 正则限制策略偏离参考模型，但只约束分布移动而不保证代理奖励代表真实目标"
  - "KL regularization limits policy divergence from a reference model, but constrains only distributional movement and does not guarantee that proxy reward represents the true objective"
student_os: knowledge-atom
atom_id: LLM-PT-019
atom_set: llm-post-training
atom_type: guarantee-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[KL参考约束]]"
related:
  - "[[DPO边界]]"
  - "[[奖励代理过优化]]"
  - "[[偏好分数解释边界]]"
leads_to:
  - "[[LLM 评测]]"
part_of:
  - "[[LLM 后训练.canvas|LLM 后训练]]"
---

# KL 正则限制策略偏离参考模型，但只约束分布移动而不保证代理奖励代表真实目标
<!-- bilingual-en:start -->
*KL regularization limits policy divergence from a reference model, but constrains only distributional movement and does not guarantee that proxy reward represents the true objective*
<!-- bilingual-en:end -->

> [!summary] KL 回答“走了多远”，不回答“方向是否正确”
> 在奖励最大化目标中，常见做法是加入
> $$
> -\beta\,D_{\mathrm{KL}}\!\left(
> \pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)
> \right),
> \qquad \beta>0,
> $$
> 惩罚当前策略 $\pi_\theta$ 相对参考策略 $\pi_{\mathrm{ref}}$ 的分布变化。它让大幅改变付出更高代价，却没有检验奖励模型 $r_\phi(x,y)$ 是否把正确性、安全性或用户效用排对了。
>
> <!-- bilingual-en:start -->
> Reward-maximization objectives often include $-\beta D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x))$, with $\beta>0$, to penalize distributional movement away from a reference policy. Large changes become more costly, but the term does not test whether the reward model $r_\phi(x,y)$ ranks correctness, safety, or user utility properly.
> <!-- bilingual-en:end -->

## 约束来自概率比，而不是内容判定
<!-- bilingual-en:start -->
*The constraint comes from probability ratios, not content judgments*
<!-- bilingual-en:end -->

对离散回答分布，
$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=\sum_y \pi_\theta(y\mid x)
\log\frac{\pi_\theta(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}.
$$
若当前策略把大量概率移到参考模型原本很少生成的回答上，KL 通常会上升；若概率重排较小，KL 通常较低。式子只看两个分布的概率比，不读取回答是否真实、有用或有害。内容质量只能通过奖励、数据和独立评测进入判断。

<!-- bilingual-en:start -->
For discrete response distributions, $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})=\sum_y\pi_\theta(y\mid x)\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}$. Moving substantial probability to responses that the reference rarely produces generally raises KL; a small probability rearrangement generally yields a smaller value. The expression reads probability ratios, not whether an answer is true, useful, or harmful. Content quality must enter through reward data and independent evaluation.
<!-- bilingual-en:end -->

系数 $\beta$ 决定代理奖励与保守程度的交换。$\beta$ 较大时，策略更难离开参考模型，可能减少不稳定和某些奖励利用，也可能压住真实改进；$\beta$ 较小时，策略更容易追逐奖励，也更容易进入奖励模型缺少比较数据的区域。没有一个仅由公式给出的“正确 $\beta$”；它必须连同奖励尺度、优化算法和部署评测校准。

<!-- bilingual-en:start -->
The coefficient $\beta$ trades proxy reward against conservatism. A larger $\beta$ makes departure from the reference harder, which may reduce instability and some reward exploitation but can also suppress genuine improvements. A smaller $\beta$ lets the policy pursue reward more aggressively and move into regions with sparse comparison data. The formula supplies no universally correct $\beta$; it must be calibrated with the reward scale, optimizer, and deployment evaluations.
<!-- bilingual-en:end -->

即使 KL 很小，策略仍可能利用参考模型附近的错误排序。例如奖励模型在措辞稍微更自信时总是多给一点分，策略只需把概率从“审慎但正确”轻微移向“自信且过度断言”，就可能在很小的分布移动内稳定提高代理奖励。反过来，较大的 KL 也不自动表示行为变坏：参考模型若有系统缺陷，修复它本来就需要移动。

<!-- bilingual-en:start -->
Even a small KL can permit exploitation of a local ranking error. If the reward model systematically gives slightly higher scores to more confident wording, the policy may shift modest probability from cautious correct answers to confident overclaims and raise proxy reward while remaining close to the reference. Conversely, a large KL does not automatically mean behavior worsened: correcting a systematic reference-model defect may require substantial movement.
<!-- bilingual-en:end -->

> [!warning] 口径与实现边界
> KL 有方向，$D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$ 与反向 KL 一般不同；序列级精确 KL、采样得到的逐 token 对数比，以及带自适应系数的近似也不是完全相同的量。阅读实验时要核对实际惩罚项和估计方法。本卡的共同结论只到“限制相对参考策略的分布移动”，不能把 KL 误称为真实性证明、安全保证或偏好代表性检验。
>
> <!-- bilingual-en:start -->
> KL is directional: $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$ generally differs from reverse KL. Exact sequence-level KL, sampled token-level log ratios, and approximations with adaptive coefficients are also not identical quantities. An experiment must specify the penalty and estimator it actually uses. The common conclusion stops at limiting distributional movement relative to a reference; KL is not a proof of truthfulness, a safety guarantee, or a test of preference representativeness.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 两个候选策略与参考模型的 KL 完全相同。策略 A 的回答更准确；策略 B 学会了奖励模型偏爱的冗长措辞。只看 KL，能判断哪一个更符合真实目标吗？
>
> <!-- bilingual-en:start -->
> Two candidate policies have exactly the same KL from the reference. Policy A is more accurate; policy B has learned verbose wording favored by the reward model. Can KL alone determine which better serves the true objective?
> <!-- bilingual-en:end -->
>
> **答案：** 不能。相同 KL 只表示在所用口径下分布移动量相同，不说明移动方向的内容价值。要区分 A 与 B，必须检查奖励有效性并做准确性、风格和真实使用结果的独立评测。
>
> <!-- bilingual-en:start -->
> **Answer:** No. Equal KL means equal movement under the chosen divergence measure, not equal value of the direction taken. Distinguishing A from B requires validation of the reward and independent evaluation of accuracy, style, and real use outcomes.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Ouyang et al. (2022), *Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155), §3.4 与 Appendix C：给出 PPO 目标中的参考策略 KL 惩罚及其系数设置。
- [Rafailov et al. (2023), *Direct Preference Optimization*](https://arxiv.org/abs/2305.18290), §3：从 KL 正则的奖励最大化问题推导最优策略与 DPO 目标，清楚区分奖励项和参考策略约束。
- [Touvron et al. (2023), *Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/abs/2307.09288), §3.2：提供奖励优化、策略迭代与参考分布控制的另一公开工程语境；具体实现口径须以原文为准。

<!-- bilingual-en:start -->
- Ouyang et al. (2022), §3.4 and Appendix C, documents a reference-policy KL penalty and its coefficient in PPO training.
- Rafailov et al. (2023), §3, derives the optimal policy and DPO objective from KL-regularized reward maximization, separating the reward term from the reference-policy constraint.
- Touvron et al. (2023), §3.2, provides another public engineering context for reward optimization, policy iteration, and control relative to a reference distribution; its exact implementation must be read from the paper.
<!-- bilingual-en:end -->
