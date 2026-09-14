---
aliases:
  - "DPO 直接提高 chosen 相对 rejected 且相对参考策略的对数概率优势，无需显式训练奖励模型或运行在线强化学习"
  - "DPO directly increases the chosen response's log-probability advantage over the rejected response relative to a reference policy, without explicitly training a reward model or running online reinforcement learning"
student_os: knowledge-atom
atom_id: LLM-PT-007
atom_set: llm-post-training
atom_type: objective
status: source-checked
mastery_state: unassessed
requires:
  - "[[Bradley–Terry模型]]"
  - "[[KL参考约束]]"
contrasts_with:
  - "[[显式奖励模型RLHF]]"
leads_to:
  - "[[DPO边界]]"
part_of:
  - "[[LLM 后训练.canvas|LLM 后训练]]"
---

# DPO 直接提高 chosen 相对 rejected 且相对参考策略的对数概率优势，无需显式训练奖励模型或运行在线强化学习
<!-- bilingual-en:start -->
*DPO directly increases the chosen response's log-probability advantage over the rejected response relative to a reference policy, without explicitly training a reward model or running online reinforcement learning*
<!-- bilingual-en:end -->

> [!summary] 它优化的是“相对参考策略的成对优势”
> 对提示 $x$、被偏好的回答 $y_w$（chosen）和未被偏好的回答 $y_l$（rejected），DPO 比较
> $$
> \Delta_\theta=
> \log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
> -
> \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
> $$
> 训练让 $\Delta_\theta$ 更大：当前策略相对于参考策略，应该更偏向 chosen 而不是 rejected。这个损失可以直接由固定的偏好对计算，因此标准离线 DPO 不必先拟合一个显式奖励模型，也不必在每轮更新后从当前策略采样并运行 PPO。
>
> <!-- bilingual-en:start -->
> For a prompt $x$, a preferred response $y_w$, and a rejected response $y_l$, DPO compares the current policy's chosen-versus-rejected log-probability margin with the same margin under a reference policy. Training increases this relative margin. Because the loss is computed directly from a fixed preference pair, standard offline DPO need not fit an explicit reward model or repeatedly sample from the current policy for PPO updates.
> <!-- bilingual-en:end -->

## 从 KL 正则目标到一个分类式损失
<!-- bilingual-en:start -->
*From a KL-regularised objective to a classification-style loss*
<!-- bilingual-en:end -->

Rafailov et al. 从 KL 正则的奖励最大化问题出发，把最优策略写成奖励与参考策略的函数；由于奖励在 [[提示内奖励平移不变|同一提示内只识别到共同平移]]，再将其等价类代入 [[Bradley–Terry模型|Bradley–Terry 式成对偏好模型]]。对于一组偏好数据 $\mathcal D$，得到常用的 DPO 损失：

$$
\mathcal L_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[
\log\sigma\!\left(\beta\Delta_\theta\right)
\right].
$$

其中 $\sigma$ 是 logistic 函数，$\beta$ 是奖励—策略关系中的尺度参数，也对应原始 KL 正则目标的惩罚系数。损失看起来像二分类，但模型并不是只给整对回答贴标签：梯度会穿过两条完整回答的 token 对数概率，改变语言模型本身。

<!-- bilingual-en:start -->
Rafailov et al. begin with KL-regularised reward maximisation, express the optimal policy as a function of reward and a reference policy, use the [[提示内奖励平移不变|prompt-wise reward-shift equivalence class]], and substitute the relation into a [[Bradley–Terry模型|Bradley–Terry pairwise-preference model]]. The resulting DPO loss is a logistic loss on $\beta\Delta_\theta$. Although it resembles binary classification, gradients pass through the token log probabilities of both complete responses and update the language model itself.
<!-- bilingual-en:end -->

## 这个目标不要求 chosen 的绝对概率每一步都上升
<!-- bilingual-en:start -->
*The objective does not require the chosen response's absolute probability to rise at every step*
<!-- bilingual-en:end -->

DPO 关心的是两个“相对参考策略的对数概率变化”之差。假设 chosen 的对数概率相对参考策略下降 $0.2$，rejected 下降 $0.8$，则 $\Delta_\theta=0.6$；当前策略仍比参考策略更强地偏向 chosen。反过来，只提高 chosen 的概率却把 rejected 提高得更多，也会使成对优势变差。

这一区分很重要，因为序列概率随长度快速变小，单看 $\pi_\theta(y_w\mid x)$ 的数值没有可比意义。有效诊断应同时查看 chosen 与 rejected 的 token 对数概率、相对参考策略的变化以及成对 margin，而不是把“chosen loss 下降”当作 DPO 的完整目标。

<!-- bilingual-en:start -->
DPO depends on a difference of two changes relative to the reference. If the chosen response's log probability falls by $0.2$ relative to the reference while the rejected response's falls by $0.8$, the relative margin is still $0.6$. Conversely, raising both probabilities can worsen the margin if the rejected response rises more. Diagnostics should therefore examine both responses, their changes relative to the reference, and the pairwise margin rather than treating the chosen loss alone as the objective.
<!-- bilingual-en:end -->

> [!question]- 自检
> 对同一偏好对，当前策略相对参考策略的 chosen 对数概率变化为 $-0.3$，rejected 的变化为 $-0.9$。这组数值是否与 DPO 希望的方向一致？能否据此说 chosen 的绝对概率提高了？
>
> <!-- bilingual-en:start -->
> For one preference pair, the current policy changes the chosen log probability by $-0.3$ relative to the reference and the rejected log probability by $-0.9$. Is this the direction favoured by DPO? Does it show that the chosen response's absolute probability increased?
> <!-- bilingual-en:end -->
>
> **答案：** 相对 margin 为 $-0.3-(-0.9)=0.6$，所以当前策略相对参考策略更偏向 chosen，方向与 DPO 一致；但 chosen 相对参考策略的对数概率其实下降了，不能说它的绝对概率提高。DPO 判断的是相对优势，不是 chosen 单边概率。
>
> <!-- bilingual-en:start -->
> **Answer:** The relative margin is $-0.3-(-0.9)=0.6$, so the current policy favours the chosen response more strongly than the reference does. But the chosen log probability itself fell relative to the reference. DPO concerns the relative advantage, not a one-sided increase in chosen probability.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Rafailov et al. (2023), “Direct Preference Optimization: Your Language Model is Secretly a Reward Model”](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)：核对 KL 正则奖励目标到 DPO 损失的推导、参考策略与 $\beta$ 的位置，以及无需显式奖励模型和在线策略采样的训练管线主张。

<!-- bilingual-en:start -->
- Rafailov et al. (2023) was checked for the derivation from KL-regularised reward maximisation to the DPO loss, the roles of the reference policy and $\beta$, and the claim that DPO avoids an explicit reward model and online policy sampling during fine-tuning.
<!-- bilingual-en:end -->
