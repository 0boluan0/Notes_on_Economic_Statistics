---
aliases:
  - "显式奖励模型 RLHF 是先从成对比较拟合标量奖励模型，再以该代理奖励优化策略的两阶段后训练管线"
  - "Explicit-reward-model RLHF is a two-stage post-training pipeline that fits a scalar reward model from pairwise comparisons and then optimizes a policy against that proxy reward"
  - "典型 RLHF 先用成对比较拟合奖励模型，再用策略优化提高该代理奖励"
student_os: knowledge-atom
atom_id: LLM-PT-005
atom_set: llm-post-training
atom_type: process
status: source-checked
mastery_state: unassessed
requires:
  - "[[成对偏好模型]]"
related:
  - "[[SFT目标]]"
  - "[[Bradley–Terry模型]]"
leads_to:
  - "[[KL参考约束]]"
part_of:
  - "[[LLM 后训练.canvas|LLM 后训练]]"
---

# 显式奖励模型 RLHF 是先从成对比较拟合标量奖励模型，再以该代理奖励优化策略的两阶段后训练管线
<!-- bilingual-en:start -->
*Explicit-reward-model RLHF is a two-stage post-training pipeline that fits a scalar reward model from pairwise comparisons and then optimizes a policy against that proxy reward*
<!-- bilingual-en:end -->

> [!summary] 显式奖励模型管线有两个不同的学习问题
> 第一阶段收集同一提示下的回答比较，用 $(x,y_w,y_l)$ 拟合奖励模型 $r_\phi(x,y)$；第二阶段固定或阶段性更新这个代理，再调整语言模型策略 $\pi_\theta(y\mid x)$，使它生成的回答获得更高奖励。前一阶段问“标注者在这些候选中会选谁”，后一阶段问“怎样生成能被当前奖励模型打高分的回答”。两步相连，却不是同一个目标。
>
> <!-- bilingual-en:start -->
> The first stage collects comparisons among responses to the same prompt and fits a reward model $r_\phi(x,y)$ from $(x,y_w,y_l)$. The second stage freezes or periodically updates that proxy and adjusts the language-model policy $\pi_\theta(y\mid x)$ so that generated responses receive higher reward. The first asks which candidate annotators would choose; the second asks how to generate responses that the current reward model scores highly. The stages are connected but solve different optimization problems.
> <!-- bilingual-en:end -->

## 从比较数据到可优化的标量
<!-- bilingual-en:start -->
*From comparison data to an optimizable scalar*
<!-- bilingual-en:end -->

人类比较本来是离散观察：在几个候选回答中选一个或给出排序。[[成对偏好模型]] 规定怎样学习这种相对选择；显式奖励模型管线再选用一个可泛化的标量分数，其中 [[Bradley–Terry模型|Bradley–Terry 分数差模型]] 是常见形式。这样，策略不必为每个新回答实时询问标注者，而可以用奖励模型近似评价大量采样结果。

<!-- bilingual-en:start -->
Human comparisons are discrete observations: select one candidate or rank several candidates. A [[成对偏好模型|pairwise preference model]] specifies how to learn those relative choices; the explicit-reward pipeline then adopts a generalizable scalar score, commonly with a [[Bradley–Terry模型|Bradley–Terry score-difference model]]. The policy can then evaluate many sampled responses approximately without asking an annotator about every new generation.
<!-- bilingual-en:end -->

一个典型的策略目标可以概括为
$$
\max_\theta\;
\mathbb E_{x\sim D,\;y\sim\pi_\theta(\cdot\mid x)}
\bigl[r_\phi(x,y)\bigr]
-\beta\,
\mathbb E_{x\sim D}
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)
\right).
$$
工程上常用 PPO 一类算法，根据采样回答的奖励估计更新策略；参考策略常取 SFT 模型。具体实现可能把 KL 惩罚放进逐 token 奖励，也可能用单独损失或自适应系数。共同结构是：策略接受来自代理奖励的优化压力，同时受到某种偏离约束。

<!-- bilingual-en:start -->
A typical policy objective maximizes expected reward while subtracting $\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$. Engineering pipelines often use PPO-like algorithms to update the policy from rewards on sampled responses, with the SFT model commonly serving as the reference. Implementations may place the KL penalty in token-level rewards, use a separate loss, or adapt its coefficient. The shared structure is optimization pressure from a proxy reward together with a constraint on policy movement.
<!-- bilingual-en:end -->

奖励模型一旦进入策略优化，就面对与最初训练不同的数据。开始时的比较通常来自基座或 SFT 策略产生的候选；更新后的策略会主动寻找奖励更高的新回答。若它走到比较数据稀少的区域，奖励模型的外推误差也会变成优化方向。于是“奖励提高”是算法按当前代理成功优化的证据，不等于真实用户目标已经提高。

<!-- bilingual-en:start -->
Once used for policy optimization, the reward model encounters data different from its original training set. Initial comparisons usually concern candidates from a base or SFT policy; the updated policy actively seeks new high-reward responses. If it moves into regions with sparse comparison data, reward-model extrapolation error becomes part of the optimization direction. Rising reward therefore demonstrates successful optimization of the current proxy, not automatic improvement in the underlying human objective.
<!-- bilingual-en:end -->

> [!warning] 这是典型显式 RM 管线，不是 RLHF 的唯一实现
> “来自人类反馈的强化学习”在不同文献和系统中覆盖范围并不完全一致。人类反馈还可以进入 SFT、拒绝采样、在线比较、过程奖励或其他策略学习方法；DPO 则使用偏好对直接更新策略，不显式训练供策略调用的奖励模型，也不运行这条在线 RL 管线。PPO 是 InstructGPT 与 Llama 2 等系统采用的具体算法，不是“使用人类反馈”这件事的定义。
>
> <!-- bilingual-en:start -->
> The label reinforcement learning from human feedback is used with different scopes across papers and systems. Human feedback can also enter SFT, rejection sampling, online comparison, process rewards, or other policy-learning methods. DPO uses preference pairs to update a policy directly, without explicitly training a reward model for policy optimization or running this online RL pipeline. PPO is a concrete algorithm used in systems such as InstructGPT and Llama 2, not the definition of learning from human feedback.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 一个系统收集 chosen/rejected 对，然后用 DPO loss 直接更新语言模型；它既没有训练独立奖励模型，也没有用 PPO 采样优化。它是否走完了本卡描述的典型显式 RM 管线？
>
> <!-- bilingual-en:start -->
> A system collects chosen/rejected pairs and updates the language model directly with a DPO loss. It trains no separate reward model and runs no PPO sampling loop. Has it implemented the typical explicit-reward-model pipeline described here?
> <!-- bilingual-en:end -->
>
> **答案：** 没有。它使用了人类偏好信号，但绕过了“先拟合显式奖励模型，再用策略优化提高该模型分数”这条两阶段管线。
>
> <!-- bilingual-en:start -->
> **Answer:** No. It uses human preference signals but bypasses the two-stage route of first fitting an explicit reward model and then optimizing a policy against that model.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Ouyang et al. (2022), *Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155), §3 与 Figure 2：给出 demonstration SFT、比较数据奖励模型、再以 PPO 优化策略的完整公开管线。
- [Touvron et al. (2023), *Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/abs/2307.09288), §3.2：给出安全与有用性奖励模型、拒绝采样和 PPO 迭代使用的具体系统。
- [Rafailov et al. (2023), *Direct Preference Optimization*](https://arxiv.org/abs/2305.18290), §1 与 §3：用于界定 DPO 如何省去显式奖励模型拟合与在线 RL，而不是把它误写成这条管线的同义词。

<!-- bilingual-en:start -->
- Ouyang et al. (2022), §3 and Figure 2, presents the full public pipeline of demonstration SFT, a reward model from comparison data, and PPO policy optimization.
- Touvron et al. (2023), §3.2, gives a concrete system using helpfulness and safety reward models together with rejection sampling and PPO.
- Rafailov et al. (2023), §1 and §3, establishes how DPO removes explicit reward-model fitting and online RL, which marks the boundary of the pipeline described here.
<!-- bilingual-en:end -->
