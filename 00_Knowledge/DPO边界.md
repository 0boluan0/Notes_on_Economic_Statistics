---
aliases:
  - "DPO 简化训练管线却仍依赖偏好数据、参考策略、β 和成对偏好建模假设"
  - "DPO simplifies the training pipeline but still depends on preference data, a reference policy, beta, and pairwise-preference modeling assumptions"
student_os: knowledge-atom
atom_id: LLM-PT-008
atom_set: llm-post-training
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[DPO目标]]"
related:
  - "[[Bradley–Terry模型]]"
  - "[[偏好分数解释边界]]"
  - "[[KL不保证奖励有效]]"
  - "[[长度代理偏差]]"
  - "[[迎合偏好偏差]]"
  - "[[奖励代理过优化]]"
part_of:
  - "[[LLM 后训练.canvas|LLM 后训练]]"
---

# DPO 简化训练管线却仍依赖偏好数据、参考策略、β 和成对偏好建模假设
<!-- bilingual-en:start -->
*DPO simplifies the training pipeline but still depends on preference data, a reference policy, beta, and pairwise-preference modeling assumptions*
<!-- bilingual-en:end -->

> [!summary] 少一个显式奖励模型，不等于没有奖励假设
> DPO 把“拟合奖励模型，再用在线 RL 优化策略”折叠为一个直接偏好损失。这减少了训练部件，却没有消除四个决定结果的输入：哪些回答被配成一对、谁把哪一个标为 chosen、用哪个参考策略衡量偏移，以及用什么 $\beta$ 缩放隐式奖励差。它的标准推导还把成对标签解释为对潜在奖励差的 logistic 选择。
>
> <!-- bilingual-en:start -->
> DPO folds “fit a reward model, then optimise a policy with online RL” into a direct preference loss. This removes components from the pipeline, not the assumptions that determine the result: pair construction, preference labels, the reference policy, and the scale parameter $\beta$. Its standard derivation also models a pairwise label as a logistic choice based on a latent reward difference.
> <!-- bilingual-en:end -->

## 四个依赖分别在回答什么
<!-- bilingual-en:start -->
*What each dependency decides*
<!-- bilingual-en:end -->

1. **偏好数据决定学习方向。** 数据只说明特定提示、候选回答与评估者条件下哪个回答被选中；[[偏好分数解释边界|这种标签不能自动解释成事实性、安全性或广泛用户效用]]。未出现的行为、评估者没识别出的事实错误，以及数据中系统性的长度或语气差异，都不会被公式自动纠正。
2. **参考策略定义“偏离”。** 同一当前策略配上不同的 $\pi_{\mathrm{ref}}$，会产生不同的对数概率比。参考策略不是只为数值稳定而存在；它参与定义 DPO 所比较的优势。
3. **$\beta$ 定义损失尺度与隐式奖励—策略关系。** 它不能脱离数据、优化过程和参考策略被解释为一个通用的“对齐强度”旋钮。
4. **成对模型压缩了偏好结构。** 标准推导使用 [[Bradley–Terry模型|Bradley–Terry 型概率]]，只由两个回答的标量奖励差决定选择概率。评估者异质性、平局、循环偏好和依上下文改变的标准若未显式建模，就会被压进同一个标签。

<!-- bilingual-en:start -->
1. **Preference data sets the direction.** A label concerns particular prompts, candidates, and evaluators. Unseen behaviour, unnoticed factual errors, and systematic length or style differences are not repaired by the loss.
2. **The reference policy defines deviation.** The same current policy produces different log ratios under a different reference. The reference is part of the comparison, not merely a numerical convenience.
3. **$\beta$ sets the loss scale and the implicit reward–policy relationship.** Its effect cannot be interpreted as a universal alignment-strength knob independently of data and optimisation.
4. **The pairwise model compresses preference structure.** The standard derivation uses a Bradley–Terry probability driven by a scalar reward difference. Heterogeneous evaluators, ties, cycles, and context-dependent criteria require additional modeling rather than disappearing into one label.
<!-- bilingual-en:end -->

## 离线配对还带来支持范围问题
<!-- bilingual-en:start -->
*Offline pairs also impose a support boundary*
<!-- bilingual-en:end -->

偏好对通常来自某个采样策略，而训练后的策略可能移动到另一种回答分布。数据若只比较短回答，就不能直接决定很长回答之间的排序；若 chosen 与 rejected 来自不同模型或模板，模型还可能学到来源痕迹。DPO 可以在固定数据上稳定优化，却不能从未覆盖的比较中恢复评估者真正会怎样选择。

因此，评价 DPO 不应只看训练 loss。至少要在独立提示和重新采样的回答上，分开检查任务质量、长度、事实性、安全与 KL 偏移；若部署分布会继续变化，还要收集覆盖新分布的偏好证据。这里的问题是证据范围，不是算法名字。

<!-- bilingual-en:start -->
Preference pairs are normally sampled from a particular policy, while the trained policy may move to a different response distribution. Pairs containing only short answers cannot identify preferences among much longer answers, and candidates produced by different templates or models may expose source artifacts. Stable optimisation on a fixed dataset cannot recover comparisons the data never made. Evaluation therefore needs newly sampled, held-out responses and separate measurements of task quality, length, factuality, safety, and KL shift.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两个团队在同一偏好数据上训练同一模型，但使用不同的参考策略。两者都把训练 loss 降得很低。能否据此断定它们学到了相同偏好，且部署行为会相同？
>
> <!-- bilingual-en:start -->
> Two teams train the same model on the same preference pairs but use different reference policies. Both obtain a very low training loss. Does this establish that they learned the same preference and will behave identically in deployment?
> <!-- bilingual-en:end -->
>
> **答案：** 不能。参考策略参与定义每个回答的相对对数概率优势；低训练 loss 只说明各自模型在各自参照下拟合了这些已观察配对。还要比较实际输出分布、独立偏好、质量维度和偏移范围。
>
> <!-- bilingual-en:start -->
> **Answer:** No. The reference policy helps define every relative log-probability advantage. Low loss only shows that each model fit the observed pairs under its own reference. Actual response distributions, held-out preferences, quality dimensions, and distributional shift still need comparison.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Rafailov et al. (2023), “Direct Preference Optimization: Your Language Model is Secretly a Reward Model”](https://papers.nips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)：核对参考策略、$\beta$、Bradley–Terry 偏好模型和 KL 正则推导。论文证明的是其建模设定下的等价重参数化；它并未使偏好标签成为无偏的人类效用测量。

<!-- bilingual-en:start -->
- Rafailov et al. (2023) was checked for the reference policy, $\beta$, Bradley–Terry preference model, and KL-regularised derivation. The paper establishes a reparameterisation under its modeling setup; it does not make preference labels unbiased measurements of human utility.
<!-- bilingual-en:end -->
