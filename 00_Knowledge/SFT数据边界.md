---
aliases:
  - "SFT 的行为覆盖取决于示范质量、分布、对话模板和训练强度，增加样本量本身不保证改进"
  - "SFT behavior coverage depends on demonstration quality, distribution, conversation templates, and training intensity; increasing sample count alone does not guarantee improvement"
student_os: knowledge-atom
atom_id: LLM-PT-003
atom_set: llm-post-training
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[SFT目标]]"
related:
  - "[[开放模型生态]]"
  - "[[SFT损失掩码]]"
leads_to:
  - "[[后训练能力归因边界]]"
part_of:
  - "[[LLM 后训练.canvas|LLM 后训练]]"
---

# SFT 的行为覆盖取决于示范质量、分布、对话模板和训练强度，增加样本量本身不保证改进
<!-- bilingual-en:start -->
*SFT behavior coverage depends on demonstration quality, distribution, conversation templates, and training intensity; increasing sample count alone does not guarantee improvement*
<!-- bilingual-en:end -->

> [!summary] 数据量只是一条坐标
> SFT 优化的是示范数据分布上的平均 token loss。多加一条样本是否有用，取决于它是否扩展了目标任务和用户的覆盖，是否提供了可信的回答，是否与其他示范相容，以及模板和训练设置是否让模型学到了预期的回答边界。样本计数变大，只能证明优化器看到了更多记录，不能单独推出部署行为更好。
>
> <!-- bilingual-en:start -->
> SFT optimizes average token loss on its demonstration distribution. Whether an additional example helps depends on whether it expands coverage of target tasks and users, supplies a trustworthy response, remains compatible with other demonstrations, and is represented by templates and training settings that expose the intended answer boundary. A larger count proves only that the optimizer saw more records; it does not by itself establish better deployed behavior.
> <!-- bilingual-en:end -->

## 增加样本也在重加权训练分布
<!-- bilingual-en:start -->
*Adding examples also reweights the training distribution*
<!-- bilingual-en:end -->

假设原数据有 10,000 条，新增 100,000 条都是同一种简单问答的轻微改写。样本数扩大了，但新任务的覆盖几乎没变，这个狭窄区域反而在平均 loss 中获得更大权重。若新增数据含有事实错误、相互冲突的规则，或用表面线索就能猜中的答案，模型会忠实地拟合这些混合信号；规模不会自动完成去噪。

<!-- bilingual-en:start -->
Suppose a dataset of 10,000 examples receives 100,000 lightly rewritten versions of the same easy question type. The count grows, but task coverage barely changes, and the narrow region now receives much more weight in the average loss. If the added data contain factual errors, conflicting rules, or answers recoverable from superficial cues, the model fits that mixture; scale does not automatically denoise it.
<!-- bilingual-en:end -->

示范质量也不是单一标签。一个回答可以事实正确却与问题无关，可以语言流畅却隐去必要条件，也可以对一个基座模型很有教学价值，却对另一个已经掌握该行为的模型几乎重复。判断数据价值时，至少要把正确性、任务与用户覆盖、答案边界、风格一致性，以及相对基座模型的新信息分开看。

<!-- bilingual-en:start -->
Demonstration quality is not a single label. A response can be factually correct yet irrelevant, fluent yet omit a necessary condition, or highly informative for one base model but redundant for another that already exhibits the behavior. Data assessment should separate correctness, task and user coverage, answer boundaries, stylistic consistency, and marginal information relative to the base model.
<!-- bilingual-en:end -->

## 模板和训练强度改变模型实际面对的问题
<!-- bilingual-en:start -->
*Templates and training intensity change the effective learning problem*
<!-- bilingual-en:end -->

对话模板决定 system、user 和 assistant 文本如何分隔，回答从哪里开始、在哪里结束，以及部署提示与训练表示是否同构。若角色标记或结束 token 配错，原本正确的内容也可能被学成续写用户、泄露格式标记或不停在应停止处。这不是展示层的小问题，因为模板直接参与每个目标 token 的条件上下文。

<!-- bilingual-en:start -->
The conversation template determines how system, user, and assistant text are separated, where an answer begins and ends, and whether deployment prompts match the training representation. Incorrect role markers or end tokens can turn otherwise correct content into continuation of the user, exposed formatting markers, or failure to stop. This is not a cosmetic display issue because the template enters the conditioning context for every target token.
<!-- bilingual-en:end -->

学习率、更新步数、batch 组成和混合权重则决定同一份数据实际把参数推多远。训练太弱，目标行为可能尚未显现；训练过强或分布过窄，可能损害保留能力。因此比较两份 SFT 数据时，还要固定或报告基座模型、模板、优化预算和评测分布。只报“用了几条”不足以解释结果。

<!-- bilingual-en:start -->
Learning rate, update count, batch composition, and mixture weights determine how far the same dataset moves the parameters. Too little training may not expose the target behavior; excessive training or a narrow distribution can damage retained capabilities. A comparison of two SFT datasets must therefore control or report the base model, template, optimization budget, and evaluation distribution. The number of examples alone does not explain the result.
<!-- bilingual-en:end -->

> [!warning] LIMA 给出的是条件性证据
> LIMA 显示，在该论文的基座模型、精选示范、训练设置和评测下，少量高质量示范可以诱导出很强的对齐行为。它反驳了“高样本数必然是有效 SFT 的先决条件”，却不证明“小数据在任何模型、任务和评测上都优于大数据”。把这个结果写成普遍定律，正好会抹掉证据成立的条件。
>
> <!-- bilingual-en:start -->
> LIMA shows that, for its base model, curated demonstrations, training setup, and evaluations, a small high-quality set can induce strong aligned behavior. It challenges the claim that a very large example count is a necessary condition for effective SFT. It does not establish that small datasets universally outperform large ones across models, tasks, and evaluations. Turning the result into such a law would erase the conditions that make the evidence interpretable.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 团队把一份 SFT 数据扩大十倍，但新样本几乎全是原有简单问答的改写。训练 loss 更低，新领域测试却没有改善。这与本命题矛盾吗？
>
> <!-- bilingual-en:start -->
> A team expands an SFT dataset tenfold, mostly with paraphrases of existing easy questions. Training loss falls, but tests in new domains do not improve. Does this contradict the proposition of this atom?
> <!-- bilingual-en:end -->
>
> **答案：** 不矛盾。样本数增加了，任务分布和新领域覆盖却几乎没有扩展。更低的训练 loss 只说明模型更好地拟合了这个重复分布，不是它获得了新领域行为。
>
> <!-- bilingual-en:start -->
> **Answer:** No. The count increased while task-distribution and new-domain coverage barely changed. Lower training loss shows better fit to the repeated distribution, not acquisition of behavior in unseen domains.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Zhou et al. (2023), *LIMA: Less Is More for Alignment*](https://arxiv.org/abs/2305.11206)：提供“少量精选示范也能在特定设置下产生强对齐行为”的条件性实验；本卡不将其外推为小数据普遍更好。
- [Touvron et al. (2023), *Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/abs/2307.09288), §3.1：提供优先追求高质量 SFT 示范、并报告具体数据与训练设置的公开案例。
- [Ouyang et al. (2022), *Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155), §3.2 与 §4：提供 demonstration 数据、SFT 设置与保留能力评测的具体上下文。

<!-- bilingual-en:start -->
- Zhou et al. (2023) provides conditional experimental evidence that a small curated demonstration set can induce strong aligned behavior in the paper's setting; this atom does not extrapolate it into a universal small-data law.
- Touvron et al. (2023), §3.1, provides a public case that prioritizes high-quality SFT demonstrations while reporting the associated data and training setup.
- Ouyang et al. (2022), §3.2 and §4, provides a concrete context for demonstration data, SFT choices, and evaluation of retained capabilities.
<!-- bilingual-en:end -->
