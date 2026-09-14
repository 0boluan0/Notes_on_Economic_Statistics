---
aliases:
  - "SFT 在给定提示下最小化示范回答 token 的条件负对数似然，从而模仿目标回答而不是直接优化相对偏好"
  - "Given a prompt, SFT minimizes the conditional negative log-likelihood of demonstration-response tokens, thereby imitating a target response rather than directly optimizing relative preference"
student_os: knowledge-atom
atom_id: LLM-PT-002
atom_set: llm-post-training
atom_type: objective
status: source-checked
mastery_state: unassessed
requires:
  - "[[训练信号差异]]"
related:
  - "[[SFT损失掩码]]"
  - "[[Teacher forcing]]"
  - "[[Token损失均值不等价]]"
leads_to:
  - "[[SFT数据边界]]"
part_of:
  - "[[LLM 后训练.canvas|LLM 后训练]]"
---

# SFT 在给定提示下最小化示范回答 token 的条件负对数似然，从而模仿目标回答而不是直接优化相对偏好
<!-- bilingual-en:start -->
*Given a prompt, SFT minimizes the conditional negative log-likelihood of demonstration-response tokens, thereby imitating a target response rather than directly optimizing relative preference*
<!-- bilingual-en:end -->

> [!summary] SFT 的基本问题是“示范回答怎样写”
> 为了把样本长度权重写清，本卡对一个 batch $\mathcal B$ 中的提示 $x$、示范回答 $y^*=(y_1^*,\ldots,y_T^*)$ 和二值目标 mask $m_{x,y,t}$ 采用全局 token micro-average：
> $$
> \mathcal L_{\mathrm{SFT}}(\theta)
> =-\frac{\sum_{(x,y^*)\in\mathcal B}\sum_{t=1}^{T}
> m_{x,y,t}\log \pi_\theta(y_t^*\mid x,y_{<t}^*)}
> {\sum_{(x,y^*)\in\mathcal B}\sum_{t=1}^{T}m_{x,y,t}},
> $$
> 其中分母必须大于零。这是本卡为展示公式选定的全局 token micro-average，不是 SFT 的普遍定义；换成逐样本均值会改变长短回答权重，见 [[Token损失均值不等价]]。这个 loss 提高模型在给定提示和示范前缀时产生目标 token 的概率，但没有“这个回答比另一个好多少”这一项，所以不直接优化相对偏好。
>
> <!-- bilingual-en:start -->
> To make sample-length weighting explicit, this card adopts a global token micro-average over prompts $x$, demonstration responses $y^*$, and binary target masks $m_{x,y,t}$. This is an accounting convention for the displayed equation, not a universal definition of SFT; per-sample averaging changes sequence-length weights, as detailed in [[Token损失均值不等价|Token-loss means are not interchangeable]]. The loss raises demonstration-token likelihood but contains no term saying how much better one answer is than another, so it does not directly optimize relative preference.
> <!-- bilingual-en:end -->

## 模型学的是条件分布，不是一条抽象的“好答案”规则
<!-- bilingual-en:start -->
*The model learns a conditional distribution, not an abstract rule for good answers*
<!-- bilingual-en:end -->

训练时通常使用 [[Teacher forcing|teacher forcing]]：预测 $y_t^*$ 时，条件中放的是数据里真实的前缀 $y_{<t}^*$，而不是模型自己上一步采样的 token。因此，SFT 能把回答格式、语气、任务到输出的映射以及示范中包含的解题行为推向模型，但它没有单独鉴别示范里的哪一个特征真正造成了质量。

<!-- bilingual-en:start -->
Training commonly uses [[Teacher forcing|teacher forcing]]: when predicting $y_t^*$, the conditioning prefix is the ground-truth $y_{<t}^*$ from the demonstration rather than tokens sampled from the model's previous steps. SFT can therefore push answer format, tone, task-to-output mappings, and demonstrated solution behavior into the model. The objective does not separately identify which feature of the demonstration was responsible for its quality.
<!-- bilingual-en:end -->

“模仿”也不等于部署时逐字复读。参数在大量样本之间共享，生成时又会在新提示和模型自己的前缀上运行，所以模型可以组合、概括，也可能偏离示范。这里的“模仿”准确描述了训练目标，不是对最终行为的保证。

<!-- bilingual-en:start -->
“Imitation” does not mean verbatim repetition at deployment. Parameters are shared across many examples, and generation runs on new prompts and the model's own generated prefix, so the model can combine, generalize, or depart from demonstrations. Imitation precisely describes the training target; it is not a guarantee about final behavior.
<!-- bilingual-en:end -->

SFT 也可以有多个可接受参考、样本权重或不同目标区段；只要梯度来自指定示范 token 的似然，它仍属于监督模仿。相反，如果 loss 明确比较 chosen 与 rejected 回答的相对分数或相对概率，训练信号已经变为偏好比较，不能只因数据里也有“好回答”就称为普通 SFT。

<!-- bilingual-en:start -->
SFT may use multiple acceptable references, example weights, or different target spans; as long as the gradient comes from the likelihood of designated demonstration tokens, it remains supervised imitation. If the loss explicitly contrasts chosen and rejected responses through relative scores or probabilities, the signal has become preference comparison and should not be called ordinary SFT merely because a “good answer” is present.
<!-- bilingual-en:end -->

> [!warning] 目标位置取决于实现
> 许多对话 SFT 管线只让 assistant 回答 token 进入求和；用户提示提供条件，却不直接贡献目标 loss。但 assistant-only masking 是常见实现选择，不是 SFT 在所有代码库中的定义。有的管线会训练整段序列或选定的多个角色。具体差别见 [[SFT损失掩码]]；不能仅凭“做了 SFT”推断哪些位置直接收到梯度。
>
> <!-- bilingual-en:start -->
> Many conversational SFT pipelines include only assistant-response tokens in the sum: user tokens condition the response but do not directly contribute target loss. Assistant-only masking is a common implementation choice, not the definition of SFT across all codebases. Some pipelines train on the full sequence or on selected spans from several roles. See [[SFT损失掩码|SFT loss masking]] for the distinction; the label SFT alone does not reveal which positions directly receive gradient.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 一份数据对每个提示都保留 chosen 回答，丢弃 rejected 回答，然后只最小化 chosen token 的负对数似然。这个 loss 本身是否直接学到了“chosen 比 rejected 更好”？
>
> <!-- bilingual-en:start -->
> A dataset retains the chosen answer for each prompt, discards the rejected answer, and minimizes only the chosen tokens' negative log-likelihood. Does this loss directly learn that the chosen answer is better than the rejected one?
> <!-- bilingual-en:end -->
>
> **答案：** 没有。它把 chosen 当作示范做 SFT，只提高该回答的条件概率；rejected 既未进入 loss，也没有与 chosen 形成差值，所以相对偏好没有被直接优化。
>
> <!-- bilingual-en:start -->
> **Answer:** No. This is SFT on the chosen response as a demonstration. It raises that answer's conditional likelihood, but the rejected answer neither enters the loss nor forms a contrast with the chosen answer, so relative preference is not directly optimized.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Ouyang et al. (2022), *Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155), §3.2 与 Appendix C：对应在人类 demonstrations 上对 GPT-3 做监督微调的目标与工程设置。
- [Touvron et al. (2023), *Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/abs/2307.09288), §3.1：对应高质量指令数据上的自回归 SFT 目标；该论文是一个具体管线实例，不替代对任意实现 label mask 的核对。

<!-- bilingual-en:start -->
- Ouyang et al. (2022), §3.2 and Appendix C, documents supervised fine-tuning of GPT-3 on human demonstrations and its training setup.
- Touvron et al. (2023), §3.1, documents autoregressive SFT on high-quality instruction data. It is one concrete pipeline and does not substitute for inspecting the label mask in an arbitrary implementation.
<!-- bilingual-en:end -->
