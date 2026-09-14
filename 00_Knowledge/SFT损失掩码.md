---
aliases:
  - "SFT 的损失掩码决定哪些 token 直接贡献训练梯度，assistant-only masking 不等于 causal mask 或 padding mask"
  - "An SFT loss mask determines which tokens directly contribute training loss; assistant-only masking is not the causal mask or the padding mask"
student_os: knowledge-atom
atom_id: LLM-PT-014
atom_set: llm-post-training
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[SFT目标]]"
related:
  - "[[SFT数据边界]]"
  - "[[Transformer自回归约束]]"
  - "[[Token损失均值不等价]]"
part_of:
  - "[[LLM 后训练.canvas|LLM 后训练]]"
---

# SFT 的损失掩码决定哪些 token 直接贡献训练梯度，assistant-only masking 不等于 causal mask 或 padding mask
<!-- bilingual-en:start -->
*An SFT loss mask determines which tokens directly contribute training loss; assistant-only masking is not the causal mask or the padding mask*
<!-- bilingual-en:end -->

> [!summary] 三种 mask 回答三个不同问题
> **Loss mask（损失掩码）**问：哪些位置的预测误差进入目标函数？**Causal mask（因果注意力掩码）**问：当前位置能否在注意力中看到未来 token？**Padding mask（填充掩码）**问：批处理中为了补齐长度而加入的虚拟位置是否应被当成真实上下文？assistant-only masking 只是在第一问中选择 assistant 目标位置，不能代替后两种结构。
>
> <!-- bilingual-en:start -->
> A **loss mask** asks which positions' prediction errors enter the objective. A **causal attention mask** asks whether a position may attend to future tokens. A **padding mask** asks whether dummy positions added to equalize sequence lengths should be treated as real context. Assistant-only masking selects assistant targets for the first question; it cannot replace the other two structures.
> <!-- bilingual-en:end -->

## Loss mask 选择目标项，不会删除提示上下文
<!-- bilingual-en:start -->
*A loss mask selects objective terms; it does not remove the prompt context*
<!-- bilingual-en:end -->

把 batch 中第 $i$ 条目标序列记作 $s_{i,1:n_i}$。[[Transformer自回归约束|训练输入与目标错开一位]]：与目标 $s_{i,t}$ 对齐的 logit 读取 $(\mathrm{BOS},s_{i,1},\ldots,s_{i,t-1})$，因此每个目标位置的自回归负对数似然为
$$
\ell_{i,t}(\theta)=-\log p_\theta(s_{i,t}\mid s_{i,<t}).
$$
给每个位置一个损失权重 $m_{i,t}^{\mathrm{loss}}\ge 0$。为把权重口径写清，本卡把 batch-level 加权归一化定义为
$$
\mathcal L(\theta)
=\frac{\sum_i\sum_t m_{i,t}^{\mathrm{loss}}\ell_{i,t}(\theta)}
{\sum_i\sum_t m_{i,t}^{\mathrm{loss}}}.
$$
并要求 $\sum_{i,t}m_{i,t}^{\mathrm{loss}}>0$。这是本卡用来解释 loss mask 的加权归一化；全零 batch 不定义该损失，应跳过或在预处理中剔除。当 $m_{i,t}^{\mathrm{loss}}\in\{0,1\}$ 时，上式是有效 token 的 micro-average；改为逐样本均值会另行改变长度权重，见 [[Token损失均值不等价]]。
当用户提示位置的 $m_{i,t}^{\mathrm{loss}}=0$、assistant 回答位置为 1 时，提示 token 自己的“下一个 token 是否预测正确”不形成独立 loss 项；回答 token 的预测误差才直接进入求和。这就是常说的 assistant-only loss。

<!-- bilingual-en:start -->
For batch item $i$, let the target sequence be $s_{i,1:n_i}$. [[Transformer自回归约束|Inputs and targets are shifted]] so the logit aligned with $s_{i,t}$ reads only the earlier sequence. This card uses the displayed weighted normalization to explain loss masking; an all-zero batch does not define it and should be skipped or prevented in preprocessing. With a binary mask, the equation is a valid-token micro-average; per-sample averaging changes length weights separately, as detailed in [[Token损失均值不等价|Token-loss means are not interchangeable]]. Setting prompt positions to zero and assistant-response positions to one means that prompt-token prediction errors supply no separate loss terms, while answer-token errors enter the sum directly. This is assistant-only loss.
<!-- bilingual-en:end -->

被 loss mask 掉不等于从计算图消失。提示 token 仍是后续回答的条件，回答位置的 loss 可以通过注意力和共享参数反向传播到处理提示的计算。准确说法是“提示位置没有自己的目标损失”，而不是“提示对梯度或学习完全没有影响”。

<!-- bilingual-en:start -->
Being loss-masked does not remove a token from the computation graph. Prompt tokens still condition later answers, and answer losses can backpropagate through attention and shared parameters involved in processing the prompt. The precise statement is that prompt positions have no target loss of their own, not that prompts have no effect on gradients or learning.
<!-- bilingual-en:end -->

## Causal mask 限制信息方向
<!-- bilingual-en:start -->
*The causal mask restricts information flow*
<!-- bilingual-en:end -->

沿用 [[Transformer自回归约束|输入—标签错位与 causal visibility]] 的约定，与目标 $s_{i,t}$ 对齐的 logit 只能使用移位输入中的 $(\mathrm{BOS},s_{i,<t})$。这个信息约束通常同时作用于提示和回答位置，与某个位置是否计 loss 是两回事：

- 用户提示可以不计 loss，却仍被后面的 assistant token 看见；
- assistant token 可以计 loss，却仍不能看见其后的答案 token；
- 即使 loss mask 完全正确，若 causal mask 允许看未来，训练目标仍会发生标签泄漏。

<!-- bilingual-en:start -->
Under the shift convention above, the logit aligned with target $s_{i,t}$ may read only $(\mathrm{BOS},s_{i,<t})$. The input–label shift keeps $s_{i,t}$ out of its own logit position, while the causal mask blocks that input position from reading positions to its right. Together they implement $p_\theta(s_{i,t}\mid s_{i,<t})$; the causal mask does not do so alone. It normally applies across prompt and response positions independently of which positions carry loss. A prompt can be loss-masked yet visible to later assistant tokens; a loss-bearing assistant token still cannot see later answer tokens; and a correct loss mask cannot prevent target leakage if the shift or causal visibility is wrong.
<!-- bilingual-en:end -->

## Padding mask 排除批处理中的虚拟位置
<!-- bilingual-en:start -->
*The padding mask excludes dummy batch positions*
<!-- bilingual-en:end -->

同一 batch 内的样本长度不同，系统常在较短样本末尾补 PAD token。Padding attention mask 用来阻止真实 token 把这些占位符当作上下文；训练标签还通常用 ignore index 或相应 loss 权重，使 PAD 位置不贡献目标损失。两步经常一起配置，所以在代码里容易被统称为“mask”，但概念上仍不同：padding mask 区分真实位置与补齐位置，loss mask 可以进一步在真实位置中区分 user、assistant 或其他目标区段。

<!-- bilingual-en:start -->
Sequences of unequal length are commonly padded within a batch. A padding attention mask prevents real tokens from treating those placeholders as context; labels also commonly use an ignore index or loss weight so PAD positions contribute no target loss. These settings are often configured together and casually called a mask, but they remain conceptually distinct: padding separates real positions from batch fillers, while a loss mask can further select user, assistant, or other target spans among real positions.
<!-- bilingual-en:end -->

实现名称不能替代语义核对。有的库把二维有效位置向量叫 attention mask，内部再与 causal mask 合并；有的接口分别接收 key padding mask 和 attention mask；loss mask 又可能藏在 labels 的 ignore value 中。检查代码时要沿着张量最后进入注意力分数还是 token loss 来判断，而不是只看变量名。

<!-- bilingual-en:start -->
Implementation names do not replace semantic inspection. Some libraries call a two-dimensional valid-position vector an attention mask and combine it internally with a causal mask; others accept a key-padding mask and an attention mask separately; the loss mask may be encoded as an ignored label value. Trace whether the tensor ultimately modifies attention scores or token losses rather than relying on its variable name.
<!-- bilingual-en:end -->

> [!warning] Assistant-only 是常见选择，不是 SFT 定义
> Llama 2 的公开 SFT 实现把 user prompt 的 loss 置零，只在 answer token 上反向传播；这是 assistant-only 的明确实例。它不能推出所有 SFT 都应如此。关于 prompt loss weight 的实验发现，给提示 token 非零权重的效果会随回答长度和评测任务改变。由此能得到的稳健结论是“loss mask 是需要报告和验证的训练设定”，而不是“全 mask”或“全不 mask”无条件最优。
>
> <!-- bilingual-en:start -->
> Llama 2 explicitly zeroes the loss on user-prompt tokens and backpropagates only on answer tokens, providing a clear assistant-only example. It does not establish that every SFT pipeline must do the same. Experiments on prompt-loss weights find that the effect of nonzero prompt loss changes with completion length and evaluation task. The defensible conclusion is that the loss mask is a training choice to report and validate, not that either complete prompt masking or no masking is universally optimal.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 一条训练样本的 user token 不计 loss，但 assistant token 仍可看见全部 user prompt，同时每个位置都看不到未来答案，PAD 位置也不参与上下文。这里分别是哪三种 mask 在起作用？
>
> <!-- bilingual-en:start -->
> In one training example, user tokens carry no loss, assistant tokens can still see the full user prompt, no position can see future answer tokens, and PAD positions do not enter the context. Which three masks implement these effects?
> <!-- bilingual-en:end -->
>
> **答案：** Loss mask 把 user 目标项置零并保留 assistant 目标项；causal mask 阻止看未来；padding mask 排除 PAD 上下文。User token 仍可被 assistant 看见，正说明 loss mask 没有承担 causal 或 padding mask 的工作。
>
> <!-- bilingual-en:start -->
> **Answer:** The loss mask zeros user target terms and retains assistant targets; the causal mask blocks future attention; the padding mask excludes PAD context. The fact that assistant tokens can still see user tokens shows precisely that the loss mask is not doing the work of the causal or padding mask.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Touvron et al. (2023), *Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/html/2307.09288v2#S3.SS1), §3.1 “Fine-Tuning Details”：明确说明把 user prompt token 的 loss 置零，只在 answer token 上反向传播；它支持“常见实例”，不支持“普遍定义”。
- [Huerta-Enochian and Ko (2024), *Instruction Fine-Tuning: Does Prompt Loss Matter?*](https://aclanthology.org/2024.emnlp-main.1267/), EMNLP 2024：直接比较不同 prompt loss token weights，报告效果随短/长 completion 和评测类型变化，为“mask 选择需按设置验证”提供条件性实证。
- [Vaswani et al. (2017), *Attention Is All You Need*](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need), §3.2.3：核对 decoder causal mask 通过阻止当前位置关注后续位置来保持自回归性质。
- [PyTorch, *MultiheadAttention* reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)：直接核对 key padding mask 与一般 attention mask 在实现接口中的不同职责；具体库仍须阅读其版本文档与代码。

<!-- bilingual-en:start -->
- Touvron et al. (2023), §3.1 “Fine-Tuning Details,” explicitly zeroes user-prompt loss and backpropagates only on answer tokens. It supports a common example, not a universal definition.
- Huerta-Enochian and Ko (2024) directly compares prompt-loss token weights and reports effects that vary with completion length and evaluation type, providing conditional evidence that masking choices require validation.
- Vaswani et al. (2017), §3.2.3, establishes the decoder causal mask as the mechanism preventing attention to subsequent positions.
- The PyTorch MultiheadAttention reference distinguishes a key-padding mask from a general attention mask at the implementation interface; the exact library version and code must still be inspected.
<!-- bilingual-en:end -->
