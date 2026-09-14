---
aliases:
  - "在并行训练 decoder-only Transformer 时，输入—标签错位排除当前目标，因果掩码排除右侧输入，二者共同实现只依赖历史的 next-token 预测"
  - When a decoder-only Transformer is trained in parallel, input-label shifting excludes the current target and causal masking excludes rightward inputs, jointly enforcing history-only next-token prediction
student_os: knowledge-atom
atom_id: LLM-PRE-015
atom_set: llm-pretraining
atom_type: mechanism
status: source-checked
mastery_state: unassessed
requires:
  - "[[自回归目标]]"
related:
  - "[[Transformer]]"
  - "[[Teacher forcing]]"
  - "[[SFT损失掩码]]"
part_of:
  - "[[LLM 预训练.canvas|LLM 预训练]]"
---

# 在并行训练 decoder-only Transformer 时，输入—标签错位排除当前目标，因果掩码排除右侧输入，二者共同实现只依赖历史的 next-token 预测

<!-- bilingual-en:start -->
*When a decoder-only Transformer is trained in parallel, input-label shifting excludes the current target and causal masking excludes rightward inputs, jointly enforcing history-only next-token prediction*
<!-- bilingual-en:end -->

> [!summary] 并行 Transformer 需要两道约束
> 在 decoder-only Transformer 的并行训练中，若某个 logit 以 $x_t$ 为标签，它只能依赖 $x_{<t}$。输入—标签错位先让这个 logit 对齐到 $(\mathrm{BOS},x_1,\ldots,x_{t-1})$；causal mask 再阻止该位置读取右侧输入。只做其中一步，不能完整保证
> $$
> p_\theta(x_t\mid x_{<t}).
> $$
>
> <!-- bilingual-en:start -->
> During parallel training of a decoder-only Transformer, a logit labelled with $x_t$ may depend only on $x_{<t}$. Input-label shifting aligns that logit with $(\mathrm{BOS},x_1,\ldots,x_{t-1})$, while the causal mask blocks inputs to its right. Neither mechanism alone fully enforces $p_\theta(x_t\mid x_{<t})$ in this implementation.
> <!-- bilingual-en:end -->

## 张量位置为何容易看错

常见实现把同一 token 张量同时传为 `input_ids` 和 `labels`，然后在模型内部用位置 $j$ 的 logit 预测标签位置 $j+1$。因此“两个参数张量相同”不等于没有错位；真正要核对的是哪个 logit 与哪个标签进入 loss。

若位置 $j$ 的输入是 $x_j$，它读取自身并不泄漏，只要该 logit 的标签是 $x_{j+1}$。相反，若同一 logit 直接以 $x_j$ 为标签，目标 token 已进入条件，即使注意力仍是下三角，训练也接近 $p_\theta(x_j\mid x_{\le j})$，loss 可以异常低而没有学会从过去预测未来。

<!-- bilingual-en:start -->
Many libraries accept identical `input_ids` and `labels` tensors but shift the logits and labels internally, so tensor equality does not imply missing alignment. Reading input token $x_j$ at position $j$ is safe when that logit predicts $x_{j+1}$. Labelling the same logit with $x_j$ leaks the target even under a lower-triangular attention mask.
<!-- bilingual-en:end -->

## 不要把三种 mask 合并

Causal mask 限制可见上下文；loss mask 决定哪些预测误差进入目标；padding mask 排除补齐位置。它们可能在库内部合并成一个张量，但语义职责仍不同。SFT 中更详细的损失选择见 [[SFT损失掩码]]。

> [!question]- 自检
> 一个 decoder-only Transformer 训练脚本让位置 $t$ 的输入和标签都是 $x_t$，同一位置的 logit 直接计算该标签的交叉熵，并只屏蔽右侧注意力。它实现了 next-token 目标吗？
>
> **答案：** 没有。目标 $x_t$ 已经进入产生该 logit 的输入条件；必须重新核对 logit—label 对齐，而不是只看 mask 的形状。

## 来源与核验

- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)，§3.1、§3.2.3：核对输出错位与 decoder causal mask 的不同职责。
- [Hugging Face Transformers, causal language model labels](https://huggingface.co/docs/transformers/model_doc/codegen)：核对常见接口在模型内部移动 labels，因而允许调用方传入相同的 `input_ids` 与 `labels`。
