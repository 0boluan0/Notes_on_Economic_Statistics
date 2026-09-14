---
aliases:
  - "Teacher forcing 在训练时用真实历史而不是模型生成历史作为下一 token 的条件"
  - Teacher forcing conditions each training-time next-token prediction on the ground-truth history rather than model-generated history
student_os: knowledge-atom
atom_id: LLM-PRE-002
atom_set: llm-pretraining
atom_type: definition
status: source-checked
mastery_state: unassessed
related:
  - "[[自回归目标]]"
  - "[[Transformer自回归约束]]"
  - "[[SFT目标]]"
leads_to:
  - "[[暴露偏差]]"
part_of:
  - "[[LLM 预训练.canvas|LLM 预训练]]"
---

# Teacher forcing 在训练时用真实历史而不是模型生成历史作为下一 token 的条件

<!-- bilingual-en:start -->
*Teacher forcing conditions each training-time next-token prediction on the ground-truth history rather than model-generated history*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 预测训练序列中的 $x_t$ 时，teacher forcing 使用数据里的真实前缀 $x_{<t}$，而不是把模型先前生成的 $\hat x_{<t}$ 反馈为条件。它规定的是**条件历史来自哪里**，不是损失函数，也不是注意力 mask。
>
> <!-- bilingual-en:start -->
> When predicting training token $x_t$, teacher forcing conditions on the observed prefix $x_{<t}$ rather than feeding back a model-generated prefix $\hat x_{<t}$. It specifies the source of the conditioning history, not the loss function or attention mask.
> <!-- bilingual-en:end -->

## 训练与自由生成的前缀不同

对真实序列 $(x_1,\ldots,x_T)$，训练时每一步都能使用已观察到的前缀。即使模型在预测 $x_t$ 时把另一个 token 排在最高概率，下一训练位置仍使用真实的 $x_t$。自由生成则没有真实未来输出：模型一旦生成 $\hat x_t$，下一步就必须以包含 $\hat x_t$ 的模型历史为条件。

在 Transformer 中，整段真实训练序列预先已知，所以配合正确的 [[Transformer自回归约束|输入—标签错位与 causal mask]]，各位置可在一次前向计算中并行求 loss。并行性来自 Transformer 计算结构与已知前缀的组合，不是 teacher forcing 对所有序列模型的普遍加速保证。

<!-- bilingual-en:start -->
During training, every step can use the observed prefix. A wrong highest-probability token is not written back into the next training condition. During free generation, later steps must condition on tokens the model actually produced. In a Transformer, known target sequences and correct autoregressive information constraints permit parallel position-wise loss computation; teacher forcing alone does not make every sequence architecture parallel.
<!-- bilingual-en:end -->

## 最小例子

训练片段是 “France's capital is Paris .”。预测 “Paris” 时，模型也许把 “Lyon” 排在第一，但预测句点时仍看见真实的 “Paris”。自由生成若真的输出 “Lyon”，下一步看见的就是 “Lyon”。由此产生的条件分布差及其边界属于 [[暴露偏差]]。

Teacher forcing 可以和不同 loss 配合。序列回归可在真实历史上用均方误差；scheduled sampling 也可保留 next-token 交叉熵，却把部分真实历史换成模型历史。因此，改变历史来源不等于改变 loss 或 mask。

> [!question]- 自检
> 一个 decoder 保持 next-token NLL 与 causal mask 不变，只把部分真实上一 token 换成模型生成 token。哪一层发生了变化？
>
> **答案：** 条件历史来源发生了变化，从纯 teacher forcing 转为混合真实历史与模型历史；loss 和可见性约束可以保持不变。

## 来源与核验

- [Williams and Zipser (1989), *A Learning Algorithm for Continually Running Fully Recurrent Neural Networks*](https://doi.org/10.1162/neco.1989.1.2.270)，§2.3：核对 teacher forcing 的原始术语与定义。
- [Bengio et al. (2015), *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*](https://arxiv.org/abs/1506.03099)：核对训练真实历史与推理模型历史的协议差异。
