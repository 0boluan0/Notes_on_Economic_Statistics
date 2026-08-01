---
aliases:
  - "Large Language Models Hub"
  - "LLM Hub"
status: source-checked
---

# 大语言模型 Hub
<!-- bilingual-en:start -->
*Large Language Models Hub*
<!-- bilingual-en:end -->

> [!summary] 这张 Hub 解决什么
> 论文笔记按论文保存，知识文件按可恢复的主题保存；本页只负责在两者之间导航。
> 先用 [[06_paper/LLM/LLM Big Picture.canvas|LLM Big Picture]] 看整体，再进入一个主题。需要追溯论文时，打开相应专题 Canvas 或 [[06_paper/LLM/LLM Map Index|论文索引]]。
> <!-- bilingual-en:start -->
> Paper notes are organised by paper, while Knowledge files are organised as recoverable topics. This page only connects the two layers.
> Begin with [[06_paper/LLM/LLM Big Picture.canvas|LLM Big Picture]], then enter one topic below. To trace a claim back to papers, open the relevant thematic Canvas or the [[06_paper/LLM/LLM Map Index|Paper Index]].
> <!-- bilingual-en:end -->

## 主干
<!-- bilingual-en:start -->
*Backbone*
<!-- bilingual-en:end -->

1. [[Transformer 与注意力机制]]：模型怎样在序列中读取信息，是后续训练、长上下文和多模态的共同计算骨架。
2. [[LLM 预训练]] → [[LLM 后训练：SFT、RLHF 与 DPO]]：先学习续写分布中的通用能力，再把这些能力调成可用、可控的助手行为。
3. [[Scaling laws 与计算最优训练]] → [[Mixture of Experts（MoE）]] / [[大模型分布式训练]]：先决定参数、数据与计算的预算，再分别理解条件计算和跨设备执行。
4. [[LLM 推理与验证]] → [[LLM Agent 与工具调用]]：先理解模型怎样分解并验证答案，再看它怎样把推理落实为环境中的多步行动。
5. [[RAG（检索增强生成）]] ↔ [[长上下文语言模型]]：两者都把参数外知识交给模型，但一个强调检索、更新与溯源，另一个强调一次容纳和利用更长输入。
6. [[LLM 推理效率]]：训练完成后怎样控制首 token 延迟、吞吐和显存。
7. [[LLM 评测]]：怎样证明能力、可靠性和成本真的满足用途。
8. [[多模态大模型]]：视觉等模态怎样与语言表示对齐并共同生成。
9. [[开放模型生态]]：怎样从权重、数据、许可证、资源和任务效果选择可用模型。
<!-- bilingual-en:start -->

&nbsp;
**1.** [[Transformer 与注意力机制|Transformers and Attention]]: the shared computational backbone through which models read sequences, supporting later training, long context, and multimodality.<br>
**2.** [[LLM 预训练|LLM Pretraining]] → [[LLM 后训练：SFT、RLHF 与 DPO|LLM Post-Training: SFT, RLHF, and DPO]]: first learn general capabilities from the continuation distribution, then shape those capabilities into useful and controllable assistant behaviour.<br>
**3.** [[Scaling laws 与计算最优训练|Scaling Laws and Compute-Optimal Training]] → [[Mixture of Experts（MoE）|Mixture of Experts (MoE)]] / [[大模型分布式训练|Distributed Training for Large Models]]: first allocate the parameter, data, and compute budget, then understand conditional computation and execution across devices.<br>
**4.** [[LLM 推理与验证|LLM Reasoning and Verification]] → [[LLM Agent 与工具调用|LLM Agents and Tool Use]]: first understand how a model decomposes and verifies an answer, then how reasoning becomes a sequence of actions in an external environment.<br>
**5.** [[RAG（检索增强生成）|Retrieval-Augmented Generation (RAG)]] ↔ [[长上下文语言模型|Long-Context Language Models]]: both provide knowledge outside the parameters, but RAG emphasizes retrieval, updating, and provenance, while long context emphasizes holding and using more input at once.<br>
**6.** [[LLM 推理效率|LLM Inference Efficiency]]: controlling time to first token, throughput, and memory after training.<br>
**7.** [[LLM 评测|LLM Evaluation]]: establishing whether capability, reliability, and cost actually satisfy the intended use.<br>
**8.** [[多模态大模型|Multimodal Large Models]]: aligning visual and other modalities with language representations for joint generation.<br>
**9.** [[开放模型生态|Open Model Ecosystems]]: choosing a usable model from its weights, data, licence, resource requirements, and task performance.<br>
<!-- bilingual-en:end -->

## 论文专题入口

- [[06_paper/LLM/canvas/01 架构与预训练.canvas|架构与预训练]]
- [[06_paper/LLM/canvas/02 规模化与模型家族.canvas|规模化与模型家族]]
- [[06_paper/LLM/canvas/03 对齐与偏好优化.canvas|对齐与偏好优化]]
- [[06_paper/LLM/canvas/04 推理与Agent.canvas|推理与 Agent]]
- [[06_paper/LLM/canvas/05 RAG与知识增强.canvas|RAG 与知识增强]]
- [[06_paper/LLM/canvas/06 长上下文与效率.canvas|长上下文与效率]]
- [[06_paper/LLM/canvas/07 评测与数据.canvas|评测与数据]]
- [[06_paper/LLM/canvas/08 多模态.canvas|多模态]]
- [[06_paper/LLM/canvas/09 开源模型技术报告.canvas|开放模型技术报告]]

## 边界
<!-- bilingual-en:start -->
*Boundary*
<!-- bilingual-en:end -->

Hub 不重复解释论文，也不为每个模型家族再建一层分类。稳定的主题结论进入上面的知识文件；单篇论文的方法、实验和限制留在 `06_paper/LLM/papers/`。
<!-- bilingual-en:start -->
This Hub does not repeat paper explanations or add another classification layer for every model family. Stable thematic conclusions belong in the Knowledge files above; the methods, experiments, and limitations of individual papers remain in `06_paper/LLM/papers/`.
<!-- bilingual-en:end -->
