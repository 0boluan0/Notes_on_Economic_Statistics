---
aliases:
  - "Large Language Models Hub"
  - "LLM Hub"
status: source-checked
---

# 大语言模型 Hub

> [!summary] 这张 Hub 解决什么
> 论文笔记按论文保存，知识文件按可恢复的主题保存；本页只负责在两者之间导航。
> 先用 [[06_paper/LLM/LLM Big Picture.canvas|LLM Big Picture]] 看整体，再进入一个主题。需要追溯论文时，打开相应专题 Canvas 或 [[06_paper/LLM/LLM Map Index|论文索引]]。

## 主干

1. [[Transformer 与注意力机制]]：模型怎样在序列中读取信息，是后续训练、长上下文和多模态的共同计算骨架。
2. [[预训练、指令微调与偏好优化]]：基础模型怎样从续写分布变成可用的指令助手。
3. [[规模化、MoE 与分布式训练]]：训练预算怎样在参数、数据和设备间分配。
4. [[推理模型与 LLM Agent]]：模型怎样分解、验证并在真实环境中调用工具。
5. [[RAG 与长上下文]]：任务知识怎样在参数之外进入模型，并保持可更新和可追溯。
6. [[LLM 推理效率]]：训练完成后怎样控制首 token 延迟、吞吐和显存。
7. [[LLM 评测]]：怎样证明能力、可靠性和成本真的满足用途。
8. [[多模态大模型]]：视觉等模态怎样与语言表示对齐并共同生成。
9. [[开放模型生态]]：怎样从权重、数据、许可证、资源和任务效果选择可用模型。

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

Hub 不重复解释论文，也不为每个模型家族再建一层分类。稳定的主题结论进入上面的知识文件；单篇论文的方法、实验和限制留在 `06_paper/LLM/papers/`。
