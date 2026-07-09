---
aliases:
  - "SFT流程"
  - "Supervised Fine-Tuning Procedure"
tags:
  - llm
  - llm/procedure
---

# SFT Procedure

## Step 1: 准备指令数据
收集 prompt-response 格式的高质量任务样本。

## Step 2: 格式化对话模板
统一 system、user、assistant 等角色格式。

## Step 3: 监督微调
用标准语言建模损失训练模型模仿目标回答。

## Step 4: 检查泛化和风格
评估是否过拟合模板、是否损伤基础能力。

## 输出
一个会按指令格式回答的模型。

## 相关卡片
[[Instruction Tuning and RLHF]]、[[Pretraining Paradigm]]
