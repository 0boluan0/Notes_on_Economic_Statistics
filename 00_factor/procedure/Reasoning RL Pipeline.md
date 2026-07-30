---
aliases:
  - "推理RL流程"
  - "Reasoning Reinforcement Learning Pipeline"
tags:
  - llm
  - llm/procedure
type: procedure
---

# Reasoning RL Pipeline

## 输入

- 有可靠答案或验证器的数学、代码、逻辑等任务。
- 初始模型、采样策略、奖励/验证器和推理质量评测集。

## Step 1: 定义可验证任务
选择数学、代码、逻辑等能判断结果正确性的任务。

## Step 2: 采样推理轨迹
让模型生成多个中间推理过程和答案。

## Step 3: 用结果或验证器给奖励
根据最终正确性、过程质量或规则反馈计算奖励。

## Step 4: 强化学习更新
提高能产生正确推理行为的轨迹概率。

## 检查点

- 验证器必须难以被格式、长度或奖励漏洞欺骗，并单独抽样人工复核。
- 比较直接回答、搜索/采样和 RL 后模型；监控正确率、推理长度、成本与能力迁移。

## 输出
一个更倾向于展开推理、检查和修正的模型。

## 相关卡片
[[Reasoning Models]]、[[Chain-of-Thought]]
