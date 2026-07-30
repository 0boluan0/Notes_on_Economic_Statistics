---
aliases:
  - "知识蒸馏"
  - "Distillation"
tags:
  - llm
  - llm/concept
  - concept
---

# Knowledge Distillation

Knowledge Distillation 是让小模型学习大模型输出或行为的压缩方法。

## 最小定义
教师模型产生软标签、推理轨迹或偏好数据，学生模型用更低成本拟合它。

## 最小例子
用强模型生成指令数据训练小模型，使小模型获得部分助手能力。

## 相关卡片
[[SFT Procedure]]、[[LLM Efficiency Engineering]]
## 符号表达

将本概念记为 $C_{KnowledgeDis}$；使用时先明确对象、条件与输出，再判断 $C$ 是否满足定义。
## 概念边界

本卡片只回答“它是什么”，集中在定义、核心关系与最小例子；具体估计步骤、证明和诊断流程应分别放在 procedure、proof 或 system 卡片中。
