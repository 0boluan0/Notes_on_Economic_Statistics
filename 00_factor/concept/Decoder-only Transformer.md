---
aliases:
- Decoder-only Transformer
- Decoder-only
- 仅解码器Transformer
tags:
- llm
- llm/concept
- concept
---

# Decoder-only Transformer

Decoder-only Transformer 是只使用自回归解码器、按 next-token prediction 训练和生成的 Transformer。

## 最小定义
模型只能看当前位置之前的 token，并预测下一个 token。

## 最小例子
GPT 系列把任意任务转成“给定上下文继续写”的问题。

## 相关卡片
[[Next-token Prediction]]、[[Pretraining Paradigm]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Next-token Prediction]]、[[Pretraining Paradigm]]。
