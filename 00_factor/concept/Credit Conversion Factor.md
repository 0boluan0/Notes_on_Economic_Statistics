---
aliases:
- Credit Conversion Factor
- CCF
- 信用转换系数
- 信用风险转换系数
tags:
- concept
- credit-risk
- regulation
---

# Credit Conversion Factor

## 先记一句话

Credit Conversion Factor 把表外承诺或衍生品名义金额转换成监管认可的信用等价敞口。

## 它是什么

CCF 表示表外项目在违约时可能转化为真实信用暴露的比例。常见形式：

$$
EAD=\text{drawn exposure}+\text{undrawn commitment}\times CCF
$$

## 解决什么判断

它回答：“这笔表外业务不能直接算 RWA，那应该先折成多少信用敞口？”

## 最小例子

未提款授信额度 1 亿，CCF=20%，则转换出的表外信用暴露为 2000 万。

## 易混点

- CCF 不是风险权重；先用 CCF 转成 [[EAD]] 或信用等价额，再乘风险权重得到 [[Risk-Weighted Assets]]。
- 衍生品暴露还要考虑当前暴露、潜在未来暴露和 [[Netting]]。
- CCF 不应被拆成 Credit + Factor Analysis；它是一个固定监管术语。

## 来自课程位置

- [[15_《巴塞尔协议I II》和 偿付能力法案II]]

## 关联卡片

- [[EAD]]
- [[Risk-Weighted Assets]]
- [[Basel Accords]]
- [[Credit Risk]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[EAD]]、[[Risk-Weighted Assets]]、[[Netting]]、[[15_《巴塞尔协议I II》和 偿付能力法案II]]、[[Basel Accords]]、[[Credit Risk]]。
