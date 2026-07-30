---
aliases:
- 有效市场假说
- EMH
- Efficient Market Hypothesis
tags:
- 金融学
- 资产定价
- 金融
- concept
---

# Efficient Market Hypothesis

有效市场假说（Efficient Market Hypothesis, EMH）认为金融市场价格完全反映所有可用信息，投资者无法持续获得超额收益。

## 三种形式

### 1. 弱式有效（Weak-form Efficiency）

>[!note] 定义
> 价格完全反映所有历史价格和交易量信息。

**含义**：
- 技术分析无法获得超额收益
- 价格变化遵循[[Random Walk|随机游走]]

**检验方法**：
- 游程检验
- 序列相关检验
- 过滤规则检验

### 2. 半强式有效（Semi-strong-form Efficiency）

>[!note] 定义
> 价格完全反映所有公开信息（历史信息、财务报表、新闻公告等）。

**含义**：
- 基本面分析无法获得超额收益
- 公告事件（如分红、并购）的影响会迅速反映在价格上

**检验方法**：
- 事件研究法（Event Study）
- 检验超额收益

### 3. 强式有效（Strong-form Efficiency）

>[!note] 定义
> 价格完全反映所有信息（包括公开信息和内幕信息）。

**含义**：
- 即使拥有内幕信息也无法获得超额收益
- 市场价格是资产价值的完全反映

**检验方法**：
- 检验专业投资者是否获得超额收益
- 检验公司内部人交易

## 市场异象（对EMH的挑战）

1. **规模效应**：小公司股票收益高于大公司股票
2. **价值效应**：价值型股票收益高于成长型股票
3. **动量效应**：近期表现好的股票继续表现好
4. **反转效应**：极端表现股票倾向于反向变动
5. **日历效应**：一月效应、周末效应等

## 行为金融学解释

行为金融学认为投资者非理性行为导致市场不完全有效：
- **过度反应**：对新信息过度反应
- **反应不足**：对新信息反应不足
- **羊群效应**：跟随他人行为
- **损失厌恶**：对损失比收益更敏感

相关链接: [[Random Walk|随机游走]], [[CAPM|资本资产定价模型]], [[Behavioral Finance|行为金融学]]
]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Random Walk]]、[[CAPM]]、[[Behavioral Finance]]。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
