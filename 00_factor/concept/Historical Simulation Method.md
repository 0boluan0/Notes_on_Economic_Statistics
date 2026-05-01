---
aliases:
- Historical Simulation Method
- Historical Simulation
- 历史模拟法
tags:
- concept
- 金融风险
- VaR
---
# Historical Simulation Method

## 先记一句话

历史模拟法就是：**把过去每一天的市场变动当成未来可能情景，重新计算组合损失分布**。

它不先假设正态分布。

## 它是什么

基本做法：

1. 收集过去 $N$ 天风险因子变化；
2. 把每一天的变化套到今天的组合上；
3. 得到 $N$ 个模拟损益；
4. 按损失排序；
5. 取对应分位数作为 VaR。

具体步骤见 [[Historical Simulation VaR]]。

## 它解决什么判断

适合：

- 不想假设正态分布；
- 希望保留历史厚尾、偏度、相关结构；
- 组合可以在每个历史情景下重新定价。

## 主要边界

- 强依赖历史窗口，历史没有出现过的风险就看不到。
- 高置信水平需要很长样本。
- 市场结构变化时，“历史重演”假设会失效。

## 常见误区

- 历史模拟不是没有模型；它的模型就是“历史情景代表未来”。
- 非线性组合要重新定价，不能只用线性收益加权。
- 样本太短时，99% VaR 只由少数几个尾部观测决定。

## 来自课程位置

- [[12_VAR风险#2.2 历史模拟法|金融风险管理 12：历史模拟法]]
- [[13_历史模拟法和极值理论|金融风险管理 13：历史模拟和 EVT]]

## 关联卡片

- [[VaR]]
- [[Historical Simulation VaR]]
- [[Weighted Historical Simulation]]
- [[Bootstrap Simulation]]
- [[EVT]]

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
