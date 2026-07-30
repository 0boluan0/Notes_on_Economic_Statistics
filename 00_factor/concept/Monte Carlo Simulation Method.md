---
aliases:
- Monte Carlo Simulation Method
- Monte Carlo Simulation
- Monte Carlo VaR
- 蒙特卡罗模拟方法
- 蒙特卡罗模拟法
tags:
  - concept
  - 金融风险
  - VaR
---
# Monte Carlo Simulation Method

## 先记一句话

蒙特卡罗模拟法就是：**先设定风险因子的随机模型，再大量生成未来情景，重估组合损益分布**。

它最灵活，也最依赖模型。

## 它是什么

基本做法：

1. 设定风险因子分布和相关结构；
2. 生成大量随机情景；
3. 每个情景下重新定价组合；
4. 得到模拟损失分布；
5. 取分位数作为 VaR。

具体步骤见 [[Monte Carlo Simulation VaR]]。

## 它解决什么判断

适合：

- 非线性衍生品；
- 路径依赖产品；
- 高维风险因子；
- 历史中没有出现过但模型能生成的情景。

## 主要边界

- 结果高度依赖模型和参数。
- 计算量大。
- 相关性、波动率、尾部分布假设错误会直接传导到 VaR。

## 常见误区

- 模拟次数多不等于模型正确。
- Monte Carlo 可以算复杂产品，但前提是定价模型也可靠。
- 置信水平越高，尾部分位数需要越多模拟次数才稳定。

## 来自课程位置

- [[12_VAR风险#2.3 蒙特卡罗模拟方法|金融风险管理 12：蒙特卡罗模拟]]
- [[14_VaR参数法和模拟法|金融风险管理 14：模拟法]]

## 关联卡片

- [[VaR]]
- [[Monte Carlo Simulation VaR]]
- [[Historical Simulation Method]]
- [[Variance-Covariance Method]]
- [[Scenario Analysis]]


## 最小例子

把 **Monte Carlo Simulation Method** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
## 符号表达

将本概念记为 $C_{MonteCarloSi}$；使用时先明确对象、条件与输出，再判断 $C$ 是否满足定义。
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
