---
aliases:
- 大样本与小样本
- Large and Small Samples
tags:
  - concept
---
# 大样本与小样本

* 大小不由数量决定，而是由其数字特征与总体的数字特征的相似程度决定。
## 大样本
数据趋于无穷，样本信息结构无限接近于总体。
## 小样本
样本容量过于小，样本信息结构发生扭曲，无法解释总体样本的数字特征
理论上推导出来的分布不适用。


## 最小例子

把 **Large and Small Samples** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
## 符号表达

将本概念记为 $C_{LargeandSmal}$；使用时先明确对象、条件与输出，再判断 $C$ 是否满足定义。
## 概念边界

本卡片只回答“它是什么”，集中在定义、核心关系与最小例子；具体估计步骤、证明和诊断流程应分别放在 procedure、proof 或 system 卡片中。
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
