---
aliases: [Proof Strategy Selection, 证明方法选择]
tags: [framework, discrete-mathematics]
type: framework
---
# Proof Strategy Selection

## 何时用什么

| 目标结构 | 优先考虑 | 失败信号 |
|---|---|---|
| $P\Rightarrow Q$ | [[Direct Proof]] 或 [[Proof by Contrapositive]] | 从 $Q$ 反推 $P$ |
| 否定存在性 | [[Proof by Contradiction]] | 矛盾来自未声明假设 |
| 有限个互斥情形 | [[Proof by Cases]] | 情形未穷尽或重叠造成重复计数 |
| 对所有自然数 | [[Induction Proof Procedure]] | 归纳假设太弱或基例缺失 |
| 状态转移永不越界 | [[Invariant Proof Procedure]] | 只验证若干轨迹 |
| 递归生成对象 | [[Structural Induction]] | 没有覆盖全部构造规则 |

## 选择原则

先把命题写出论域、量词、假设和结论，再根据逻辑形状选证明法。方法服务于命题结构，而不是由题目表面关键词决定。

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
