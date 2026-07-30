---
aliases: [Direct Proof, 直接证明]
tags: [procedure, discrete-mathematics]
type: procedure
---
# Direct Proof

## 输入与输出

输入是形如 $\forall x\in D, P(x)\Rightarrow Q(x)$ 的命题；输出是在任意满足 $P$ 的对象上推出 $Q$ 的推导链。

## Step 1–4

1. 固定任意 $x\in D$，假设 $P(x)$。
2. 展开定义，列出允许使用的已知条件。
3. 每一步注明代数、定义或已知定理，直至得到 $Q(x)$。
4. 因 $x$ 任意，恢复全称量词并结束证明。

不得把目标 $Q$ 当作已知，也不得只验证若干例子。

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
