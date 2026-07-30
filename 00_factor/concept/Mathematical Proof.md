---
aliases: [Mathematical Proof, Proof, 数学证明, 证明]
tags: [concept, discrete-mathematics, proof]
type: concept
---
# Mathematical Proof

数学证明是从明确假设出发，经过每一步都由定义、已知事实或有效推理规则支持的有限论证，并最终推出目标命题。证明的职责是排除所有满足假设却不满足结论的情形，而不是只展示若干支持性例子。

方法选择见 [[Proof Strategy Selection]]；执行型模板见 [[Direct Proof]]、[[Proof by Contrapositive]]、[[Proof by Contradiction]]、[[Proof by Cases]] 与 [[Induction Proof Procedure]]。

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
