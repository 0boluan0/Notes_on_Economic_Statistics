---
aliases: [Set Operations, Set, 集合运算, 集合]
tags: [concept, discrete-mathematics]
---
# Set Operations

集合由其元素决定。对集合 $A,B$：$A\cup B$ 是至少属于一个集合的元素，$A\cap B$ 是同时属于二者的元素，$A\setminus B$ 是属于 $A$ 但不属于 $B$ 的元素，$\mathcal P(A)$ 是 $A$ 的全部子集组成的幂集。

笛卡尔积
$$
A\times B=\{(a,b):a\in A,b\in B\}
$$
由有序对组成，所以一般 $A\times B\ne B\times A$。

## 关联

- [[Binary Relation]] 是笛卡尔积的子集。
- 若有限集 $|A|=n$，则 $|\mathcal P(A)|=2^n$。

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
