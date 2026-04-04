---
aliases:
- Testing Positive Definiteness
- Positive Definite Test Order
- 正定矩阵判别流程
tags:
- procedure
- 线性代数
---
# Testing Positive Definiteness

>[!note] 何时使用
> - 题目要你判断矩阵是否 positive definite、classify a quadratic form，或解释某个临界点为何是 minimum。
> - 这张卡默认你面对的是课程里的对称实矩阵语境。

## Step 1. 先检查是否在“对称矩阵语境”里

- 先看 $A=A^T$ 是否成立。
- 若题目已经说“symmetric matrix”，直接进入下一步。
- 若不是对称矩阵，不要直接套主元判据或顺序主子式判据。

## Step 2. 选最快的判别入口

- 若已经给出 eigenvalues，就直接看它们是否全大于 0。
- 若矩阵维度小、数值明确，就优先看顺序主子式或 elimination 主元。
- 若题目给的是 quadratic form，就尝试配方、换基或转到特征向量方向。

## Step 3. 执行对应判别

- 特征值路线：所有特征值都大于 0，则 positive definite。
- 主元路线：无换行交换的消元中，所有主元都大于 0，则 positive definite。
- 顺序主子式路线：所有 leading principal minors 都大于 0，则 positive definite。
- 二次型路线：若对任意非零 $x$ 都有 $x^TAx>0$，则 positive definite。

## Step 4. 区分相近但不同的结论

- 若允许某些方向取到 0，只能说 positive semidefinite。
- 若既能取正也能取负，就是 indefinite。
- 若所有特征值都小于 0，则是 negative definite。

## Step 5. 写结论时必须带判据

- 不要只写“所以它正定”。
- 要明确写出“因为它对称且所有特征值大于 0”或“因为顺序主子式全为正”等依据。
- 若题目和极小值有关，要补一句：positive definite quadratic form implies a strict minimum at the origin。

## 输出检查

- 我有没有先说明矩阵是否对称。
- 我用的判据是否和题目给出的信息匹配。
- 我有没有区分 strict positivity 与 semidefinite 的 `\ge 0`。

## 关联卡片

- [[Positive Definite Matrix]]
- [[Symmetric Matrix]]
- [[Spectral Decomposition]]
- [[Choosing Matrix Decompositions]]

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
