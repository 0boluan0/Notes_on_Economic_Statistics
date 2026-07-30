---
aliases: [Proof by Contrapositive, Proof by Contraposition, 逆否证明]
tags: [procedure, discrete-mathematics]
type: procedure
---
# Proof by Contrapositive

要证明 $P\Rightarrow Q$，改证逻辑等价的 $\neg Q\Rightarrow\neg P$：

1. 正确否定 $Q$；
2. 假设 $\neg Q$；
3. 逐步推出 $\neg P$；
4. 由逆否命题等价性得到原命题。

它不同于证明逆命题 $Q\Rightarrow P$，后者通常不与原命题等价。

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
