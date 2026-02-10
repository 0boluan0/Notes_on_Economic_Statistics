---
aliases:
- Taylor Expansion
- Taylor Formula
- 泰勒公式
tags:
- concept
---
# Taylor Expansion

## 它是什么
- 「Taylor Expansion」是指在某点附近用多项式近似函数的展开。

## 最小可检索信息
- 定义：在某点附近用多项式近似函数的展开。
- 符号/公式：$f(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2}(x-a)^2+\cdots。$
- 最小例子：用二阶泰勒近似期权价格。

## 关联卡片
- [[Lagrange Mean Value Theorem]]

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
