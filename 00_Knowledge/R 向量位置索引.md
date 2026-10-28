---
student_os: knowledge-atom
atom_id: r-base-vector-position-index
aliases:
  - "Positive R indices select positions in order and negative indices omit positions"
  - "R 正索引按给定顺序选位置，负索引排除位置"
status: source-checked
---
<!-- bilingual-en:start -->
# Positive R indices select positions in order and negative indices omit positions
<!-- bilingual-en:end -->

R 正索引按给定顺序选位置，负索引排除位置

<!-- bilingual-en:start -->
For a plain vector, positions start at 1. A vector of positive integer indices selects those positions in the supplied order, including repetitions. Negative integer indices omit positions. Zero selects nothing and is not a first-element index. Do not mix positive and negative indices in one selection.
<!-- bilingual-en:end -->

普通向量的位置从 1 开始。正整数索引向量按给出的顺序选择位置，允许重复；负整数索引排除位置。0 不选择元素，不是首元素索引。一次选择不能混用正、负索引。

```r
v <- c(10, 20, 30, 40)
v[c(3, 1, 3)]          # 30 10 30
v[-2]                  # 10 30 40
v[0]                   # numeric(0)
```
<!-- bilingual-en:start -->
This rule concerns positional extraction with integer indices, not matching names or interpreting a logical condition.
<!-- bilingual-en:end -->

这条规则针对整数位置提取；按名称匹配和按逻辑条件筛选有各自的解释。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/03_Seminars/Week_01/week1 (3).R|Week 1 script]], positive positions / 正位置索引。
- [R: extraction](https://stat.ethz.ch/R-manual/R-patched/library/base/html/Extract.html): positional selection and exclusion / 位置选择与排除。
