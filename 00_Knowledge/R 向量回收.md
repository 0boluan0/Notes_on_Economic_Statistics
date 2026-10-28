---
student_os: knowledge-atom
atom_id: r-base-arithmetic-recycling
aliases:
  - "Ordinary R vector arithmetic recycles the shorter nonempty operand"
  - "R 普通向量算术会回收较短的非空操作数"
status: source-checked
---
<!-- bilingual-en:start -->
# Ordinary R vector arithmetic recycles the shorter nonempty operand
<!-- bilingual-en:end -->

R 普通向量算术会回收较短的非空操作数

<!-- bilingual-en:start -->
Ordinary vector arithmetic works element by element. For two nonempty vectors, the shorter one repeats to match the longer length. A fractional final repeat produces a warning, but still returns values. Even when no warning occurs, the pairing must make sense for the analysis.
<!-- bilingual-en:end -->

普通向量算术逐元素运算。两个非空向量长度不同时，短向量重复到长向量的长度；最后一次重复不完整会给警告，但仍返回结果。即使没有警告，也需要确认配对符合分析意图。

```r
c(1, 2, 3) + c(4, 5, 6)      # 5 7 9
c(1, 2, 3) - c(8, 7.5, -2)   # -7 -5.5 5
c(1, 2, 3) * c(1, 3, 3)      # 1 6 9
12 / c(1, 2, 3)                # 12 6 4
c(1, 2, 4) * c(1, 2, 4, 5, 7) # 1 4 16 5 14; warning
numeric(0) + 1:3              # numeric(0)
```
<!-- bilingual-en:start -->
A zero-length operand yields a zero-length result. These rules describe ordinary vector arithmetic; arrays and special classes have additional rules. `*` is element-wise multiplication, while `%*%` denotes matrix multiplication.
<!-- bilingual-en:end -->

操作数之一长度为 0 时，结果长度也为 0。以上限定普通向量算术；数组和特殊类还可能有额外规则。`*` 是逐元素乘法，`%*%` 表示矩阵乘法。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/03_Seminars/Week_01/week1 (3).R|Week 1 script]], vector calculations / 向量运算。
- [R: arithmetic](https://stat.ethz.ch/R-manual/R-patched/library/base/html/Arithmetic.html): recycling, warning and zero-length boundary / 回收、警告与空向量边界。
