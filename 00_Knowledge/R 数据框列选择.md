---
student_os: knowledge-atom
atom_id: r-base-dataframe-subset-shape
aliases:
  - "R data-frame selection can return a column vector or preserve a data frame"
  - "R 数据框选择既可返回列向量，也可保留数据框"
status: source-checked
---
<!-- bilingual-en:start -->
# R data-frame selection can return a column vector or preserve a data frame
<!-- bilingual-en:end -->

R 数据框选择既可返回列向量，也可保留数据框

<!-- bilingual-en:start -->
A data frame is a rectangular collection of equal-length columns whose types may differ. In `d[rows, columns]`, a blank side selects all entries on that side. Selecting a single ordinary column with two-index `[` normally drops the data-frame wrapper; use `drop = FALSE` to retain it.
<!-- bilingual-en:end -->

数据框由等长的列组成，各列类型可以不同。`d[行, 列]` 的某侧留空表示选择该侧全部。双下标 `[` 只选一个普通列时，通常会去掉数据框外层；`drop = FALSE` 可保留数据框。

```r
d <- data.frame(a = c(1, 5, 9), b = c(2, 6, 10))
d[, 2]                         # 2 6 10
class(d[, 2])                  # "numeric"
class(d[, 2, drop = FALSE])     # "data.frame"
dim(d[1, ])                    # 1 2
d[2, 2]                        # 6
```
<!-- bilingual-en:start -->
`d$b` extracts this named column. `d["b"]` keeps a one-column data frame. A one-row selection with several columns remains a data frame; do not assume every row selection becomes a vector.
<!-- bilingual-en:end -->

`d$b` 提取这里名为 b 的列；`d["b"]` 保留单列数据框。选一行且保留多列时仍是数据框，不能认为取一行总会变成向量。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=11|Chapter 1, pp. 10–12]], rectangular input and indexing / 矩形输入与索引。
- [R: data-frame extraction](https://stat.ethz.ch/R-manual/R-patched/library/base/html/Extract.data.frame.html): return shape and drop / 返回形状。
