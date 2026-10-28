---
student_os: knowledge-atom
atom_id: r-base-logical-filter
aliases:
  - "R logical indexing selects values using a condition for each position"
  - "R 逻辑索引用每个位置的条件结果筛选值"
status: source-checked
---
<!-- bilingual-en:start -->
# R logical indexing selects values using a condition for each position
<!-- bilingual-en:end -->

R 逻辑索引用每个位置的条件结果筛选值

<!-- bilingual-en:start -->
For a logical mask of the same length as a vector, `TRUE` selects the corresponding value and `FALSE` omits it. `&`, `|` and `!` combine or negate conditions element by element. Build each condition from the same rows as the values being selected.
<!-- bilingual-en:end -->

逻辑掩码与向量等长时，`TRUE` 保留对应值，`FALSE` 排除。`&`、`|`、`!` 逐元素组合或取反条件；每个条件都应来自与待选值相同的观测行。

```r
numbers <- c(1, 2, 3)
numbers > 1                      # FALSE TRUE TRUE
numbers[numbers > 1]             # 2 3
numbers[numbers > 1 & numbers < 3] # 2
```
<!-- bilingual-en:start -->
An `NA` condition leaves an unknown selected value rather than silently discarding it. If missing values should be excluded, say so explicitly. Short logical indices can recycle; an equally long mask avoids unintended positional pairing.
<!-- bilingual-en:end -->

`NA` 条件会留下未知值，不会自动把它排除。若要排除缺失值，应明确写出条件。较短逻辑索引可能回收；使用等长掩码可以避免无意的位置配对。

```r
z <- c(1, NA, 3)
z[!is.na(z) & z > 1]             # 3
```

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/03_Seminars/Week_01/week1 (3).R|Week 1 script]], vector and salary filtering / 向量及工资筛选。
- [R: extraction](https://stat.ethz.ch/R-manual/R-patched/library/base/html/Extract.html): logical and missing indices / 逻辑与缺失索引。
