---
student_os: knowledge-atom
atom_id: r-base-matrix-barplot
aliases:
  - "R barplot stacks matrix columns unless beside is TRUE"
  - "R 的 barplot 默认按矩阵列堆叠，beside 为 TRUE 时才并列"
status: source-checked
---
<!-- bilingual-en:start -->
# R barplot stacks matrix columns unless beside is TRUE
<!-- bilingual-en:end -->

R 的 barplot 默认按矩阵列堆叠，beside 为 TRUE 时才并列

<!-- bilingual-en:start -->
For a matrix supplied to `barplot()`, each column forms one group. By default, entries within that column stack into one bar; `beside = TRUE` places those entries side by side. Decide whether bar heights represent counts or normalized proportions before comparing groups.
<!-- bilingual-en:end -->

给 `barplot()` 输入矩阵时，每一列组成一组。默认把该列各项堆成一根柱；`beside = TRUE` 才把它们并列。比较组别前，需要确认柱高是计数还是归一化比例。

```r
heights <- rbind(group_a = c(6, 4), group_b = c(3, 7))
barplot(heights, legend.text = row.names(heights))
barplot(heights, beside = TRUE, legend.text = row.names(heights))
```
<!-- bilingual-en:start -->
With a vector input, there is one bar per element. In a matrix chart, `names.arg` labels column groups and `legend.text` can identify row components.
<!-- bilingual-en:end -->

输入向量时，每个元素对应一根柱。矩阵图里，`names.arg` 标注列组，`legend.text` 可以标识各行成分。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=18|Chapter 1, p. 18]], the stacked Jobs chart / Jobs 堆叠图。
- [R: bar plots](https://stat.ethz.ch/R-manual/R-patched/library/graphics/html/barplot.html): matrix input and beside / 矩阵输入与排列。
