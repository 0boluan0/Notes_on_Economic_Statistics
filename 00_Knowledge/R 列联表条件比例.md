---
student_os: knowledge-atom
atom_id: r-base-table-conditional-proportion
aliases:
  - "The denominator of a contingency-table proportion defines its reference group"
  - "列联表比例的分母确定参照群体"
status: source-checked
---
<!-- bilingual-en:start -->
# The denominator of a contingency-table proportion defines its reference group
<!-- bilingual-en:end -->

列联表比例的分母确定参照群体

<!-- bilingual-en:start -->
In a two-way count table, dividing a cell by its row total conditions on that row; dividing by its column total conditions on that column. Name the reference group before interpreting the proportion. A zero denominator does not define an ordinary conditional proportion.
<!-- bilingual-en:end -->

二维频数表中，某格除以行合计，参照群体就是该行；除以列合计，参照群体就是该列。解释比例前先说明群体。分母为 0 时，普通条件比例没有定义。

```r
counts <- rbind(A = c(yes = 6, no = 4),
                B = c(yes = 3, no = 7))
counts[1, 1] / sum(counts[1, ])   # 0.6: yes among group A
counts[1, 1] / sum(counts[, 1])   # 2/3: group A among yes
row_prop <- prop.table(counts, 1)
dim(row_prop)                    # 2 2
100 * row_prop[1, ]              # 60 40
```
<!-- bilingual-en:start -->
`prop.table(counts, 1)` normalizes every row and returns the whole table of fractions. To obtain the first row’s percentages, select `[1, ]` and multiply by 100. Use margin `2` for column proportions; omit the margin for proportions of the entire table total.
<!-- bilingual-en:end -->

`prop.table(counts, 1)` 对每一行归一化，返回整张小数比例表。要取得第一行百分比，还需选择 `[1, ]` 并乘 100。参数 2 表示列内比例；省略该参数则除以全表总数。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=17|Chapter 1, p. 17]], within-group percentages / 组内百分比。
- [R: proportions](https://stat.ethz.ch/R-manual/R-patched/library/base/html/proportions.html): margin and result shape / 分母与结果形状。
