---
student_os: knowledge-atom
atom_id: r-base-factor-definition
aliases:
  - "An R factor represents categories with integer codes and level labels"
  - "R 因子用整数编码和水平标签表示类别"
status: source-checked
---
<!-- bilingual-en:start -->
# An R factor represents categories with integer codes and level labels
<!-- bilingual-en:end -->

R 因子用整数编码和水平标签表示类别

<!-- bilingual-en:start -->
A factor represents categorical observations using integer codes and a set of level labels. The labels carry the category meaning; a numeric-looking label is not automatically a quantitative measurement. `table(f)` counts observations by level. `ordered = TRUE` additionally declares an ordering of those levels.
<!-- bilingual-en:end -->

factor 用整数编码及一组水平标签表示分类观测。类别含义由标签表达；标签看起来像数字，不会自动把类别变成定量测量。`table(f)` 按水平计数；`ordered = TRUE` 才额外声明水平顺序。

```r
f <- factor(c("high", "low", "high"), levels = c("low", "high"))
levels(f)                      # "low" "high"
as.integer(f)                  # 2 1 2
is.ordered(f)                  # FALSE
```
<!-- bilingual-en:start -->
A category may have a meaningful natural order without that order being encoded in an ordinary factor. Inspect the declared levels and ordering before interpreting comparisons or a model.
<!-- bilingual-en:end -->

类别在现实中有自然顺序，并不意味着普通 factor 已把该顺序编码进去。解释比较或模型前，应检查声明的水平及其有序性。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=16|Chapter 1, p. 16]], categorical coding / 分类编码。
- [R: factors](https://stat.ethz.ch/R-manual/R-patched/library/base/html/factor.html): representation and ordering / 表示方式与有序性。
