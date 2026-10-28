---
student_os: knowledge-atom
atom_id: r-base-factor-label-numeric
aliases:
  - "Converting a factor to numbers requires distinguishing codes from labels"
  - "把因子转成数值前必须区分内部编码和标签"
status: source-checked
---
<!-- bilingual-en:start -->
# Converting a factor to numbers requires distinguishing codes from labels
<!-- bilingual-en:end -->

把因子转成数值前必须区分内部编码和标签

<!-- bilingual-en:start -->
`as.numeric(f)` returns a factor’s integer codes. When the labels themselves represent numbers, convert the labels with `as.numeric(as.character(f))`. Labels that are not numeric do not become meaningful measurements through this operation.
<!-- bilingual-en:end -->

`as.numeric(f)` 返回因子的内部整数编码。标签本身表示数字时，应用 `as.numeric(as.character(f))` 转换标签；非数字标签不会通过这一步变成有意义的测量值。

```r
f <- factor(c("10", "20"))
as.numeric(f)                  # 1 2
as.numeric(as.character(f))    # 10 20
```
<!-- bilingual-en:start -->
In some datasets the labels happen to equal their codes. That coincidence does not make direct numeric conversion safe for other factors. Use this distinction when exporting or preparing numerical calculations.
<!-- bilingual-en:end -->

某些数据的标签碰巧与内部编码相等，这不能保证其他因子也可以直接转数值。导出或准备数值计算时，应先确认需要的是哪一种表示。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=25|Chapter 1, p. 25]], the numerical export example / 数值导出示例。
- [R: factors](https://stat.ethz.ch/R-manual/R-patched/library/base/html/factor.html): code-versus-label conversion / 编码与标签转换。
