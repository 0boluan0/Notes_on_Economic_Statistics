---
student_os: knowledge-atom
atom_id: r-base-number-character-literals
aliases:
  - "R distinguishes a numeric value from its character representation"
  - "R 区分数值和表示该数值的字符"
status: source-checked
---
<!-- bilingual-en:start -->
# R distinguishes a numeric value from its character representation
<!-- bilingual-en:end -->

R 区分数值和表示该数值的字符

<!-- bilingual-en:start -->
`1`, `1L` and `"1"` represent a double, an integer and a one-character string. `class()` reports an object class; `typeof()` reports its storage type. Quotation marks delimit the string in the source code: the value of `"1"` contains one character, not the surrounding quotation marks.
<!-- bilingual-en:end -->

`1`、`1L`、`"1"` 分别是 double、integer 和单字符字符串。`class()` 查看对象类，`typeof()` 查看存储类型。代码中的引号划定字符串边界；`"1"` 的值只有一个字符，不包含外围引号。

```r
class(1)                       # "numeric"
typeof(1)                      # "double"
typeof(1L)                     # "integer"
typeof(1:100)                  # "integer"
is.numeric(1L)                 # TRUE
class("1")                     # "character"
is.character("1")              # TRUE
nchar("1")                     # 1
as.numeric("1")                # 1
as.numeric(TRUE)               # 1
as.numeric("A")                # NA, with a warning
```
<!-- bilingual-en:start -->
Implicit conversion can also occur in comparison: both `TRUE == 1` and `TRUE == "TRUE"` evaluate to `TRUE`. Equality after conversion does not establish identical original types.
<!-- bilingual-en:end -->

比较也可能发生隐式转换：`TRUE == 1` 和 `TRUE == "TRUE"` 都为 `TRUE`。转换后相等，不能说明原始类型相同。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/03_Seminars/Week_01/week1 (3).R|Week 1 script]], class checks and conversion / 类型检查与转换。
- [R: quotes](https://stat.ethz.ch/R-manual/R-patched/library/base/html/Quotes.html): string delimiters / 字符串边界。
- [R: numeric vectors](https://stat.ethz.ch/R-manual/R-patched/library/base/html/numeric.html): integer and double distinctions / integer 与 double。
