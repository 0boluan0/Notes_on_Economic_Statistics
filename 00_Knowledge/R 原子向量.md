---
student_os: knowledge-atom
atom_id: r-base-atomic-vector
aliases:
  - "An R atomic vector is an ordered collection with one element type"
  - "R 原子向量是所有元素共享一种类型的有序集合"
status: source-checked
---
<!-- bilingual-en:start -->
# An R atomic vector is an ordered collection with one element type
<!-- bilingual-en:end -->

R 原子向量是所有元素共享一种类型的有序集合

<!-- bilingual-en:start -->
An atomic vector stores elements of one underlying type. For ordinary atomic inputs, `c()` concatenates values and coerces them to a common type when needed. Combining text with numbers therefore changes the numeric entries into character strings; selecting them later does not undo that conversion.
<!-- bilingual-en:end -->

原子向量的元素具有同一种底层类型。对普通原子输入，`c()` 合并各值，必要时转换为共同类型；数字与文本合并后，数字也会变成字符。随后取出这些元素不会自动恢复数值类型。

```r
numbers <- c(1, 2, 3)
combined <- c(numbers, "Cat")
combined[1:3]                    # "1" "2" "3"
as.numeric(combined[1:3])         # 1 2 3
length(combined)                  # 4
```
<!-- bilingual-en:start -->
This describes atomic vectors. A list can hold components of different types; special classes may define their own `c()` methods.
<!-- bilingual-en:end -->

这里限定原子向量。list 可以包含不同类型的成分；特殊类也可以定义自己的 `c()` 方法。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/03_Seminars/Week_01/week1 (3).R|Week 1 script]], vector construction and coercion / 向量构造与转换。
- [R: combine values](https://stat.ethz.ch/R-manual/R-patched/library/base/html/c.html): default coercion and method boundary / 默认转换与方法边界。
