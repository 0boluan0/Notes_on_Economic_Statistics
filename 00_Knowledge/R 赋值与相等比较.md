---
student_os: knowledge-atom
atom_id: r-base-assignment-comparison
aliases:
  - "R assignment changes a binding and equality comparison tests values"
  - "R 赋值改变名称绑定，相等比较检验值"
status: source-checked
---
<!-- bilingual-en:start -->
# R assignment changes a binding and equality comparison tests values
<!-- bilingual-en:end -->

R 赋值改变名称绑定，相等比较检验值

<!-- bilingual-en:start -->
For an ordinary name, `a <- expression` evaluates the right side and binds the result to `a`. `a == value` compares values. Assignment also returns its assigned value, usually invisibly: absence of automatic printing does not mean absence of a return value.
<!-- bilingual-en:end -->

普通名称赋值先计算右侧，再把结果绑定给名称；`==` 才是值相等比较。赋值也返回所赋的值，通常以不可见方式返回；控制台没有自动打印，不等于没有返回值。

```r
a <- 3
a == 3                 # TRUE
(a <- 3) + 2           # 5
withVisible(a <- 3)    # value: 3; visible: FALSE
```
<!-- bilingual-en:start -->
Use `<-` for assignments in these examples. `=` can also assign in permitted contexts, but in a function call `by = 2` names an argument; the two symbols are not interchangeable everywhere.
<!-- bilingual-en:end -->

本讲用 `<-` 写赋值。`=` 在允许的语法位置也能赋值，但函数调用里的 `by = 2` 是指定参数，不能认为两种符号在任何位置都能互换。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/03_Seminars/Week_01/week1 (3).R|Week 1 script]], object assignment and comparison / 对象赋值与比较。
- [R: assignment operators](https://stat.ethz.ch/R-manual/R-patched/library/base/html/assignOps.html): assignment result and syntax / 返回值与语法。
