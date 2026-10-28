---
student_os: knowledge-atom
atom_id: r-base-table-header-rownames
aliases:
  - "A table header names variables and a row-name column leaves the data columns"
  - "表头命名变量，行名列移出数据列"
status: source-checked
---
<!-- bilingual-en:start -->
# A table header names variables and a row-name column leaves the data columns
<!-- bilingual-en:end -->

表头命名变量，行名列移出数据列

<!-- bilingual-en:start -->
With `read.table()`, `header = TRUE` interprets the first input line as column names. `row.names = 1` uses the first input column as row identifiers, so it is no longer a data-frame column. Check `dim()`, `names()` and column classes after import.
<!-- bilingual-en:end -->

`read.table()` 中，`header = TRUE` 把输入首行解释为列名；`row.names = 1` 把第一列用作行标识，此列不再是数据框的数据列。导入后检查 `dim()`、`names()` 和列类型。

```r
txt <- "id value
a 10
b 20"
d <- read.table(text = txt, header = TRUE, row.names = 1)
dim(d)                         # 2 1
names(d)                       # "value"
row.names(d)                   # "a" "b"
```
<!-- bilingual-en:start -->
`colClasses` describes input columns, including a column consumed as row names. A full specification avoids accidental alignment errors; shorter unnamed specifications recycle, while named entries match column names. It is not a universal requirement to supply one entry per column.
<!-- bilingual-en:end -->

`colClasses` 对应输入列，也包括被用于行名的列。完整指定能避免无意错位；较短的无名向量会回收，有名条目按列名匹配。因此并非任何情况下都必须逐列写一个条目。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=14|Chapter 1, pp. 14–16]], Jobs import / Jobs 导入。
- [R: data input](https://stat.ethz.ch/R-manual/R-patched/library/utils/html/read.table.html): header, row names and column classes / 表头、行名、列类型。
