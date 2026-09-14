---
aliases:
  - "awk条件聚合在逐记录验证与筛选后累积并输出汇总结果"
  - "Conditional aggregation with awk"
student_os: knowledge-atom
atom_id: CS-TEXT-010
atom_type: method
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[awk文本处理]]"
related:
  - "[[字段切分不等于格式解析]]"
  - "[[文本处理的locale边界]]"
  - "[[管道退出状态]]"
---

# awk条件聚合在逐记录验证与筛选后累积并输出汇总结果

<!-- bilingual-en:start -->
*Conditional aggregation in awk validates and selects records before accumulating and reporting a result*
<!-- bilingual-en:end -->

使用 `BEGIN` 初始化累积状态，在普通规则中验证字段、筛选记录并更新计数或总和，最后由 `END` 输出汇总。数值转换不是验证：文本可能按数字前缀转换，非数字内容也可能被转成零；需要数值的字段应先按输入契约检查完整内容。

<!-- bilingual-en:start -->
Use `BEGIN` to initialize aggregate state, ordinary rules to validate fields and update counts or sums for selected records, and `END` to report the result. Numeric conversion is not validation: a numeric prefix may be accepted, and nonnumeric text may become zero. Validate the complete field according to its input contract before treating it as numeric data.
<!-- bilingual-en:end -->

以下输入契约是每行一个三位 ASCII 状态码，范围为 `100` 至 `599`；聚合目标是统计其中 `400` 至 `599` 的记录数。任意坏行使整次统计失败，不把“部分计数”当成最终结果。

<!-- bilingual-en:start -->
The input contract below is one three-digit ASCII status code per line, from `100` to `599`. The aggregate counts records from `400` to `599`. Any malformed line invalidates the whole calculation rather than presenting a partial count as the final result.
<!-- bilingual-en:end -->

```bash
printf '%s\n' 200 404 403 | LC_ALL=C awk '
BEGIN { failures = 0; invalid = 0 }
$0 !~ /^[1-5][0-9][0-9]$/ { invalid = 1; exit 2 }
$0 + 0 >= 400 { failures++ }
END {
    if (invalid) exit 2
    print failures
}'
```

输出为 `2`；空输入输出 `0`，这是这里明确选择的计数契约。把任意一行改为 `oops` 时，`awk` 以 `2` 退出且不输出计数。关键是 `END` 在普通规则调用 `exit` 后仍会执行，所以必须在 `END` 检查失败标记。此程序检查输入内容，不宣称能探测上游未报告的数据缺失；整个[[管道退出状态|管道的失败]]仍须由调用方检查。

<!-- bilingual-en:start -->
The output is `2`; empty input produces `0`, an explicit choice for this counting contract. Replacing any line with `oops` makes awk exit with `2` and no count output. Crucially, `END` still runs after `exit` in an ordinary rule, so it must check the failure flag. This program validates received content; it does not detect unreported missing data upstream. The caller must still check [[管道退出状态|pipeline failures]].
<!-- bilingual-en:end -->

普通 `awk` 数值运算通常使用有限精度数值表示，不能把这种小计数例子推广成任意大整数或精确金额计算的保证。需要精确算术时应另选适当表示与工具。

<!-- bilingual-en:start -->
Ordinary awk arithmetic generally uses finite-precision numbers. This small counting example does not guarantee arbitrary-size integer arithmetic or exact monetary calculation. Choose an appropriate representation and tool when exact arithmetic is required.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [POSIX awk, Overall Program Structure、Special Patterns、Statements and Expressions](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/awk.html)：支持 `BEGIN/END`、逐记录动作、数字与字符串转换，以及普通规则 `exit` 后执行 `END` 的控制流；示例据此显式阻止失败后的汇总输出。
  <!-- bilingual-en:start -->
  These sections support `BEGIN/END`, record actions, numeric/string conversion, and execution of `END` after an ordinary-rule `exit`. The example consequently blocks aggregate output after validation failure.
  <!-- bilingual-en:end -->
- [Missing Semester 2020, Data Wrangling](https://missing.csail.mit.edu/2020/data-wrangling/)：支持筛选后计数的课程范围；本卡补全输入有效性、空输入和失败输出契约。
  <!-- bilingual-en:start -->
  The lecture provides the scope of counting selected records; this card makes validity, empty-input, and failure-output contracts explicit.
  <!-- bilingual-en:end -->
