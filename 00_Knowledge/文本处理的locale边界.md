---
aliases:
  - "locale会改变文本比较和字符解释且多阶段处理需明确各阶段设置"
  - "Locale affects text comparison and character interpretation across a workflow"
student_os: knowledge-atom
atom_id: CS-TEXT-014
atom_type: boundary
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
related:
  - "[[正则方言边界]]"
  - "[[文本排序键选择]]"
  - "[[环境变量]]"
  - "[[环境变量导出]]"
  - "[[相邻行去重]]"
  - "[[awk条件聚合]]"
---

# locale会改变文本比较和字符解释且多阶段处理需明确各阶段设置

<!-- bilingual-en:start -->
*Locale can change text comparison and character interpretation, so specify the settings at each stage of a workflow*
<!-- bilingual-en:end -->

文本命令的结果不仅由输入字节和选项决定，还可能受 locale 影响：`LC_COLLATE` 控制排序及相关范围解释，`LC_CTYPE` 影响字符编码解释和字符分类，`LC_NUMERIC` 影响遵循该设置的数值读写规则。具体工具也可能有自己的模式或扩展，不能只凭语言环境名称推断全部行为。

<!-- bilingual-en:start -->
Text-command results can depend on locale as well as input bytes and options. `LC_COLLATE` controls collation and associated range interpretation, `LC_CTYPE` affects character encoding interpretation and classification, and `LC_NUMERIC` affects numeric input/output rules in tools that honor it. Tool modes and extensions may add further conditions, so a locale name alone does not determine every behavior.
<!-- bilingual-en:end -->

若任务需要可重复的机器文本处理，且数据契约适合 `C` locale，可在局部子 Shell 中设置并[[环境变量导出|导出]] `LC_ALL=C`，让其中所有命令使用相同设置。`LC_ALL` 覆盖分类别的 `LC_*` 与 `LANG` 设置；子 Shell 结束后，不会把该赋值留在调用它的 Shell 中。

<!-- bilingual-en:start -->
For reproducible machine-oriented text processing whose data contract suits the `C` locale, set and [[环境变量导出|export]] `LC_ALL=C` inside a local subshell so all its commands share that setting. `LC_ALL` overrides individual `LC_*` and `LANG` settings. The assignment does not remain in the invoking Shell after the subshell finishes.
<!-- bilingual-en:end -->

```bash
(
    export LC_ALL=C
    printf '%s\n' b a b | sort | uniq -c
)
```

只写 `LC_ALL=C sort | uniq -c` 时，命令前赋值只作用于 `sort`，并没有为 `uniq` 单独设置环境。若不同阶段使用不同的字符或相等规则，分组与去重的含义可能不一致。

<!-- bilingual-en:start -->
In `LC_ALL=C sort | uniq -c`, the command-prefixed assignment applies to `sort` only; it does not independently configure uniq's environment. Different character or equality rules across stages can make grouping and deduplication disagree.
<!-- bilingual-en:end -->

`C` locale 提供适合字节导向处理的可预测规则，不等于自然语言排序，也不执行 Unicode 归一化。需要面向人的语言排序时，应选择并记录适当的 locale 与实现；不能把“固定环境以便复现”误写成“所有任务都必须使用 C”。

<!-- bilingual-en:start -->
The `C` locale supplies predictable rules suitable for byte-oriented processing, not natural-language collation or Unicode normalization. Human-facing linguistic ordering requires an appropriate, recorded locale and implementation. Fixing the environment for reproducibility does not mean every task should use `C`.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [POSIX grep, Environment Variables](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/grep.html)；[POSIX sort, Environment Variables](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/sort.html)；[POSIX awk, Environment Variables](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/awk.html)：支持字符、排序、数值环境与 `LC_ALL` 覆盖关系；`C` 的选择以任务数据契约为条件。
  <!-- bilingual-en:start -->
  These specifications support character, collation, and numeric environments and the precedence of `LC_ALL`. Choosing `C` remains conditional on the task's data contract.
  <!-- bilingual-en:end -->
- 本机 GNU Bash 3.2 `bash(1)`，`ENVIRONMENT` 与 `COMMAND EXECUTION ENVIRONMENT`：核对外部命令前赋值的范围、导出继承，以及括号子 Shell 不改写调用方环境。
  <!-- bilingual-en:start -->
  The installed GNU Bash 3.2 manual supports command-prefixed assignment scope, exported inheritance, and the isolation of parenthesized subshell environment changes from the invoking Shell.
  <!-- bilingual-en:end -->
