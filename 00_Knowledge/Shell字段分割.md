---
aliases:
  - "Bash字段分割按IFS把适用的未引用展开结果分成多个词"
  - "Bash word splitting"
student_os: knowledge-atom
atom_id: CS-SHELL-004
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell引用]]"
related:
  - "[[Shell文件名展开]]"
leads_to:
  - "[[文件名安全传参]]"
---

# Bash字段分割按IFS把适用的未引用展开结果分成多个词

字段分割是 Bash 在适用的上下文中，按 `IFS` 对未受双引号保护的参数展开、命令替换或算术展开结果进行分割的步骤。默认 `IFS` 包含空格、制表符和换行。它不同于读取命令源码时识别词和操作符的语法分析，也不能推广为 Zsh 等其他 Shell 的默认规则。
<!-- bilingual-en:start -->
Word splitting is the Bash step that, in applicable contexts, splits unquoted parameter, command-substitution, or arithmetic-expansion results according to `IFS`. Default `IFS` contains space, tab, and newline. This is distinct from parsing words and operators in the command source, and is not a statement about other shells' defaults.
<!-- bilingual-en:end -->

```bash
item='red blue'
printf '<%s>\n' $item
printf '<%s>\n' "$item"
```

默认 `IFS` 下，第一条 `printf` 接收两个数据参数，第二条接收一个。未经引用的空变量还可能消失，而 `"$item"` 在变量为空时保留空参数。分割完成后，留下的通配模式还可能进行[[Shell文件名展开]]。
<!-- bilingual-en:start -->
With default `IFS`, the first `printf` receives two data arguments and the second receives one. An unquoted empty expansion can disappear, whereas `"$item"` preserves an empty argument. Any remaining unquoted wildcard patterns may then undergo [[Shell文件名展开|pathname expansion]].
<!-- bilingual-en:end -->

这不是“所有未引用变量都分割”：普通赋值的右侧与 Bash 的 `[[ … ]]` 内部不做这种字段分割。`IFS` 为空会禁止字段分割，但不会因此同时禁止文件名展开。变量值中的 `|` 或 `;` 也不会仅因普通展开就重新成为管道或命令分隔符。
<!-- bilingual-en:start -->
Not every unquoted variable is split: ordinary assignment values and Bash `[[ … ]]` expressions suppress this step. Empty `IFS` disables word splitting, not pathname expansion. A `|` or `;` introduced by ordinary variable expansion is not reparsed as a pipeline or command separator.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`Word Splitting`、`PARAMETERS`、`Compound Commands: [[ … ]]` 与 `SIMPLE COMMAND EXPANSION`；[官方字段分割章节](https://www.gnu.org/software/bash/manual/html_node/Word-Splitting.html)：支持适用展开、IFS、空参数与上下文例外。
<!-- bilingual-en:start -->
The local Bash manual's word-splitting, assignment, conditional-command, and simple-command sections support the applicable expansions, `IFS` behavior, empty arguments, and context exceptions. Its parsing/expansion separation explains why ordinary expansion does not create new control operators.
<!-- bilingual-en:end -->
