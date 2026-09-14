---
aliases:
  - "Bash引用通过转义或引号限制字符的特殊解释"
  - "Bash quoting"
student_os: knowledge-atom
atom_id: CS-SHELL-003
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell简单命令]]"
related:
  - "[[Shell文件名展开]]"
leads_to:
  - "[[Shell字段分割]]"
  - "[[Shell位置参数]]"
---

# Bash引用通过转义或引号限制字符的特殊解释

在 Bash 中，引用是用反斜杠或引号限制字符的 Shell 特殊含义。单引号保留内部字符的字面值；双引号仍允许变量、命令替换和算术展开，但保护这些结果不再进行通常的字段分割与文件名展开。普通双引号中的单个标量变量即使为空，也能保留为一个参数。
<!-- bilingual-en:start -->
In Bash, quoting uses backslashes or quotation marks to restrict shell-special interpretation. Single quotes preserve literal characters. Double quotes still allow parameter, command, and arithmetic expansion, but protect the results from ordinary word splitting and pathname expansion. A single scalar variable within double quotes remains one argument even when empty.
<!-- bilingual-en:end -->

```bash
label='two words'
printf '<%s>\n' "$label" '$label'
```

两行输出分别是 `<two words>` 与 `<$label>`。单引号内部不能直接包含单引号，即使前面加反斜杠；双引号中的反斜杠也只对特定字符有转义作用。交互式历史展开启用时，`!` 另有规则。
<!-- bilingual-en:start -->
The two output lines are `<two words>` and `<$label>`. A single quote cannot occur inside single quotes, even with a preceding backslash. Inside double quotes, backslash escapes only specific characters. Interactive history expansion introduces additional rules for `!` when enabled.
<!-- bilingual-en:end -->

引用保护的是 Shell 层的解释和参数边界，不是通用输入验证。`"$path"` 仍可能被目标程序当成选项或特殊操作数；`eval` 则把得到的文本再次当命令解释。文件名传递应继续使用[[文件名安全传参]]的规则。
<!-- bilingual-en:start -->
Quoting protects shell interpretation and argument boundaries; it is not general input validation. The receiving program may still interpret `"$path"` as an option or a special operand. `eval` reparses text as commands. Continue with the rules for [[文件名安全传参|safe filename arguments]].
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`QUOTING`、`Word Splitting` 与内建 `eval`；[官方引用章节](https://www.gnu.org/software/bash/manual/html_node/Quoting.html)：支持三种引用机制、双引号保留的展开、空参数和重新解释的边界。
- [Missing Semester 2020：Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)，单引号与双引号示例：支持课程中的基础对照。
<!-- bilingual-en:start -->
The local Bash manual supplies the exact quoting, empty-argument, and `eval` rules. The course provides the introductory single-versus-double-quote contrast. Argument preservation does not determine how a receiving utility interprets an operand.
<!-- bilingual-en:end -->
