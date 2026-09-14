---
aliases:
  - "Bash位置参数按序保存脚本或函数接收的实参"
  - "Bash positional parameters"
student_os: knowledge-atom
atom_id: CS-SHELL-007
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell引用]]"
related:
  - "[[Shell函数]]"
  - "[[Shell脚本]]"
leads_to:
  - "[[文件名安全传参]]"
---

# Bash位置参数按序保存脚本或函数接收的实参

Bash 位置参数是按编号保存调用实参的参数：`$1` 是第一个，`${10}` 是第十个，`$#` 给出数量。调用函数时，位置参数暂时替换为函数实参，函数结束后恢复。`$0` 是独立的特殊参数，不是第一个实参，也不会因普通函数调用而变成函数名。
<!-- bilingual-en:start -->
Bash positional parameters hold invocation arguments by position: `$1` is the first, `${10}` the tenth, and `$#` gives their count. A function call temporarily installs its own arguments and restores the previous positional parameters on return. `$0` is a separate special parameter, not the first argument, and an ordinary function call does not change it to the function name.
<!-- bilingual-en:end -->

要逐个转发已有实参，在参数位置单独写 `"$@"`：每个原实参仍是一个词，空字符串也保留；没有实参时产生零个词。`"$*"` 则把实参按 `IFS` 的首字符连接成一个词，不能用来保持原来的参数边界。
<!-- bilingual-en:start -->
To forward arguments individually, use `"$@"` as a standalone argument word. Each original argument remains one word, including empty strings; no arguments produce zero words. `"$*"` joins the arguments into one word using the first character of `IFS`, so it does not preserve their original boundaries.
<!-- bilingual-en:end -->

```bash
set -- 'two words' '' '*'
printf 'argc=%s\n' "$#"
for arg in "$@"; do
    printf '<%s>\n' "$arg"
done
```

这在当前练习 Shell 中设置三个位置参数，依次显示 `<two words>`、`<>`、`<*>`。循环中的 `*` 是数据，不再匹配文件。未经引用的 `$@` 不能保证同样的结果。
<!-- bilingual-en:start -->
This installs three positional parameters in the practice shell and displays `<two words>`, `<>`, and `<*>`. The asterisk is data rather than a filename pattern. Unquoted `$@` does not guarantee the same result.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`Positional Parameters`、`Special Parameters` 与 `FUNCTIONS`；[官方特殊参数章节](https://www.gnu.org/software/bash/manual/html_node/Special-Parameters.html)：支持编号、函数内恢复、`$0`、`"$@"`、`"$*"` 与零实参行为。
- [Missing Semester 2020：Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)，特殊参数与 `for file in "$@"` 示例：支持课程中的参数遍历用途。
<!-- bilingual-en:start -->
The local Bash manual establishes numbering, function-call restoration, `$0`, and the quoted argument-list rules. The course demonstrates iterating over `"$@"`; the example here additionally checks empty and wildcard-containing arguments.
<!-- bilingual-en:end -->
