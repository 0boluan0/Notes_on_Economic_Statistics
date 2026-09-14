---
aliases:
  - "Bash文件名展开把未引用的模式替换为匹配的路径名"
  - "Bash pathname expansion"
  - "Shell globbing"
student_os: knowledge-atom
atom_id: CS-SHELL-005
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell引用]]"
related:
  - "[[Bash大括号展开]]"
  - "[[Make 模式规则]]"
leads_to:
  - "[[文件名安全传参]]"
---

# Bash文件名展开把未引用的模式替换为匹配的路径名

文件名展开，也称 globbing，是 Bash 在适用上下文中将含未引用 `*`、`?` 或 `[` 的模式替换为匹配路径名的步骤。`*` 匹配任意长度字符串，`?` 匹配一个字符，方括号表达式匹配其中规定的一个字符；这些不是正则表达式的同名运算。
<!-- bilingual-en:start -->
Pathname expansion, or globbing, is the Bash step that replaces applicable patterns containing unquoted `*`, `?`, or `[` with matching pathnames. `*` matches a string of any length, `?` one character, and a bracket expression one character from its specified set. These are not the corresponding regular-expression operators.
<!-- bilingual-en:end -->

若当前目录只有 `a.py`、`b.py`、`note.txt` 三个普通文件，则 `printf '%s\n' *.py` 得到两个数据参数，而 `printf '%s\n' '*.py'` 只输出字面量 `*.py`。匹配所得的每个路径名作为一个词保留，文件名内部的空格不会在 glob 后再分割。
<!-- bilingual-en:start -->
If the current directory contains only the ordinary files `a.py`, `b.py`, and `note.txt`, `printf '%s\n' *.py` receives two data arguments. Quoting `'*.py'` prints the literal pattern instead. Each matched pathname remains one word; spaces inside matched filenames are not split again after globbing.
<!-- bilingual-en:end -->

默认 Bash 在无匹配时保留原模式，`nullglob` 可使它消失，`failglob` 可使命令报错而不执行。默认模式不会隐式匹配路径组件开头的点，`*` 也不跨 `/`；相关选项能改变部分行为。因此不能无条件把 `*` 称为“所有文件”。[[Bash大括号展开]]则不检查路径是否存在。
<!-- bilingual-en:start -->
By default, Bash leaves an unmatched pattern unchanged. `nullglob` can remove it, while `failglob` can reject the command. Default matching does not implicitly match a leading dot in a pathname component, and `*` does not cross `/`; options change some of these rules. Thus `*` does not unconditionally mean all files. [[Bash大括号展开|Brace expansion]] does not check file existence.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`EXPANSION`、`Pathname Expansion` 与 `Pattern Matching`；[官方对应章节](https://www.gnu.org/software/bash/manual/html_node/Filename-Expansion.html)：支持展开顺序、无匹配选项、点与斜杠边界以及模式语义。
- [Missing Semester 2020：Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)，Wildcards：支持课程中的文件匹配用途；本文将其与大括号文本生成区分。
<!-- bilingual-en:start -->
The local Bash manual supports expansion order, pattern syntax, and the unmatched-pattern, dotfile, and slash boundaries. The course motivates filename matching; brace-based text generation is treated separately here.
<!-- bilingual-en:end -->
