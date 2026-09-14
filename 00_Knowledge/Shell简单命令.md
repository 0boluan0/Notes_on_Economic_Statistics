---
aliases:
  - "Bash简单命令由可选赋值、命令词与参数词及重定向组成"
  - "Bash simple command"
student_os: knowledge-atom
atom_id: CS-SHELL-002
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell]]"
related:
  - "[[Shell命令查找]]"
leads_to:
  - "[[Shell引用]]"
  - "[[Shell重定向]]"
---

# Bash简单命令由可选赋值、命令词与参数词及重定向组成

在 Bash 中，简单命令是一组可选的变量赋值、命令词与参数词以及重定向。Shell 先识别语法，再展开适用的词；展开后留下的第一个命令词确定要调用的命令，其余词作为分开的参数传入。重定向语法不作为普通参数传给命令。
<!-- bilingual-en:start -->
In Bash, a simple command consists of optional variable assignments, command and argument words, and redirections. The shell recognizes the syntax and then expands the applicable words. The first remaining command word names the command; the rest become separate arguments. Redirection syntax is not passed as ordinary arguments.
<!-- bilingual-en:end -->

```bash
printf '<%s>\n' 'two words' ''
```

这里命令词为 `printf`，三个参数依次是格式串、`two words`、空字符串；引号本身不进入参数。输出两行 `<two words>` 和 `<>`。因此命令行的可见空格数不能决定实际参数数目。
<!-- bilingual-en:start -->
The command word is `printf`. Its three arguments are the format string, `two words`, and an empty string; the quote characters are not included. The output is `<two words>` followed by `<>`. Visible spaces in the command line therefore do not determine the argument count.
<!-- bilingual-en:end -->

`topic=value` 可以是没有命令名的赋值命令；`topic = value` 却把 `topic` 放在命令位置。[[Shell引用]]影响词如何形成，[[Shell字段分割]]与[[Shell文件名展开]]还可能改变展开后的词数。不要把整个过程简化成“按空格切开后运行第一个单词”。
<!-- bilingual-en:start -->
`topic=value` can be an assignment without a command name, whereas `topic = value` puts `topic` in command position. [[Shell引用|Quoting]] affects word formation, and [[Shell字段分割|word splitting]] and [[Shell文件名展开|pathname expansion]] can change the resulting word count. The process is more than splitting the line at spaces.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`SHELL GRAMMAR: Simple Commands` 与 `SIMPLE COMMAND EXPANSION`；[官方对应章节](https://www.gnu.org/software/bash/manual/html_node/Simple-Command-Expansion.html)：支持赋值、展开后命令与参数、重定向的处理顺序。
- [Missing Semester 2020：Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)，Shell Scripting 开头：支持赋值两侧空格会改变语法的例子。
<!-- bilingual-en:start -->
The local Bash 3.2 manual supports the simple-command components and their processing. The course's opening scripting section supports the assignment-versus-command contrast. The `printf` example applies these rules to preserved argument boundaries.
<!-- bilingual-en:end -->
