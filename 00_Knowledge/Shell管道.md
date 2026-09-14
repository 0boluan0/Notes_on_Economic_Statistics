---
aliases:
  - "Shell管道把前一命令的标准输出连接到后一命令的标准输入"
  - "Shell pipeline"
student_os: knowledge-atom
atom_id: CS-SHELL-013
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[标准流]]"
related:
  - "[[重定向顺序]]"
leads_to:
  - "[[管道退出状态]]"
  - "[[文件名安全传参]]"
---

# Shell管道把前一命令的标准输出连接到后一命令的标准输入

管道是用 `|` 组合命令的结构：除最后一段外，每段的 stdout 连接到下一段的 stdin。本页具体执行行为以 Bash 为准。普通 `|` 不自动传递 stderr，也不把前一命令输出的词变成后一命令的参数。
<!-- bilingual-en:start -->
A pipeline combines commands with `|`: each stage except the last connects its standard output to the next stage's standard input. Concrete execution behavior here is Bash. Ordinary `|` neither forwards standard error automatically nor turns output words into the next command's arguments.
<!-- bilingual-en:end -->

```bash
printf '%s\n' pear apple pear | sort | uniq -c
```

`sort` 从 stdin 读取三行并排序；`uniq -c` 从自己的 stdin 接收排序结果并统计相邻重复行。Shell 不需要先把整份输出保存为一个文本参数，也不会要求上一段完全结束才启动下一段。
<!-- bilingual-en:start -->
`sort` reads and sorts three lines from standard input. `uniq -c` receives the sorted stream on its own standard input and counts adjacent duplicates. The shell need not save the complete output as one text argument or wait for one stage to finish before starting the next.
<!-- bilingual-en:end -->

管道传输的是字节，不天然保证“每次读到一整行”；行、字段或记录的边界由相邻程序的格式契约决定。显式重定向还可覆盖默认管道连接，见[[重定向顺序]]。整条管道的成功判断则另见[[管道退出状态]]。
<!-- bilingual-en:start -->
A pipe transports bytes; it does not inherently guarantee one complete line per read. Adjacent programs' format contracts determine lines, fields, or records. Explicit redirections can override the pipe connection, as explained by [[重定向顺序|redirection order]]. Success of the whole pipeline is a separate [[管道退出状态|status rule]].
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`Pipelines`；[官方管道章节](https://www.gnu.org/software/bash/manual/html_node/Pipelines.html)：支持 stdin/stdout 连接、独立阶段执行及重定向顺序。
- 本机 `pipe(2)`，`DESCRIPTION`：支持读写描述符之间的数据流接口，而不是自动构造行记录或参数。
- [Missing Semester 2020：The Shell](https://missing.csail.mit.edu/2020/course-shell/)，Connecting programs；[Shell Tools 练习 4](https://missing.csail.mit.edu/2020/shell-tools/)：支持命令组合以及 stdin 与参数接口的区别。
<!-- bilingual-en:start -->
The local Bash manual establishes stream connections, separate pipeline stages, and redirection order. The local `pipe(2)` description supplies the data-flow interface rather than automatic line or argument construction. The course introduces composition, and its fourth shell-tools exercise distinguishes standard-input interfaces from argument interfaces. The sorting example applies those connections to line-oriented tools.
<!-- bilingual-en:end -->
