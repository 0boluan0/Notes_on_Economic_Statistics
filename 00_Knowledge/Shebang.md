---
aliases:
  - "Shebang是可执行脚本首行用于指定解释器的井号叹号标记"
  - "Script interpreter directive"
student_os: knowledge-atom
atom_id: CS-SHELL-020
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell脚本]]"
related:
  - "[[Unix权限位]]"
  - "[[Shell命令查找]]"
  - "[[脚本执行与载入]]"
---

# Shebang是可执行脚本首行用于指定解释器的井号叹号标记

Shebang 是 Unix 类系统可执行脚本首行的 `#!` 标记，其后指定解释器。直接执行脚本路径时，支持该格式的系统据此启动解释器。它不是 `.sh` 扩展名，也不保证文件具有执行权限或解释器一定可用。
<!-- bilingual-en:start -->
A shebang is the first-line `#!` marker in an executable script on Unix-like systems, followed by its interpreter. When the script pathname is executed directly, a system supporting this format uses it to select the interpreter. It is not a filename extension, nor a guarantee of execute permission or interpreter availability.
<!-- bilingual-en:end -->

```bash
#!/bin/bash
printf '%s\n' hello
```

在 `/bin/bash` 存在且脚本具有适当权限的环境中，直接调用该脚本会请求 Bash。若改为 `bash file` 或 `sh file`，解释器已经由命令行选定；文件中的这行对 Shell 只是注释，不能把显式 `sh` 调用变成 Bash。载入文件也使用当前 Shell，而不按 shebang 切换解释器。
<!-- bilingual-en:start -->
Where `/bin/bash` exists and the script has appropriate permissions, direct invocation requests Bash. With `bash file` or `sh file`, the command line has already selected the interpreter; the shell treats the line as a comment. It cannot turn explicit `sh` invocation into Bash. Sourcing likewise uses the current shell rather than switching according to the shebang.
<!-- bilingual-en:end -->

`#!/usr/bin/env bash` 让 `env` 按 PATH 查找 `bash`，可减少固定安装路径的依赖，但不能保证选中同一版本。解释器参数的处理还受操作系统影响，不能把一行任意复杂的 Shell 命令当作可移植 shebang。
<!-- bilingual-en:start -->
`#!/usr/bin/env bash` asks `env` to find `bash` through PATH, reducing dependence on a fixed installation path without guaranteeing a particular version. Interpreter-argument handling also depends on the operating system; an arbitrary shell command line is not a portable shebang.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`ARGUMENTS`、`COMMENTS` 与 `COMMAND EXECUTION` 的 `#!` 段；[官方脚本章节](https://www.gnu.org/software/bash/manual/html_node/Shell-Scripts.html)：支持直接执行的解释器标记与显式解释器调用的区别。
- 本机 `env(1)`，`DESCRIPTION`；[Missing Semester 2020：Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)，Python script 与 shebang：支持 `env` 经 PATH 选择解释器的用途。
<!-- bilingual-en:start -->
The local Bash manual distinguishes direct script execution from explicitly reading a file with a chosen interpreter. The local `env` manual and the course's shebang example support PATH-based interpreter selection, which does not pin a version.
<!-- bilingual-en:end -->
