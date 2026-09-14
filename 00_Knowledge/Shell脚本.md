---
aliases:
  - "Shell脚本是将Shell命令保存在文本文件中供解释执行的程序"
  - "Shell script"
student_os: knowledge-atom
atom_id: CS-SHELL-018
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell]]"
related:
  - "[[Shell位置参数]]"
leads_to:
  - "[[脚本执行与载入]]"
  - "[[Shebang]]"
---

# Shell脚本是将Shell命令保存在文本文件中供解释执行的程序

Shell 脚本是保存 Shell 命令的文本程序，可包含变量、条件、循环和函数。本页例子由 Bash 解释；文件扩展名 `.sh` 本身既不选择解释器，也不保证内容符合任意 `sh` 的语法。
<!-- bilingual-en:start -->
A shell script is a text program containing shell commands, possibly including variables, conditions, loops, and functions. Bash interprets the examples here. A `.sh` extension neither selects the interpreter nor guarantees compatibility with every `sh`.
<!-- bilingual-en:end -->

例如将下面内容保存为练习文件 `greet.sh`：
<!-- bilingual-en:start -->
For example, save the following as the practice file `greet.sh`:
<!-- bilingual-en:end -->

```bash
printf 'hello, %s\n' "$1"
```

运行 `bash greet.sh 'Ada Lovelace'` 时，新 Bash 从文件读取命令，实参通过 `$1` 提供，输出 `hello, Ada Lovelace`。这里是显式选择解释器；直接以 `./greet.sh` 调用还涉及权限与 [[Shebang]]。
<!-- bilingual-en:start -->
With `bash greet.sh 'Ada Lovelace'`, a new Bash reads the file, receives the argument through `$1`, and prints `hello, Ada Lovelace`. This explicitly selects the interpreter. Direct invocation as `./greet.sh` additionally involves permissions and the [[Shebang|shebang]].
<!-- bilingual-en:end -->

脚本也可以被当前 Shell 载入；这不是与执行脚本完全相同的环境行为。复用一段文本之前，应明确选择[[脚本执行与载入|执行还是载入]]，并知道它会读取或修改哪些输入、文件及环境状态。
<!-- bilingual-en:start -->
A current shell can also source a script, which has different environment behavior from executing it. Before reusing the text, choose [[脚本执行与载入|execution or sourcing]] deliberately and identify which inputs, files, and environment state it accesses or changes.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`ARGUMENTS`、`COMMAND EXECUTION` 与内建 `source`；[官方脚本章节](https://www.gnu.org/software/bash/manual/html_node/Shell-Scripts.html)：支持文件输入、参数、解释执行与载入的区别。
- [Missing Semester 2020：Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)，Shell Scripting：支持脚本作为包含控制流和函数的命令程序。
<!-- bilingual-en:start -->
The local Bash manual supports reading commands from a file, argument passing, and the distinction from sourcing. The course introduces scripts as command programs containing control flow and functions.
<!-- bilingual-en:end -->
