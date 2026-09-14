---
aliases:
  - "Shell是读取、解释并执行命令的程序"
  - "Command shell"
student_os: knowledge-atom
atom_id: CS-SHELL-001
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
related:
  - "[[标准流]]"
leads_to:
  - "[[Shell简单命令]]"
  - "[[Shell脚本]]"
---

# Shell是读取、解释并执行命令的程序

Shell 是命令解释器：它从交互输入、文件或字符串读取命令，按自己的语言规则解释并执行。这里的具体语法以 Bash 为准；终端是输入输出界面，Shell 是运行在其中的程序，两者不等同。
<!-- bilingual-en:start -->
A shell is a command interpreter: it reads commands interactively, from a file, or from a string, then interprets and executes them according to its language. The concrete syntax here is Bash. A terminal is an input/output interface; a shell is a program that can run through it.
<!-- bilingual-en:end -->

执行并不总是“启动一个外部程序”。Bash 可以执行自己的内建命令、[[Shell函数]]和外部程序；例如 `cd` 需要改变调用 Shell 的当前目录。具体名称怎样被解析到命令，由[[Shell命令查找]]决定。
<!-- bilingual-en:start -->
Execution does not always mean launching an external program. Bash can execute builtins, [[Shell函数|functions]], and external programs. For example, `cd` must change the calling shell's current directory. [[Shell命令查找|Command lookup]] determines what a name resolves to.
<!-- bilingual-en:end -->

```bash
printf '%s\n' 'hello'
```

在通常的 Bash 环境中，这调用内建 `printf`，输出一行 `hello`。后续可以通过[[Shell管道]]连接其他命令，或将同样的语言保存在脚本中。
<!-- bilingual-en:start -->
In an ordinary Bash environment, this calls the `printf` builtin and prints one line. The same language can connect commands with a [[Shell管道|pipeline]] or be saved in a script.
<!-- bilingual-en:end -->

## 来源与核验

- [Missing Semester 2020：The Shell](https://missing.csail.mit.edu/2020/course-shell/)，What is the shell? 与 Using the shell：支持命令解释器和课程采用 Bash 的范围。
- GNU Bash 3.2 本机 `bash(1)`，`DESCRIPTION`、`COMMAND EXECUTION`；[官方对应说明](https://www.gnu.org/software/bash/manual/html_node/What-is-a-shell_003f.html)：支持交互与文件输入、内建命令和外部执行的区别。
<!-- bilingual-en:start -->
The course establishes the interpreter model and its Bash scope. The locally supplied GNU Bash 3.2 manual, under `DESCRIPTION` and `COMMAND EXECUTION`, supports the input modes and the distinction between builtins and external execution; the link points to the corresponding official explanation.
<!-- bilingual-en:end -->
