---
aliases:
  - "Bash函数是以名称保存并调用一组Shell命令的结构"
  - "Bash shell function"
student_os: knowledge-atom
atom_id: CS-SHELL-017
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell简单命令]]"
related:
  - "[[Shell位置参数]]"
  - "[[Shell别名]]"
leads_to:
  - "[[脚本执行与载入]]"
---

# Bash函数是以名称保存并调用一组Shell命令的结构

Bash 函数把一组命令与名称绑定，供以后像命令一样调用。定义函数只保存命令，不立即执行函数体。普通前台调用在调用它的 Shell 环境中执行，不会仅因函数调用就创建一个新的解释进程。
<!-- bilingual-en:start -->
A Bash function binds a command sequence to a name for later invocation as a command. Defining it stores the commands without executing the body. An ordinary foreground call runs in its caller's shell environment; the call itself does not create a new interpreter process.
<!-- bilingual-en:end -->

```bash
greet() {
    local who="$1"
    printf 'hello, %s\n' "$who"
}
greet 'Ada Lovelace'
```

这里实参通过[[Shell位置参数|位置参数]]进入函数，`local` 将 `who` 限制在该次函数调用的动态局部作用域内：被它调用的函数也能看到这个局部值，调用结束后恢复外层同名变量。`return` 可提前返回状态；没有显式 `return` 时，函数通常返回最后执行命令的状态，而不是把打印内容当成返回值。
<!-- bilingual-en:start -->
The argument enters through [[Shell位置参数|positional parameters]]. `local` makes `who` dynamically local to this invocation: functions called from it can also see the local value, and the outer binding is restored afterward. `return` can return a status early; without it, a function normally returns its last executed command's status, not its printed text.
<!-- bilingual-en:end -->

函数并不自动隔离目录变化或普通变量赋值；这正是函数能封装 `cd` 的原因。但函数若被放在子 Shell、后台或 Bash 3.2 管道阶段中，改变的是那个环境。当前环境与独立环境的边界见[[脚本执行与载入]]。
<!-- bilingual-en:start -->
A function does not automatically isolate directory changes or ordinary variable assignments, which is why it can wrap `cd`. If invoked inside a subshell, in the background, or as a Bash 3.2 pipeline stage, however, it changes that environment. See [[脚本执行与载入|execution versus sourcing]] for the environment boundary.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`Shell Function Definitions`、`FUNCTIONS`、内建 `local` 与 `return`；[官方函数章节](https://www.gnu.org/software/bash/manual/html_node/Shell-Functions.html)：支持定义与调用、动态局部变量、参数与退出状态。
- [Missing Semester 2020：Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)，`mcd` 与 Functions versus scripts：支持把目录变化保留在调用环境中的用途；子环境限制由 Bash 手册补足。
<!-- bilingual-en:start -->
The local Bash manual establishes definition versus invocation, dynamic local variables, arguments, and status. The course's `mcd` example motivates preserving directory changes in the caller; the manual supplies the subshell qualification.
<!-- bilingual-en:end -->
