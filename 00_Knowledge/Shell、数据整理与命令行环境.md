---
aliases:
  - "Shell and Data Wrangling"
  - "Command-Line Environment"
  - "命令行工具"
status: source-checked
---

# Shell、数据整理与命令行环境
<!-- bilingual-en:start -->
*Shell, Data Wrangling, and the Command-Line Environment*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用小而可组合的程序、文件流和自动化脚本高效操作计算环境。
> **具体锚点：** `producer | filter | summary` 让前一个程序输出直接成为后一个输入，不需手工复制临时文件。
> **核心难点：** shell 会展开空格、通配符、变量和命令替换；未正确引用的路径或不可信输入可能改变命令含义。
> **为什么重要：** 搜索、批处理、远程工作和可复现流程都建立在明确的数据流上。
> **继续：** 先理解路径、stdin/stdout/stderr、管道和引用，再写最小脚本；危险命令先解析精确目标。
> <!-- bilingual-en:start -->
> **Problem addressed:** Operate a computing environment efficiently with small composable programs, file streams, and automation scripts.
> **Concrete anchor:** `producer | filter | summary` connects one program's output directly to the next program's input without manually copying temporary files.
> **Central difficulty:** A shell expands spaces, globs, variables, and command substitutions; an unquoted path or untrusted input can change the meaning of a command.
> **Why it matters:** Search, batch processing, remote work, and reproducible workflows all depend on explicit data flow.
> **Continue with:** Understand paths, stdin/stdout/stderr, pipelines, and quoting before writing a small script; resolve exact targets before a destructive command.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [MIT Missing Semester official course](https://missing.csail.mit.edu/2020/) 与本地练习：支持 shell、Git、调试、构建与安全工作流。
> <!-- bilingual-en:start -->
> - The [official MIT Missing Semester course](https://missing.csail.mit.edu/2020/) and local exercises support shell, Git, debugging, build, and security workflows.
> <!-- bilingual-en:end -->

## Shell 模型
<!-- bilingual-en:start -->
*The Shell Model*
<!-- bilingual-en:end -->

Shell 是连接程序、文件和操作系统的命令解释环境。它先解析命令行中的程序名、参数、变量、重定向与管道，再让操作系统启动相应进程；因此同一行里既有“要运行什么”，也有“数据从哪里来、到哪里去”。把复杂任务拆成多个单一用途程序并用文件流连接，是命令行工作的基本模型。
<!-- bilingual-en:start -->
A shell is a command interpreter that connects programs, files, and the operating system. It parses a program name, arguments, variables, redirections, and pipelines before asking the operating system to start processes. One command line therefore expresses both what runs and where its data comes from and goes. Decomposing work into single-purpose programs connected by streams is the basic command-line model.
<!-- bilingual-en:end -->

## 文件系统与路径
<!-- bilingual-en:start -->
*Filesystems and Paths*
<!-- bilingual-en:end -->

绝对路径从根开始，相对路径基于当前目录；`.`/`..` 有具体含义。权限区分读写执行。空格和通配符路径需引用，脚本不要假设当前目录，先确定输入根和输出目标。
<!-- bilingual-en:start -->
An absolute path starts at the filesystem root, while a relative path is interpreted from the current directory; `.` and `..` have specific meanings. Permissions distinguish reading, writing, and execution. Quote paths containing spaces or glob characters, and make a script establish its input root and output target instead of assuming the current directory.
<!-- bilingual-en:end -->

## stdin、stdout、stderr
<!-- bilingual-en:start -->
*Standard Input, Standard Output, and Standard Error*
<!-- bilingual-en:end -->

程序从标准输入读，正常输出到 stdout，诊断到 stderr。`>` 覆盖、`>>` 追加、`<` 输入；覆盖前确认目标。管道 `|` 只连接前者 stdout 到后者 stdin。
<!-- bilingual-en:start -->
A program reads standard input, writes normal output to stdout, and writes diagnostics to stderr. `>` overwrites, `>>` appends, and `<` supplies input; verify the destination before overwriting. A pipeline `|` connects only the former command's stdout to the latter command's stdin.
<!-- bilingual-en:end -->

管道默认返回最后一个命令的状态，前段失败可能被掩盖。可靠脚本要逐步检查，并在 Bash 中按需要启用 `pipefail`；但 shell 选项不能替代明确处理预期失败。
<!-- bilingual-en:start -->
A pipeline normally returns the status of its last command, so an earlier failure may be hidden. A reliable script checks stages and can enable `pipefail` in Bash when appropriate, although shell options do not replace explicit handling of expected failures.
<!-- bilingual-en:end -->

## 文本处理
<!-- bilingual-en:start -->
*Text Processing*
<!-- bilingual-en:end -->

搜索、排序、去重、列选择和正则工具组合处理行流。先抽样检查，再逐步加管道；保留结构化格式时优先专用解析器而非脆弱正则。
<!-- bilingual-en:start -->
Search, sorting, deduplication, column selection, and regular-expression tools compose over line streams. Inspect a sample before extending a pipeline; when structure must be preserved, prefer a format-aware parser to a brittle regular expression.
<!-- bilingual-en:end -->

## shell 引用与变量
<!-- bilingual-en:start -->
*Shell Quoting and Variables*
<!-- bilingual-en:end -->

双引号允许变量展开，单引号按字面；未引用变量会分词和 glob。不要用 `eval` 处理不可信文本。命令替换输出会去尾换行且再次参与分词，需谨慎。
<!-- bilingual-en:start -->
Double quotes allow variable expansion, while single quotes preserve literal text. An unquoted variable undergoes word splitting and globbing. Never use `eval` on untrusted text. Command substitution removes trailing newlines, and its result may be split again unless quoted.
<!-- bilingual-en:end -->

## Worked example：安全统计日志状态码
<!-- bilingual-en:start -->
*Worked Example: Count Log Status Codes Safely*
<!-- bilingual-en:end -->

先用只读管道抽取小样本，确认字段，再对完整文件统计。文件路径始终作为一个参数传递；若日志是 JSON，应改用 JSON parser 而不是假设空格列永远稳定。
<!-- bilingual-en:start -->
Start with a read-only pipeline over a small sample, verify the field, and then count the full file. Pass the path as one argument throughout. If the log is JSON, use a JSON parser instead of assuming that whitespace-delimited columns remain stable.
<!-- bilingual-en:end -->

```bash
log_file='access log.txt'
awk '{print $9}' "$log_file" | sort | uniq -c | sort -nr
```

## 作业控制与终端复用
<!-- bilingual-en:start -->
*Job Control and Persistent Terminals*
<!-- bilingual-en:end -->

前台/后台、signals、`jobs/bg/fg` 管理进程；tmux 等让远程会话可恢复。终止先用正常 signal，确认进程和资源后再升级。
<!-- bilingual-en:start -->
Foreground and background execution, signals, and `jobs`, `bg`, and `fg` manage processes; tools such as tmux make remote sessions recoverable. Terminate with the normal signal first, confirm the process and its resources, and escalate only when necessary.
<!-- bilingual-en:end -->

## 脚本可靠性
<!-- bilingual-en:start -->
*Reliable Shell Scripts*
<!-- bilingual-en:end -->

验证参数、处理失败、用临时目录、准确引用变量并给 dry-run/日志。幂等脚本重复运行不破坏状态；批量删除/覆盖前列出确切对象。
<!-- bilingual-en:start -->
Validate arguments, handle failures, use temporary directories, quote variables accurately, and provide a dry run or log. An idempotent script can be rerun without corrupting state; list exact objects before batch deletion or overwrite.
<!-- bilingual-en:end -->

## 环境与 dotfiles
<!-- bilingual-en:start -->
*Environment and Dotfiles*
<!-- bilingual-en:end -->

PATH 决定命令解析，环境变量传配置。dotfiles 版本化时避免 secrets；不同机器差异用小条件而非复制多套配置。
<!-- bilingual-en:start -->
`PATH` determines command lookup, and environment variables carry configuration. Keep secrets out of versioned dotfiles; express small machine-specific differences as conditions instead of copying entire configurations.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 路径含空格后命令参数变多：检查变量是否始终写成 `"$var"`，而不是事后转义部分字符。
  <!-- bilingual-en:start -->
  A path containing spaces becomes several arguments: verify that the variable is consistently written as `"$var"` instead of escaping selected characters afterward.
  <!-- bilingual-en:end -->
- 管道输出为空：逐段运行并观察 stdout/stderr 与退出状态，找到第一次丢失数据的位置。
  <!-- bilingual-en:start -->
  A pipeline produces no output: run each stage and inspect stdout, stderr, and exit status to locate the first point where data disappears.
  <!-- bilingual-en:end -->
- 脚本在另一目录失败：查找对当前工作目录、PATH、locale 和工具版本的隐含假设。
  <!-- bilingual-en:start -->
  A script fails from another directory: inspect hidden assumptions about the working directory, `PATH`, locale, and tool versions.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 管道 `a | b` 传递的是什么？
<!-- bilingual-en:start -->
*What does the pipeline `a | b` transmit?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 把 a 的标准输出作为 b 的标准输入；a 的 stderr 默认不进入管道。
> <!-- bilingual-en:start -->
> It makes the standard output of `a` the standard input of `b`; by default, `a`'s stderr does not enter the pipe.
> <!-- bilingual-en:end -->

### 为什么 shell 变量通常要写成 `"$var"`？
<!-- bilingual-en:start -->
*Why should a shell variable normally be written as `"$var"`?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 避免值中的空格、通配符或空字符串被再次分词/glob，改变参数边界。
> <!-- bilingual-en:start -->
> Quoting prevents spaces, glob characters, or an empty value from undergoing word splitting or filename expansion and changing argument boundaries.
> <!-- bilingual-en:end -->

### 批量修改前最小安全步骤是什么？
<!-- bilingual-en:start -->
*What is the minimum safety step before a batch modification?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 用只读命令解析并列出精确目标，确认路径不宽泛，再执行可恢复或有备份的操作。
> <!-- bilingual-en:start -->
> Resolve and list exact targets with a read-only command, verify that the path is not overly broad, and then use a recoverable or backed-up operation.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [MIT Missing Semester official course](https://missing.csail.mit.edu/2020/) 与本地练习：支持 shell、Git、调试、构建与安全工作流。
  <!-- bilingual-en:start -->
  The [official MIT Missing Semester course](https://missing.csail.mit.edu/2020/) and local exercises support shell, Git, debugging, build, and security workflows.
  <!-- bilingual-en:end -->
- [GNU Bash Reference Manual](https://www.gnu.org/software/bash/manual/bash.html)：核验解析顺序、引用、管道、重定向、环境与作业控制。
  <!-- bilingual-en:start -->
  The [GNU Bash Reference Manual](https://www.gnu.org/software/bash/manual/bash.html) verifies parsing order, quoting, pipelines, redirections, environments, and job control.
  <!-- bilingual-en:end -->
