---
aliases:
  - "Bash重定向在执行命令前改变其文件描述符的连接"
  - "Bash redirection"
student_os: knowledge-atom
atom_id: CS-SHELL-011
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[标准流]]"
related:
  - "[[Unix权限位]]"
leads_to:
  - "[[重定向顺序]]"
---

# Bash重定向在执行命令前改变其文件描述符的连接

Bash 重定向是由 Shell 解释、在命令执行前打开、复制、关闭或重新连接文件描述符的语法。`< file` 默认连接 stdin，`> file` 默认连接 stdout，`2> file` 连接 stderr；这些符号与文件操作数不会作为普通参数交给被调用命令。
<!-- bilingual-en:start -->
Bash redirection is shell-interpreted syntax that opens, duplicates, closes, or reconnects file descriptors before command execution. `< file` defaults to standard input, `> file` to standard output, and `2> file` targets standard error. The operators and their file operands are not ordinary command arguments.
<!-- bilingual-en:end -->

对可写的普通文件，默认 `>` 会创建不存在的目标或将已有目标截断为零长度；`>>` 则以追加方式打开，目标不存在时也可创建。`noclobber` 等选项会改变已有文件的处理。打开失败时，重定向失败，不能据此假定命令已运行。
<!-- bilingual-en:start -->
For writable regular files, ordinary `>` creates a missing target or truncates an existing target to zero length. `>>` opens for appending and can also create a missing target. Options such as `noclobber` change handling of existing files. An open failure makes the redirection fail; it does not imply that the command ran.
<!-- bilingual-en:end -->

只在临时练习目录中运行下面的写入例子：先写入 `first`，再追加 `second`，最后通过 stdin 读取两行。
<!-- bilingual-en:start -->
Run this writing example only in a temporary practice directory. It writes `first`, appends `second`, then reads both lines through standard input.
<!-- bilingual-en:end -->

```bash
printf '%s\n' first > report.txt
printf '%s\n' second >> report.txt
cat < report.txt
```

因为文件由当前 Shell 打开，给后面的程序加 `sudo` 不会自动赋予前面的重定向更高权限。`command > file` 也不是“成功后才保存”：命令失败前目标就可能已经被截断。
<!-- bilingual-en:start -->
Because the current shell opens the file, placing `sudo` before the invoked program does not automatically elevate the redirection. Nor does `command > file` mean save only on success: the target may already be truncated when the command fails.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`REDIRECTION`、`Redirecting Output`、`Appending Redirected Output`；[官方对应章节](https://www.gnu.org/software/bash/manual/html_node/Redirections.html)：支持打开时点、截断、追加、noclobber 与打开失败。
- [Missing Semester 2020：The Shell](https://missing.csail.mit.edu/2020/course-shell/)，Connecting programs 与 A versatile and powerful tool：支持重定向由 Shell 处理以及 `sudo echo … > …` 的权限边界。
<!-- bilingual-en:start -->
The local Bash manual supports when files are opened, truncation, appending, `noclobber`, and redirection failure. The course's redirection and privilege example establishes that the shell, not the invoked utility, handles the redirection.
<!-- bilingual-en:end -->
