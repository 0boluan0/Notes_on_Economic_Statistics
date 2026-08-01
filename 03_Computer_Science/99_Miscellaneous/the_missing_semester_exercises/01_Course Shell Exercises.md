---
aliases:
  - The Missing Semester Lecture 1 Exercises
  - Missing Semester Course Shell Exercises
tags:
  - computer-science
  - tools
  - exercises
  - shell
  - the-missing-semester
---

# 第 1 讲 课程概览与 shell 练习
<!-- bilingual-en:start -->
*Lecture 1: Course Overview and Shell Exercises*
<!-- bilingual-en:end -->

>[!note]
> 对应主笔记：[[the_missing_semester#第 1 讲 课程概览与 shell]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/course-shell/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//course-shell-solution

## 练习清单
<!-- bilingual-en:start -->
*Exercise Checklist*
<!-- bilingual-en:end -->

1. 确认你在使用类 Unix shell。
   使用 `echo $SHELL` 检查；若在 Windows 上，不要用 `cmd` 或 PowerShell，改用 WSL 或 Linux 虚拟机。
<!-- bilingual-en:start -->

&nbsp;
**1.** Confirm that you are using a Unix-like shell. Check with `echo $SHELL`. On Windows, use WSL or a Linux virtual machine rather than `cmd` or PowerShell.<br>
<!-- bilingual-en:end -->
2. 在 `/tmp` 下新建一个名为 `missing` 的文件夹。
<!-- bilingual-en:start -->

&nbsp;
**2.** Create a directory named `missing` under `/tmp`.<br>
<!-- bilingual-en:end -->
3. 用 `man` 查看 `touch` 的使用手册。
<!-- bilingual-en:start -->

&nbsp;
**3.** Use `man` to read the manual page for `touch`.<br>
<!-- bilingual-en:end -->
4. 用 `touch` 在 `missing` 文件夹中新建一个叫 `semester` 的文件。
<!-- bilingual-en:start -->

&nbsp;
**4.** Use `touch` to create a file named `semester` inside the `missing` directory.<br>
<!-- bilingual-en:end -->
5. 将以下内容一行一行写入 `semester`：

```sh
#!/bin/sh
curl --head --silent https://missing.csail.mit.edu
```
<!-- bilingual-en:start -->

&nbsp;
**5.** Write the two lines shown above to `semester`, one line at a time.<br>
<!-- bilingual-en:end -->

6. 尝试直接执行 `./semester`。如果不能执行，用 `ls` 观察权限位并理解失败原因。
<!-- bilingual-en:start -->

&nbsp;
**6.** Try to run `./semester` directly. If it does not execute, inspect its permission bits with `ls` and explain why it fails.<br>
<!-- bilingual-en:end -->
7. 查看 `chmod` 的手册，例如 `man chmod`。
<!-- bilingual-en:start -->

&nbsp;
**7.** Read the manual page for `chmod`, for example with `man chmod`.<br>
<!-- bilingual-en:end -->
8. 使用 `chmod` 让 `./semester` 可以直接执行，不要使用 `sh semester`。
   进一步思考：shell 为什么知道这个文件应由 `sh` 解析？关键词：`shebang`。
<!-- bilingual-en:start -->

&nbsp;
**8.** Use `chmod` to make `./semester` directly executable; do not invoke it with `sh semester`. Then explain how the shell knows that `sh` should interpret the file. Keyword: `shebang`.<br>
<!-- bilingual-en:end -->
9. 使用 `|` 和 `>`，将 `semester` 输出中的“最后修改日期”写入主目录下的 `last-modified.txt`。
<!-- bilingual-en:start -->

&nbsp;
**9.** Use `|` and `>` to extract the “Last-Modified” date from `semester`'s output and write it to `last-modified.txt` in your home directory.<br>
<!-- bilingual-en:end -->
10. 写一条命令，从 `/sys` 读取你的笔记本电量信息，或者台式机 CPU 温度。
    macOS 没有 `sysfs`，可跳过。
<!-- bilingual-en:start -->

&nbsp;
**10.** Write a command that reads your laptop's battery information, or a desktop CPU's temperature, from `/sys`. You may skip this exercise on macOS because it does not provide `sysfs`.<br>
<!-- bilingual-en:end -->
