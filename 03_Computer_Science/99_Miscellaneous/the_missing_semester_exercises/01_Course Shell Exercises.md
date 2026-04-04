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

>[!note]
> 对应主笔记：[[the_missing_semester#第 1 讲 课程概览与 shell]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/course-shell/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//course-shell-solution

## 练习清单

1. 确认你在使用类 Unix shell。
   使用 `echo $SHELL` 检查；若在 Windows 上，不要用 `cmd` 或 PowerShell，改用 WSL 或 Linux 虚拟机。
2. 在 `/tmp` 下新建一个名为 `missing` 的文件夹。
3. 用 `man` 查看 `touch` 的使用手册。
4. 用 `touch` 在 `missing` 文件夹中新建一个叫 `semester` 的文件。
5. 将以下内容一行一行写入 `semester`：

```sh
#!/bin/sh
curl --head --silent https://missing.csail.mit.edu
```

6. 尝试直接执行 `./semester`。如果不能执行，用 `ls` 观察权限位并理解失败原因。
7. 查看 `chmod` 的手册，例如 `man chmod`。
8. 使用 `chmod` 让 `./semester` 可以直接执行，不要使用 `sh semester`。
   进一步思考：shell 为什么知道这个文件应由 `sh` 解析？关键词：`shebang`。
9. 使用 `|` 和 `>`，将 `semester` 输出中的“最后修改日期”写入主目录下的 `last-modified.txt`。
10. 写一条命令，从 `/sys` 读取你的笔记本电量信息，或者台式机 CPU 温度。
    macOS 没有 `sysfs`，可跳过。
