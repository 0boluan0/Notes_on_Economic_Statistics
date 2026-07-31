---
aliases:
  - The Missing Semester Lecture 2 Exercises
  - Missing Semester Shell Tools Exercises
tags:
  - computer-science
  - tools
  - exercises
  - shell
  - bash
  - the-missing-semester
---

# 第 2 讲 Shell 工具和脚本 练习
<!-- bilingual-en:start -->
*Lecture 2: Shell Tools and Scripting Exercises*
<!-- bilingual-en:end -->

>[!note]
> 对应主笔记：[[the_missing_semester#第 2 讲 Shell 工具和脚本]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/shell-tools/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//shell-tools-solution

## 练习清单
<!-- bilingual-en:start -->
*Exercise Checklist*
<!-- bilingual-en:end -->

1. 阅读 `man ls`，写出一条 `ls` 命令，使其输出同时满足：
   - 包含隐藏文件
   - 文件大小使用人类可读格式
   - 按最近修改时间排序
   - 输出带颜色
<!-- bilingual-en:start -->
1. Read `man ls`, then write an `ls` command whose output includes hidden files, uses human-readable file sizes, is ordered by most recent modification time, and is colorized.
<!-- bilingual-en:end -->
2. 编写两个 bash 函数 `marco` 和 `polo`：
   - 执行 `marco` 时，记录当前工作目录
   - 执行 `polo` 时，无论当前在哪，都能 `cd` 回 `marco` 记录的目录
   - 可将函数写到 `marco.sh`，再用 `source marco.sh` 重载
<!-- bilingual-en:start -->
2. Write two Bash functions, `marco` and `polo`. Running `marco` should record the current working directory; running `polo` from anywhere should `cd` back to that directory. You may place the functions in `marco.sh` and reload them with `source marco.sh`.
<!-- bilingual-en:end -->
3. 写一段 bash 脚本，反复运行下面这段“很少失败”的脚本，直到它失败为止：
   - 记录标准输出和标准错误到文件
   - 失败后打印所有捕获内容
   - 附加题：统计失败前总共运行了多少次

```bash
#!/usr/bin/env bash

n=$(( RANDOM % 100 ))

if [[ n -eq 42 ]]; then
   echo "Something went wrong"
   >&2 echo "The error was using magic numbers"
   exit 1
fi

echo "Everything went according to plan"
```
<!-- bilingual-en:start -->
3. Write a Bash script that repeatedly runs the “rarely failing” program shown above until it fails. Capture both standard output and standard error in files, print everything captured after the failure, and, as an extension, report how many successful runs occurred before the failure.
<!-- bilingual-en:end -->

4. 写一条命令，递归查找一个目录下所有 HTML 文件，并把它们打包成 zip。
   - 必须能正确处理带空格的文件名
   - 提示：`find`、`xargs`、`-print0`、`-0`
   - macOS 用户注意 BSD `find` 与 GNU 版本差异
<!-- bilingual-en:start -->
4. Write a command that recursively finds every HTML file under a directory and packages the files into a zip archive. It must handle filenames containing spaces correctly. Useful tools and options include `find`, `xargs`, `-print0`, and `-0`; on macOS, account for differences between BSD and GNU `find`.
<!-- bilingual-en:end -->
5. 进阶：写一条命令或脚本，递归找出某目录中最近修改的文件。
   更进一步：能否按修改时间列出全部文件？
<!-- bilingual-en:start -->
5. Advanced: write a command or script that recursively finds the most recently modified file in a directory. As a further extension, list every file in modification-time order.
<!-- bilingual-en:end -->
