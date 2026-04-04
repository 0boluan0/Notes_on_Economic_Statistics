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

>[!note]
> 对应主笔记：[[the_missing_semester#第 2 讲 Shell 工具和脚本]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/shell-tools/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//shell-tools-solution

## 练习清单

1. 阅读 `man ls`，写出一条 `ls` 命令，使其输出同时满足：
   - 包含隐藏文件
   - 文件大小使用人类可读格式
   - 按最近修改时间排序
   - 输出带颜色
2. 编写两个 bash 函数 `marco` 和 `polo`：
   - 执行 `marco` 时，记录当前工作目录
   - 执行 `polo` 时，无论当前在哪，都能 `cd` 回 `marco` 记录的目录
   - 可将函数写到 `marco.sh`，再用 `source marco.sh` 重载
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

4. 写一条命令，递归查找一个目录下所有 HTML 文件，并把它们打包成 zip。
   - 必须能正确处理带空格的文件名
   - 提示：`find`、`xargs`、`-print0`、`-0`
   - macOS 用户注意 BSD `find` 与 GNU 版本差异
5. 进阶：写一条命令或脚本，递归找出某目录中最近修改的文件。
   更进一步：能否按修改时间列出全部文件？
