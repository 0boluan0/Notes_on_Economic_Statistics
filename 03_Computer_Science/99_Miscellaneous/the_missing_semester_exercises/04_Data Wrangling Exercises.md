---
aliases:
  - The Missing Semester Lecture 4 Exercises
  - Missing Semester Data Wrangling Exercises
tags:
  - computer-science
  - tools
  - exercises
  - data
  - shell
  - the-missing-semester
---

# 第 4 讲 数据整理 练习

>[!note]
> 对应主笔记：[[the_missing_semester#第 4 讲 数据整理]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/data-wrangling/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//data-wrangling-solution

## 练习清单

1. 完成这份简短的交互式正则教程：
   https://regexone.com/
2. 统计 `/usr/share/dict/words` 中满足下列条件的单词：
   - 至少包含三个 `a`
   - 不以 `'s` 结尾
   然后继续回答：
   - 这些单词中最常见的三个末尾两个字母组合是什么
   - 一共存在多少种词尾两字母组合
   - 挑战题：哪种组合从未出现
3. 思考为什么下面这种“原地替换”是坏主意：

```sh
sed s/REGEX/SUBSTITUTION/ input.txt > input.txt
```

   它是不是 `sed` 独有的问题？请查 `man sed` 找出正确做法。
4. 找出最近十次开机的：
   - 平均启动时间
   - 中位数
   - 最长启动时间
   Linux 用 `journalctl`，macOS 用 `log show`，从开机开始与启动完成对应日志中提取时间戳。
5. 比较前三次重启启动日志中“不共享”的信息。
   建议拆成几步：
   - 提取前三次启动日志
   - 去掉总会变化的部分，例如时间戳
   - 去重并计数
   - 删除出现 3 次的行
6. 在网上找一个公开数据集，用 `curl` 获取并提取两列数值数据。
   - HTML 数据可尝试 `pup`
   - JSON 数据可尝试 `jq`
   然后：
   - 用一条命令找出某一列的最小值和最大值
   - 用另一条命令求两列差值的总和
