---
aliases:
  - The Missing Semester Lecture 7 Exercises
  - Missing Semester Debugging and Profiling Exercises
tags:
  - computer-science
  - tools
  - exercises
  - debugging
  - profiling
  - performance
  - the-missing-semester
---

# 第 7 讲 调试及性能分析 练习
<!-- bilingual-en:start -->
*Lecture 7: Debugging and Profiling Exercises*
<!-- bilingual-en:end -->

>[!note]
> 对应主笔记：[[the_missing_semester#第 7 讲 调试及性能分析]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/debugging-profiling/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//debugging-profiling-solution

## 调试
<!-- bilingual-en:start -->
*Debugging*
<!-- bilingual-en:end -->

1. 在 Linux 上用 `journalctl`、或在 macOS 上用 `log show`，找出最近一天里超级用户登录及执行命令的信息。
   如果没有记录，可以先执行一些无害命令，例如 `sudo ls` 再看。
<!-- bilingual-en:start -->

&nbsp;
**1.** On Linux, use `journalctl`, or on macOS, use `log show`, to find records of superuser logins and commands executed during the past day. If no records exist, first run a harmless command such as `sudo ls` and inspect the logs again.<br>
<!-- bilingual-en:end -->
2. 完成一份 `pdb` 实践教程，并熟悉常用命令：
   - https://github.com/spiside/pdb-tutorial
   - 深入版：https://realpython.com/python-debugging-pdb
<!-- bilingual-en:start -->

&nbsp;
**2.** Complete a hands-on `pdb` tutorial and learn its common commands. Start with https://github.com/spiside/pdb-tutorial and use https://realpython.com/python-debugging-pdb for a more detailed treatment.<br>
<!-- bilingual-en:end -->
3. 安装 `shellcheck`，并检查下面这段脚本：

```sh
#!/bin/sh
## Example: a typical script with several problems
for f in $(ls *.m3u)
do
  grep -qi hq.*mp3 $f \
    && echo -e 'Playlist $f contains a HQ file in mp3 format'
done
```

   找出并修复其中的问题；同时给你的编辑器安装 linter 插件。
<!-- bilingual-en:start -->

&nbsp;
**3.** Install `shellcheck` and use it to inspect the script shown above. Identify and correct its problems, and install a linter plugin for your editor.<br>
<!-- bilingual-en:end -->
4. 进阶：阅读可逆调试资料，并用 `rr` 或 `RevPDB` 做一个可运行示例。
<!-- bilingual-en:start -->

&nbsp;
**4.** Advanced: read about reversible debugging and build a working example with `rr` or `RevPDB`.<br>
<!-- bilingual-en:end -->

## 性能分析
<!-- bilingual-en:start -->
*Profiling*
<!-- bilingual-en:end -->

1. 使用 `cProfile`、`line_profiler`、`memory_profiler` 比较插入排序和快速排序的性能与内存消耗，并继续观察原地快排版本。
   附加题：用 `perf` 看循环次数、缓存命中与丢失。
<!-- bilingual-en:start -->

&nbsp;
**1.** Use `cProfile`, `line_profiler`, and `memory_profiler` to compare the runtime and memory use of insertion sort and quicksort, including an in-place quicksort implementation. As an extension, use `perf` to inspect loop counts and cache hits and misses.<br>
<!-- bilingual-en:end -->
2. 把官方给出的斐波那契 Python 代码保存成可执行文件，安装 `pycallgraph` 与 `graphviz`，生成调用图并比较：
   - 原始版本里 `fib0` 被调用多少次
   - 加上 memoization 之后，每个 `fibN` 被调用多少次
<!-- bilingual-en:start -->

&nbsp;
**2.** Save the provided Fibonacci Python code as an executable file, install `pycallgraph` and `graphviz`, and generate call graphs. Compare how often `fib0` is called in the original version with how often each `fibN` is called after memoization.<br>
<!-- bilingual-en:end -->
3. 用 `python -m http.server 4444` 占用端口，再用 `lsof | grep LISTEN` 找到对应 PID，并用 `kill <PID>` 停掉它。
<!-- bilingual-en:start -->

&nbsp;
**3.** Occupy port 4444 with `python -m http.server 4444`, find the corresponding PID using `lsof | grep LISTEN`, and stop the process with `kill <PID>`.<br>
<!-- bilingual-en:end -->
4. 用 `stress -c 3` 配合 `htop` 观察 CPU 占用。
   再执行 `taskset --cpu-list 0,2 stress -c 3`，观察为什么没有用满 3 个 CPU。
   附加题：用 `cgroups` 实现类似资源限制，并限制 `stress -m` 的内存使用。
<!-- bilingual-en:start -->

&nbsp;
**4.** Run `stress -c 3` and observe CPU usage in `htop`. Then run `taskset --cpu-list 0,2 stress -c 3` and explain why three CPUs are not fully utilized. As an extension, impose a similar resource restriction with `cgroups` and limit the memory available to `stress -m`.<br>
<!-- bilingual-en:end -->
5. 进阶：执行 `curl ipinfo.io`，用 Wireshark 抓取请求与回复报文，并用 `http` 过滤器观察流量。
<!-- bilingual-en:start -->

&nbsp;
**5.** Advanced: run `curl ipinfo.io`, capture the request and response packets in Wireshark, and inspect the traffic with the `http` display filter.<br>
<!-- bilingual-en:end -->
