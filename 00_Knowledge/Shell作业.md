---
student_os: knowledge-atom
atom_id: CS-CLI-007
atom_type: definition
aliases:
  - Shell作业是shell作为一个控制单位追踪的管道进程集合
  - Shell job
status: source-checked
requires:
  - "[[Unix进程]]"
  - "[[Shell管道]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# Shell作业是shell作为一个控制单位追踪的管道进程集合
<!-- bilingual-en:start -->
*A shell job is a collection of pipeline processes tracked as one control unit by a shell.*
<!-- bilingual-en:end -->

在支持作业控制的 Bash 中，每条管道关联一个作业；单条命令也可以形成作业。Shell 在自己的作业表里追踪其运行、暂停或完成状态，用 `jobs` 查询，用 `%1` 这样的 jobspec 指向该表中的编号。
<!-- bilingual-en:start -->
In Bash with job control, each pipeline has an associated job, and a single command can form one. The shell tracks running, stopped, and completed states in its own job table. `jobs` queries that table, while a jobspec such as `%1` names an entry in it.
<!-- bilingual-en:end -->

例如 `producer | consumer` 可能包含两个进程，但作为一个作业一起接受前台、后台和暂停控制。`%1` 不是 PID `1`；另一 shell 的 `%1` 也不是同一作业。启动输出中的 PID、`$!` 与整个作业的进程集合不能不加区别地混用。
<!-- bilingual-en:start -->
For example, `producer | consumer` may contain two processes while forming one job for foreground, background, and stop control. `%1` is not PID `1`, and `%1` in another shell does not identify the same job. Do not conflate a displayed PID or `$!` with the complete set of processes belonging to a job.
<!-- bilingual-en:end -->

`jobs` 不是整台机器的进程清单。先确定需要管理的是当前 shell 的作业还是系统中的某个[[Unix进程]]，再选择查询和控制对象。
<!-- bilingual-en:start -->
`jobs` is not a machine-wide process list. First determine whether the target is a job owned by the current shell or a particular system [[Unix进程|process]].
<!-- bilingual-en:end -->

## 来源与核验

[GNU Bash Manual §7.1, Job Control Basics](https://www.gnu.org/software/bash/manual/html_node/Job-Control-Basics.html)：核对每条管道关联作业、jobspec 与进程组；本机 `bash(1)` JOB CONTROL 与 `help jobs` 交叉核对。
<!-- bilingual-en:start -->
[GNU Bash Manual §7.1: Job Control Basics](https://www.gnu.org/software/bash/manual/html_node/Job-Control-Basics.html) supports pipeline jobs, jobspecs, and process groups, cross-checked with installed `bash(1)` and `help jobs`.
<!-- bilingual-en:end -->
