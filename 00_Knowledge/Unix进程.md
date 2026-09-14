---
student_os: knowledge-atom
atom_id: CS-CLI-006
atom_type: definition
aliases:
  - Unix进程是带有执行状态与系统资源的程序执行实例
  - Unix process
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# Unix进程是带有执行状态与系统资源的程序执行实例
<!-- bilingual-en:start -->
*A Unix process is a program execution instance with execution state and system resources.*
<!-- bilingual-en:end -->

进程包含执行线程及其共享的地址空间、身份、当前目录和打开的资源；PID 在该系统的同时存活进程之间标识它。磁盘上的同一程序文件可以对应多个进程，进程也可能等待、暂停或已退出待回收，并非始终占用 CPU 运行。
<!-- bilingual-en:start -->
A process contains execution threads and shared resources such as an address space, identity, working directory, and open resources. Its PID identifies it among concurrently existing processes on that system. One program file may produce many processes, and a process may wait, stop, or have exited while awaiting collection rather than continuously running on a CPU.
<!-- bilingual-en:end -->

PID、shell 作业编号与进程组 ID 是不同层次的标识；进程组 ID 可以等于组长的 PID。Shell 可以把一个管道中的多个进程视为一个[[Shell作业]]；[[Unix信号]]则可以按明确的进程或进程组目标发送。
<!-- bilingual-en:start -->
PIDs, shell job numbers, and process-group IDs identify different levels; a process-group ID can equal its leader's PID. A shell can treat several pipeline processes as one [[Shell作业|job]], while [[Unix信号|signals]] can target a specified process or process group.
<!-- bilingual-en:end -->

进程结束后 PID 可以被重用。保存了一个旧数字不等于永久保存了任务身份；操作前还需核对所属用户、命令和生命周期，不能仅凭一个名称匹配就批量终止进程。
<!-- bilingual-en:start -->
PIDs may be reused after processes terminate. Retaining an old number does not retain a permanent task identity. Check ownership, command, and lifecycle before acting rather than terminating a group of processes solely by a name match.
<!-- bilingual-en:end -->

## 来源与核验

[OpenBSD intro(2), Definitions](https://man.openbsd.org/intro.2#DEFINITIONS)：支持进程、PID、父进程、进程组和会话的区别；不把该平台的 PID 数值范围推广到其他 Unix 系统。
<!-- bilingual-en:start -->
[OpenBSD intro(2): Definitions](https://man.openbsd.org/intro.2#DEFINITIONS) supports distinctions among processes, PIDs, parents, process groups, and sessions. Its platform-specific PID range is not generalized.
<!-- bilingual-en:end -->

[OpenBSD wait(2)](https://man.openbsd.org/wait.2)：支持子进程退出状态回收及退出不等于立即移除全部记录的边界。
<!-- bilingual-en:start -->
[OpenBSD wait(2)](https://man.openbsd.org/wait.2) supports collection of child exit status and the distinction between exit and immediate removal of all process records.
<!-- bilingual-en:end -->
