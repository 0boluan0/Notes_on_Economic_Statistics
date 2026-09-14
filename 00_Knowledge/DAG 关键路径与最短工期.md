---
aliases:
  - "单位时长且处理器无限时 DAG 最短工期等于最长 chain 大小"
  - "Critical path gives exact makespan for unit tasks and unlimited processors"
  - "DAG 关键路径与无限处理器调度"
student_os: knowledge-atom
atom_id: MCS-GRAPH-037
atom_set: acyclic-graphs-trees-dags
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[DAG 与拓扑序等价]]"
  - "[[拓扑序的线性扩张]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

# 单位时长且处理器无限时 DAG 最短工期等于最长 chain 大小
<!-- bilingual-en:start -->
*With unit-duration tasks and unlimited processors, a DAG's minimum makespan equals its longest-chain size*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 设 finite DAG 的 vertices 是任务，每项恰耗时一单位，edge $u\to v$ 表示 $u$ 必须先完成，且有无限多个相同 processors、没有其他资源约束。若最大 chain 含 $L$ 个任务，则最短总工期恰为 $L$ 个 time steps。
> <!-- bilingual-en:start -->
> Let the vertices of a finite DAG be unit-duration tasks, with $u\to v$ meaning that $u$ must finish before $v$. With unlimited identical processors and no other resource constraints, if the largest chain has $L$ tasks, the minimum makespan is exactly $L$ time steps.
> <!-- bilingual-en:end -->

下界来自 chain：其中任务两两受先后约束，不能共享 time step，所以任何 schedule 至少用 $L$ 步。上界用 depth 分层：令 $d(v)$ 是以 $v$ 结尾的最长 chain 大小，把 $v$ 安排在第 $d(v)$ 步。每条 $u\to v$ 都有 $d(v)\ge d(u)+1$，所以所有依赖都被满足；最大 depth 正好是 $L$，下界可达。
<!-- bilingual-en:start -->
A chain gives the lower bound because its tasks must occupy distinct steps. For the upper bound, let $d(v)$ be the size of a longest chain ending at $v$ and schedule $v$ at step $d(v)$. Every edge $u\to v$ satisfies $d(v)\ge d(u)+1$, so all precedence constraints hold, and the maximum depth is exactly $L$.
<!-- bilingual-en:end -->

三个条件决定了“等号”而不只是下界：unit duration、unlimited processors、只有 precedence constraints。若只有 $P$ 个 processors，总工作量还给下界 $\lceil n/P\rceil$，但两个下界的最大值也未必可达；若任务时长或其他资源约束不同，还需另做 scheduling optimization。单独给一个 topological order 只是串行合法顺序，不是最优 parallel schedule。
<!-- bilingual-en:start -->
Unit durations, unlimited processors, and precedence-only constraints make the lower bound exact. With only $P$ processors, total work adds the lower bound $\lceil n/P\rceil$, whose maximum with the critical-path bound need not be attainable. Unequal durations or other resources require an additional scheduling problem. One topological order is merely a legal sequential order.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一个 unit-task DAG 的最长 chain 有 4 个任务。为什么“至少 4 步”在只有两台处理器时不必等于“恰好 4 步”？
>
> **答案：**critical path 只给依赖下界；两台处理器还限制每步最多完成两项，其他任务的总工作量与排布可能把工期推到 4 步以上。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=443|MIT Mathematics for Computer Science, §10.5.2]] 与 [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=445|Theorem 10.5.8]]：核对 unit-duration、unlimited-processor 假设、critical-path 下界与 depth schedule 的可达性。
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#17.2 调度、chain 与 antichain|Session 17.2]]：核对有限处理器、总工作量与非单位时长时仅保留下界的课程边界。
