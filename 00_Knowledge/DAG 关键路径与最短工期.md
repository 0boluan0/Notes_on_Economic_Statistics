---
aliases:
  - "Critical-path makespan theorem"
  - "无限处理器下的关键路径定理"
  - "DAG 关键路径与无限处理器调度"
student_os: knowledge-atom
atom_id: MCS-GRAPH-037
atom_set: acyclic-graphs-trees-dags
atom_type: theorem
status: source-checked
requires:
  - "[[DAG 与拓扑序等价]]"
  - "[[拓扑序的线性扩张]]"
part_of:
  - "[[无环图：树、生成树、DAG 与拓扑排序.canvas]]"
---

<!-- bilingual-en:start -->
# Critical paths and minimum completion time
<!-- bilingual-en:end -->

*关键路径与最短工期*

<!-- bilingual-en:start -->
Some tasks can run together; others must wait for earlier tasks to finish. A **critical path** is a dependency route with the largest total task duration. Its tasks cannot overlap, so it sets a lower bound on the time needed to finish the whole project. With enough processors and only these precedence constraints, starting each task as soon as its predecessors finish attains that bound. Task durations do not have to be equal.
<!-- bilingual-en:end -->

有些任务可以一起做，有些必须等前面的任务完成。**关键路径**是任务时长总和最大的依赖路线。路上的任务不能重叠，因此它给整个项目的完成时间一个下界。若处理器足够，且只有这些先后约束，让每项任务在前驱完成后尽早开始，就能达到这个下界；任务时长不必相同。

<!-- bilingual-en:start -->
## Read a schedule before the formula
<!-- bilingual-en:end -->

*先看一个具体调度*

<!-- bilingual-en:start -->
Suppose tasks $A,B$ prepare two inputs and task $C$ combines them. The [[有向无环图|DAG]] has edges $A\to C$ and $B\to C$: each arrow means its tail must finish before its head starts. Let the durations be $2,3,4$ hours for $A,B,C$ respectively. With enough workers, start $A,B$ at time $0$. They finish at times $2,3$, so $C$ starts at time $3$ and finishes at time $7$.
<!-- bilingual-en:end -->

设任务 $A,B$ 准备两个输入，$C$ 合并它们。[[有向无环图|DAG]] 的边是 $A\to C$、$B\to C$：每条箭头表示起点任务必须完成，终点任务才能开始。三项时长依次为 $2,3,4$ 小时。人手足够时，让 $A,B$ 都在时刻 $0$ 开始；它们分别在时刻 $2,3$ 完成，于是 $C$ 在时刻 $3$ 开始，时刻 $7$ 完成。

> [!example] Waiting for the last prerequisite · 等最后一个前驱完成
> <!-- bilingual-en:start -->
> The route $A\to C$ takes $2+4=6$ hours; $B\to C$ takes $3+4=7$ hours and is critical. Adding the durations of all tasks gives $9$ hours of total work, but the project takes only $7$ hours because $A$ and $B$ overlap. A [[拓扑序|topological order]] such as $(A,B,C)$ specifies precedence; its positions are not the tasks' starting times.
> <!-- bilingual-en:end -->
>
> 路线 $A\to C$ 用时 $2+4=6$ 小时；$B\to C$ 用时 $3+4=7$ 小时，是关键路径。全部任务相加是 $9$ 小时工作量，但项目只用 $7$ 小时，因为 $A,B$ 可以重叠。$(A,B,C)$ 这样的[[拓扑序]]只规定先后，列表位置不是任务开始时刻。

<!-- bilingual-en:start -->
## State the scheduling model
<!-- bilingual-en:end -->

*明确调度模型*

<!-- bilingual-en:start -->
Use a finite DAG $G=(V,E)$ whose vertices are tasks. Task $v$ has a specified finite duration $w(v)\ge0$ and, once started, runs without interruption on one processor. All tasks are available from time $0$ subject to their predecessors. An edge $u\to v$ imposes only the finish-before-start inequality: $v$ may start when $u$ finishes, without an additional delay. Processors are identical and sufficiently numerous for every ready task to run; there are no further resource, release-time or deadline constraints. These assumptions describe the theorem's model, not every practical project.
<!-- bilingual-en:end -->

取有限 DAG $G=(V,E)$，顶点是任务。任务 $v$ 有指定的有限时长 $w(v)\ge0$，开始后在一台处理器上不间断运行。所有任务从时刻 $0$ 起可用，只需等待前驱。边 $u\to v$ 只施加“完成后才能开始”的不等式：$u$ 一完成，$v$ 就可以开始，没有额外等待时间。处理器相同且足够多，能让全部就绪任务同时运行；没有其他资源、释放时间或截止日期约束。这些是本定理的模型前提，并非所有实际项目都满足。

<!-- bilingual-en:start -->
A [[游走路径与环|directed path]] follows dependency arrows without repeating a vertex. Its weight is the **sum of its vertex durations**, including both endpoints. Let $L$ be the largest such sum, allowing a path consisting of one task. The project's **makespan** $T$ is the time by which all tasks have finished, measured from time $0$; $T^*$ denotes the smallest feasible makespan. For the empty task set, define both $L$ and $T^*$ to be $0$.
<!-- bilingual-en:end -->

[[游走路径与环|有向路径]]沿依赖箭头走，不重复顶点。本卡的路径权重是**顶点任务的时长和**，包括两个端点。允许只含一个任务的路径，并用 $L$ 表示最大的这种总和。项目的**总工期** $T$ 是从时刻 $0$ 起，到全部任务完成所需的时间；$T^*$ 表示最小可行工期。空任务集约定 $L=T^*=0$。

> [!theorem] The critical-path bound is attainable · 关键路径下界可达
> <!-- bilingual-en:start -->
> Under the model above, the minimum makespan equals the largest sum of task durations along a directed path:
> <!-- bilingual-en:end -->
>
> $$T^*=L.$$
>
> <!-- bilingual-en:start -->
> Unit durations give a special case: $L$ is the number of tasks on a longest path, not its number of edges. A one-task project already takes that task's duration.
> <!-- bilingual-en:end -->
>
> 在上述模型下，最短工期等于有向路径上最大的任务时长总和。单位时长是特例：此时 $L$ 数最长路径上的任务数，不是边数；只有一项任务的项目也需要该任务本身的时间。

<!-- bilingual-en:start -->
## Why no schedule can beat the path
<!-- bilingual-en:end -->

*为什么任何调度都不能快过这条路径*

<!-- bilingual-en:start -->
On a path $v_1\to v_2\to\cdots\to v_k$, each task must wait for the previous one to finish. If the first task starts no earlier than $0$, repeated use of these constraints shows that the last task cannot finish before $w(v_1)+\cdots+w(v_k)$. The whole project cannot finish earlier either. This applies to every path, so every feasible schedule satisfies $T\ge L$, even when processor capacity is limited.
<!-- bilingual-en:end -->

路径 $v_1\to v_2\to\cdots\to v_k$ 上，每项任务都要等前一项完成。第一项不能早于 $0$ 开始，逐次使用这些约束，就知道最后一项不可能早于 $w(v_1)+\cdots+w(v_k)$ 完成。整个项目也不能更早结束。这个结论适用于每条路径，所以任何可行调度都满足 $T\ge L$，处理器受限时也一样。

<!-- bilingual-en:start -->
## Start every task as early as its prerequisites allow
<!-- bilingual-en:end -->

*在前驱允许的最早时刻开始*

<!-- bilingual-en:start -->
Process vertices in a [[拓扑序|topological order]], so every predecessor has already been handled. Let $s(v)$ and $f(v)$ be the assigned start and finish times. With the maximum of an empty predecessor set interpreted as $0$, set
<!-- bilingual-en:end -->

按[[拓扑序]]处理顶点，保证前驱已经处理。用 $s(v),f(v)$ 分别表示安排的开始和完成时刻；把空前驱集合的最大值约定为 $0$，令

$$s(v)=\max_{u\to v} f(u),\qquad f(v)=s(v)+w(v).$$

<!-- bilingual-en:start -->
A source therefore starts at $0$. For every edge $u\to v$, the formula gives $s(v)\ge f(u)$, so all dependencies hold. Enough processors let every task run over its assigned interval, even if many intervals overlap. Thus these times are feasible without pausing any task.
<!-- bilingual-en:end -->

源点因此在 $0$ 开始。对每条边 $u\to v$，公式都保证 $s(v)\ge f(u)$，所以全部依赖成立。处理器足够，便能让每项任务占用安排好的时间段，即使许多时间段重叠，也无需中断任务。因此这是一份可行调度。

<!-- bilingual-en:start -->
The same recurrence computes the largest total duration of a path ending at $v$. For a source that path is just $v$. At a non-source, starting a path at $v$ alone cannot be better than extending an incoming path, since all previous durations are nonnegative. A path arriving through a predecessor $u$ has its best previous total $f(u)$ plus $w(v)$; maximizing over predecessors gives $f(v)$. Induction along the topological order proves this for every vertex. Hence the schedule finishes at $\max_v f(v)=L$, reaching the lower bound and proving the theorem. A path attaining the maximum can be recovered by remembering a predecessor attaining each maximum; ties can give several critical paths.
<!-- bilingual-en:end -->

同一个递推也计算“以 $v$ 结尾的路径的最大时长和”。源点的路径只有自身。对非源点，由于前面的时长均非负，只取 $v$ 本身不会优于延长一条入路径。经前驱 $u$ 到达的最佳路径总时长是 $f(u)+w(v)$；对前驱取最大，就得到 $f(v)$。沿拓扑序归纳，这对每个顶点成立。因此整份调度在 $\max_v f(v)=L$ 完成，达到了下界，也证明了定理。每次记下一个实现最大值的前驱，便能回溯出关键路径；并列最大值可能给出多条关键路径。

<!-- bilingual-en:start -->
Zero-duration tasks are allowed as instantaneous tasks: a successor may start at the same clock time as such a predecessor, while the dependency order is still respected. For positive durations, time strictly increases along each task. The zero-duration extension follows directly from the non-strict timing inequalities above.
<!-- bilingual-en:end -->

零时长任务可以理解为瞬时任务：后继可以与这样的前驱处于同一钟表时刻，依赖顺序仍然得到遵守。时长为正时，每项任务都会推进时间。零时长扩展直接来自上面的非严格时间不等式。

<!-- bilingual-en:start -->
## Unit tasks recover the depth-layer schedule
<!-- bilingual-en:end -->

*单位任务恢复深度分层*

<!-- bilingual-en:start -->
When every $w(v)=1$, write $d(v)=f(v)$ for the number of tasks on a longest path ending at $v$. Then $d(v)=1$ at a source and $d(v)=1+\max_{u\to v}d(u)$ otherwise. Run $v$ during $[d(v)-1,d(v)]$, or call this step $d(v)$. Vertices with the same value can run in parallel because each edge increases $d$ by at least one.
<!-- bilingual-en:end -->

若每个 $w(v)=1$，用 $d(v)=f(v)$ 表示以 $v$ 结尾的最长路径上的任务数。源点满足 $d(v)=1$，其他顶点满足 $d(v)=1+\max_{u\to v}d(u)$。把 $v$ 放在 $[d(v)-1,d(v)]$ 运行，或称为第 $d(v)$ 步。每条边都会让 $d$ 至少增加一，所以相同 $d$ 的任务可以并行。

<!-- bilingual-en:start -->
In [[拓扑序的线性扩张|the reachability partial order]], a **chain** is a set of tasks every two of which are comparable. The maximum chain size equals the maximum path's vertex count: each path is a chain, and a chain can be joined in its precedence order by directed paths; the concatenation cannot repeat a vertex in a DAG. This recovers the textbook's longest-chain statement. MIT MCS uses depth $0$ at a source, so its depth is $d(v)-1$ in the notation here.
<!-- bilingual-en:end -->

在[[拓扑序的线性扩张|可达偏序]]中，**链**是任意两项都可比的一组任务。最大链大小等于最长路径的顶点数：每条路径都是链；反过来，可按链的先后次序，用有向路径把任务连接起来，DAG 中这样的连接不会重复顶点。因此得到教材的最长链表述。MIT MCS 把源点深度记为 $0$，所以教材的 depth 对应本卡的 $d(v)-1$。

> [!warning] Limited processors can prevent equality · 处理器有限时可能达不到下界
> <!-- bilingual-en:start -->
> With $P\ge1$ processors, a schedule also needs at least $W/P$ time, where $W=\sum_v w(v)$ is total work. For unit tasks this becomes the integer bound $\lceil |V|/P\rceil$. Even the larger of this bound and $L$ can be too small. Consider three independent unit tasks, then one task waiting for all three, then three unit tasks each waiting for that middle task. With two processors, the three stages need at least $2+1+2=5$ units. Here $L=3$ and $\lceil7/2\rceil=4$, yet a five-unit schedule is necessary and attainable.
> <!-- bilingual-en:end -->
>
> 若只有 $P\ge1$ 台处理器，还需至少 $W/P$ 时间，其中 $W=\sum_v w(v)$ 是总工作量。单位任务时得到整数下界 $\lceil |V|/P\rceil$。即使取这个下界与 $L$ 中较大的一个，也可能不够。考虑三个独立单位任务，接着一个等待三者全完成的任务，最后再接三个都等待中间任务的单位任务。两台处理器下，三阶段至少需要 $2+1+2=5$ 个单位；这里 $L=3$、$\lceil7/2\rceil=4$，但五单位既必要又可达到。

<!-- bilingual-en:start -->
Unequal durations alone do not destroy the theorem: the weighted recurrence already handles them. Restricted processors or additional constraints can. A particular project may still attain $L$ with finitely many processors, but the unlimited-capacity theorem does not promise that for an arbitrary capacity limit. A legal [[拓扑序的线性扩张|linear order]] supplies a way to perform tasks sequentially; it does not by itself supply an optimal parallel schedule.
<!-- bilingual-en:end -->

时长不等本身不会破坏定理，上面的加权递推已经处理了它。处理器受限或增加其他约束，才可能使关键路径下界无法达到。具体项目也可能只用有限台处理器就达到 $L$，但本定理不会对任意容量上限作出保证。合法的[[拓扑序的线性扩张|线性顺序]]提供串行执行方案，本身并没有给出最优并行调度。

> [!question]- Four tasks on a critical chain · 关键链有四项任务
> <!-- bilingual-en:start -->
> Why might a unit-task project with longest chain size $4$ take more than four steps on two processors? Besides precedence, at most two tasks can occupy a step. For example, nine tasks consisting of a four-task chain and five isolated tasks need at least $\lceil9/2\rceil=5$ steps. The extra tasks do not lengthen the chain, but they consume capacity.
> <!-- bilingual-en:end -->
>
> 单位任务项目的最长链有四项，为什么两台处理器仍可能超过四步？先后关系之外，每步最多容纳两项任务。例如，一条四任务链外加五个孤立任务，共九项，至少需要 $\lceil9/2\rceil=5$ 步。额外任务虽不延长链，却占用处理能力。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=443|MIT Mathematics for Computer Science, PDF443–446]], §10.5.2, definitions10.5.5–10.5.7, theorem10.5.8 and corollary10.5.9: equal-duration parallel scheduling, chains, lower bound and the depth-layer construction. Its source-depth convention is $0$; the unit-step index above is shifted by one.
- Sedgewick and Wayne, [Algorithms §4.4, Critical path method](https://algs4.cs.princeton.edu/44sp/), and [CPM.java](https://algs4.cs.princeton.edu/44sp/CPM.java.html): specified, possibly unequal job durations with as many identical processors as needed; earliest starts obtained through longest paths in a DAG. The source puts durations on job-start/job-end edges; this card gives the equivalent vertex-duration recurrence and its proof. Empty and zero-duration cases are explicitly handled by this card's conventions.
- [[01_Math/07-Mathematics for Computer Science/02_Structures.md#17.2 调度、chain 与 antichain|Course Session17.2]]: unit-task layering and finite-processor workload boundaries. The weighted exact result here uses the stated unlimited-capacity model; the two-processor counterexamples are checked directly.
<!-- bilingual-en:end -->

- MIT《Mathematics for Computer Science》，PDF443–446，§10.5.2、定义10.5.5–10.5.7、定理10.5.8及推论10.5.9：等时长并行调度、链、下界与深度分层构造。其源点深度为 $0$，本卡单位步骤编号比它大一。
- Sedgewick 与 Wayne，[Algorithms §4.4 的 Critical path method](https://algs4.cs.princeton.edu/44sp/) 及 [CPM.java](https://algs4.cs.princeton.edu/44sp/CPM.java.html)：指定且可不相等的任务时长、足量相同处理器，以及由 DAG 最长路径求最早开始时间。来源将任务时长放在任务开始与结束之间的边上，本卡给出等价的顶点时长递推及证明。空集和零时长按本卡明确约定处理。
- 课程 Session17.2：单位任务分层及有限处理器工作量边界。本卡的加权精确结论采用已列明的无限容量模型；两处理器反例直接核算。
