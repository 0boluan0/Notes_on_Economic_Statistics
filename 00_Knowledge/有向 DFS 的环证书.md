---
aliases:
  - "有向 DFS 遇到当前递归栈中的祖先就获得有向环证书"
  - Directed DFS back edge certifies a directed cycle
  - 有向 DFS 回边与环
student_os: knowledge-atom
atom_id: MCS-GRAPH-017
atom_type: theorem
status: source-checked
mastery_state: unassessed
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[DFS 可达性与非最短路径]]"
leads_to:
  - "[[无向 DFS 的父边例外]]"
---

# 有向 DFS 遇到当前递归栈中的祖先就获得有向环证书
<!-- bilingual-en:start -->
*An edge to an ancestor on the active directed-DFS stack certifies a directed cycle*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 在 directed DFS 中，若探索边 $u\to v$ 时，$v$ 仍是当前 recursion stack 上的灰色祖先，那么从 $v$ 沿 parent edges 到 $u$ 的 directed tree path，再接 $u\to v$，构成一个 directed cycle。这样的边称为 back edge，并同时给出可检查的 cycle certificate。
> <!-- bilingual-en:start -->
> In directed DFS, suppose edge $u\to v$ is examined while $v$ is a gray ancestor still on the active recursion stack. The directed parent path from $v$ to $u$, followed by $u\to v$, forms a directed cycle. This back edge therefore supplies an explicit cycle certificate.
> <!-- bilingual-en:end -->

活动栈之所以关键，是因为 parent edge 方向恰从祖先指向后代。只有 $v$ 仍是 $u$ 的祖先时，已有 $v\leadsto u$ tree path 才能与新边 $u\to v$ 闭合。若 $v$ 已经完成并变黑，它不在当前祖先链上；$u\to v$ 可能只是指向一个已处理的共享依赖，本身不能宣布成环。
<!-- bilingual-en:start -->
The active stack matters because parent edges point from ancestors to descendants. Only when $v$ remains an ancestor of $u$ is there a parent path $v\leadsto u$ that the new edge $u\to v$ can close. If $v$ is black and finished, it is no longer on the current ancestor chain. The edge may merely point to a completed shared dependency and does not by itself prove a cycle.
<!-- bilingual-en:end -->

## 反向：有环时 full DFS 必遇到祖先边
<!-- bilingual-en:start -->
*Conversely, full DFS encounters an ancestor edge when a cycle exists*
<!-- bilingual-en:end -->

在一个 directed cycle 上取最早被 full DFS 发现的顶点 $v_0$。DFS 在完成 $v_0$ 前会沿 cycle 的可达后继继续访问，最终检查从 cycle 最后访问的顶点回到某个仍活动祖先的边。因此 finite digraph acyclic 当且仅当 full DFS 不出现通向活动祖先的 back edge。
<!-- bilingual-en:start -->
Choose the first vertex $v_0$ discovered on a directed cycle. Before finishing $v_0$, DFS follows reachable successors around the cycle and eventually examines an edge from a later cycle vertex to an ancestor that is still active. Thus a finite digraph is acyclic exactly when full DFS finds no back edge to an active ancestor.
<!-- bilingual-en:end -->

仅用 `visited` 无法区分 gray 与 black。cycle detection 至少还要维护 `active/on_stack` 状态，或等价的 discovery/finish interval；否则看到任何旧顶点都报环会产生 false positive，忽略已访问点又会漏掉真正 back edge 的证书。
<!-- bilingual-en:start -->
A single `visited` bit cannot distinguish gray from black. Directed-cycle detection also needs an `active/on_stack` state or equivalent discovery/finish intervals. Reporting every edge to an old vertex gives false positives, while discarding all visited vertices loses the back-edge certificate.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> DAG 中 $a\to c$ 与 $b\to c$ 共享后继。DFS 完成 $c$ 后再从 $b$ 看到 $c$，为什么不是 cycle？
>
> **答案：**$c$ 已完成，不是 $b$ 的活动祖先；现有 parent edges 不提供 $c\leadsto b$ 的 directed path，所以 $b\to c$ 无法闭合。共享后继不等于循环依赖。
> <!-- bilingual-en:start -->
> In a DAG, $a\to c$ and $b\to c$ share a successor. Why is seeing finished vertex $c$ from $b$ not a cycle?
>
> **Answer:** Finished $c$ is not an active ancestor of $b$, and the parent edges provide no directed path $c\leadsto b$ for $b\to c$ to close. A shared successor is not a cyclic dependency.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT 6.006 Lecture 10: Depth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f3e349e0eb3288592289d2c81e0c4f4d_MIT6_006S20_lec10.pdf)，pp. 4–5：用 active ancestor set 返回 directed cycle，并证明有 cycle 时 full DFS 必遍历到 ancestor edge。
  <!-- bilingual-en:start -->
  Lecture 10 uses the active ancestor set to return a directed cycle and proves that full DFS encounters an ancestor edge whenever a cycle exists.
  <!-- bilingual-en:end -->
