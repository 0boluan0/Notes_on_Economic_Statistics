---
aliases:
  - BFS shortest-path tree
  - BFS 最短路父树
student_os: knowledge-atom
atom_id: MCS-GRAPH-015
atom_type: distinction
status: source-checked
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[BFS 分层与无权最短路]]"
  - "[[搜索标记与父森林]]"
contrasts_with:
  - "[[DFS 可达性与非最短路径]]"
---

<!-- bilingual-en:start -->
# BFS shortest-path trees
<!-- bilingual-en:end -->

*BFS 最短路树的顺序依赖*

<!-- bilingual-en:start -->
A shortest distance tells us how many edges are needed; a **BFS shortest-path tree** also records one route that attains that distance. There can be several equally short routes. BFS selects one by remembering which vertex first discovered each new vertex, so its chosen tree can depend on processing order even when every computed distance stays the same.
<!-- bilingual-en:end -->

最短距离告诉我们需要经过多少条边；**BFS 最短路树**还记录一条达到该距离的路线。同一个终点可能有多条同样短的路线。BFS 记住每个新顶点首次由谁发现，因此即使计算出的距离完全相同，所选择的树仍可能随处理顺序改变。

<!-- bilingual-en:start -->
Consider a finite directed or undirected graph and a source $s$. Use the FIFO, mark-on-discovery version of [[BFS 分层与无权最短路|BFS]], with distance labels $d[v]$. When processing $u$ first discovers $v$, record $\operatorname{parent}[v]=u$ along with $d[v]=d[u]+1$; later encounters do not overwrite either value. The [[搜索标记与父森林|parent record]] means that the actual edge $u\to v$ was used to discover $v$. The source has distance zero and no parent.
<!-- bilingual-en:end -->

考虑一个有限有向图或无向图，选定源点 $s$。使用 FIFO、发现时即标记的[[BFS 分层与无权最短路|BFS]]，距离标签记为 $d[v]$。处理 $u$ 时若首次发现 $v$，就在写入 $d[v]=d[u]+1$ 的同时记录 $\operatorname{parent}[v]=u$；后续再次遇到 $v$ 不改写这两个值。这个[[搜索标记与父森林|父指针记录]]表示：发现 $v$ 时实际使用了边 $u\to v$。源点距离为零，而且没有父顶点。

<!-- bilingual-en:start -->
## Equal distances can give different parents
<!-- bilingual-en:end -->

*距离相同，父顶点可以不同*

<!-- bilingual-en:start -->
Take the directed edges $s\to a$, $s\to b$, $a\to t$, $b\to t$, and $t\to w$. Both $a$ and $b$ have distance $1$; $t$ has distance $2$ and $w$ has distance $3$. If $s$ lists $a$ before $b$, the FIFO queue processes $a$ first, so $\operatorname{parent}[t]=a$. If it lists $b$ before $a$, then $\operatorname{parent}[t]=b$. The two selected shortest paths to $t$ are $s,a,t$ and $s,b,t$. Both use two edges, and $w$ has parent $t$ in either run. The example also works with undirected edges.
<!-- bilingual-en:end -->

取有向边 $s\to a$、$s\to b$、$a\to t$、$b\to t$、$t\to w$。$a,b$ 的距离都为 $1$，$t$ 的距离为 $2$，$w$ 的距离为 $3$。若 $s$ 的邻接表先列 $a$ 后列 $b$，FIFO 队列先处理 $a$，得到 $\operatorname{parent}[t]=a$；若先列 $b$ 后列 $a$，就得到 $\operatorname{parent}[t]=b$。选中的两条最短路径分别为 $s,a,t$ 与 $s,b,t$，都含两条边；两次运行中 $w$ 的父顶点都是 $t$。无向边也给出同样的例子。

<!-- bilingual-en:start -->
In general, the possible shortest-path predecessors of a reachable $v\ne s$ lie in the **previous** layer: they have an edge to $v$ and distance $d[v]-1$. They are in the same layer as one another, not in the same layer as $v$. Whichever eligible predecessor discovers $v$ first becomes its parent. Changing adjacency order can change this choice when alternatives exist; it need not change it in every graph. Distances are fixed by the graph and source, whereas one selected shortest-path tree need not be unique.
<!-- bilingual-en:end -->

一般地，可达点 $v\ne s$ 的最短路前驱位于**前一层**：它们有边指向 $v$，距离为 $d[v]-1$。这些候选前驱彼此同层，但并不与 $v$ 同层。哪个合法前驱最先发现 $v$，就成为它的父顶点。有多个候选时，改变邻接顺序可能改变选择，但并非每张图都会改变。距离由图和源点确定，被选中的一棵最短路树则未必唯一。

<!-- bilingual-en:start -->
## Why parent records give shortest paths
<!-- bilingual-en:end -->

*父指针为何给出最短路径*

<!-- bilingual-en:start -->
For every discovered non-source vertex, $d[\operatorname{parent}[v]]=d[v]-1$. Following parent records therefore reduces the nonnegative integer label by one at each step. It reaches the only distance-zero vertex, $s$, in exactly $d[v]$ steps and cannot cycle. Reversing the recorded list gives a source-to-$v$ route of that length; the BFS distance theorem shows that this route is shortest.
<!-- bilingual-en:end -->

对每个已发现的非源顶点，都有 $d[\operatorname{parent}[v]]=d[v]-1$。所以沿父指针查询时，非负整数标签每步减少一。恰好经过 $d[v]$ 步就到达唯一距离为零的顶点 $s$，不可能形成循环。将记录下来的顶点列表反转，就得到长度为 $d[v]$ 的源点到 $v$ 的路线；BFS 距离定理保证它最短。

> [!warning] Backtracking is a record lookup · 回溯是在查询记录
> <!-- bilingual-en:start -->
> In a directed graph, a parent edge is directed from parent to child. Looking up $t\mapsto a\mapsto s$ does not assert that edges $t\to a$ or $a\to s$ exist. Reverse that list to obtain the valid forward path $s\to a\to t$.
> <!-- bilingual-en:end -->
> 在有向图中，父边从父顶点指向子顶点。查询 $t\mapsto a\mapsto s$ 的记录，并不声称存在边 $t\to a$ 或 $a\to s$。将列表反转，才得到合法的正向路径 $s\to a\to t$。

<!-- bilingual-en:start -->
After a complete search, let $R$ be the set of vertices reachable from $s$, and let $r$ be its size. Every vertex of $R$ except $s$ has exactly one recorded parent, giving $r-1$ parent edges. The chains connect all of $R$ to $s$, so these edges form a tree on $R$ when directions are ignored; in a directed graph they point outward from the root. Unreachable vertices are outside this tree. If only $s$ is reachable, the tree consists of that single vertex with no edges. For parallel edges, one discovering edge can be selected; a parent-vertex record alone does not distinguish parallel edge identities.
<!-- bilingual-en:end -->

完整搜索结束后，令 $R$ 为从 $s$ 可达的顶点集，$r$ 为其顶点数。除 $s$ 外，$R$ 中每个顶点恰有一条记录的父边，共有 $r-1$ 条。父链把这些顶点全部连到 $s$，因此忽略方向后，它们构成 $R$ 上的一棵树；有向图中这些边从根向外延伸。不可达点不属于这棵树。若只有 $s$ 可达，树就只有一个顶点、没有边。存在平行边时，可选定其中一条发现边；仅记录父顶点并不能区分平行边的身份。

<!-- bilingual-en:start -->
## Compressed output and its limits
<!-- bilingual-en:end -->

## 输出压缩与边界

<!-- bilingual-en:start -->
Keeping every complete source-to-vertex path can duplicate much of the same information. On a chain of $n$ vertices starting at one end, the paths contain $0,1,\ldots,n-1$ edges, so separate lists store $n(n-1)/2$ edge occurrences in total. One parent entry per vertex instead uses only linear space, $O(n)$, rather than potentially quadratic space, $\Omega(n^2)$. Reconstruct a requested path when needed; a path of $k$ edges takes $k$ parent steps.
<!-- bilingual-en:end -->

把源点到每个顶点的完整路径都单独保存，会大量重复信息。对从一端出发的 $n$ 顶点链，各路径分别含 $0,1,\ldots,n-1$ 条边，单独列表合计保存 $n(n-1)/2$ 次边的出现。每个顶点只存一个父记录则只需线性空间 $O(n)$，不必承担可能达到二次量级 $\Omega(n^2)$ 的路径列表。需要某条路径时再重建；含 $k$ 条边的路径只需查询 $k$ 次父记录。

<!-- bilingual-en:start -->
A single parent array represents **one shortest path per reachable target**, not all shortest paths. The diamond example discards one of the two choices at $t$. To enumerate every shortest path, one possible approach is to retain all predecessor edges from the previous layer and explore their combinations. Counting paths need not store all those edges: it can instead accumulate counts while processing the layers. Neither operation is provided by the single-parent output alone.
<!-- bilingual-en:end -->

单个父数组为**每个可达终点表示一条最短路径**，不表示所有最短路径。上面的菱形例在 $t$ 处舍弃了一个候选。若要枚举所有最短路径，一种办法是保存从前一层进入的全部合法前驱边，再展开它们的组合；若只计数，则可以在分层处理时累加数量，不必保存所有这些边。仅有单父指针输出，本身并不完成这两项工作。

<!-- bilingual-en:start -->
A valid parent chain proves reachability immediately. Failure to discover a vertex proves unreachability only after the whole reachable search has finished. The root is a special case: it is reachable by the zero-length path despite having no parent. Also distinguish the [[搜索树与 MST 的目标差异|shortest-path tree objective from minimum spanning tree weight]]: BFS minimizes the number of edges on each root-to-vertex path, not the total weight of all tree edges. In an undirected triangle with weights $w(sa)=100$, $w(sb)=1$, $w(ab)=1$, BFS selects both direct edges from $s$, of total weight $101$, while the two unit-weight edges give a spanning tree of total weight $2$.
<!-- bilingual-en:end -->

合法父链可以立即证明可达；未发现某个顶点，则只有在搜索完全部可达区域后才能证明不可达。根是特殊情况：虽然没有父顶点，它仍可通过零长度路径到达自己。另外要区分[[搜索树与 MST 的目标差异|最短路树与最小生成树的目标]]：BFS 最小化每条根到顶点路径的边数，而非整棵树全部边的权重和。若无向三角形的权重为 $w(sa)=100$、$w(sb)=1$、$w(ab)=1$，BFS 会选从 $s$ 出发的两条直达边，总权重为 $101$；但两条单位权重边给出的生成树总权重只有 $2$。

> [!question]- Can two different parent arrays both be correct? · 两个不同父数组能都正确吗？
> <!-- bilingual-en:start -->
> Yes. With the correct BFS distances, each non-root parent must be joined to its child by a valid forward edge and have distance one less. These conditions make the parent chain reach $s$ in the shortest number of steps. Equal distances do not require identical parents.
> <!-- bilingual-en:end -->
> 能。已知距离是正确的 BFS 距离时，每个非根顶点的父顶点必须通过合法正向边连接到它，并且距离恰少一。这样父链就会以最少步数回到 $s$。距离相同并不要求父顶点相同。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.006, Lecture 9: Breadth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/196a95604877d326c6586e60477b59d4_MIT6_006S20_lec9.pdf), pp. 3–4: parent representation, reconstructing one shortest path, the quadratic cost of storing all source paths separately, and the layer proof. The directed diamond above derives the effect of tie-breaking explicitly.
- [MIT 6.006, Recitation 13: Breadth-First Search](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/6d8aaabf67d391877f978172b293d7be_MIT6_006F11_rec13.pdf), pp. 1–2: distance and reachability conventions and FIFO processing. The first-discovery parent rule is explained and proved above.
<!-- bilingual-en:end -->

- MIT 6.006 Lecture 9，第 3–4 页：父指针表示、重建一条最短路径、分别保存全部源点路径的二次空间成本与分层证明；上面的有向菱形例直接推导并列情况下的选择差异。
- MIT 6.006 Recitation 13，第 1–2 页：距离、可达性约定与 FIFO 处理；上文解释并证明了保留首次发现父记录的规则。
