---
aliases:
  - "Hall 缺额集合是不存在全覆盖匹配的证书"
  - Hall 瓶颈
  - Hall bottleneck certificate
  - Deficient set
student_os: knowledge-atom
atom_id: MCS-MATCH-009
atom_type: certificate
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hall 定理]]"
part_of:
  - "[[二分图匹配与 Hall 定理.canvas]]"
---

# Hall 缺额集合是不存在全覆盖匹配的证书

<!-- bilingual-en:start -->
*A Hall-deficient set certifies that no matching can cover the designated side*
<!-- bilingual-en:end -->

> [!summary] 原子证书
> 若某个 $S\subseteq L$ 满足
> $$
> |N(S)|<|S|,
> $$
> 那么 $S$ 是 Hall-deficient set 或 bottleneck。它直接证明不存在覆盖 $L$ 的 matching：$|S|$ 个左点只能争用更少的 $|N(S)|$ 个右点。
>
> <!-- bilingual-en:start -->
> If some $S\subseteq L$ satisfies
> $$
> |N(S)|<|S|,
> $$
> then $S$ is a Hall-deficient set or bottleneck. It directly certifies that no matching can cover $L$: $|S|$ left vertices have access to fewer than $|S|$ right vertices.
> <!-- bilingual-en:end -->

这个证书不需要枚举所有可能的 matchings。pigeonhole principle 已经足够：matching 不允许两个左点共用一个右点，所以 $S$ 中至多 $|N(S)|$ 个点能够被覆盖。
<!-- bilingual-en:start -->
The certificate avoids enumerating possible matchings. The pigeonhole principle is enough: because a matching cannot assign two left vertices to one right vertex, at most $|N(S)|$ vertices of $S$ can be covered.
<!-- bilingual-en:end -->

例如 $a$ 与 $b$ 都只能连接项目 $1$。虽然每个左点的 degree 都是 $1$，但对 $S=\{a,b\}$ 有 $N(S)=\{1\}$，于是 $|N(S)|=1<2=|S|$。这说明“每个人至少有一个选择”不是全覆盖条件。
<!-- bilingual-en:start -->
For example, suppose both $a$ and $b$ can use only project $1$. Each left vertex has degree one, yet for $S=\{a,b\}$ we have $N(S)=\{1\}$, so $|N(S)|=1<2=|S|$. Thus “everyone has at least one option” is not a full-coverage condition.
<!-- bilingual-en:end -->

发现缺额集合立即证明不可行；反过来，当最大 matching 未覆盖 $L$ 时，[[交替搜索的增广与缺额证书|alternating search]] 可以从搜索前沿构造一个这样的集合，而不必盲目尝试全部子集。
<!-- bilingual-en:start -->
Finding one deficient set proves infeasibility immediately. Conversely, when a maximum matching fails to cover $L$, [[交替搜索的增广与缺额证书|alternating search]] constructs such a set from its search frontier instead of blindly testing all subsets.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 若某个 $S$ 有七个左点但只有六个邻居，这个集合证明了什么？
>
> **答案：** 任何 matching 至多覆盖其中六个左点，因此不存在覆盖整个 $L$ 的 matching。

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]]，Section 12.5.2：核验 bottleneck 的定义以及 pigeonhole 必要性论证。
  <!-- bilingual-en:start -->
  Section 12.5.2 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] verifies the bottleneck definition and its pigeonhole necessity argument.
  <!-- bilingual-en:end -->
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_halls_thorem.pdf|MIT 6.042J Hall's Theorem slides]]：交叉核验 no-bottleneck condition 与 matching existence 的关系。
  <!-- bilingual-en:start -->
  The [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_halls_thorem.pdf|MIT 6.042J Hall's Theorem slides]] cross-check the relation between the no-bottleneck condition and matching existence.
  <!-- bilingual-en:end -->
