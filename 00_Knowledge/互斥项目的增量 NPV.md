---
aliases:
  - "互斥项目要比较可行方案的增量 NPV 而不能按 IRR 或 PI 比率直接排序"
  - "Mutually exclusive projects and incremental NPV"
  - "NPV versus IRR and PI ranking"
  - "互斥项目的增量净现值"
student_os: knowledge-atom
atom_id: CORP-CB-006
atom_set: capital-budgeting-investment-decisions
atom_type: decision-rule
status: source-checked
mastery_state: unassessed
requires:
  - "[[净现值决策]]"
  - "[[IRR 的根与失效边界]]"
related:
  - "[[不同寿命项目的替换假设]]"
  - "[[资本限额下的项目组合]]"
part_of:
  - "[[资本预算与投资决策.canvas]]"
---

# 互斥项目要比较可行方案的增量 NPV 而不能按 IRR 或 PI 比率直接排序
<!-- bilingual-en:start -->
*Mutually exclusive projects require incremental NPV comparison rather than direct IRR or PI ranking*
<!-- bilingual-en:end -->

> [!summary] 选择规则
> 独立项目可以分别接受所有正 NPV 方案；互斥项目却只能从可行集合中选一个。在共同评价时点、共同资源边界和匹配贴现率下，把“全部拒绝”记作 $NPV=0$，再选择总 NPV 最高的可行方案；若最高项目 NPV 为负就全部拒绝，若等于零则该项目与全部拒绝在财务价值上无差异。两个方案成对比较时，升级方案相对基准方案的增量 NPV 为正，才说明升级增加价值。
>
> <!-- bilingual-en:start -->
> Independent projects can all be accepted when NPV is positive, but mutually exclusive alternatives require one choice from the feasible set, including rejecting all. Choose the feasible alternative with the highest positive NPV; a negative maximum means reject all, while a zero maximum is financially indifferent to rejecting all.
> <!-- bilingual-en:end -->

IRR 和现值指数（PI）都是比率。小项目可以有更高 IRR/PI，却创造更少绝对价值；早回款项目也可能有更高 IRR，却被较晚的大额现金流项目在合理资本成本下超过。若 B 比 A 多投入 100、带来额外现金流，可直接对 $CF_B-CF_A$ 求 NPV：增量 NPV 为正说明从 A 升级到 B 增加价值。增量 IRR 只能帮助找到两个 NPV 曲线的交点，最终仍须以匹配机会成本处的增量 NPV 判断。

PI 可用于检验一个常规独立项目是否大于 1，但“互斥项目中 PI 最大”并非一般规则。只有在单期硬资本约束、项目可分割、单一稀缺投入且无依赖或互斥等狭窄条件下，单位稀缺资本的价值排序才有直接意义；离散项目的正确问题转为组合优化。

> [!question]- 自检
> A 投 10 得 NPV 4，B 投 100 得 NPV 20，两者互斥且都可融资。A 的单位投资价值更高，应选谁？
>
> **答案：** 若现金流和贴现率口径一致、没有其他约束，选 B；互斥选择最大化总增加价值 20，而不是比率。

## 来源与核验

- [MIT 15.401 Capital Budgeting, slides 3–4, 23–31 and 35](https://ocw.mit.edu/courses/15-401-finance-theory-i-fall-2008/06b2ce70ad26ae3d3be8d94145495be6_MIT15_401F08_lec18.pdf)：核对互斥项目最高正 NPV、PI 的规模偏误及 IRR 的规模/时序冲突。
- [[02_Economy/05_财务管理/2023年注册会计师全国统一考试辅导教材---财务成本管理 (中国注册会计师协会) (Z-Library).pdf#page=151|CPA《财务成本管理》第五章第二节，第 143–145 页]]：核对互斥项目、规模冲突与增量价值口径。
