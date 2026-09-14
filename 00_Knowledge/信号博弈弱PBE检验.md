---
aliases:
  - "标准信号博弈的 weak-PBE 检验应依次核对完整策略、路径内外 posterior、接收者最佳反应和每个发送者类型的所有偏离"
  - "Procedure for checking weak PBE in a standard signaling game"
  - "Signaling-game weak-PBE checklist"
student_os: knowledge-atom
atom_id: GT-PBE-003
atom_set: signaling-games-pbe
atom_type: procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[信号博弈]]"
  - "[[完美贝叶斯均衡]]"
  - "[[PBE定义口径]]"
  - "[[策略信念评估]]"
  - "[[路径外反应可支持性]]"
related:
  - "[[路径上一致性]]"
  - "[[序贯理性]]"
  - "[[合并均衡]]"
  - "[[分离均衡]]"
  - "[[半分离均衡]]"
part_of:
  - "[[信号博弈与PBE.canvas]]"
---

# 标准信号博弈的 weak-PBE 检验应依次核对完整策略、路径内外 posterior、接收者最佳反应和每个发送者类型的所有偏离
<!-- bilingual-en:start -->
*A weak-PBE check in a standard signaling game should proceed through complete strategies, on- and off-path posteriors, receiver best responses, and every sender type's deviations*
<!-- bilingual-en:end -->

> [!summary] 从候选到 weak-PBE assessment 的六步检验
> 这套流程把标准两阶段信号博弈的 weak-PBE 定义转成可重复执行的检查顺序。每一步都使用前一步已经固定的对象，最后得到完整 assessment 及其均衡分类；任何一步失败都应回到候选策略或 belief，而不是跳过失败条件继续命名均衡。若采用更强的 PBE 或序贯均衡口径，还须另加相应的路径外 consistency 条件。
> <!-- bilingual-en:start -->
> This procedure turns the weak-PBE definition for a standard two-stage signaling game into a repeatable sequence. Each step uses objects fixed by the previous step and ends with a complete assessment and equilibrium classification. Failure at any step sends the candidate back for revision rather than allowing an equilibrium label to be assigned prematurely. Stronger PBE conventions and sequential equilibrium require additional off-path consistency conditions.
> <!-- bilingual-en:end -->

## 六步 weak-PBE 检查
<!-- bilingual-en:start -->
*Six-step weak-PBE check*
<!-- bilingual-en:end -->

1. **写完整策略。** 对每个发送者类型 $\theta$ 写出 $\sigma_1(\cdot\mid\theta)$，并对每个可能信号 $s$ 写出接收者反应 $\sigma_2(\cdot\mid s)$，包括候选路径上不会出现的信号。
2. **判定路径内外。** 用先验 $\mu_0$ 与发送者策略计算
   $$
   \Pr_\sigma(s)=\sum_{\theta}\mu_0(\theta)\sigma_1(s\mid\theta).
   $$
   严格正概率的信号属于路径上，零概率信号属于路径外。
3. **填写 posterior。** 对每个路径上信号按 Bayes 法则计算 $p(\theta\mid s)$；对每个路径外信号写出一份合法 belief，而不是留下空白。
4. **检查接收者最佳反应。** 在每个信号后，用对应 posterior 计算接收者各行动的期望收益；混合反应的支持只能包含最优行动。
5. **逐类型检查发送者偏离。** 固定接收者在所有信号后的反应，对每个类型比较候选信号与每个其他信号的收益。混合类型还须在支持内无差异，支持外没有更优信号。
6. **报告 assessment 与分类。** 同时列出 $(\sigma,p)$、路径内外 beliefs、关键最优性不等式，并依据发送者类型的信号分布判定 pooling、separating 或 semi-separating。
<!-- bilingual-en:start -->
&nbsp;

**1.** Write complete strategies: specify $\sigma_1(\cdot\mid\theta)$ for every sender type and $\sigma_2(\cdot\mid s)$ after every possible signal, including unused signals.<br>
**2.** Classify signals as on or off path by computing $\Pr_\sigma(s)=\sum_\theta\mu_0(\theta)\sigma_1(s\mid\theta)$. Positive probability is on path; zero probability is off path.<br>
**3.** Fill in posteriors: use Bayes' rule after every on-path signal and specify a legal belief after every off-path signal.<br>
**4.** Check receiver best responses under the corresponding posterior at every signal. A mixed response may place weight only on optimal actions.<br>
**5.** Check sender deviations type by type, holding fixed the receiver's response after every signal. A mixing type must be indifferent within its support and must not prefer a signal outside it.<br>
**6.** Report the assessment, the on- and off-path beliefs, the key optimality inequalities, and the resulting pooling, separating, or semi-separating classification.<br>
<!-- bilingual-en:end -->

## 最容易漏掉的交叉检查
<!-- bilingual-en:start -->
*The cross-check most often omitted*
<!-- bilingual-en:end -->

路径外 posterior、接收者反应和发送者不偏离不是三项彼此独立的填写题。Posterior 必须使规定反应成为最佳反应，而该反应又必须使所有类型的不偏离不等式成立。[[路径外反应可支持性]]正是对这三个对象做联合存在性检验。
<!-- bilingual-en:start -->
An off-path posterior, the receiver's response, and sender deterrence are not three independent entries. The posterior must make the prescribed response optimal, and that response must satisfy every type's no-deviation inequality. [[路径外反应可支持性|Off-path response supportability]] is the joint existence test for these objects.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一个答案正确算出了所有路径上 posterior，也验证了接收者路径上最优，但没有检查 low type 是否愿意改发 high type 的信号。能否判定为 separating PBE？
>
> **答案：** 不能。分离产生的退化 posterior 只完成了推断步骤；还必须逐类型检查模仿和其他信号偏离。
> <!-- bilingual-en:start -->
> **Self-check.** An answer computes every on-path posterior and verifies the receiver's on-path optimality, but never checks whether the low type wants to imitate the high type. Has it established a separating PBE?
>
> **Answer:** No. Degenerate posteriors complete only the inference step; every imitation and other signal deviation must still be checked type by type.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [MIT OCW 14.126 Spring 2024, Lecture 3: *Signaling Games*, slides 4–8](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/resources/mit14_126_s24_lecture_3_signaling_pdf/)：核对标准信号博弈的完整策略空间、发送者逐类型最优、路径上 Bayes posterior、接收者最佳反应与路径外存在性条件。
- [MIT OCW 14.126 Spring 2024, Lecture 2: *Equilibrium Refinements*, slide 11](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/mit14_126_s24_lecture_2_refinements.pdf)：核对本流程采用的 weak-PBE 一致性口径。
- [[01_Math/03_game theory/07_子博弈不完全信息#4.2. 策略与 belief|本地课程：Beer–Quiche 的完整策略与 beliefs]]与[[01_Math/03_game theory/07_子博弈不完全信息#6.3. 作答格式|本地课程：PBE 作答格式]]：核对课程手算顺序。
<!-- bilingual-en:start -->
- MIT Lecture 3, slides 4–8, supplies the standard signaling game's complete strategy spaces and the sender, receiver, on-path Bayes, and off-path support conditions.
- MIT Lecture 2, slide 11, identifies the weak-PBE consistency convention used by this procedure.
- The local Beer–Quiche sections anchor the checklist in the course's calculation and answer format.
<!-- bilingual-en:end -->
