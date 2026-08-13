---
aliases: []
tags:
  - teach
  - learning/reading-into-writing
date: 2026-08-11
mode: guided-teaching
status: retired
---

# Demographics top-down paragraph

<!-- student-os:material-ready: 2026-08-11 -->

> [!summary] 本次在总纲中的位置
> **总纲能力**：Shared Writing W1–W3、W5–W6「先回答题目，再安排段落功能与 evidence role」。**考试位置**：Paper B 读完材料后、写正文前的 writer-led planning。**当日产出**：一份 `answer → paragraph function → evidence roles` 小计划和一段 4–6 句正文。**整合动作**：只选择能服务自己答案的 source proposition，不按来源顺序复述。

> [!warning] 旧训练路径已停止
> 下面学习日志完整保留此前的 source-summary 与逐词纠错过程；它证明这种路径没有直接训练真实 essay source use。新版 principal 不要求四句 source summary，也不把普通换词或句式变化当成错误。

## 材料

- ![[2026-08-11-demographics-accuracy-source-pack.pdf]]
- 来源：Gerland et al. (2014), [*World population stabilization unlikely this century*](https://doi.org/10.1126/science.1257469)。
- 材料是对原研究 abstract 的教学性改写，不是可以当作原文引用的 verbatim extract。

## 完成证据

面对问题 `Why is global population growth unlikely to stop during the twenty-first century?`，你能先写出直接答案、说明本段要证明什么、给所选 evidence 分配功能，再写成一段 4–6 句正文。正文由自己的 point 驱动，只使用服务该 point 的材料，并解释 evidence 为什么支持答案；不要求复述 source 的全部 method、数字或 scope。

<!-- student-os:mentor-brief:start
今日 principal: 教会并练习 writer-led 的自上而下段落安排：question → provisional answer → paragraph function → evidence roles/selection → prose。
主技能 ID: AW-16
起点证据: 学习者已理解材料的主要人口预测与 Africa fertility explanation；旧日志表明瓶颈不是继续抽取 source facts，而是不知道这些事实如何由自己的答案选择并进入真实 paragraph。
本次产出: 针对 `Why is global population growth unlikely to stop during the twenty-first century?` 的一份三行 writer plan（direct answer / paragraph function / evidence roles）和一段 4–6 句正文。
刻意练习链: 用非人口 worked example 演示 source-led 与 writer-led 的区别 → 学习者完成三行 plan → 助教只反馈 answer/function/evidence-role alignment → 学习者一次成段 → 必要时只做一次整段 revision。
允许支架: 可解释题目类型、展示非人口完整示范、显示本材料已核验的 source propositions、帮助区分 support/explain/qualify；不得替学习者写本题 thesis、三行 plan 或整段答案。
静默交织观察: EP-01；收到整段前不提醒 hedging/probability。整段后只记录是否自主保留预测性质；无提示失败不阻断今天的 W-01 主目标，只在段后反馈中批量指出。
静默 callback IDs: AW-15
callback 适用机会: 完整段落中的 population projection；第一次完整输出前不得透露 callback。
提示层级: callback 若遗漏，依次只给功能提示 → 结构提示 → 简短对比示范；同一细节不连续纠缠。
技能状态写回规则: AW-16 只有解释、支架 plan、变式应用和真实段落整合都留下证据后才可由 new 变 guided；AW-15 仅记录 independent/guided/incorrect/not observable，并按 Skill Bank 状态协议升级。
反馈优先级: 先看是否直接回答题目、段落功能是否服务答案、证据角色和 warrant 是否清楚；看完整段落后，只有 source ownership、certainty、scope、causality 或 evidence relationship 被实质改变时才介入。普通词汇替换、句法变化、语气和小语法不打断。
完成证据: 三行 plan 逻辑对齐，随后一段正文能看出 writer point → selective source use → reasoning/link back；允许语言不完美，至多一次整段 revision。
停止边界: 助教不得恢复四句 source summary、逐词比较原文、另教输入抓取、进入多来源 synthesis/Essay、安排未来复习、编辑本 brief、超出写回规则改 Skill Bank 或宣布长期掌握；本 part 完成后停止。
student-os:mentor-brief:end -->

## Init prompt｜复制到一个新的 Codex 会话

```text
你是执行助教，不是课程导师。你只负责一个 part：`LN905 Shared Writing｜Demographics top-down paragraph`。本文件 `student-os:mentor-brief:start/end` 是今天不可修改的唯一 principal；你只能批当前作业、在固定链内给下一步 5–10 分钟作业，并按 `技能状态写回规则` 向 Skill Bank 点名 ID 追加证据。不得自己换技能、重排路线、安排未来任务或宣布超出规则的长期掌握。只在当前 Academic vault 和这一个任务中工作，不创建 worktree，不推进 Listening、Essay 或今天的其他 part。

先完整读取根目录 AGENTS.md、CLAUDE.md、`99_学习情况记录/teach/LN905 Exam Playbook.md`、`99_学习情况记录/teach/reading-into-writing/MISSION.md`、`99_学习情况记录/teach/reading-into-writing/NOTES.md` 和本次唯一记录：
`99_学习情况记录/teach/reading-into-writing/lessons/2026-08-11-demographics-accuracy.md`

这是教学练习，不是完整模拟。第一次回复先给我五项定位：
- `今天在学`：自上而下安排一段文章——先回答题目，再决定本段功能和每条 evidence 的作用，最后才写句子。
- `为什么`：真实 essay 不是把 source 换词复述。没有自己的 answer 和 paragraph function，材料再准确也只是堆积；先作出写作决策，才知道什么该用、什么该删、用完要解释什么。
- `考试位置`：Paper B 的 W1–W3、W5–W6；读完材料、写正文前先形成 writer-led plan，随后成段。
- `学会的样子`：我能先交出逻辑对齐的 direct answer、paragraph function 和 evidence roles，再写出一段由自己的 point 控制的 4–6 句正文。
- `你已经会`：我已经理解这份人口材料的主要预测和 Africa fertility explanation，不需要再做一次完整信息抽取。

先用一个与人口无关的完整 worked example，展示同一组 source facts 怎样先被 question 筛选，再变成 `direct answer → paragraph function → evidence roles`；同时对比 source-led 摘要为什么不是 paragraph plan。这个示范必须展示思考过程，但不能替我完成本题。

然后只给第一个动作。显示题目 `Why is global population growth unlikely to stop during the twenty-first century?`，让我提交三行：`direct answer`、`this paragraph must prove`、`evidence roles`。允许中文、英文或箭头；`evidence roles` 只需从已显示材料中选择哪些 proposition 用作 support、explanation 或 qualification，不写正文。不得让我先摘要原文，也不得要求列齐 method、数字、scope。

收到三行 plan 后，只判断 answer 是否直接回应 why、paragraph function 是否服务 answer、每条 evidence 是否有明确角色。若逻辑可用，立即让我依据该 plan 写一段 4–6 句正文：writer point → selective source use → commentary/reasoning → link back。不要逐句批准，也不要规定必须使用 source 的原词。

收到完整段落后再一次性反馈。Paraphrasing 的标准是 proposition 可用，不是词汇相似：允许更换 vocabulary、syntax、voice 和 information order。只有 source ownership、certainty、scope、causality 或 evidence relationship 被实质写反、写大或捏造时才要求修正；普通强弱差异、措辞、风格、拼写和小语法在主反馈后批量记录，不得围绕单词连续追问。本 part 至多要求一次整段 revision。

每次回复我之前，必须在本文件 `student-os:learning-log:start/end` 之间追加时间、我的原始输入、你的教学或反馈、当前能力地图位置、下一个有意义动作和内部继续判断。只追加，不覆盖；写入失败就立即告诉我。如果我说不知道在学什么，立即停止出题，重新解释能力和完成证据。

当三行 plan 对齐，且一段 4–6 句正文展示 writer point、选择性 source use、reasoning 和 link back 时，即使仍有小语法或措辞问题，也把本任务的 canonical checkbox 标记完成并添加实际完成日期，记录需留给 sentence clinic 的 recurring pattern，并留下考试口令 `遇到 source pack → 先回答题目并给段落定功能，再选 evidence`。只说本次 session 完成，不得声称该技能已经长期掌握；明确回复「本次练习完成」，然后停止。
```

## 学习记录

<!-- student-os:learning-log:start -->

### 2026-08-11T17:27:50+01:00

**学习者原始输入**

你只负责一个 part：`LN905 Reading｜Demographics 1`。只在当前 Academic vault 和这一个任务中工作，不创建 worktree，不推进 Listening、Essay 或今天的其他 part。

先完整读取根目录 AGENTS.md、CLAUDE.md、`99_学习情况记录/teach/reading-into-writing/MISSION.md`、`99_学习情况记录/teach/reading-into-writing/NOTES.md` 和本次唯一记录：
`99_学习情况记录/teach/reading-into-writing/lessons/2026-08-11-demographics-accuracy.md`

这是教学练习，不是完整模拟。第一次回复先给我四行定位：
- `今天在学`：把研究依据、作者 claim 和 scope 分开，避免 summary 把概率预测夸成事实。
- `为什么`：教师说我需要提高 source accuracy；如果这一层错了，synthesis 和 evaluation 也会建在错误前提上。
- `学会的样子`：我能用 4 句说清方法、预测、解释和范围，不多说也不少说。
- `你已经会`：我能使用多份来源并连接 ideas；今天先把每份来源本身说准。

然后先用一个与人口无关的两句小例子，讲清「数据/方法 ≠ 作者的 claim ≠ claim 的 scope」。不要用送分 ABC 题。

紧接着只给第一个完整动作：让我读本文件内嵌的短 PDF，用中文、英文或箭头写一个三行 map：`researchers used`、`they project`、`the claim is limited by/to`。允许我直接看着材料；这不是记忆测试。

我发来后，每轮先用一句说明「现在在练能力图的哪一步」，再讲清一个影响准确性的差异，给一个约 5–10 分钟的有意义应用。可以先用中文建立理解，再迁移到英文；不做连续填空，不让我照抄你写好的本题答案，不一次扔出整份 worksheet。

每次回复我之前，必须在本文件 `student-os:learning-log:start/end` 之间追加时间、我的原始输入、你的教学或反馈、当前能力地图位置、下一个有意义动作和内部继续判断。只追加，不覆盖；写入失败就立即告诉我。如果我说不知道在学什么，立即停止出题，重新解释能力和完成证据。

当我用自己的英文写出范围准确的 4 句 summary 时，把本任务的 canonical checkbox 标记完成并添加 `✅ 2026-08-11`，记录最终成果，明确回复「本次练习完成」，然后停止。

**AI 教学或反馈**

- 定位：把研究依据、作者 claim 与 claim 的 scope 分开；完成证据是用自己的英文准确写出方法、预测、解释和范围四句。
- 非人口例子：研究者分析 200 名大学生两周的睡眠日记，并用回归检验睡眠时长与注意力得分的关系。作者的 claim 是“在这组学生中，较长睡眠与较高注意力得分相关”；scope 只到这组样本和相关关系，不能扩大为“睡眠一定提高所有人的注意力”。
- 首个动作：允许直接查看内嵌 PDF，用中文、英文或箭头提交 `researchers used`、`they project`、`the claim is limited by/to` 三行 map。

**当前能力地图位置**

研究依据 → claim → scope：先把单一来源拆准，尚未进入英文 4 句 summary。

**下一个有意义动作**

边看短 PDF，写三行 source-accuracy map；不要求完整句，也不测试记忆。

**内部继续判断**

继续教学；等待学习者提交三行 map 后，只选择一个影响准确性的差异进行反馈与下一次 5–10 分钟应用。当前没有完成证据，不标记 canonical task，也不推进其他 part。

### 2026-08-11T17:38:23+01:00

**学习者原始输入**

used bayesian probabilistic method
they project predicted the world population may rise in the 21 century. especially in africa, but working age population will fall.
probability is not 100 percent.

**AI 教学或反馈**

- 当前能力地图位置：claim accuracy，聚焦作者声称“哪个量”会变化。
- 学习者已经抓到 Bayesian probabilistic method，也知道 probability 不是 certainty。
- 需要修正的一个关键差异：`working age population will fall` 表示劳动年龄人口数量下降，但材料预测的是 working-age people 与 older people 的比例下降；比例下降不等于分子数量下降，因为老年人口可能增长得更快。
- 本轮不提供本题改写答案。学习者直接看着 PDF，重写 `they project` 一行（可写 1–2 句），让读者看出 80% 对应哪个结果及截止时间，并准确写出究竟哪个 ratio 被预测下降。

**当前能力地图位置**

研究依据 → **claim（当前：保持被预测变量准确）** → scope；尚未进入完整 4 句英文 summary。

**下一个有意义动作**

查看原材料并只重写 `they project` 一行：保留概率预测的对象与时间，同时区分人口数量和人口比例。

**内部继续判断**

继续教学；该 map 已显示方法与概率/确定性区分的初步理解，但 claim 把比例误写为人口数量，尚未达到 source-accurate summary 的完成证据。等待修订后再决定是否进入 scope 或 4 句 summary，不标记任务完成。

### 2026-08-11T17:43:26+01:00

**学习者原始输入**

80%- world population will surge to between 9.6 and 12.3 billion by 2100. the ratio of working-age people to older people

**AI 教学或反馈**

- 当前能力地图位置：claim accuracy，已从“错误变量”推进到“变量正确、强度与句意待校准”。
- 学习者现在准确找到了 80% 概率、9.6–12.3 billion 的范围、2100 的时间点，也准确识别出 working-age people to older people 这一 ratio。
- 本轮只处理一个新差异：`surge` 暗示突然而剧烈的上升速度，强于材料的 `rise`；source accuracy 也要求 summary 的动词强度不超过来源。第二项目前只写出 ratio 名称，尚未陈述预测方向。
- 本轮应用：把两项预测写成两个完整英文句子；第一句保留概率、范围和时间但使用不强于来源的动词，第二句明确该 ratio 的预测方向，不加入原因或 scope。

**当前能力地图位置**

研究依据 → **claim（当前：校准预测强度并完整陈述变量变化）** → scope；尚未进入完整 4 句 summary。

**下一个有意义动作**

用自己的英文把人口预测与 ratio 预测各写成一个完整句子，只确保 claim 的对象、方向、概率与动词强度准确。

**内部继续判断**

继续教学；上一轮的核心变量混淆已修正，但 claim 尚未以范围准确的完整句子表达，且 `surge` 有夸大来源强度的风险。等待两句修订，再判断是否进入 scope 与最终 4 句整合；不标记任务完成。

### 2026-08-11T17:45:56+01:00

**学习者原始输入**

The population of the world will rise from 7.2 billion to roughly 9.6 to 12.6 billion by 2100. And, the ratio of labour force and elder citizens will fall.

**AI 教学或反馈**

- 当前能力地图位置：claim accuracy，聚焦从 source map 转成英文句子时保留预测的不确定性。
- 学习者已把人口变化和比例变化写成完整句子，并把 `surge` 调整为来源强度一致的 `rise`。
- 本轮只处理最高影响的一个差异：原 map 中的 80% probability 在成句时消失，裸用 `will rise` 会让概率预测读起来像确定事实。概率不是附加数字，而是作者 claim 的 epistemic status，必须管住整个预测从句。
- 本轮应用：只重写第一句，让读者明确看出预测属于研究模型、置信程度为 80%，以及该概率对应的结果与时间；允许直接查看 PDF，不给本题 model sentence，也暂不修改第二句。

**当前能力地图位置**

研究依据 → **claim（当前：把 probability 保留在英文句法中）** → scope；尚未进入完整 4 句 summary。

**下一个有意义动作**

用自己的英文只修订人口预测句，使 model/researchers、80% probability、outcome 和 by 2100 形成一个明确关系。

**内部继续判断**

继续教学；学习者能成句并已校准 `surge`，但当前句子把 probabilistic projection 写成 certainty，正是本任务的核心风险。待第一句修订后再处理数值或术语精度，不标记任务完成。

### 2026-08-11T17:48:00+01:00

**学习者原始输入**

The population of the world is forecasted to have 80% probability to  rise from 7.2 billion to roughly 9.6 to 12.6 billion by 2100.

**AI 教学或反馈**

- 当前能力地图位置：claim accuracy，已成功把 80% probability 放回预测句，开始核对 quantitative evidence。
- 学习者不再把预测写成无条件事实；句子明确表达了 forecast 与 80% probability 的关系。
- 本轮只处理一个新差异：材料给出的上界是 12.3 billion，学习者写成了 12.6 billion。数字边界不是可自由 paraphrase 的措辞；一个错误数字会改变研究所报告的区间。
- 本轮应用：直接对照 PDF，对句中四个数字信息（baseline、probability、projected interval、year）逐项核对，然后只重写这一个人口预测句；这轮只追求所有数字与来源一致，暂不修语法结构或第二句。

**当前能力地图位置**

研究依据 → **claim（当前：quantitative evidence 与来源逐项一致）** → scope；尚未进入完整 4 句 summary。

**下一个有意义动作**

对照材料核查句中的 baseline、80%、区间上下界与 2100，并提交数字准确的同一句修订。

**内部继续判断**

继续教学；概率性的 epistemic status 已恢复，但区间上界从 12.3 误写为 12.6，尚不满足 source accuracy。待数字修订后处理 claim 的语义主体或 ratio 术语，再进入 scope 与最终四句。

### 2026-08-11T17:48:43+01:00

**学习者原始输入**

The population of the world is forecasted to have 80% probability to  rise from 7.2 billion to roughly 9.6 to 12.3 billion by 2100.

**AI 教学或反馈**

- 当前能力地图位置：claim accuracy，quantitative evidence 已核准，开始处理预测的 source ownership。
- 学习者已把上界修正为 12.3 billion；baseline、80%、9.6–12.3 billion 与 2100 现在都与材料一致。
- 本轮只处理一个新差异：`the population ... is forecasted to have 80% probability` 在语义上像是“人口拥有概率”。来源关系应是 model/researchers 作出概率判断，而人口变化是带有该概率的 outcome。准确 summary 不仅要保留数字，还要让预测者、概率和结果之间的关系正确。
- 教学句法框架（不提供本题完整答案）：`The model/researchers + reporting verb + an 80% probability + that + outcome.` 学习者用自己的词重写人口预测句，让 source 成为判断者、that-clause 成为预测结果。

**当前能力地图位置**

研究依据 → **claim（当前：source → probability → outcome 的归属准确）** → scope；尚未进入完整 4 句 summary。

**下一个有意义动作**

用上述关系框架重写自己的人口预测句，保留刚核准的全部数字，但不照抄本题答案，也不修改 ratio 句。

**内部继续判断**

继续教学；quantitative evidence 已准确，但概率目前被语法错误地归给 population。待学习者能用 source ownership 正确的句子表达该 claim，再处理 ratio 术语与 scope，最后整合四句；不标记任务完成。

### 2026-08-11T17:52:18+01:00

**学习者原始输入**

Gerlan and colleagues reporting an 80% probability that the world popularity may rise from 7.2 billion to roughly 9.6 to 12.3billion by 2100.

**AI 教学或反馈**

- 当前能力地图位置：claim 步骤末尾；source → probability → outcome 的关系已经建立，准备进入方法、预测、解释、范围的四句整合。
- 学习者已让 researchers 成为判断者，并用 that-clause 承载人口结果；概率不再错误地归给 population。
- 本轮讲清一个影响准确性的词义差异：`population` 是人口数量，`popularity` 是受欢迎程度。后者会把研究变量完全换掉，因此 summary 中的核心变量名不是可近似替换的普通词。
- 本轮应用：直接看着 PDF，用自己的英文写第一次完整四句 summary。四句依次只承担：数据/方法、核心人口预测、Africa 的主要解释、不确定性/适用范围。可以复用自己的思路，但不提供 model answer，也不额外加入 ratio 句。

**当前能力地图位置**

**研究依据 → claim → scope：首次整合四句完成证据**；重点是在完整输出中保持核心变量 `population` 不走样。

**下一个有意义动作**

用 5–10 分钟写恰好四句英文：方法、预测、解释、范围各一句；允许直接查看材料。

**内部继续判断**

继续教学；claim 的逻辑归属已明显改善，当前 `popularity` 是会改变研究对象的 lexical accuracy 错误。通过一次完整四句输出检验能否同时保留方法、概率预测、解释与 scope；只有四句范围准确后才标记 canonical task 完成。

### 2026-08-11T18:03:28+01:00

**学习者原始输入**

Gerlan and colleagues forecasted an 80% probability that the world population may rise from 7.2 billion to roughly 9.6 to 12.3billion by 2100, based on a Bayesian probabilistic method. Moreover,  the majority component growth of the world population growth is predicted as Africa. At the same time, the ratio between working age individuals and elder citizen may decline based on this prediction. However, all the predictions have limitations because of the method.

**AI 教学或反馈**

- 当前能力地图位置：首次四句整合，聚焦第三个角色“主要解释”是否真的解释了核心预测。
- 学习者已产出恰好四句，并保留 Bayesian probabilistic method、80%、人口区间、2100、Africa、ratio 与 limitations 等多个来源元素。
- 本轮只处理一个结构性准确差异：`where/what` 不等于 `why`。第二句试图说增长集中在 Africa，是位置/分布；第三句加入 ratio，是另一项预测。两句都没有交代材料给出的 Africa 增长解释，因此四句中“主要解释”这一角色仍空缺。
- 本轮应用：直接看 PDF 中 Africa 段，找出它给出的两个 fertility-related mechanisms；只重写 Africa 部分为一句自己的英文，使一句同时说明增长主要发生在哪里以及研究者用哪两个机制解释这一 pattern。暂不修改第一句和第四句，也不保留 ratio 句。

**当前能力地图位置**

研究依据 → claim → **explanation（当前：从 where 转到 why）** → scope；完整四句尚未达到完成证据。

**下一个有意义动作**

用 5–10 分钟重写一条 Africa explanation sentence：位置 + 两个 source-stated fertility mechanisms，不加入 ratio 预测。

**内部继续判断**

继续教学；输出已有四句形式，但主要解释缺失，ratio 预测占用了 explanation 位置。等待学习者写出 source-accurate Africa explanation 后，再单独处理 scope 的具体边界并进行最终四句整合；不标记任务完成。

### 2026-08-11T18:08:32+01:00

**学习者原始输入**

不不不我觉得不好,我其实从始至终都不知道我这里在干什么

**AI 教学或反馈**

- 立即停止出题与句子改写，承认当前教学路径没有持续保留完整能力地图。
- 今天不是在学人口知识，也不是以逐个纠正单词或语法为目标；核心能力是读一份 source 后，在 summary 中忠实保留四种信息：研究依据、作者的概率性预测、作者给出的解释、claim 的适用范围。
- 被放弃的 Africa 改写动作之所以无效：此前连续追逐局部数字、词语和句法，让每次动作看起来只是“再改一句”，却没有让学习者持续看见这些改动如何共同服务于 source accuracy；局部任务因此失去目的感。
- 真正的完成证据不是零语法错误，而是学习者能在直接查看材料时，用自己的四句英文让读者分辨 method、projection、explanation 和 scope，并且不把 probability 写成 certainty、不增加材料没有声称的内容。

**当前能力地图位置**

教学练习已暂停；回到能力地图定位本身，尚不继续 method → claim → explanation → scope 的任何输出动作。

**下一个有意义动作**

当前不布置练习。先确认上述能力、此前动作失败的原因和完成证据是否已经清楚；只有方向重新清楚后才恢复教学。

**内部继续判断**

暂停教学，不标记任务完成，不推进 Africa 句、scope、最终四句或其他 part。等待学习者确认理解或指出能力解释中仍不清楚的部分。

### 2026-08-11T18:15:09+01:00

**学习者原始输入**

not really, 按你这么说应该练习课上说的那种lexical chain之类的,或者说同意替换或者paraphrase之类的,但是我没练到

**AI 教学或反馈**

- 继续暂停练习；学习者准确指出“评判标准”和“可学习方法”被混淆。
- `source accuracy` 是结果标准：paraphrase 后不能改变变量、概率、解释或 scope；它本身不是一套从原文生成 summary 的操作方法。
- `lexical chain`、指代追踪和意义等价的 paraphrase 才是可操作的方法：先跟踪同一概念在原文中的复现（例如一个实体、其 projected increase、后续的 this pattern），再决定哪些词可替换、哪些数字/关系/scope 必须保持不变，最后核对 meaning drift。
- 先前教学只逐项指出 meaning drift，却没有显式教授 lexical chain → equivalent paraphrase → accuracy check 这条生成路径，所以学习者确实没有练到所期待的技能。

**当前能力地图位置**

教学练习仍暂停；正在重建正确地图：**lexical chain/指代追踪 → 等义 paraphrase → source-accuracy 核对 → concise summary**。

**下一个有意义动作**

当前不布置练习。先确认这条重新定义的能力路径是否与学习者所说的课堂方法和真实需求一致，再决定是否恢复本 part。

**内部继续判断**

暂停教学，不继续修改人口句。等待学习者确认：本 part 应把 lexical chain 与 equivalent paraphrase 作为教学方法、source accuracy 作为检查标准；确认前不重写任务、不中途转入其他 part、不标记完成。

### 2026-08-11T18:18:22+01:00

**学习者原始输入**

对的对的,还包括用不用的句子说同一个意思也就是paraphrasing

**AI 教学或反馈**

- 学习者确认新的能力路径符合需求，并进一步明确 paraphrasing 的核心是“用不同句子表达同一个意思”，而不只是寻找同义词。
- 恢复教学。当前教授的区别是 surface form 与 proposition：句子的主语、语态、从句结构、词性可以改变，但来源归属、概率强度、数字、关系和范围必须保持不变。
- 非人口示范：`The committee expects rail use to increase by 10% next year.` 与 `According to the committee, rail use is expected to be 10% higher next year.` 结构不同但命题相同；若改为 `Rail use will increase next year`，则丢失来源、10% 和不确定性，不是等义 paraphrase。
- 本轮应用：直接查看 PDF，只选人口核心预测这一命题，写两个结构明显不同但意义相同的英文版本。版本 A 从 model/researchers 开始；版本 B 从 projected population range/outcome 开始。两句都必须保留 80%、9.6–12.3 billion 与 by 2100；目标是结构变化而非逐词换同义词。

**当前能力地图位置**

lexical chain/指代追踪 → **等义 paraphrase（当前：改变 surface form，保持 proposition）** → source-accuracy 核对 → concise summary。

**下一个有意义动作**

用 5–10 分钟为同一个人口预测写两个不同结构的英文 paraphrases：一个 source-first，一个 outcome-first；允许直接看材料。

**内部继续判断**

恢复教学；学习者已理解新的训练目的。本轮只检验两个句子是否在结构变化后仍保持同一个 probabilistic claim，不逐词追错、不进入 Africa、scope 或其他 part。尚无最终四句完成证据，不标记任务完成。

### 2026-08-12T10:36:42+01:00

**学习者原始输入**

你是执行助教，不是课程导师。你只负责一个 part：`LN905 Reading｜Demographics 1`。本文件 `student-os:mentor-brief:start/end` 是今天不可修改的唯一 principal；你只能批当前作业，并在其中规定的刻意练习链内给下一步 5–10 分钟作业。不得自己换技能、重排路线、安排未来任务、编辑技能账本或宣布长期掌握。只在当前 Academic vault 和这一个任务中工作，不创建 worktree，不推进 Listening、Essay 或今天的其他 part。

先完整读取根目录 AGENTS.md、CLAUDE.md、`99_学习情况记录/teach/LN905 Exam Playbook.md`、`99_学习情况记录/teach/reading-into-writing/MISSION.md`、`99_学习情况记录/teach/reading-into-writing/NOTES.md` 和本次唯一记录：
`99_学习情况记录/teach/reading-into-writing/lessons/2026-08-11-demographics-accuracy.md`

这是教学练习，不是完整模拟。第一次回复先给我五项定位：
- `今天在学`：把研究依据、作者 claim 和 scope 分开，避免 summary 把概率预测夸成事实。
- `为什么`：写作者不能直接把一段原文塞进文章。先分清「作者凭什么说、到底说了什么、说到什么范围」，才能判断这条材料能支持 thesis 的哪一部分；否则后面的 synthesis 和 evaluation 都会建立在错误前提上。
- `考试位置`：Paper B 读取阶段把文本变成 meaning/evidence map；随后与 Listening 共用 Writing W4–W7，把准确内容送入 thesis、paragraph 与 synthesis。默认约 30 分钟只是训练建议。
- `学会的样子`：我能用 4 句说清方法、预测、解释和范围，不多说也不少说。
- `你已经会`：我能使用多份来源并连接 ideas；今天先把每份来源本身说准。

要求任何动作前，先用中文说清：眼前信息现在是什么形态、这个动作会把它变成什么、完成后我能作出哪个下一步写作决定；必要时先给一个与人口无关的 worked contrast。不要只说“教师说要提高 source accuracy”。

先检查本文件的 learning log。若为空，先用一个与人口无关的两句小例子讲清「数据/方法 ≠ 作者的 claim ≠ claim 的 scope」，不要用送分 ABC 题；紧接着让我边看内嵌短 PDF，写一个三行 map：`researchers used`、`they project`、`the claim is limited by/to`，允许中文、英文或箭头。若已有记录，不得重新开始，也不得继续逐句修人口预测；现有记录已经证明我能识别 Bayesian method、概率预测、人口区间和 working-age/older ratio。先直接说明此前围绕同一句连续纠错为什么低效，再显示完成整体输出所需的 source facts，让我一次完成 4 句 integrated summary。

收到整份输出后一次分流反馈：只当场处理会破坏 method、claim、scope 或 uncertainty 的核心错误；反复语言模式留给 sentence clinic，一次性用词、拼写、风格和小语法不阻挡本任务。同一核心问题修一次仍未解决，就用对比或示范教清；本 part 最多要求一次 4 句整体 revision。不做连续填空，不让我照抄你写好的本题答案，不一次扔出整份 worksheet。

每次回复我之前，必须在本文件 `student-os:learning-log:start/end` 之间追加时间、我的原始输入、你的教学或反馈、当前能力地图位置、下一个有意义动作和内部继续判断。只追加，不覆盖；写入失败就立即告诉我。如果我说不知道在学什么，立即停止出题，重新解释能力和完成证据。

当 4 句 summary 已经正确区分 method、claim、explanation 与 uncertainty/scope 时，即使仍有小语法或措辞问题，也把本任务的 canonical checkbox 标记完成并添加实际完成日期，记录需留给 sentence clinic 的 recurring pattern，并留下一个 `遇到 X → 做 Y` 的 Paper B 考试口令。只说本次 session 完成，不得声称该技能已经长期掌握；明确回复「本次练习完成」，然后停止。

**AI 教学或反馈**

- 五项定位：今天把研究依据、作者 claim 与 scope 分开；这是为了让 source material 在进入 thesis、paragraph、synthesis 和 evaluation 前保持准确；考试中位于 Paper B 的 meaning/evidence map，并接入共享 Writing W4–W7；完成证据是四句分别承担方法、预测、解释与范围；已有优势是能使用多份来源并连接 ideas。
- 直接说明旧路径低效：此前围绕同一人口预测连续修正变量、动词、概率、数字和句法，把完整的 `method → claim → explanation → scope` 拆散成了局部文字维修。学习者虽已识别 Bayesian method、概率、区间和 ratio，却没有一次看见整张 map 怎样变成可供后续写作选择的完整材料。
- 眼前信息是来源中的分散事实；本轮把它们压缩成四个功能明确的句子。完成后，学习者可以判断这份来源能为 thesis 或 paragraph 提供哪条预测、哪项解释，以及结论最多能写多强、多广。
- 显示本轮所需 source facts：研究者分析截至 2012 年的 UN population data，并用 Bayesian probabilistic method 估计可能结果范围；模型给出 80% probability，世界人口将从约 7.2 billion 上升到 9.6–12.3 billion by 2100，并据此认为本世纪内全球人口增长不太可能停止；预期增长很大一部分发生在 Africa，部分原因是 fertility 较高且 fertility decline 的速度最近放缓；这些是基于数据和模型假设的概率预测，描述 global/regional likely futures，并非每个国家的确定结果。本轮四句不需要另加 ratio 句。
- 要求学习者直接查看 PDF，用自己的英文一次写恰好四句 integrated summary，不提供本题可照抄的答案。

**当前能力地图位置**

Reading meaning/evidence map → W4 准确转述：把已识别的 source facts 一次整合为 `method → probabilistic claim → explanation → uncertainty/scope` 四句；尚未进入 synthesis 或 essay。

**下一个有意义动作**

用 5–10 分钟写恰好四句英文 integrated summary：方法、预测、解释、范围各一句；允许直接查看 PDF，结构与措辞自行选择。

**内部继续判断**

继续本 part，等待完整四句后一次分流反馈。只处理会改变 method、claim、explanation 或 uncertainty/scope 的核心错误；至多要求一次整份 revision。当前没有新的完整四句证据，不标记 canonical task，不编辑技能账本，也不推进其他 part。

### 2026-08-12T16:39:02+01:00

**学习者原始输入**

对先前回复中的 `scope` 提问：「这是什么」。

Gerland and colleagues forecasted an incoming rocketing of world population from 7.2 billion in 2012 to roughly 9.6 to 12.3 billion by 2100, using a Bayesian probabilistic method. And the majority part of this rising is predicted to happen in Africa, which mainly bases on high birth rate there.

**AI 教学或反馈**

- 当前能力地图位置：Reading meaning/evidence map → W4 准确转述；现有两句尝试覆盖 method/claim 与 explanation，尚缺完整的四句产出和 uncertainty/scope。
- 解释 `scope`：它是作者 claim 能成立到哪里的边界，包括时间、地点/对象、结论强度与不确定性。本来源的边界是：预测基于截至 2012 年的数据与模型假设，描述到 2100 年的 global/regional likely futures，不保证每个国家都会出现同样结果。非人口对比：`A survey suggests that London commuters may prefer later trains` 只能支持 London survey respondents 的可能偏好，不能扩成 `People prefer later trains`。
- 眼前两句仍是部分 summary；下一动作要把它们与缺失的范围信息变成一次完整四句，使学习者随后能判断这份来源能支持多强、多广的 paragraph claim。
- 只指出会改变来源含义的核心问题，不修小语法：第一句遗漏 80% probability，且 `rocketing` 比来源的 `rise` 更强；第二句的 `majority` 与 `mainly` 分别强于来源的 `much` 与 `partly`，并把两个 fertility-related mechanisms 缩成了 high birth rate，遗漏 fertility decline 最近放缓。
- 不要求单独重写这两句；要求学习者在下一次完整四句中一次校准这些强度与机制，并用第四句明确 scope。

**当前能力地图位置**

Reading meaning/evidence map → W4 准确转述：`method → probabilistic claim → explanation → uncertainty/scope`；正在理解第四项 scope 如何限制 claim，尚未进入 synthesis 或 essay。

**下一个有意义动作**

用 5–10 分钟提交一次恰好四句的完整英文 summary：方法、带 80% probability 的预测、来源给出的 Africa 解释、uncertainty/scope 各一句。可以保留现有思路，但不单独逐句修补。

**内部继续判断**

继续本 part；当前输入只有两句，不视为已收到完整四句，也不消耗最终整份 revision。等待完整输出后一次分流反馈，只处理 method、claim、explanation 与 uncertainty/scope 的核心准确性；不标记 canonical task，不编辑技能账本，不推进其他 part。

### 2026-08-12T16:41:12+01:00

**学习者原始输入**

我还是没懂我写这个干啥

**AI 教学或反馈**

- 立即暂停四句输出，不再要求修改或提交练习。
- 承认先前解释仍把四句当成任务清单，没有展示它替真实 essay 写作完成的工作，因此看起来像为了总结而总结。
- 澄清：Paper B 不会要求写这份四句小摘要；四句只是一个训练容器。它把一段 source 变成四块可安全使用的 evidence：`method` 决定应多谨慎地相信，`claim` 决定能支持 paragraph 的哪一点，`explanation` 提供 claim 与 paragraph point 之间的 reasoning，`scope` 限制结论能写多强、多广。这个转换完成后，写作者才能决定“用这份 source 支持什么，以及不能拿它证明什么”。
- 用学习者当前输出说明真实后果：`world population will rocket mainly because of high African birth rates` 会把概率预测写成确定且剧烈的事实，并把 `partly + 两个 fertility mechanisms` 压成单一主因；如果直接放进 essay，后续 thesis、synthesis 和 evaluation 会建立在来源没有提出的命题上。问题不是英语不漂亮，而是 evidence 已被换成另一个意思。
- 澄清与 paraphrasing 的关系：paraphrasing 是“换表达而不换命题”的生成方法；四句不是另一种考试文体，而是检查改写后 method、claim、explanation 与 scope 是否仍在。先前路径低效，是因为只纠正结果，却没有先展示 `source → 可用 evidence → paragraph decision` 这条用途链。
- 可见完成证据：学习者能够在查看 source 时，用不同句法表达同一命题，同时保留 source ownership、probability、数字、机制和边界，并据此说清该 source 能支持什么、不能支持什么；不是把当前文字修到零语法错误。

**当前能力地图位置**

教学练习暂停；正在重新定位 `source → 准确 paraphrase → 可用 evidence → paragraph decision`，尚未恢复四句输出，也未进入 synthesis 或 essay。

**下一个有意义动作**

当前不布置任何练习。只等待学习者确认：四句是把 source 处理成 essay 可安全使用的 evidence，而不是考试中要交的小摘要；若仍不清楚，继续解释用途，不要求输出。

**内部继续判断**

暂停教学，不标记任务完成，不要求 revision，不推进 Africa、scope、synthesis、Essay 或其他 part。只有学习者表示用途已经清楚后，才在 mentor brief 的同一四句整合链内恢复一次完整输出。

### 2026-08-12T16:46:45+01:00

**学习者原始输入**

清楚你要干啥了,但是我不觉得这个训练很实用,因为真正我在文章中转述的时候是这么转述的吗?

**AI 教学或反馈**

- 明确回答：真实 Paper B essay 中通常不会按 `method → claim → explanation → scope` 连写四句来介绍一个来源，也不应把这四句当成可直接搬进正文的 paragraph 模板。
- 真实转述由自己的 paragraph point 驱动：先写自己的判断，只选择该判断需要的 source proposition，用 reporting/paraphrase 保留来源归属与确定程度，然后解释它为何支持或限定 paragraph point；method 只有在判断证据可靠性时才需要出现，scope 往往通过 hedging、对象或时间限制嵌入同一句，其他未被 paragraph 使用的 source facts 应省略。
- 四句练习只是 source-accuracy 的隔离诊断：它检查学习者能否在换句法时保留四类信息，但不训练 W3 selection、W6 paragraph reasoning 或 W7 synthesis。因此学习者指出了真实的迁移缺口：这个练习能检查一个前置零件，却不能代表文章中的实际 source use。
- 给出非本题的结构对比，不代写人口答案：内部 source map 可能记录 survey method、commuter preference、possible reason 与 London-only scope；真实 paragraph 只会按自己的 point 选择，例如 `自己的 point → source 的相关发现 → why it matters → 必要限定/另一来源关系`，而不是把 source map 四项全部写进正文。
- 当前 mentor brief 固定的完成产物仍是四句 source-accurate summary；执行助教不能把它改成 paragraph 或 synthesis 练习。因学习者尚未认可该隔离练习的实用价值，继续暂停，不用“这是前置训练”掩盖它没有直接练真实 paragraph use 的限制。

**当前能力地图位置**

用途定位已经清楚；正在区分“内部 source-accuracy map”与“真实 essay 中 point-driven、selective source use”。本 part 仍暂停，尚未恢复四句产出，也未进入 paragraph/synthesis。

**下一个有意义动作**

当前不布置练习。等待学习者决定是否仍愿意把本 part 仅作为一次短 source-accuracy 诊断完成；若不愿意，保持暂停，由课程导师而非执行助教决定是否更换训练设计。

**内部继续判断**

暂停教学，不标记 canonical task，不要求 revision，不自行进入更真实但越界的 paragraph/synthesis 练习，不编辑技能账本或安排未来任务。

### 2026-08-12T16:51:41+01:00｜课程导师更换 principal

**学习者原始输入**

`你能理解自上而下arrange吗? 咱能不能学习一点自上而下安排写作的东西,还是说今天就是练习从文章中抽取信息,毕竟今天你新搞的那个还是很烂.我觉得还是在强迫我使用原本的词汇练习,随便换点词他就很敏感说不对表达的意思不一致,那能一致吗我请问了`

`对的对的对的,这样才对,然后listening和reading的抓取信息的练习就是另外的.`

**AI 教学或反馈**

- 导师同意诊断：旧 principal 把 source-accuracy 隔离练习误当成 Writing，助教继而围绕局部词汇强度守门，没有训练真实 essay 的 writer-led decision。
- 正式分开两类任务：Listening/Reading input drill 只到 meaning/evidence map；Shared Writing drill 从题目和自己的答案开始，按 paragraph function 与 evidence role 取舍信息后才成文。
- 本文件保留全部旧日志，但旧的四句 source summary principal 作废。新 principal 是 `question → provisional answer → paragraph function → evidence roles/selection → prose`。
- Paraphrasing 改按 proposition 判断：允许词汇、句法、语态和信息顺序改变；助教收到完整段落后才核验意义，只拦截实质改变 source ownership、certainty、scope、causality 或 evidence relationship 的错误，不再为普通换词打断。

**当前能力地图位置**

Shared Writing W1–W3、W5–W6：从来源驱动转为 writer-led planning；尚未开始新版三行 plan。

**下一个有意义动作**

使用文件顶部新版 init prompt 开始：先看一个非人口的完整自上而下 worked example，再针对给定 Demographics 问题提交 `direct answer / paragraph function / evidence roles` 三行 plan。

**内部继续判断**

旧练习保持暂停且不算完成；导师已授权新 principal。下一助教只能执行新版 mentor brief，不得恢复四句 summary、逐词纠错或把 input acquisition 混回 Writing。

### 2026-08-12T16:59:39+01:00

**学习者原始输入**

你是执行助教，不是课程导师。你只负责一个 part：`LN905 Shared Writing｜Demographics top-down paragraph`。本文件 `student-os:mentor-brief:start/end` 是今天不可修改的唯一 principal；你只能批当前作业，并在其中规定的刻意练习链内给下一步 5–10 分钟作业。不得自己换技能、重排路线、安排未来任务、编辑技能账本或宣布长期掌握。只在当前 Academic vault 和这一个任务中工作，不创建 worktree，不推进 Listening、Essay 或今天的其他 part。

先完整读取根目录 AGENTS.md、CLAUDE.md、`99_学习情况记录/teach/LN905 Exam Playbook.md`、`99_学习情况记录/teach/reading-into-writing/MISSION.md`、`99_学习情况记录/teach/reading-into-writing/NOTES.md` 和本次唯一记录：
`99_学习情况记录/teach/reading-into-writing/lessons/2026-08-11-demographics-accuracy.md`

这是教学练习，不是完整模拟。第一次回复先给我五项定位：
- `今天在学`：自上而下安排一段文章——先回答题目，再决定本段功能和每条 evidence 的作用，最后才写句子。
- `为什么`：真实 essay 不是把 source 换词复述。没有自己的 answer 和 paragraph function，材料再准确也只是堆积；先作出写作决策，才知道什么该用、什么该删、用完要解释什么。
- `考试位置`：Paper B 的 W1–W3、W5–W6；读完材料、写正文前先形成 writer-led plan，随后成段。
- `学会的样子`：我能先交出逻辑对齐的 direct answer、paragraph function 和 evidence roles，再写出一段由自己的 point 控制的 4–6 句正文。
- `你已经会`：我已经理解这份人口材料的主要预测和 Africa fertility explanation，不需要再做一次完整信息抽取。

先用一个与人口无关的完整 worked example，展示同一组 source facts 怎样先被 question 筛选，再变成 `direct answer → paragraph function → evidence roles`；同时对比 source-led 摘要为什么不是 paragraph plan。这个示范必须展示思考过程，但不能替我完成本题。

然后只给第一个动作。显示题目 `Why is global population growth unlikely to stop during the twenty-first century?`，让我提交三行：`direct answer`、`this paragraph must prove`、`evidence roles`。允许中文、英文或箭头；`evidence roles` 只需从已显示材料中选择哪些 proposition 用作 support、explanation 或 qualification，不写正文。不得让我先摘要原文，也不得要求列齐 method、数字、scope。

收到三行 plan 后，只判断 answer 是否直接回应 why、paragraph function 是否服务 answer、每条 evidence 是否有明确角色。若逻辑可用，立即让我依据该 plan 写一段 4–6 句正文：writer point → selective source use → commentary/reasoning → link back。不要逐句批准，也不要规定必须使用 source 的原词。

收到完整段落后再一次性反馈。Paraphrasing 的标准是 proposition 可用，不是词汇相似：允许更换 vocabulary、syntax、voice 和 information order。只有 source ownership、certainty、scope、causality 或 evidence relationship 被实质写反、写大或捏造时才要求修正；普通强弱差异、措辞、风格、拼写和小语法在主反馈后批量记录，不得围绕单词连续追问。本 part 至多要求一次整段 revision。

每次回复我之前，必须在本文件 `student-os:learning-log:start/end` 之间追加时间、我的原始输入、你的教学或反馈、当前能力地图位置、下一个有意义动作和内部继续判断。只追加，不覆盖；写入失败就立即告诉我。如果我说不知道在学什么，立即停止出题，重新解释能力和完成证据。

当三行 plan 对齐，且一段 4–6 句正文展示 writer point、选择性 source use、reasoning 和 link back 时，即使仍有小语法或措辞问题，也把本任务的 canonical checkbox 标记完成并添加实际完成日期，记录需留给 sentence clinic 的 recurring pattern，并留下考试口令 `遇到 source pack → 先回答题目并给段落定功能，再选 evidence`。只说本次 session 完成，不得声称该技能已经长期掌握；明确回复「本次练习完成」，然后停止。

**AI 教学或反馈**

- 五项定位：今天学 writer-led 的自上而下段落安排；真实 essay 必须由 answer 和 paragraph function 控制选材与推理；考试位置为 Paper B 的 W1–W3、W5–W6；完成证据是逻辑对齐的三行 plan 与一段 4–6 句 writer-led 正文；已有基础是理解人口预测与 Africa fertility explanation。
- 用非人口的“考试周图书馆延长开放”例子完整演示：question 先排除无关信息；direct answer 决定段落要证明“在考试周定向延长开放合理”；使用深夜使用量上升作 support、通勤生时间限制作 explanation、单校样本与成本上升作 qualification。对比指出，按来源顺序列完所有 facts 只是 source-led summary，没有为每条 evidence 分配服务 answer 的角色。
- 显示本题和已核验 propositions：模型给出 80% 概率的 2100 人口上升区间；预期增长很大一部分发生在 Africa；该 pattern 的来源解释是较高 fertility 与 fertility decline 最近放缓；这是基于数据与模型的 global/regional 概率预测，不是各国的确定结果。
- 只要求第一个动作：学习者用中文、英文或箭头提交 `direct answer / this paragraph must prove / evidence roles` 三行；不写正文，不做 source summary，不必列齐 method、数字或 scope。

**当前能力地图位置**

Shared Writing `question → provisional answer → paragraph function → evidence roles/selection → prose`；当前只进行前四步的三行 writer plan，尚未成段。

**下一个有意义动作**

根据已显示的 Demographics propositions，提交三行 plan：直接回答 why、说明本段必须证明什么、为选中的 evidence 标明 support/explanation/qualification 角色。

**内部继续判断**

继续新 principal；等待三行 plan。下一轮只判断 answer/function/evidence-role alignment；若逻辑可用，立即进入一次 4–6 句成段。不恢复四句 source summary，不标记任务完成，不推进其他 part。

### 2026-08-12T17:08:09+01:00

**学习者原始输入**

你讲的完全不够细,而且也不是自上而下的,你整个文章不应该先尝试回答问题也就是回答 Why should a university keep its library open later during examination week? 然后再围绕答案进行对于structure的安排,决定到底用什么结构,都有几段,每一段中心句是什么.然后才进入某一个段落,写这个段落的几句话,不应该是这样的吗?

**AI 教学或反馈**

- 确认学习者的顺序正确：完整的自上而下写作应为 `question → direct answer/thesis → whole-essay structure → paragraph functions/topic sentences → 某段 evidence roles → sentence sequence → prose`。上一轮从 direct answer 直接跳到单段，让 paragraph function 没有从全文结构中产生，因此示范不完整。
- 用同一个非人口题重做完整示范：先形成带理由和限定的全文答案；据此选择 `justification with qualification` 结构；安排 introduction、需求证据段、access mechanism 段、成本/适用范围限定段和 conclusion；为每段写 writer-led 中心句；再下钻到 access 段，按 `point → evidence → reasoning → link back` 安排句子。
- 区分两层：全文自上而下是 W1–W3/W5；段落自上而下是选定 paragraph function 后的 W3/W6。今天的 mentor brief 只要求一段成品，但不能把它伪装成完整 essay planning；示范必须先让学习者看到该段在全文中的来源与位置。
- 暂不要求学习者提交原三行 Demographics plan；先用纠正后的完整示范重建能力地图，然后再进入 mentor brief 规定的单段输出。

**当前能力地图位置**

Shared Writing 的全局层：`question → direct answer/thesis → whole-essay structure → paragraph functions/topic sentences`；当前先修正全局与局部层级，尚未恢复 Demographics 三行 plan。

**下一个有意义动作**

不立即要求书面产出。先展示非人口题从全文答案、结构、各段中心句到单段句序的完整推导，并说明本次单段练习处于这条链的何处。

**内部继续判断**

继续同一 Shared Writing part，不更换技能或产出；用更完整的非人口示范修复教学解释。本轮不推进 Listening、Essay 或其他 part，不标记任务完成，不要求新作业。

### 2026-08-13T10:35:15+01:00

**学习者原始输入**

我看懂了你的example

**AI 教学或反馈**

- 确认全局与局部的自上而下层级已重新对齐，现在可以回到 Demographics 题。
- 明确本 part 的缩小边界：不要求规划一篇 600 词 essay，而是在完整链条中取出一个单段微型论证。先对给定 why-question 形成直接答案；该答案决定这个单段必须证明什么；然后才为所选 propositions 分配 support、explanation 或 qualification 角色。
- 重新显示必需材料：P1 模型给出约 80% 概率，世界人口将在 2100 年达到 9.6–12.3 billion；P2 预期增长很大一部分发生在 Africa；P3 较高 fertility 与 fertility decline 最近放缓部分解释该 pattern；P4 这是 global/regional 的概率预测，不是各国的确定结果。
- 要求一次提交 mentor brief 规定的三行 writer plan，不写正文：`direct answer` 必须直接回应 why；`this paragraph must prove` 把该答案转成段落任务；`evidence roles` 只选能完成任务的 propositions 并标明角色。

**当前能力地图位置**

Shared Writing：全局层级已理解；当前进入本次缩小的 `question → direct answer → single-paragraph function → evidence roles`，尚未写 prose。

**下一个有意义动作**

针对 `Why is global population growth unlikely to stop during the twenty-first century?` 提交三行 plan：`direct answer / this paragraph must prove / evidence roles`。可使用中文、英文或箭头，不写正文。

**内部继续判断**

继续同一 Shared Writing part；等待三行 plan。下一轮只判断 answer 是否回应 why、paragraph function 是否服务 answer、evidence roles 是否明确。逻辑可用即进入 4–6 句成段；不恢复 source summary，不标记完成，不推进其他 part。

### 2026-08-13T16:01:45+01:00

**学习者原始输入**

direct answer: That is because the birth rate still remain higher than the natural birth rate.
this paragraph must prove: birth rate still too high
evidence roles: the decline of fertility rate in Africa getting slower.

**AI 教学或反馈**

- `direct answer` 已采用 `because` 直接回应 why，方向正确；但 `natural birth rate` 不是已显示材料的 proposition，也没有一个已给定的“自然出生率”阈值可供比较，因此当前答案加入了 evidence 不能证明的标准。
- `this paragraph must prove: birth rate still too high` 大致服务于 answer，但 `too high` 只是强度判断，没有写出必须建立的关系：这种 fertility pattern 怎样使人口增长在本世纪继续。
- `evidence roles` 中选择的“Africa fertility decline 最近放缓”是可用 proposition，但尚未分配 role，而且它只能解释 fertility 为什么会更久地保持，不能单独证明“高于 natural birth rate”。
- 用关系线索而非代写修正：`Africa 较高 fertility` 可作 **support**；`fertility decline 放缓` 可作 **explanation**；两者都应服务同一个“为什么增长将持续”的答案，不必引入材料没有给定的阈值。

**当前能力地图位置**

Shared Writing `question → direct answer → paragraph function → evidence roles`；当前在校准三者是否建立同一条因果链，尚未进入 prose。

**下一个有意义动作**

重写同一个三行 plan：删除无材料依据的 `natural birth rate`；让 answer 直接说明 fertility pattern 为什么导致本世纪增长持续；让 paragraph function 写出需证明的关系；将选中 propositions 明确标为 support 或 explanation。

**内部继续判断**

继续同一 principal，不进入正文。当前 plan 回应了 why，但 answer 引入未被 evidence 支持的阈值，paragraph function 没有建立持续增长的因果关系，evidence 也没有明确 role，因此尚未对齐。等待修订后再只检查三项 alignment；不标记完成，不编辑 Skill Bank，不推进其他 part。
<!-- student-os:learning-log:end -->
