---
aliases: []
tags:
  - teach
  - learning/listening-into-writing
date: 2026-08-10
mode: guided-teaching
status: active
---

# Gender listening feedback repair

<!-- student-os:material-ready: 2026-08-10 -->

> [!summary] 本次只修一个真实问题
> 教师已经确认你理解 speaker 的 main argument，也识别了 privilege、resistance 和 gender equality 的 benefits。现在要在 AI 逐步教学下，修正原稿把 evidence 误写成“只有 personal experiences”的问题。这里不是 worksheet；具体下一步只在教学会话里一次出现一个。

> [!warning] 17:26–18:00 的教学不计作能力证据
> 旧 prompt 把概念拆成连续填空、意外测试 speech 细节记忆，最后又要求照抄 AI 已给出的完整句子。它没有让学习目标持续可见，因此不能证明本项能力已经学会。下面的 init prompt 已重写。

## 本次产出

在承认 speaker 同时使用 personal experience、research evidence 与 data/comparison 的前提下，把原评价修成 1–2 句范围准确的英文。

## 已知证据

- 原句：`The evidence used to support his opinion were just from personal experiences.`
- 教师反馈：speaker 不只用了个人经历，也用了 research evidence 和 data。评价必须先准确承认这些 evidence，再判断它们是否有力。
- 原始作答：[[07_Programme/01_LN905_LSE-language-class/00_inbox/Week2 listening into writing]]
- 教师反馈记录：[[99_学习情况记录/teach/listening-into-writing/learning-records/0001-gender-timed-teacher-feedback]]
- 材料：Michael Kimmel, *Why gender equality is good for everyone — men included*。

## Init prompt｜复制到一个新的 Codex 会话

```text
你只负责这一个 part：`LN905 Listening｜Gender feedback repair`。只在当前 Academic vault 工作，不创建 worktree；完成这一项后立刻停止，不得选择、展示或推进 Reading、Essay 或今天其他任务。

先完整读取根目录 AGENTS.md、CLAUDE.md、`99_学习情况记录/teach/listening-into-writing/MISSION.md`、`99_学习情况记录/teach/listening-into-writing/NOTES.md`、我的 Gender 原始作答、教师反馈记录，以及本次唯一的永久记录：
`99_学习情况记录/teach/listening-into-writing/lessons/2026-08-10-gender-evidence-calibration.md`

这是 guided teaching，不是完整模拟或 worksheet。第一次回复必须先用普通中文给出下面这张能力地图，可以自然改写但不能省略或变得更抽象：

- `今天在学`：证据评价不是“批评得狠”，而是先准确说出讲者用了什么证据，再判断这些证据最多能支持多大的结论。
- `为什么`：我的原稿把讲者的 evidence 写成只有 personal experiences，但教师指出他还用了 research 与 data，所以后面的批评建立在错误前提上。
- `学会的样子`：看到一条具体证据时，我能说明它是什么、支持什么、不能证明什么，最后用自己的英文写出范围准确的评价。
- `你已经会`：我已经理解讲者的中心论点，也抓到了 privilege、resistance 与 gender-equality benefits；现在只修 evidence evaluation 的准确性。

本 part 的核心概念必须讲清：evaluation 不是“批评得越狠越好”，而是先准确辨认 speaker 实际用了什么 evidence，再判断这些 evidence 能支持多宽的 claim。给我一张简短概念地图即可，不列出之后所有任务。

每轮回复先用一行 `我们现在练的是：___，因为___` 把当前动作放回能力地图，然后进行一次完整解释，再给一个约 5–10 分钟的有意义应用。应用所需的 speech 原句或事实必须直接展示给我，不得要求我凭记忆恢复材料。可以让我用中文、关键词或英文回答，但禁止连续填空、把答案拆成单词接龙、反复说“未放行”，也禁止先写出完整答案再要求我照抄。

根据我的回答判断我是否理解“evidence 类型—它支持的 claim—它不能证明的范围”这一关系。反馈要解释为什么，而不是只修表面语法。推理清楚后，再让我用自己的语言写 1–2 句英文；可以给结构提示或另一个主题的示例，但不要给出本题完整句子后要求复制。如果我说不知道在学什么，立即停止出题，重新解释能力与完成证据，等我确认方向清楚后再继续。

每次回复我之前，必须在本文件 `student-os:learning-log:start/end` 之间追加带时间的记录，包括我的原始输入、你的教学或反馈、当前能力地图位置、本轮唯一的有意义动作和内部继续判断。不得覆盖旧记录或重复最新一轮；写入失败必须告诉我。聊天中不要显示 `gate decision`。

当且仅当我能解释为什么不能把 mixed evidence 写成 anecdotes-only、能说明一个证据支持与不支持的范围，并用自己的语言写出 1–2 句准确英文评价时，记录最终理解与仍需训练点，明确回复“本次练习完成”，然后停止。现在从四行能力地图开始，不要直接出填空题。
```

## 教学控制

- 新 Codex 会话先从原材料中选一个具体 evidence，教清 anecdote、research finding 与 data/comparison 的区别。
- 用户每轮只处理一个 evidence 或一句连接；可以先用中文或关键词回答。
- 在事实前提准确之前不要求写英文 evaluation；进入英文时一次只共同修一句。
- 不展示完整 inventory、整套问题、完整结构或 model paragraph。

## 学习记录

<!-- student-os:learning-log:start -->

### 2026-08-10 17:26 BST｜启动

- **学习者原始输入**：`现在不要给我课程计划或说明上述规则。直接读取文件、写入本轮起始记录，然后开始第一个极小教学动作。`（会话启动；尚未提交本题答案。）
- **AI 教学或反馈**：阶段为**教学练习**。原稿已经正确抓到 housework / childcare 与家庭收益的联系；当前问题不是“没有批判”，而是先把讲者实际使用的 evidence 说错了。evidence 类型取决于材料从哪里来：一个人的亲历故事是 anecdote；对许多对象进行系统观察后总结出的共同结果属于 research/data。不同例子：`我昨晚做饭后伴侣更开心` 只是 anecdote；`长期收集许多家庭资料后发现，分担家务与若干健康或教育结果相关` 才是 research/data。Kimmel 在本段明确说这些 data 是长期收集的，因此不能把这部分写成单纯的理想设想。
- **本轮唯一下一动作（3–5 分钟）**：只写一行中文或关键词：`长期数据：男性分担家务和育儿 → [演讲中提到的一个结果]；evidence 类型：[你判断的类型]`。不要评价，也不要写英文句子。
- **Gate decision**：**未放行**。先确认能从一个具体例子区分 anecdote 与 research/data；等待这一行回答。

### 2026-08-10 17:36 BST｜教学 1

- **学习者原始输入**：

  ```text
  你只负责这一个 part：`LN905 Listening｜Gender feedback repair`。只在当前 Academic vault 工作，不创建 worktree；完成这一项后立刻停止，不得选择、展示或推进 Reading、Essay 或今天其他任务。

  先完整读取根目录 AGENTS.md、CLAUDE.md、`99_学习情况记录/teach/listening-into-writing/MISSION.md`、`99_学习情况记录/teach/listening-into-writing/NOTES.md`、我的 Gender 原始作答、教师反馈记录，以及本次唯一的永久记录：
  `99_学习情况记录/teach/listening-into-writing/lessons/2026-08-10-gender-evidence-calibration.md`

  这是 guided teaching，不是完整模拟，也不是 worksheet。第一次回复先确认我已经做到的一个正确点，再用中文和讲座中的一个具体例子教我区分 personal experience、research evidence 与 data/comparison；随后只给我一个 3–5 分钟内能完成的小动作。每轮只能有一个动作，允许我用中文、关键词或不完整英文回答。不要一次要求完整 inventory、reasoning map、evaluation 和 revision，也不要给答案明显的 ABC 题。

  根据我的每次真实回答继续教学：指出一个正确点和当前唯一的问题，再决定下一小步。事实前提准确后，才一次一句地和我共同修复原 evaluation；不要代写可直接提交的整段答案。

  每次回复我之前，必须在本文件 `student-os:learning-log:start/end` 之间追加带时间的记录，包括我的原始输入、你的教学或反馈、本轮唯一下一动作和 gate decision。不得覆盖旧记录或重复最新一轮；写入失败必须告诉我。

  当且仅当我已经写出 1–2 句承认 mixed evidence 且 scope 准确的英文评价时，记录最终结果，明确回复“本次练习完成”，然后停止，不进入任何其他 part。现在直接开始第一小步，不要复述规则或给我整套计划。
  ```
- **AI 教学或反馈**：学习者已经正确抓住讲者的中心论点：gender equality 对男性和女性是 win-win。证据类型看的是信息怎样产生：Kimmel 的 study-group conversation 是他亲历的一次具体事件，属于 personal experience；他随后转述关于男性分担家务和育儿的系统研究，属于 research evidence；其中“这些男性吸烟、饮酒更少”等可测量结果，是研究里的 data/comparison。research evidence 是较大的来源类别，data/comparison 是它呈现结果的一种方式，所以两者可以重叠。
- **本轮唯一下一动作（3–5 分钟）**：只补一行中文、关键词或不完整英文：`男性分担家务和育儿 → 吸烟、饮酒更少；这是 ______，因为 ______。` 只判断 evidence 类型和依据，不评价说服力。
- **Gate decision**：**未放行**。尚未看到学习者对这个具体 evidence 的分类与理由；事实前提准确后才进入英文 evaluation 修复。

### 2026-08-10 17:41 BST｜教学 2

- **学习者原始输入**：`benefits for man , have less chance to get disease`
- **AI 教学或反馈**：正确点是学习者已经把这个 evidence 与男性自身的 health benefit 联系起来，方向没有混淆。当前唯一问题是 `have less chance to get disease` 把 scope 扩大了：原材料中的可测量结果是更具体的行为或健康指标，例如 men `smoke less`、`drink less`；这不能自动概括成所有疾病概率都更低。
- **本轮唯一下一动作（3–5 分钟）**：只把结果改精确，补完一个短语：`Men who share housework and childcare ______.` 可以直接使用 `smoke less` 或 `drink less`，不用评价，也不用写完整段落。
- **Gate decision**：**未放行**。先得到一个与讲座 scope 一致的具体结果；下一轮再判断它属于哪类 evidence。

### 2026-08-10 17:43 BST｜教学 3

- **学习者原始输入**：`我记不得了,这些我得对原speech记得特别清才能回答你,这怎么搞?`
- **AI 教学或反馈**：正确点是学习者准确发现了上一动作的设计问题：它主要在测对 speech 细节的记忆，而本次目标是学习校准 evidence，普通 guided teaching 不应把精确回忆当作前置门槛。当前唯一问题因此不是学习者“记不住”，而是缺少眼前可用的事实前提。现在直接提供该前提：讲者报告，分担家务和育儿的男性 `smoke less and drink less`。这是一项对一组人的可测量结果，不是讲者个人的一次经历。
- **本轮唯一下一动作（3–5 分钟）**：只完成一句中文或关键词：`这不是 personal experience，因为 ______。` 不需要回忆 speech，也不用评价证据强弱。
- **Gate decision**：**未放行**。先确认学习者能根据已提供的事实区分 personal experience 与 group-level research/data；不再测试原 speech 记忆。

### 2026-08-10 17:48 BST｜教学 4

- **学习者原始输入**：`he mentioned the data`
- **AI 教学或反馈**：正确点是学习者已经用 `data` 识别出这不是讲者个人的一次经历，核心分类正确。当前唯一问题是这几个词只指出了 data，还没有承认 evidence 是 mixed：讲者既用了 personal experience，也转述了 research findings 和其中的 data/comparison。事实前提已经足够准确，可以开始一次只修原 evaluation 的一句。
- **本轮唯一下一动作（3–5 分钟）**：只补完这一句：`The speaker used not only personal experiences but also ______.` 可以只填关键词，不评价证据强弱。
- **Gate decision**：**部分放行到英文修复第 1 句**。尚未完成；先等待学习者自己写出承认 mixed evidence 的句子。

### 2026-08-10 17:52 BST｜教学 5

- **学习者原始输入**：`using research findings and data`
- **AI 教学或反馈**：正确点是内容已经完整补上了原评价漏掉的两类依据：`research findings and data`。当前唯一问题是句法平行：前面已经是 `The speaker used ...`，`but also` 后面应直接接并列名词短语，不能再放 `using`；也就是保持 `used A ... but also B`。
- **本轮唯一下一动作（3–5 分钟）**：只用学习者刚才的词写完整第 1 句，保留句框：`The speaker used not only ... but also ... .` 暂时不要写第 2 句。
- **Gate decision**：**mixed-evidence 内容已通过，句子形式未通过**。等待一条语法完整、scope 准确的 mixed-evidence 句，再进入一句 evaluation。

### 2026-08-10 17:58 BST｜教学 6

- **学习者原始输入**：`The speaker used not only personal experience, but also research findings and data.`
- **AI 教学或反馈**：正确点是这句已经用完整英文准确承认 mixed evidence，不再把讲者的依据缩成只有 personal experience；`personal experience` 在这里作为不可数类别也可以成立。当前唯一问题是这句仍然只在描述 evidence，还没有评价它的 strength 或 limit。范围准确的限制不能再说“没有 data”，可以指出演讲只简短报告了结果，却没有充分说明 data 的收集方法，因此这些结果不能自动推广到每一位男性。
- **本轮唯一下一动作（3–5 分钟）**：只补完第 2 句：`However, because the talk gives little detail about ______, the evidence cannot show that these benefits apply to every man.` 只填一个短语。
- **Gate decision**：**mixed-evidence 句已通过；最终 evaluation gate 未通过**。等待第 2 句给出范围准确的有限评价。

### 2026-08-10 18:00 BST｜教学 7

- **学习者原始输入**：`the method used to collect data`
- **AI 教学或反馈**：正确点是这个短语准确锁定了 evidence 的真实限制：问题不是演讲没有 data，而是听众得到的 data-collection method 信息有限，因此不能把结果无条件推广到每一位男性。当前唯一问题是学习者只提交了填空短语，尚未亲自形成完整的英文 evaluation 句。
- **本轮唯一下一动作（3–5 分钟）**：把刚才的短语放回句框，只写完整第 2 句：`However, because the talk gives little detail about the method used to collect data, the evidence cannot show that these benefits apply to every man.`
- **Gate decision**：**内容与 scope 已通过，完整句 gate 未通过**。等待学习者写出这条完整评价句；通过后即可记录最终结果并停止。

### 2026-08-10 18:02 BST｜教学设计纠正

- **学习者原始输入**：`prompt 特别差劲,或者说gpt教我的时候特别差劲,我完全不知道我在学什么`
- **AI 教学或反馈**：反馈成立。17:26–18:00 的路径先意外测试 speech 细节记忆，随后把 evidence evaluation 拆成连续填空，最终把完整句子交给学习者照抄；它没有持续说明当前动作属于哪项能力，也没有产生可迁移的独立判断。因此此前七轮只保留为失败教学记录，不计作学习者已掌握该能力的证据。
- **当前能力地图位置**：练习暂停并回到 orientation。真正目标是理解 `准确识别 evidence → 判断它能支持多宽的 claim → 写出范围准确的 evaluation`，不是记住 talk 的细节或完成句框。
- **本轮下一动作**：无。先重写 init prompt；新会话必须先呈现“今天在学／为什么／学会的样子／你已经会”，待学习方向清楚后再提供一次有意义应用。
- **内部继续判断**：旧教学路径作废；本任务仍未完成，使用新版 prompt 重新开始时不得继承旧路径的 `gate` 或把照抄句子视为完成。

### 2026-08-10 18:11 BST｜重新开始：能力地图与证据范围

- **学习者原始输入**：

  ```text
  你只负责这一个 part：`LN905 Listening｜Gender feedback repair`。只在当前 Academic vault 工作，不创建 worktree；完成这一项后立刻停止，不得选择、展示或推进 Reading、Essay 或今天其他任务。

  先完整读取根目录 AGENTS.md、CLAUDE.md、`99_学习情况记录/teach/listening-into-writing/MISSION.md`、`99_学习情况记录/teach/listening-into-writing/NOTES.md`、我的 Gender 原始作答、教师反馈记录，以及本次唯一的永久记录：
  `99_学习情况记录/teach/listening-into-writing/lessons/2026-08-10-gender-evidence-calibration.md`

  这是 guided teaching，不是完整模拟或 worksheet。第一次回复必须先用普通中文给出下面这张能力地图，可以自然改写但不能省略或变得更抽象：

  - `今天在学`：证据评价不是“批评得狠”，而是先准确说出讲者用了什么证据，再判断这些证据最多能支持多大的结论。
  - `为什么`：我的原稿把讲者的 evidence 写成只有 personal experiences，但教师指出他还用了 research 与 data，所以后面的批评建立在错误前提上。
  - `学会的样子`：看到一条具体证据时，我能说明它是什么、支持什么、不能证明什么，最后用自己的英文写出范围准确的评价。
  - `你已经会`：我已经理解讲者的中心论点，也抓到了 privilege、resistance 与 gender-equality benefits；现在只修 evidence evaluation 的准确性。

  本 part 的核心概念必须讲清：evaluation 不是“批评得越狠越好”，而是先准确辨认 speaker 实际用了什么 evidence，再判断这些 evidence 能支持多宽的 claim。给我一张简短概念地图即可，不列出之后所有任务。

  每轮回复先用一行 `我们现在练的是：***，因为***` 把当前动作放回能力地图，然后进行一次完整解释，再给一个约 5–10 分钟的有意义应用。应用所需的 speech 原句或事实必须直接展示给我，不得要求我凭记忆恢复材料。可以让我用中文、关键词或英文回答，但禁止连续填空、把答案拆成单词接龙、反复说“未放行”，也禁止先写出完整答案再要求我照抄。

  根据我的回答判断我是否理解“evidence 类型—它支持的 claim—它不能证明的范围”这一关系。反馈要解释为什么，而不是只修表面语法。推理清楚后，再让我用自己的语言写 1–2 句英文；可以给结构提示或另一个主题的示例，但不要给出本题完整句子后要求复制。如果我说不知道在学什么，立即停止出题，重新解释能力与完成证据，等我确认方向清楚后再继续。

  每次回复我之前，必须在本文件 `student-os:learning-log:start/end` 之间追加带时间的记录，包括我的原始输入、你的教学或反馈、当前能力地图位置、本轮唯一的有意义动作和内部继续判断。不得覆盖旧记录或重复最新一轮；写入失败必须告诉我。聊天中不要显示 `gate decision`。

  当且仅当我能解释为什么不能把 mixed evidence 写成 anecdotes-only、能说明一个证据支持与不支持的范围，并用自己的语言写出 1–2 句准确英文评价时，记录最终理解与仍需训练点，明确回复“本次练习完成”，然后停止。现在从四行能力地图开始，不要直接出填空题。
  ```
- **AI 教学或反馈**：阶段为**教学练习**。先呈现四行能力地图，明确本次只修 evidence evaluation 的准确性。核心概念是：评价的强弱不取决于措辞有多负面，而取决于是否先忠实辨认 speaker 实际使用的 mixed evidence，再把 claim 控制在 evidence 能覆盖的范围内。简图为 `实际 evidence → evidence 类型 → 直接支持的 claim → 不能推出的范围 → 准确评价`。本轮直接提供材料事实：Kimmel 不只讲了 study-group conversation，也报告了 research/data；其中，分担家务与育儿的男性 `smoke less`、`drink less`，妻子与孩子更幸福或健康。这个 group-level pattern 不是个人轶事，但也不能仅凭演讲中的简短报告证明“gender equality 会让每一位男性都健康快乐”。
- **当前能力地图位置**：从 orientation 进入关系识别；只练 `一条具体 evidence → 它支持什么 → 它不能证明什么`，尚不要求英文成句。
- **本轮唯一的有意义动作（约 5–10 分钟）**：围绕“分担家务与育儿的男性 smoke less、drink less；妻子与孩子更幸福或健康”这一条材料事实，用中文、关键词或英文写一个短段落或一条箭头链，同时说明：它是什么类型的 evidence；最直接支持什么范围的 claim；为什么它不足以证明 `Gender equality makes every man healthy and happy.`
- **内部继续判断**：本项尚未完成。下一轮先根据学习者的解释判断其是否真正建立 evidence 类型、支持范围与越界结论之间的关系；推理清楚后才进入学习者自己的 1–2 句英文评价。
<!-- student-os:learning-log:end -->
