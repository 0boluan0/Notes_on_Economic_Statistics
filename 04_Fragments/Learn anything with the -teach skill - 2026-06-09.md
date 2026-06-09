---
aliases: []
tags:
  - fragment
  - source/youtube
  - ai/skills
  - learning/personalized-learning
  - method/teaching
date: "2026-06-09"
source_type: "youtube"
source_title: "Learn anything with the /teach skill"
source_author: "Matt Pocock"
source_url: "https://www.youtube.com/watch?v=s5T5oQJcJ6U"
published_date: "2026-06-08"
created_by: "fragments"
---

# Learn anything with the /teach skill

## 一句话总结
Matt Pocock 展示了一个名为 `/teach` 的 Codex/Claude Code 技能：它把“老师记得学生目标、进度、资源和困难点”的能力写进文件系统状态，用 mission、resources、HTML lessons、learning records、glossary、cheat sheets 和 notes 等材料，持续为学习者生成处在“最近发展区”的个性化课程。

## 来源信息
- 类型：YouTube 视频
- 标题：Learn anything with the /teach skill
- 作者 / 频道：Matt Pocock（@mattpocockuk）
- 发布时间：2026-06-08
- 时长：13:04
- 链接：https://www.youtube.com/watch?v=s5T5oQJcJ6U
- 相关链接：skills repo `https://github.com/mattpocock/skills`；teach skill 介绍页 `https://aihero.dev/s/1T2OM1`
- 可用材料：YouTube 元数据、作者描述、官方章节时间戳、英文自动字幕（本次已成功下载）
- 素材完整性说明：字幕为 YouTube 自动字幕，不是人工校订逐字稿；内容梳理以章节、字幕和作者描述互相校对，具体术语和数字以可见元数据为准。

## 核心问题
- 一个 AI “skill”什么时候应该是 stateless，什么时候应该是 stateful？
- 如果要让 AI 真正像老师一样连续教学，它需要在文件系统里记住哪些东西？
- 为什么作者认为个性化学习的核心不是一次性生成教程，而是持续诊断学生当前所处的“最近发展区”？
- HTML lessons、glossary、learning record、cheat sheet 这些产物分别解决什么学习问题？
- 这个 `/teach` 思路对代码库 onboarding、课程学习、语言学习和个人技能训练有什么可迁移价值？

## 详细梳理
### 1. 开场：把十年教学经验压进一个 skill（00:00-00:39）
视频从作者自己的教学经历开始：Matt Pocock 说自己做过 6 年 voice coach，又做了 4 年开发者教学，因此一直在想能否把“如何教人”这件事写进一个 skill，让任何人都能学任何东西。他在一次去伦敦的长途车程里写出 `/teach`，并用它学习如何复原三阶魔方。

开场的重点不是魔方本身，而是“学习体验像有一位真正的老师”。作者强调，这个 skill 之所以有效，是因为它按照他喜欢被教的方式来教，并且和他的学习目标对齐。这里已经埋下后面两个核心设计：第一，它必须知道学习者为什么学；第二，它必须持续记住学习者已经走到哪里。

### 2. Stateful vs stateless：教学型 skill 必须有状态（00:39-02:19）
作者先区分两类 skill：stateless skill 不保留上一次运行的状态，不在文件系统或 MCP server 里保存可供后续恢复的材料；stateful skill 则会把过程、记录、资源或决策保存下来，让下一次运行能接着往前走。

他一开始想把 `/teach` 做成 stateless：用户说“教我某件事”，skill 搜索资源并生成一节课。但他很快意识到，好的教学几乎总是 stateful 的。老师会记得学生已经学过什么、卡在哪里、下一步适合学什么，也会积累之前教过这个主题时有用的资源。

作者用自己的其他 skill 做对照：`grill me` 是 stateless，它只是围绕一个主题不断追问，直到用户准备好实现；`grill with docs` 是 stateful，因为它会保存本地 ADR、glossary 等文档，并随项目进展变得更有用。这个对照说明，stateful 并不天然比 stateless 高级，关键在任务类型。一次性检查或问答可以 stateless；跨多次会话的学习、onboarding、成长路径则需要 stateful。

### 3. 安装与第一次运行：空目录变成教学工作区（02:19-02:47）
安装方式很简单：进入 Matt 的 skills repo，运行快速开始里的安装脚本，然后选择 `teach` skill。之后在一个空目录里打开 coding agent，直接提出学习任务。例如作者在自己的 Rubik's cube 目录里说，让它教自己如何复原三阶魔方。

这个细节很重要：`/teach` 不是只在对话窗口里工作，而是把一个目录变成“教学工作区”。这个工作区之后会逐渐积累课程、记录、参考资料和代理自己的内部笔记。也就是说，学习不是一次聊天，而是一个可恢复、可审计、可继续的本地项目。

### 4. Mission：先定义学习目标，而不是先灌教程（02:47-03:30）
第一次运行时，skill 创建的第一个关键文件是 mission。作者认为，老师要有效，必须先理解学生为什么想学这件事。以魔方为例，mission 写得非常具体：Matt 想在无人帮助的情况下至少复原一次三阶魔方；目标是完成这件事本身，不是追求速度，也不是追求理论。

这说明 `/teach` 的教学不是“同一套课程给所有人”。如果目标是“至少复原一次”，课程就不需要从群论或速度解法讲起；如果目标是“进入 speedcubing”，课程结构可能完全不同。mission 的作用是给后续每一节课设置边界：哪些内容现在有用，哪些内容虽然正确但会偏离目标。

同一阶段，skill 还会建立 resources。它会去搜索 web，找 primary-source、高信任度的材料，并把这些材料作为后续 lesson 的来源。resources 不是一次性列表，而是可以随着学习过程继续更新的资源库。

### 5. HTML lessons：课程输出不只是 Markdown 文档（03:30-05:21）
`/teach` 生成的 lesson 存在 `lessons` 文件夹中，以编号 HTML 文件形式保存。作者选择 HTML 的理由很直接：HTML 比 Markdown 表达力更强，可以放图、callout、quiz、交互控件和动态演示。

第一个魔方 lesson 叫 anatomy, notation, and the white cross。它只讲学习者当下需要知道的东西：魔方部件、notation、白十字的基本操作。作者特别强调 lesson 里有 quiz，因为教学需要 feedback loop。quiz 不是最丰富的反馈形式，但在没有更好反馈的场景里，它至少能让学习者检查自己是否真的掌握了当前知识。

这一段还区分了 knowledge 和 skills：lesson 不只是给概念解释，也鼓励学习者练习具体技能。比如魔方学习里，“知道 notation”只是知识；“能做出 white cross”才是技能。`/teach` 的输出要同时覆盖知识获取和技能练习。

作者后来还给 lesson 底部加了“community”入口，用于提示学习者到哪里提问或交流。他的理由是：知识和技能可以通过 lesson 建立，但智慧来自与真实社区、真实实践者互动。这个设计把 AI 老师的边界说得很清楚：AI 可以准备你进入社区，但不应该替代社区。

### 6. Reference material：glossary、cheat sheet 与 notes 的分工（05:21-06:11）
除了 lesson，skill 还会维护 reference material。第一个是 glossary，用来保存术语、部件名称、notation、解法步骤中的奇怪 jargon。它的价值有两层：一是学习者忘记术语时可以回看；二是后续 lesson 可以更简洁，因为它可以引用 glossary，而不必每次重新解释。

第二个是 cheat sheet 或 solve card。魔方例子里，它提供一个单页的完整复原参考。如果学习者只是想在操作时快速查步骤，这张卡比重新打开长 lesson 更适合。

第三个是 `notes.md`。这不是给学习者看的正式课程，而是 agent 自己的内部记录：用户偏好、需要注意的问题、历史表现等。它类似一位老师课后的备课笔记，用来帮助下一次会话更快恢复上下文。

### 7. 继续一次学习会话：先读取状态，再做诊断（06:11-07:16）
作者演示了第二次学习会话：他告诉 `/teach`，自己现在基本能复原魔方，但 corner cycle 还没有形成肌肉记忆。因为当前对话上下文是空的，agent 会先检查教学工作区状态：读 solve card、lesson、learning record 等文件，判断学习者已经到哪里。

随后，agent 给出一个诊断：corner cycle 的概念已经稳固，但算法还没有进入 muscle memory。它还会查看已有的 memorization lesson 和前一节 lesson，以匹配已有课程风格。这一步体现了 stateful skill 的真正价值：不是每次从零解释 corner cycle，而是在已有学习路径上定位一个精确缺口。

### 8. 最近发展区：课程要刚好有挑战但不吓人（07:16-08:36）
作者明确提到 zone of proximal development，即最近发展区。他认为 lesson 应该落在学习者“刚好被挑战，但不会被吓到”的区域。太简单会无聊，太难会让人崩溃。因此每一节课都必须简洁、紧凑，并精准框定当下问题。

corner cycle 的新 lesson 就是这个原则的例子。它没有回到魔方基础，也没有展开所有高级技巧，而是只围绕一个算法和肌肉记忆展开：给出新的 mental model，把动作拆成“一个四步 phrase 重复两次”，并提供互动式 tap/guided mode 练习。作者看到 HTML 互动控件时很兴奋，因为这正体现了 HTML lesson 的优势：浏览器可以承载比静态 Markdown 更丰富的练习形式。

这一段可以看作 `/teach` 的教学哲学核心：AI 不是生成一大堆信息，而是根据学生当前状态设计下一步练习。

### 9. Skill 本体：用文件结构固定教学哲学（08:36-10:04）
作者随后打开 skill 本身，说明它虽然较长，但思路并不复杂。skill 明确告诉 agent：用户请求你教他们某件事，这是 stateful request；他们打算跨多个 session 学习这个主题。

skill 还规定了 teaching workspace 的形状，并给出一套 sectional philosophy。作者把学习拆成三层：knowledge、skills、wisdom。knowledge 来自高质量、高信任度资源；skills 来自高度相关、可交互的 lesson 和练习；wisdom 则来自与其他学习者和实践者互动。

这三分法很实用。很多 AI 学习工具只停在 knowledge，即生成解释和总结；好一点的会给练习，进入 skills；但 wisdom 需要把学习者送到现实环境里，让他们和真实问题、真实社区接触。作者甚至在 skill 中写道，当用户提出需要 wisdom 的问题时，默认姿态应该是尝试回答，但最终把他们引向社区。

这里也能看出 `/teach` 的边界设计：它不是为了让学习者永远依赖 agent，而是让学习者获得足够知识和技能后，有信心进入真实社区。

### 10. 工程应用：代码库 onboarding 是天然场景（10:04-11:06）
作者认为 `/teach` 对工程团队尤其有用，典型场景是代码库 onboarding。传统文档难维护，而且很难适配每个人的最近发展区：一个新人可能懂技术栈但不懂业务域，另一个人可能懂业务域但不熟 TypeScript，还有人只是需要理解这个项目的局部架构。

如果把 `/teach` 指向代码库，新人可以在自己的学习工作区里独立学习：先弄清项目目标和自己的学习目标，再逐步生成适配当前水平的 lesson、glossary、cheat sheet 和练习。理想状态下，这会让新人更快变成 productive employee，而且不会要求团队写一份对所有人都同样合适的庞大文档。

这个判断也提示了一个风险：如果 resources 和 lesson 没有和真实代码、真实约束绑定，AI 可能生成看似合理但不贴合项目的教材。因此用于 onboarding 时，最好把代码库、README、ADR、测试、关键 issue 和团队约定都纳入 resources，并保留人类 review。

### 11. 开发者作为 AI first movers：把 coding 里的经验带出去（11:06-13:04）
最后，作者把 `/teach` 上升到一个更大的观点：开发者社区是最早真正体验 AI 能力的一群人，因为 AI 当前在写代码上特别强。开发者能在一个 AI 擅长的领域里大量试错，因此会最早形成使用 AI 的直觉、模式和 skill 设计经验。

他的结论是，开发者现在在 Claude Code、Codex 等工具里积累的经验，不应该只留在 coding domain。可以把这些经验抽象成 skills，再迁移到学习语言、学习 vocal harmonies、学棋类开局、个人项目探索等非代码领域。

这个结尾和开场形成闭环：`/teach` 不是单个学习魔方的 demo，而是一个把“AI agent + 本地文件状态 + 教学法”组合起来的模板。真正值得学习的不是某一节魔方课，而是如何把可持续学习过程设计成一个有状态的工作区。

## 关键论点与依据
- 教学型 skill 应该是 stateful，因为好的老师会记住学生的目标、进度、困难和下一步；对应 00:39-02:19 的 stateful/stateless 讨论。
- mission 是教学工作区的第一块地基；它把学习目标限制到可执行范围，避免课程跑偏；对应 02:47-03:30 的 Rubik's cube mission 示例。
- resources 负责把 lesson 连接到高信任度材料，而不是完全依赖模型即兴生成；对应 02:47-03:30 的资源创建说明。
- HTML lessons 的价值在于表达力和互动性，可以放图、quiz、guided mode、tap 练习等；对应 03:30-05:21 与 07:16-08:36 的 lesson 展示。
- learning records 让 agent 能根据用户报告记录学习进展，并在下一次会话恢复诊断；对应 05:21-07:16 的 learning record 与继续会话演示。
- zone of proximal development 是课程生成的核心约束：课程应让学习者刚好被挑战，但不被吓退；对应 07:16-08:36。
- skill 设计把学习分成 knowledge、skills、wisdom；knowledge 来自资料，skills 来自练习，wisdom 来自社区和真实实践；对应 08:36-10:04。
- 代码库 onboarding 是高价值应用场景，因为团队文档很难同时适配不同新人的背景与最近发展区；对应 10:04-11:06。
- 开发者现在在 AI coding 中形成的 skill 设计经验，可以迁移到非代码学习场景；对应 11:06-13:04。

## 可复用框架
### Stateful learning skill 结构
- `mission`：学习者为什么学，成功标准是什么，不追求什么。
- `resources`：高信任度材料来源，后续 lesson 的依据。
- `lessons/`：按顺序生成的 HTML 课程，每节只解决当前最近发展区内的问题。
- `learning records`：学习者报告、已掌握内容、卡点、练习结果。
- `glossary`：术语、notation、jargon，减少后续 lesson 重复解释。
- `cheat sheets / solve cards`：操作时快速查阅的一页式参考。
- `notes.md`：agent 自己的备课笔记，记录偏好、注意事项和诊断线索。

### 设计 skill 时的 stateful/stateless 判断
- 一次性问答、面试追问、临时检查：优先 stateless。
- 跨 session 学习、项目 onboarding、长期能力建设：优先 stateful。
- 如果下一次运行需要知道“上次走到哪里”，就应该把状态写入文件系统或可信存储。
- 如果状态会影响下一步教学/诊断/资源选择，就不要只依赖聊天上下文。

### Knowledge-Skills-Wisdom 教学分层
- Knowledge：解释概念、术语、背景、规则，依赖高信任度资料。
- Skills：把知识变成可执行动作，通过练习、quiz、交互 lesson 和反馈循环形成。
- Wisdom：进入真实社区或真实场景，和实践者互动，处理教材之外的模糊问题。

### 最近发展区 lesson 模板
1. 读取当前学习记录和目标。
2. 诊断当前最小卡点。
3. 只生成解决这个卡点所需的短 lesson。
4. 给一个具体练习或反馈环。
5. 更新 learning record。
6. 如果问题需要真实经验，把学习者引向社区或实践场景。

## 对我的启发
- 这个视频和我的 Obsidian 工作流很贴：`04_Fragments` 负责保留来源和结构，`00_factor` 负责后续提炼；`/teach` 则说明“学习过程本身”也可以变成本地文件系统里的有状态项目。
- 如果以后为课程学习设计一个 Codex skill，可以借鉴 `mission + resources + lessons + learning records + glossary + notes` 的结构，让它围绕 MIT 18.01、EC400 或论文模型持续教学，而不是每次重新解释。
- 对 Academic vault 来说，`learning records` 很值得单独设计。现在日记和课程笔记记录了学习内容，但不一定记录“我已经掌握什么、卡在哪里、下一步最近发展区是什么”。
- 对代码项目 onboarding 来说，`/teach` 比传统 README 更个性化，但必须绑定真实代码证据，否则会产生漂亮但不可靠的课程。适合把 repo 文件、测试、ADR、issue 和已有文档作为 resources。
- 对 AI skill 设计来说，关键不是让模型“一次答得更好”，而是让它能留下结构化痕迹，下一次用这些痕迹做更准的诊断。

## 待提炼知识点
- `concept`：Stateful Skill / 有状态技能
- `concept`：Stateless Skill / 无状态技能
- `concept`：Zone of Proximal Development / 最近发展区
- `framework`：Stateful Learning Workspace / 有状态学习工作区
- `framework`：Knowledge-Skills-Wisdom Learning Layers / 知识-技能-智慧学习分层
- `procedure`：Design a Teach Skill / 教学型 AI skill 设计流程
- `procedure`：Codebase Onboarding with Teach Skill / 用教学型 skill 做代码库 onboarding
- `system`：AI Lesson Reliability Checks / AI 课程可靠性检查
- `writing`：How to Explain Stateful vs Stateless Skills / 如何解释有状态与无状态 skill

## 值得继续追问的问题
- `/teach` 生成的 resources 如何判断“高信任度”？是否需要固定白名单、引用检查或人工 review？
- HTML lessons 的交互内容是否容易变成炫技？什么时候 Markdown 已经足够，什么时候必须用 HTML？
- learning records 应该记录到多细？如果记录太多，会不会让 agent 误判学习者状态或过拟合过去表现？
- 对数学课、经济学课、编程课，最近发展区的诊断指标分别应该是什么？
- 如果把 `/teach` 用在 Academic vault，输出应该放在 `04_Fragments`、课程目录、还是单独的 learning workspace？
- 如何把 `/teach` 生成的 lesson 与 `00_factor` 卡片边界对齐，避免把课程内容直接塞进 concept/framework 卡？
