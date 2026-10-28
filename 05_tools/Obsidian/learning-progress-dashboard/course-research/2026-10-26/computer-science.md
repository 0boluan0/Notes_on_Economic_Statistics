# Computer science course outline audit / 计算机课程目录核验

<!-- bilingual-en:start -->
Verified on 2026-10-26. The 17 courses in `cs.json` now contain 439 actual official teaching units. Sixteen have complete published outlines; CS144 Fall 2026 remains a published-to-date outline of nine named lecture-note topics. Unit counts measure the declared course format: lecture meetings, assigned reading sessions, or individually linked instructional videos. They are not interchangeable measures of workload.
<!-- bilingual-en:end -->

2026-10-26核验：17门课程共439个官方教学单元；16门的所选版本目录完整，CS144 Fall2026仍在发布，仅列9个已公开讲义课题。讲次、阅读课和视频分别计数，不把它们视为相同学习量。

## Verified directories / 官方目录

| Course | Edition | Count | Unit | Unmapped | Official directory |
|---|---|---:|---|---:|---|
| 6.100L | Fall 2022 | 26 | 讲 | 0 | [Directory](https://ocw.mit.edu/courses/6-100l-introduction-to-cs-and-programming-using-python-fall-2022/pages/material-by-lecture/) |
| 6.0001 | Fall 2016 | 12 | 讲 | 0 | [Directory](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/lecture-slides-code/) |
| CS50x | 2025 | 12 | 讲 | 1 | [Directory](https://cs50.harvard.edu/x/2025/) |
| 6.042J / 18.062J | Spring 2015 | 35 | 阅读课 | 0 | [Directory](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/readings/) |
| 6.006 | Spring 2020 | 21 | 讲 | 0 | [Directory](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/) |
| 6.046J | Spring 2015 | 24 | 讲 | 0 | [Directory](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/pages/lecture-notes/) |
| CS61B | Spring 2021 | 40 | 讲 | 4 | [Directory](https://sp21.datastructur.es/) |
| 15-213 / 18-213 / 15-513 | Fall 2015 | 27 | 讲 | 1 | [Directory](https://www.cs.cmu.edu/afs/cs/academic/class/15213-f15/www/schedule.html) |
| 6.S081 | Fall 2020 | 24 | 讲 | 0 | [Directory](https://pdos.csail.mit.edu/6.828/2020/schedule.html) |
| CS144 | Fall 2026 · 进行中 | 9 | 讲义单元 | 0 | [Directory](https://cs144.github.io/) |
| 6.5840 / 6.824 | Spring 2026 | 22 | 讲 | 0 | [Directory](https://pdos.csail.mit.edu/6.824/schedule.html) |
| 6.031 | Spring 2021 | 30 | 阅读课 | 0 | [Directory](https://web.mit.edu/6.031/www/sp21/) |
| Missing Semester | 2020 | 11 | 讲 | 0 | [Directory](https://missing.csail.mit.edu/2020/) |
| 15-445 / 15-645 | Fall 2024 | 26 | 讲 | 0 | [Directory](https://15445.courses.cs.cmu.edu/fall2024/schedule.html) |
| CS186 | Spring 2024 | 25 | 讲 | 2 | [Directory](https://cs186berkeley.net/sp24/) |
| CS149 | Fall 2024 | 18 | 讲 | 0 | [Directory](https://gfxcourses.stanford.edu/cs149/fall24) |
| DE Zoomcamp | 2025 cohort | 77 | 视频 | 9 | [Directory](https://github.com/DataTalksClub/data-engineering-zoomcamp/blob/76fed2419e917176369a0d3a2eeff6c936f87286/cohorts/2025/README.md) |

<!-- bilingual-en:start -->
`full-outline` means that the selected official teaching directory has been enumerated; it does not certify that every concept inside every lecture has been mapped. Seventeen units retain `mapping: syllabus` and grant no shared knowledge coverage. Introductory logistics, reviews, installation-only demonstrations, Q&A and project demonstrations may have an intentionally empty reviewed mapping. All new units use new IDs, so old coarse-module completions must not be silently expanded into completed lectures. No learner progress or mastery records were changed.
<!-- bilingual-en:end -->

`full-outline`表示已逐项展开所选官方目录，并不表示每讲中的每个知识点都已穷尽。17个单元仍待知识映射，暂不授予共享覆盖；导论、复习、纯安装演示、答疑和项目展示可有明确为空的已核映射。新单元采用新ID，旧模块完成状态不能被无条件扩成已完成逐讲。本次未修改个人学习进度或掌握记录。

## Counting decisions / 计数说明

- **6.100L** — 26个官方Lecture页面；不计习题课、作业与考试。与旧私人计划的合并学习块不等价。
- **6.0001** — 12个官方lecture slide条目；不计problem sets。
- **CS50x** — 11节主课（Week0–10）加1节官方独立Artificial Intelligence补充讲座；不把shorts、sections、习题、seminars及final project算作主讲。AI讲座未据概览授予完整机器学习知识点覆盖。
- **6.042J / 18.062J** — 35个官方reading sessions，单位是阅读课而非35段视频；保留官方chapter范围。知识范围另与同版官方教材目录核对。官方第15–17项PDF指向存在重复，入口保留readings总表，mappingSource指向整本教材。
- **6.006** — 21讲：lecture-notes页提供20份讲义，calendar另列Lecture21 Algorithms—Next Steps；复习与展望不新增知识覆盖。
- **6.046J** — 24个编号讲次；同一讲的手写版与打字版讲义不重复计数。
- **CS61B** — 40个官方编号讲次；不计lab、discussion、project、exam。SWE I–IV的guide没有足够内容边界，暂保留待映射。
- **15-213 / 18-213 / 15-513** — 27个主讲次；不计15个recitations、考试与停课日。共享幻灯片的Part1/2仍是不同讲次。
- **6.S081** — 24个官方LEC编号，含3次Q&A；Q&A不自动赋予lab对应的知识。
- **CS144** — Fall2026仍在发布。只列截至核验日已经公开的9个命名讲义课题；它们不是9次完整课堂（一个课堂可能包含多个课题）。未公开未来内容不补写，作业checkpoints不并入讲义数。
- **6.5840 / 6.824** — Spring2026官方LEC1–22，含Q&A和Project demos；不计考试、project hacking及独立论文阅读作业。
- **6.031** — 30篇官方编号class readings；阅读课不是录像段数，不计problem sets、quizzes及项目。
- **Missing Semester** — 2020版11节官方lectures，含Potpourri与Q&A；Q&A不额外授予知识覆盖。
- **15-445 / 15-645** — 26讲（#00–#25）；#00线上导论和#25复习保留为真实讲次，不计project、homework和考试。
- **CS186** — 24个编号讲次（0–23）加Feb15未编号的Sorting & Hashing续讲，共25次教学单元；Jan18同一天的SQL I、II按官网分别列出。Recovery I/II的公开链接存在但当前下载返回预览HTML，细粒度知识暂不授予。
- **CS149** — 18个官方lecture页面；不计考试、停课日和编程作业。
- **DE Zoomcamp** — 2025 cohort结束后的官方固定提交：六个模块README共76段实际教学视频，加1段dlt workshop，共77段（含明确列出的可选、替代路线与故障排查）。逐一采用视频外层链接，避免过期缩略图指向旧视频。项目说明、作业、社区笔记、课前答疑及直播播放列表不计为视频课；安装与替代路线并非都必修。

## Source limits and mapping review / 来源边界与待核映射

<!-- bilingual-en:start -->
CS144 has nine named note topics across eight published class meetings. Its HTML also contains commented-out older material; that material was excluded. No guessed future schedule or historical replacement was used. The metadata therefore records `outlineComplete: false`, `coverage: topic-outline`, and the unit label “讲义单元”.
<!-- bilingual-en:end -->

CS144的9项讲义课题分布在8次已公布课堂中；网页注释内的旧内容已排除。保留2026版，不把未公开内容或旧课表补成完整课程。

<!-- bilingual-en:start -->
MIT 6.042J uses its official Spring 2015 reading table and the table of contents of the [same-edition textbook](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/mit6_042js15_textbook.pdf). Reading entries 15, 16 and 17 on the official site currently link to the same session-15 resource despite different chapter labels. Units retain the reading table as their source and the whole textbook as mapping evidence instead of silently trusting those duplicated PDF links.
<!-- bilingual-en:end -->

6.042J的第15、16、17项虽然章节名不同，官方表却链接同一session15资源；条目保留总目录入口，并以同版完整教材核对章节范围。

<!-- bilingual-en:start -->
The Data Engineering Zoomcamp is pinned to official commit `76fed2419e917176369a0d3a2eeff6c936f87286`, the selected post-cohort 2025 snapshot. Its six module READMEs contain 76 distinct linked instructional videos; the 2025 dlt workshop contributes one more. All 77 video IDs were mechanically matched against the outer video hyperlinks in those official files. Several thumbnail IDs are stale, so thumbnails were not used as video identity. Optional paths, troubleshooting and installation videos remain labelled separately. Course-launch Q&A, project instructions, homework and community notes are excluded from teaching-video totals. PyFlink event-time/window/watermark mappings were also checked against the [official aggregation example](https://github.com/DataTalksClub/data-engineering-zoomcamp/blob/76fed2419e917176369a0d3a2eeff6c936f87286/06-streaming/pyflink/src/job/aggregation_job.py).
<!-- bilingual-en:end -->

Data Engineering固定到2025届结束后的官方提交。六模块76段教学视频加dlt工作坊1段，77个视频ID与官方README的外层链接逐一一致；旧缩略图ID不作为视频身份。可选路线与排错视频保留标记，开课答疑、项目说明、作业及社区笔记不计入视频课。

<!-- bilingual-en:start -->
Fine-grained mappings were authored per teaching unit using its title, explicit official outline and, where necessary, linked official teaching material. No keyword matcher or module-wide outcome replication was used. Existing canonical outcome IDs were reused where their scope matched; narrowly scoped new outcomes cover gaps such as stable matching, Bloom filters, callbacks, Ray futures and Kafka stream testing. Reading a title does not establish learner mastery.
<!-- bilingual-en:end -->

知识映射逐节人工审核，依据讲名、官方列明范围与必要的原始材料；没有使用关键词自动匹配，也没有把整个模块的知识复制到每节。优先复用语义范围一致的已有ID，明确缺口才新建窄范围知识，如稳定匹配、Bloom过滤器、回调、Ray future、Kafka流测试。目录覆盖与个人掌握始终分开。

### Units awaiting knowledge mapping / 待知识映射单元

- **CS50x**: [Artificial Intelligence](https://cs50.harvard.edu/x/2025/weeks/ai/).
- **CS61B**: [27. Software Engineering I](https://sp21.datastructur.es/materials/lectures/lec31/lec31); [31. Software Engineering II](https://sp21.datastructur.es/materials/lectures/lec34/lec34); [33. Software Engineering III](https://sp21.datastructur.es/materials/lectures/lec37/lec37); [37. Software Engineering IV](https://sp21.datastructur.es/).
- **15-213 / 18-213 / 15-513**: [The Future of Computing](https://www.cs.cmu.edu/afs/cs/academic/class/15213-f15/www/lectures/27-future.pdf).
- **CS186**: [17. Recovery I](https://drive.google.com/file/d/1dPwqG5Jsjcf-QY6zEC3tX93wyN6QEvCc/view?usp=drive_link); [18. Recovery II](https://drive.google.com/file/d/1GvONU5hpGys2iDu8w193glNTBhdZrLZ4/view?usp=drive_link).
- **DE Zoomcamp**: [Introduction to GCP (Google Cloud Platform)](https://www.youtube.com/watch?v=18jIzE41fJ4); [What is dbt?](https://www.youtube.com/watch?v=gsKuETFJr54); [Troubleshooting: Permission denied while globbing file pattern](https://www.youtube.com/watch?v=kL3ZVNL9Y4A); [5.1.1 - Introduction to Batch Processing](https://www.youtube.com/watch?v=dcHe5Fl3MF8); [5.1.2 - Introduction to Spark](https://www.youtube.com/watch?v=FhaqbEOuQ8U); [5.6.1 - Connecting to Google Cloud Storage](https://www.youtube.com/watch?v=Yyz293hBVcQ); [5.6.3 - Setting up a Dataproc Cluster](https://www.youtube.com/watch?v=osAiAYahvh8); [6.0.2 - What is stream processing](https://www.youtube.com/watch?v=WxTxKGcfA-k); [6.6 - Kafka configuration](https://www.youtube.com/watch?v=SXQtWyRpMKs).

<!-- bilingual-en:start -->
The four CS61B Software Engineering guides provide no sufficient concept-level boundary; 15-213’s final outlook and CS50’s AI overview are likewise not promoted to broad mastery claims. Berkeley CS186 Recovery I/II are real scheduled lectures, but the shared slide download returned a Google Drive preview rather than readable PDF content during this check. Data Engineering’s remaining unmapped entries are overview, platform setup or specific troubleshooting videos whose titles alone do not justify the broader existing atoms.
<!-- bilingual-en:end -->

CS61B四节软件工程guide不足以核定细粒度边界；15-213展望和CS50 AI概览也不直接映射为完整方法。CS186两节Recovery已核实存在，但共享课件链接本次下载返回Drive预览HTML。Data Engineering剩余待映射项是概览、平台部署或具体排错，不能仅凭标题点亮范围更广的知识。

## Validation / 验证

- All 439 units have a unique course-local ID, a learner-facing label, an official source, and an explicit mapping status.
- All referenced outcome IDs resolve in the combined course catalogue.
- All outline counts equal their actual unit-array lengths.
- New unit IDs do not reuse the replaced coarse-module IDs.
- DTC fixed-snapshot video IDs match all 77 official instructional links exactly.
- The combined `course-catalog` loads successfully; no private learning-state file or Obsidian plugin deployment was changed.

## Independent boundary review · 独立边界复核

<!-- bilingual-en:start -->
The final integration review changed CS61B lecture 36 and three DE Zoomcamp operational tutorials (Codespaces preparation, Spark installation, Confluent Cloud) from empty overview mappings to pending scope verification. The CS61B guide introduces additional JIT and sorting-cost material, while the tutorials teach actual procedures. The CS group therefore has 21 pending units in the final catalog. The first and last material links for 6.100L, CS61B and DE Zoomcamp were checked successfully.
<!-- bilingual-en:end -->

最终复核将 CS61B 第 36 讲和 DTC 三段操作教程从空概览改为范围待核；它们确有新增内容。CS 组最终为 21 个待核单元。6.100L、CS61B、DTC 的首末材料入口核验成功。
