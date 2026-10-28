# Bayesian course outlines · 贝叶斯课程目录核验

<!-- bilingual-en:start -->
Checked on 2026-10-26 against the selected official editions. Directory completeness and knowledge mapping completeness are separate. Existing learning records were not changed. New unit identifiers do not inherit completion from the former broad modules.
<!-- bilingual-en:end -->

于 2026-10-26 对照所选版本官方材料核验。目录完整性与知识映射完整性分开记录。没有改动个人学习状态，新单元不继承旧大纲模块的完成状态。

| Course | Official counting basis | Result |
| --- | --- | --- |
| [Aalto BDA 2025](https://avehtari.github.io/BDA_course_Aalto/Aalto2025.html) | 12 lectures, with recordings split within lectures | 12 units; the last is project presentation advice. Assignments and project evaluation excluded. |
| [Duke STA601 Fall 2021](https://www2.stat.duke.edu/courses/Fall21/sta601.001/syllabus/) | Lectures 0–22 and 24, plus unnumbered HMC continuation | 25 units; the continuation remains unmapped. Labs, the unnumbered November 8 lab-slot review, and exams excluded. |
| [Cambridge MLBayInfer 2020–21](https://www.cl.cam.ac.uk/teaching/2021/MLBayInfer/) | 16 official public lecture recordings | All 16 lectures, in the instructor’s playlist order. |

## Evidence and limits · 依据与边界

<!-- bilingual-en:start -->
Aalto: the official page explicitly states twelve lectures. Its detailed lecture descriptions establish the mapping boundaries for Monte Carlo, MCMC, HMC, model checking, comparison, decisions and sensitivity. Numbered video parts are provided as resources within the lecture. Completing one clip is not represented as completing the whole lecture. The official 2025 index still links to the 2024 recording for 8.2; that year remains visible. Metropolis alone does not grant the complete general Metropolis–Hastings scope.

Duke: public HTML/PDF lecture slides were reopened, including lecture headings and the Bayesian adaptive regression kernels slides for lecture 22. Slides 12 and 13 distinguish hierarchical shrinkage from Bayesian multiple testing; the latter receives its own scope. Lecture 23 is not invented. The unnumbered HMC continuation has no distinct public material, so it can record unit completion but grants no knowledge automatically. Lecture 24 is review. Video access is through Sakai rather than open public recordings.

Cambridge: the final selected edition is the complete 2020–21 public recording series. The instructor's [teaching page](https://www.cl.cam.ac.uk/~sbh11/lecturing.html) identifies the edition, and his [official playlist](https://www.youtube.com/playlist?list=PLdLk2RYEiAhp9Slj6F_LCMXUv7_Fi3V_Y) contains exactly 16 numbered videos. Each video's description was reopened to establish the actual lecture boundary. The units follow these descriptions; for example, maximum margin starts in lecture 6, full SVM treatment finishes in 7, and Gaussian process regression finishes in 12. The later 2025–26 combined deck was not used to assign lecture boundaries to older videos. Two narrower scopes distinguish maximum-margin setup from complete SVMs and classifier comparison from hyperparameter selection.
<!-- bilingual-en:end -->

Aalto：官网明确为 12 讲。分段录播放在所属讲内；完成一个视频片段不等于完成整讲。8.2 仍链接 2024 年录播，保留年份。仅讲 Metropolis 不赋予一般非对称提议的完整 Metropolis–Hastings 范围。

Duke：11 月 8 日实验时段的未编号复习不计入主讲目录。重新打开逐讲 HTML/PDF，核对讲义标题与范围。第 12、13 讲区分层级收缩和贝叶斯多重检验。没有补造第 23 讲；未编号 HMC 续讲可记录课次，但范围仍待核对。第 24 讲按复习处理。录播需要 Sakai 访问。

Cambridge：最终固定到教师本人发布的 2020–21 完整 16 讲录播。教师主页确认版本，播放列表确认数量，逐个重新打开视频说明以确定内容边界。第 6 讲只标记最大间隔，第 7 讲才标记完整 SVM，第 12 讲才标记完整高斯过程回归；避免前一讲仅引入主题就提前点亮完整范围。

## Recording boundary checks · 录像分界补证

<!-- bilingual-en:start -->
Independent subtitle inspection confirmed Cambridge lecture 10 covers neural-network weight posteriors (07:30, 10:38, 27:38), and lecture 11 covers predictive uncertainty (14:18 and 18–19 minutes). K-means belongs to lecture 12 (42:34 introduction; 46:55–49:27 algorithm), while lecture 13 recalls it at 00:56 before introducing EM at 01:24. The mapping was corrected accordingly. Timestamp links are stored with each unit. Only subtitles and metadata were retrieved for this check.
<!-- bilingual-en:end -->

独立字幕抽查确认第 10 讲的神经网络权重后验与第 11 讲的预测不确定性。K-means 已据录像分界改到第 12 讲；第 13 讲只新增 EM。各讲依据中保存可直达的视频时间戳。本次补证仅读取字幕与元数据。
