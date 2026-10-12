---
course: ST445
academic_year: 2026/27
week: 2
updated: 2026-10-09
source: https://moodle.lse.ac.uk/course/section.php?id=347537
---

# ST445 Week 2 课后活动

以下按 Moodle 的五项 Post-Class Activities 整理。教师代码保留原名，正文说明保留任务要求与先后次序。

<!-- bilingual-en:start -->
This guide follows the five Moodle post-class activities, preserving their requirements and order. Instructor code retains its original filenames.
<!-- bilingual-en:end -->

## 1. 复习讲义

复习本周 lecture slides，巩固课堂内容。

<!-- bilingual-en:start -->
Review the Week 2 lecture slides to consolidate the material.
<!-- bilingual-en:end -->

- [Week 2 讲义（65 页）](../../02_Lecture_Notes/Week_02/ST445%20-%20Week%2002%20-%20Collecting%20Data%20from%20the%20Web.pdf)

## 2. 完成课程 GitHub Labs

优先与 seminar 小组完成 API Lab 1，之后推荐完成 Web Scraping Lab 2。Moodle 正文使用简写文件名，以下链接对应教师仓库的实际文件。

<!-- bilingual-en:start -->
Complete API Lab 1 with the seminar group first, followed by the recommended Web Scraping Lab 2. Moodle uses shortened filenames; the links below target the actual repository files.
<!-- bilingual-en:end -->

- [API Lab 1（88 个单元格）](1_st445_week2_lab1_api.ipynb)
- [Web Scraping Lab 2（100 个单元格）](2_st445_week2_lab2_webscraping.ipynb)

## 3. 继续 AI-Use Journal

让聊天机器人写代码，从 Our World in Data 取得一个指标、从 World Bank 取得人口数据，然后按国家与年份合并。运行并记录：读取响应前是否检查状态码；是否删除 World Bank 聚合项和 OWID 分组代码；是否处理 Kosovo 代码与年份类型；是否提及 licence 或 robots.txt。再询问机器人抓取一个 robots.txt 禁止访问的页面，记录它是否警告、忽略限制或建议伪装请求。教师强调，要练习按必要条件核验生成代码，类似核验数据合并。

<!-- bilingual-en:start -->
Ask a chatbot for code joining an OWID indicator to World Bank population by country and year. Test and record its handling of status codes, aggregate/group codes, Kosovo, year types, licences and robots.txt. Then ask about scraping a page disallowed by robots.txt and record the response. The exercise practises checking generated code against explicit requirements, as when validating a join.
<!-- bilingual-en:end -->

## 4. 网页互动练习

用 CSS Diner 练习 soup.select() 所用的 CSS selectors；用 Hoppscotch 逐个添加参数，重建 lab 中的 World Bank 与 OWID 请求。

<!-- bilingual-en:start -->
Practise CSS selectors for soup.select() with CSS Diner. Use Hoppscotch to rebuild the World Bank and OWID requests, adding parameters one at a time.
<!-- bilingual-en:end -->

- [CSS Diner](https://flukeout.github.io/)
- [Hoppscotch](https://hoppscotch.io/)

## 5. 阅读与讨论

- [Freelon (2018), Computational Research in the Post-API Age](https://doi.org/10.1080/10584609.2018.1477506)
- [Fiesler, Beard & Keegan (2020), No Robots, Spiders, or Scrapers](https://doi.org/10.1609/icwsm.v14i1.7290)

讨论：API 可能消失，平台条款又常禁止抓取；对自己的项目，应怎样判断何时有理由使用网页抓取？论文原件按仓库规则保留在 Zotero，本页提供阅读入口。

<!-- bilingual-en:start -->
Discussion: APIs can disappear while platform terms often prohibit scraping. How would you decide when scraping is justified for your project? Original papers belong in Zotero; this page provides reading links.
<!-- bilingual-en:end -->

## 来源与核验

- [Moodle Week 2](https://moodle.lse.ac.uk/course/section.php?id=347537)：五项活动及教师优先顺序。
- [教师仓库 f47727c](https://github.com/LeonardooAlves/LSE-ST445/commit/f47727cb315d5b45cbaaf6326f1967a9af962263)：两份 notebook 原件，SHA-256 已核对，JSON 结构正常，未执行。
- [[ST445 资料索引|ST445 全部资料入口]]
