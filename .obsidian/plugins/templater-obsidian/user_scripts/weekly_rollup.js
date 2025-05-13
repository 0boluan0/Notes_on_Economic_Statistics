/**
 * Obsidian Templater user-script
 * 作用：每周整合本周 Daily Notes 中 “# 今日完成内容” 区块，
 *       生成 / 更新周记，并把已复盘的 Daily Note
 *       移动到 “99_学习情况记录/已复盘/” 文件夹。
 *
 * 保留以下 5 个常量即可按需修改：
 *   DAILY_FOLDER      —— 每日日记所在文件夹
 *   ARCHIVE_SUBFOLDER —— 已复盘子文件夹
 *   WEEKLY_PREFIX     —— 周记子文件夹 (可为空字符串)
 *   SECTION_TITLE     —— 要抽取的标题
 *   FILENAME_PATTERN  —— Daily Note 文件名模板
 *   WEEK_START_ISO    —— 一周起始日 (1=周一, 0=周日 …)
 */
module.exports = async (tp) => {
  const app    = tp.app;
  const moment = window.moment;

  /* ===== 可按需修改 ===== */
  const DAILY_FOLDER      = "99_学习情况记录";
  const ARCHIVE_SUBFOLDER = "已复盘";
  const WEEKLY_PREFIX     = "周记";
  const SECTION_TITLE     = "# 今日完成内容";
  const FILENAME_PATTERN  = "YYYY-MM-DD——ddd";
  const WEEK_START_ISO    = 1;      // 1=周一
  /* ===================== */

  /* 1. 本周日期数组（7 天） */
  const today        = moment();
  const startOfWeek  = today.clone().isoWeekday(WEEK_START_ISO);
  const dateList     = [...Array(7).keys()]
                           .map(i => startOfWeek.clone().add(i, "days"));

  /* 2. 逐日提取区块并归档文件 */
  const weeklyBlocks = [];
  for (const d of dateList) {
    const dailyPath = `${DAILY_FOLDER}/${d.format(FILENAME_PATTERN)}.md`;
    const afile     = app.vault.getAbstractFileByPath(dailyPath);
    if (!afile) continue;                                        // 当天没写

    const raw = await app.vault.read(afile);
    const idx = raw.indexOf(SECTION_TITLE);
    if (idx === -1) continue;                                    // 无目标标题

    const lines = raw.slice(idx + SECTION_TITLE.length)
                     .split("\n")
                     .filter(l => l.trim().length)
                     .map(l => l.trim());
    if (lines.length) {
      weeklyBlocks.push(
`### ${d.format("YYYY-MM-DD ddd")}
${lines.join("\n")}\n`);
    }

    /* 2.1 移动到已复盘 */
    const newPath =
      `${DAILY_FOLDER}/${ARCHIVE_SUBFOLDER}/${afile.name}`;
    await app.fileManager.renameFile(afile, newPath);
  }

  if (!weeklyBlocks.length) {
    new Notice("本周没有可整合的日记条目");
    return;
  }

  /* 3. 写 / 更新周记文件 */
  const weeklyName =
    `${WEEKLY_PREFIX}/${today.format("gggg-[W]ww")}.md`;
  const weeklyPath = `${DAILY_FOLDER}/${weeklyName}`;
  const header =
`# 🗓 ${today.format("gggg年第 ww 周")}（${startOfWeek.format("YYYY.MM.DD")}–${startOfWeek.clone().add(6,"days").format("YYYY.MM.DD")}）

`;
  const content = header + weeklyBlocks.join("\n");

  const wFile = app.vault.getAbstractFileByPath(weeklyPath);
  if (wFile) {
    await app.vault.modify(wFile, content);   // 覆写
  } else {
    await app.vault.create(weeklyPath, content);
  }

  new Notice(`✅ 已整合 ${weeklyBlocks.length} 篇日记并生成周记`);
};