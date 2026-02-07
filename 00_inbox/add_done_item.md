<%*
(async () => {

  /* ========== 仅需改动这 3 行 ========== */
  const DAILY_FOLDER     = "99_学习情况记录";  // 日记文件夹
  const SECTION_TITLE    = "# 今日完成内容";   // 标题（含 # 或 ##）
  const FILENAME_PATTERN = "YYYY-MM-DD——ddd"; // 文件名模板
  /* ===================================== */

  const desc = await tp.system.prompt("完成了什么？");
  if (!desc) { new Notice("未输入内容，已取消"); return; }

  const ts = tp.date.now("YYYY-M-D  HH:mm");

  /* 1. 定位今天日记文件 */
  const dailyPath = `${DAILY_FOLDER}/${tp.date.now(FILENAME_PATTERN)}.md`;
  const file      = app.vault.getAbstractFileByPath(dailyPath);
  if (!file) { new Notice("找不到今天日记！"); return; }

  /* 2. 读取全文并定位标题 */
  let text = await app.vault.read(file);
  const idx = text.indexOf(SECTION_TITLE);
  if (idx === -1) {
    new Notice(`日记缺少「${SECTION_TITLE}」`);
    return;
  }

  /* 3. 统计已有编号 */
  const after  = text.slice(idx + SECTION_TITLE.length);
  const count  = (after.match(/^\d+\.\s/mg) || []).length + 1;

  /* 4. 组装新条目并写回（追加到末尾） */
  const entry  = `\n${count}. ==${desc}==   ${ts}`;
  const newTxt = text.slice(0, idx + SECTION_TITLE.length) + after + entry;

  await app.vault.modify(file, newTxt);
  new Notice("✓ 已写入今日完成内容");

})();
%>