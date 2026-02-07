// QuickAdd 任务添加脚本
async function addTask() {
  const taskTitle = await inputPrompt("任务标题");
  const dueDate = await inputPrompt("截止日期 (YYYY-MM-DD)");
  const priority = await selectPrompt("优先级", ["高", "中", "低"]);
  const skills = await selectPrompt("技能", ["数学", "编程", "论文", "英语"], true);

  const taskTags = skills.map(skill => `#${skill}`).join(" ");

  const taskText = `- [ ] ${taskTitle}
  - 截止日期：${dueDate}
  - 优先级：${priority}
  - 技能：${skills.join("/")}
  - 标签：#任务 ${taskTags}`;

  // 将任务添加到今日日记
  const today = moment().format("YYYY-MM-DD——ddd");
  const dailyNotePath = `99_学习情况记录/${today}.md`;

  await app.vault.adapter.read(dailyNotePath).then(content => {
    // 找到任务汇总部分并添加新任务
    const updatedContent = content.replace(
      /## 未完成任务/,
      `## 未完成任务\n${taskText}`
    );
    app.vault.adapter.write(dailyNotePath, updatedContent);
  });

  new Notice(`任务 "${taskTitle}" 已添加到今日日记`);
}

addTask();
