# News Format

Use this rubric for module 1 in the daily note.

## Time Window

- Use the latest 24 hours from execution time.
- If a source only provides a date (no exact time), keep it only when clearly inside the window.

## Coverage

- Show a single summary table beneath the map.
- The number of rows is allocated dynamically by region heat.
- Do not include sources in the table.
- Append finance and tech bullet summaries after the table.
- Finance/tech bullets should not include sources.

## Output Template

```markdown
## 模块一｜全球态势静态快照（中文为主）
> 时间窗口：{start_time} ~ {end_time} ({tz})
> 注：表格为摘要，不含来源。

![[98_attachment/dashboards/YYYY-MM-DD-map.svg]]

### 今日要点（按热度分配篇幅）
| 区域 | 热度 | 发生了什么 |
| --- | --- | --- |
| 美洲 / Americas | 86 | …… |
|  |  | …… |
| 欧洲 / Europe | 9 | …… |

### 金融要点
- ……
- ……

### 科技要点
- ……
- ……
```

## Quality Rules

- Keep each summary short and factual (1 short clause).
- Prefer the top items in each region by recency.
- For finance/tech, keep 2–4 bullets and avoid jargon unless needed.
- If no items are available, write "暂无显著新闻".
