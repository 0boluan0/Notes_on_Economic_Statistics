# Link Check Report

- Date: 2026-02-06
- Files scanned: 442
- Auto-fixes applied: 0
- Issues found: 0

## External Links (Not Checked)
- Count: 14

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
