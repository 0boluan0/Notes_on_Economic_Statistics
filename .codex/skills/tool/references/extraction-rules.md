# Extraction Rules

## URL handling

1. Accept only `http://` or `https://`.
2. Use the first valid URL in user input.
3. If the URL is invalid or unreachable, fail fast with clear error and do not write output files.

## Source priority

### When URL is GitHub

1. Repository metadata: `https://api.github.com/repos/{owner}/{repo}`
2. Release metadata:
   - first try `.../releases/latest`
   - fallback `.../releases?per_page=1`
3. README:
   - first try `.../readme` via GitHub API
   - fallback to `download_url` from API payload
4. Homepage metadata:
   - if repo `homepage` exists, fetch title/meta description from homepage

### When URL is non-GitHub

1. Fetch page HTML and extract:
   - `og:site_name`
   - `og:title` or `<title>`
   - `description` or `og:description`
2. Scan page links (`<a href="...">`) for first GitHub repo URL.
3. If GitHub repo URL is found, run GitHub extraction pipeline to enrich installation/usage details.

## Project naming fallback chain

1. README first H1 (`# Title`)
2. Website `og:site_name`
3. Website `title`
4. GitHub repo name
5. URL domain

## README section matching keywords

- Installation: `install`, `installation`, `setup`, `requirements`, `dependency`, `安装`, `部署`, `依赖`, `环境`
- First run: `quick start`, `quickstart`, `getting started`, `run`, `start`, `demo`, `快速开始`, `首次`, `启动`, `运行`
- Usage: `usage`, `example`, `examples`, `guide`, `tutorial`, `workflow`, `cli`, `command`, `使用`, `示例`, `教程`, `工作流`
- Troubleshooting: `troubleshoot`, `troubleshooting`, `faq`, `common issues`, `error`, `known issues`, `limitations`, `排错`, `问题`, `常见`, `注意事项`

Use heading-title keyword matching first, then extract nearby text and up to two code blocks.

## Missing information policy

If section data is missing, keep the section and write:

- explicit `信息不足`
- one or more `TODO` bullets for manual completion

Do not leave required sections empty.

## Filename sanitization

1. Remove illegal characters: `\ / : * ? " < > |`
2. Collapse repeated whitespace to one space.
3. Trim leading/trailing spaces and trailing dots.
4. If empty after sanitization, use `tool-note`.

Output path:

- default: `<vault-root>/05_tools/<sanitized-name>.md`
