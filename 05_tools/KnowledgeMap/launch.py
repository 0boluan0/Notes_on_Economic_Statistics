#!/usr/bin/env python3
"""Run the personal map until the user closes its Terminal window."""
import argparse
from pathlib import Path
import sys
import webbrowser

from server import create_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    url = f'http://127.0.0.1:{args.port}'
    try:
        server = create_server(Path(__file__).resolve().parents[2], args.port, args.data_dir)
    except OSError as error:
        print(f'端口 {args.port} 已被占用。若地图已启动，请打开 {url}；否则可用 --port 指定其他端口。', flush=True)
        return 1
    print(f'知识与课程：{url}\n保持此窗口打开。按 Control-C 关闭地图服务。', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n地图已关闭，已保存的关联仍保留。', flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
