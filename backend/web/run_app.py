#!/usr/bin/env python3
"""量化交易系统 Web 面板启动入口。

启动方式：
    python3 backend/web/run_app.py
"""
import sys
from pathlib import Path

# 项目根目录
_PROJECT_ROOT = Path(__file__).parent.parent.parent
# backend 根目录及各功能子目录加入 sys.path，保持模块名不变（import config / import data_source 等）
_BACKEND_DIR = Path(__file__).parent.parent
for sub in ['', 'web', 'data', 'strategy', 'pattern', 'signal', 'tracking', 'utils']:
    sys.path.insert(0, str(_BACKEND_DIR / sub))

from config import config
from web_app import create_app

app = create_app()

if __name__ == '__main__':
    # macOS 默认 spawn 多进程，子进程会重新 import/执行主模块。
    # 子进程到达此处时必须跳过 app.run()，避免端口冲突导致任务全部失败。
    from multiprocessing import current_process
    if current_process().name == 'MainProcess':
        print('=' * 50)
        print('  量化交易策略回测系统')
        print('=' * 50)
        print(f'  访问地址: http://{config.FLASK_HOST}:{config.FLASK_PORT}')
        print(f'  数据目录: {config.DATA_DIR}')
        print(f'  日志目录: {config.LOG_DIR}')
        print('=' * 50)
        app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG,
                threaded=False, use_reloader=False)
    # 子进程：不启动服务器
