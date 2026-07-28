import json
from pathlib import Path

from config import config

TEMPLATES_FILE = config.SIGNAL_UPDATE_DIR / 'templates.json'

DEFAULT_TEMPLATES = [
    {
        'id': 'default',
        'name': '默认模板',
        'description': '标准回测配置，适用于常规信号更新。覆盖指数、A股、ETF全量标的，回测周期2010年至今，观察天数2天，关闭谨慎模式。',
        'params': {
            'types': ['index', 'hs', 'cy', 'kc', 'etf'],
            'start_date': '20100104',
            'end_date': '',
            'observe_day': 2,
            'cautious': False,
            'workers': 4,
        },
        'system': True,
    },
    {
        'id': 'quick',
        'name': '快速增量',
        'description': '快速更新模板，仅回测最近1年数据，减少计算时间。适用于日常快速验证信号表现变化。',
        'params': {
            'types': ['index', 'hs', 'cy', 'kc', 'etf'],
            'start_date': '',
            'end_date': '',
            'observe_day': 2,
            'cautious': False,
            'workers': 8,
        },
        'system': True,
    },
    {
        'id': 'quarterly',
        'name': '季度全量',
        'description': '季度更新专用，回测完整历史数据，开启谨慎模式，确保结果严谨性。建议每6个月执行一次。',
        'params': {
            'types': ['index', 'hs', 'cy', 'kc', 'etf'],
            'start_date': '20100104',
            'end_date': '',
            'observe_day': 2,
            'cautious': True,
            'workers': 6,
        },
        'system': True,
    },
    {
        'id': 'risk_test',
        'name': '风控测试',
        'description': '风控策略测试模板，开启谨慎模式并使用更高的观察天数，评估策略在严格风控条件下的表现。',
        'params': {
            'types': ['index', 'hs', 'cy', 'kc'],
            'start_date': '20100104',
            'end_date': '',
            'observe_day': 10,
            'cautious': True,
            'workers': 4,
        },
        'system': True,
    },
]


def _load_templates():
    """加载模板列表"""
    if not TEMPLATES_FILE.exists():
        _save_templates(DEFAULT_TEMPLATES)
        return DEFAULT_TEMPLATES
    try:
        with open(TEMPLATES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        _save_templates(DEFAULT_TEMPLATES)
        return DEFAULT_TEMPLATES


def _save_templates(templates):
    """保存模板列表"""
    with open(TEMPLATES_FILE, 'w', encoding='utf-8') as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)


def get_templates():
    """获取所有模板"""
    return _load_templates()


def get_template(template_id):
    """获取单个模板"""
    templates = _load_templates()
    for t in templates:
        if t['id'] == template_id:
            return t
    return None


def create_template(name, description, params):
    """创建自定义模板"""
    templates = _load_templates()
    template_id = f'custom_{len(templates) + 1}'
    new_template = {
        'id': template_id,
        'name': name,
        'description': description,
        'params': params,
        'system': False,
    }
    templates.append(new_template)
    _save_templates(templates)
    return new_template


def update_template(template_id, name=None, description=None, params=None):
    """更新模板"""
    templates = _load_templates()
    for t in templates:
        if t['id'] == template_id:
            if t.get('system', False):
                return None, '系统模板不可修改'
            if name is not None:
                t['name'] = name
            if description is not None:
                t['description'] = description
            if params is not None:
                t['params'] = params
            _save_templates(templates)
            return t, None
    return None, '模板不存在'


def delete_template(template_id):
    """删除模板"""
    templates = _load_templates()
    for i, t in enumerate(templates):
        if t['id'] == template_id:
            if t.get('system', False):
                return False, '系统模板不可删除'
            templates.pop(i)
            _save_templates(templates)
            return True, None
    return False, '模板不存在'
