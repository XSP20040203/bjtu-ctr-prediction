"""Build a standalone model explorer from the previously verified report source.

Read literals with AST only; never import or execute model/report code.
"""
from pathlib import Path
import ast
import hashlib
import json

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / '四份建模报告_20261104' / '_work' / 'integrated_architecture_report.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))

def literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'block':
        return {'title': ast.literal_eval(node.args[0]), 'body': ast.literal_eval(node.args[1])}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(x) for x in node.elts]
    return ast.literal_eval(node)

sections, tables, sources = [], [], []
for node in tree.body:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Name) and call.func.id == 'page':
            title, name, rows, paragraphs = map(literal, call.args[:4])
            kw = {x.arg: literal(x.value) for x in call.keywords}
            row_ids, nodes = [], []
            for row in rows:
                ids = []
                for item in row:
                    ids.append(len(nodes))
                    nodes.append(item)
                row_ids.append(ids)
            edges = kw.get('edges')
            if edges is None:
                edges = [[a, b] for upper, lower in zip(row_ids, row_ids[1:]) for a in upper for b in lower]
            sections.append(dict(id=name, title=title, nodes=nodes, rows=row_ids, edges=edges, paragraphs=paragraphs))
        if isinstance(call.func, ast.Attribute) and call.func.attr == 'table':
            tables.append(dict(headers=literal(call.args[0]), rows=literal(call.args[1]),
                               caption=next((literal(x.value) for x in call.keywords if x.arg == 'caption'), '')))
    if isinstance(node, ast.For) and isinstance(node.iter, ast.List):
        try:
            values = literal(node.iter)
        except (ValueError, TypeError):
            continue
        if values and all(isinstance(x, list) and len(x) == 3 and isinstance(x[1], str) and x[1].startswith('analysis/') for x in values):
            sources.extend(dict(title=x[0], path=x[1], detail=x[2]) for x in values)

groups = [
    ('总览与路径', ['total', 'route']),
    ('数据输入与特征', ['alignment', 'embedding', 'input_families']),
    ('基础网络与融合', ['base_models', 'gated', 'protected_day', 'tabular_parallel', 'allocation_encoder', 'allocation_output', 'base_fusion']),
    ('用户日并行校正', ['calibration', 'parallel_day', 'group_operator', 'day_branch', 'twelve', 'merge']),
    ('行级、质量与排序', ['row_daily', 'news_response', 'quality337', 'quality_transfer', 'compact_dense', 'ranking']),
    ('任务受众与历史', ['audience', 'history', 'events']),
    ('上下文与神经表示', ['context395', 'frozen138', 'three_trees', 'embedding_family', 'final_cohorts']),
    ('运算、例子与训练', ['operators', 'example', 'training']),
]
known = {s['id'] for s in sections}
assert known == {x for _, items in groups for x in items}, (known, groups)
assert len(sections) == 35 and len(tables) == 4 and len(sources) == 24
for s in sections:
    s['title'] = s['title'].split(' ', 1)[1] if ' ' in s['title'] and all(c in '一二三四五六七八九十' for c in s['title'].split(' ', 1)[0]) else s['title']
    s['group'] = next(title for title, items in groups if s['id'] in items)
    assert all(0 <= a < len(s['nodes']) and 0 <= b < len(s['nodes']) for a, b in s['edges'])

data = dict(sections=sections, tables=tables, groups=[dict(title=t, ids=items) for t, items in groups], sources=sources,
            meta=dict(created='2026-11-04', basis='已核对的整合模型结构与维度详解', source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      diagram_count=len(sections), node_count=sum(len(s['nodes']) for s in sections)))
(OUT / 'model_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
template = (OUT / 'site_template.html').read_text(encoding='utf-8')
page = template.replace('/*__MODEL_DATA__*/', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
(OUT / '说明阅读.html').write_text(page, encoding='utf-8')
print(json.dumps(dict(path=str(OUT / '说明阅读.html'), sections=len(sections), nodes=data['meta']['node_count'], tables=len(tables), bytes=len(page.encode('utf-8'))), ensure_ascii=False))
