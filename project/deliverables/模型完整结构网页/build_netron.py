"""Create a real Netron viewer for the documented integrated prediction recipe.

This exports a structural graph, never executes project code, loads model weights,
or accesses training/prediction labels.
"""
from pathlib import Path
from urllib.request import urlopen
from collections import defaultdict
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor' / 'netron'
DATA = json.loads((ROOT / 'model_data.json').read_text(encoding='utf-8'))
SECTIONS = {s['id']: s for s in DATA['sections']}
ORIGINALS = ROOT / '_vendor_originals'
ORIGINALS.mkdir(exist_ok=True)

def patch_asset(name, transform):
    backup = ORIGINALS / name
    if not backup.exists():
        shutil.copy2(VENDOR / name, backup)
    source = backup.read_text(encoding='utf-8')
    result = transform(source)
    assert result != source, name
    (VENDOR / name).write_text(result, encoding='utf-8')

def patch_message(s):
    s = s.replace('        this.inputs = [];', "        this.name = data.name || '';\n        this.description = data.description || '';\n        this.inputs = [];", 1)
    s = s.replace("        this.type = { name: data.type.name, category: data.type.category };", "        this.type = { name: data.type.name, category: data.type.category };\n        this.description = data.description || '';\n        this.graphReference = data.type.graph || null;")
    needle = '        this.modules = modules.map((module) => new message.Module(module));'
    replacement = needle + '''
        // Local structural-model extension: preserve named modules and drill-down.
        const definitions = new Map(this.modules.map((module) => [module.name, module]));
        for (const module of this.modules) {
            for (const node of module.nodes) {
                if (node.graphReference && definitions.has(node.graphReference)) {
                    const definition = definitions.get(node.graphReference);
                    node.type = { ...definition, name: node.type.name, category: node.type.category };
                }
            }
        }'''
    return s.replace(needle, replacement)

def patch_view(s):
    s = s.replace('attributes: false,', 'attributes: true,', 1)
    s = s.replace("mousewheel: 'scroll'", "mousewheel: 'zoom'", 1)
    # Keep full Chinese module names and full dimension/formula values on nodes.
    s = s.replace('if (content.length > 21) {', 'if (content.length > 120) {', 1)
    s = s.replace("node.type.name.split('.').pop()", 'node.type.name', 1)
    s = s.replace('if (content && content.length > 12) {', 'if (content && content.length > 160) {', 1)
    # Netron's normal 15% floor is too large for this long ensemble graph.
    s = s.replace('Math.max(limit, 0.15)', 'Math.max(limit, 0.015)')
    s = s.replace("if ((name && this._term(name)) || (type && this._term(type)) || (identifier && this._term(identifier))) {", "if ((name && this._term(name)) || (type && this._term(type)) || (identifier && this._term(identifier)) || (node.attributes || []).some((attribute) => this._term(String(attribute.value)))) {")
    return s

def patch_browser(s):
    return s.replace('        await age();\n        await consent();\n        await telemetry();',
                     '        // Local offline viewer: no age gate, geolocation request, or telemetry startup.')

patch_asset('message.js', patch_message)
patch_asset('view.js', patch_view)
patch_asset('browser.js', patch_browser)
patch_asset('index.html', lambda s: s.replace('<a id="sidebar-closebutton" class="sidebar-closebutton" href="javascript:void(0)" draggable="false">&times;</a>', '<button type="button" id="sidebar-closebutton" class="sidebar-closebutton" draggable="false" aria-label="关闭属性" style="border:0;background:transparent">&times;</button>'))
if not (VENDOR / 'LICENSE').exists():
    (VENDOR / 'LICENSE').write_bytes(urlopen('https://raw.githubusercontent.com/lutzroeder/netron/main/LICENSE', timeout=30).read())

class Graph:
    def __init__(self, name, description=''):
        self.data = dict(name=name, description=description, inputs=[], outputs=[], nodes=[], values=[])
        self.values = {}
        self.ids = set()

    def value(self, name, shape, dtype='float32'):
        if name not in self.values:
            self.values[name] = len(self.data['values'])
            record = dict(name=name)
            if shape is not None:
                record['type'] = dict(dataType=dtype, shape=dict(dimensions=shape))
            self.data['values'].append(record)
        return self.values[name]

    def input(self, name, shape, dtype='float32'):
        value = self.value(name, shape, dtype)
        self.data['inputs'].append(dict(name=name, value=[value]))
        return name

    def node(self, key, title, parents, shape=('N', 1), attrs=None, section=None, category='Layer', out=None, description='', dtype='float32'):
        assert key not in self.ids, key
        self.ids.add(key)
        output = out or key
        output_id = self.value(output, list(shape) if shape is not None else None, dtype)
        inputs = []
        for i, p in enumerate(parents):
            assert p in self.values, (title, p)
            inputs.append(dict(name=f'输入{i+1}' if len(parents) > 1 else '输入', value=[self.values[p]]))
        attributes = []
        for name, value in (attrs or {}).items():
            attributes.append(dict(name=name, type='string', value=str(value)))
        kind = dict(name=title, category=category)
        if section:
            kind['graph'] = SECTIONS[section]['title']
            description = description or '\n\n'.join(SECTIONS[section]['paragraphs'])
        self.data['nodes'].append(dict(name=title, type=kind, description=description,
                                      inputs=inputs, outputs=[dict(name='输出', value=[output_id])], attributes=attributes))
        return output

    def output(self, key, label):
        self.data['outputs'].append(dict(name=label, value=[self.values[key]]))

g = Graph('完整整合模型｜从四张输入表到 pctr', '最终提交采用的完整依赖图。每个模型独立保存权重；ƒ 可进入该模块的内部详图。')
ad_history = g.input('广告历史表｜2,526,320 行', [2526320, '原始字段'], 'table')
ad_target = g.input('待预测广告表｜N=404,182', ['N', '原始字段'], 'table')
news_train = g.input('资讯历史表｜827,820 行', [827820, '原始字段'], 'table')
news_target = g.input('资讯测试表｜142,962 行', [142962, '原始字段'], 'table')
aligned = g.node('aligned', '用户日期对齐与原始行序保留', [ad_history, ad_target, news_train, news_target], ['N', '对齐资料'],
    attrs={'连接键': 'user_id ↔ u_userId；pt_d ↔ e_et', '资讯': '先按用户日聚合再连接', '原始行': '重复曝光保留；group_index 为 N×1'}, section='alignment', category='Data')
emb = g.node('embedding312', '类别索引与无序历史列表准备', [aligned], ['N',87], attrs={'索引输入': '27单值 + 12列表×最多5项 = 87个编号', '成员嵌入': '各成员查自己的嵌入表，27×8+12×8=312', '列表': '只平均有效项；PAD=0；UNK=1', '权重': '各网络的312维表示宽度相同，训练权重独立'}, section='embedding', category='Data', dtype='int64')
num = g.node('numeric', '多套数值与表格特征组装', [aligned], ['N','特征族'], attrs={'166维': '75基础 + 43曝光 + 48快照', '325维': '166 + 159多时间窗口', '500维': '166 + 159 + 54历史输入 + 41历史元数据 + 80资讯', '535维': '75 + 43 + 48 + 159 + 210资讯响应'}, section='input_families', category='Transform')

base = g.node('base_dcn', '基础有界编号深度交叉网络', [emb,num], attrs={'输入': '312 + 166 = 478', '交叉 ∥ 深层': '两层全矩阵 D→D ∥ 478→128→64', '输出': '拼接542→1→Sigmoid'}, section='base_models')
base_cat = g.node('base_cat', '基础类别提升点击树 CatBoost', [num], attrs={'输入': '253 = 27类别 + 60列表token + 166数值', '配置': '87类别列；深度6；2000轮', '输出': 'N×1 概率'}, section='base_models', category='Tree')
base_lgb = g.node('base_lgb', '统计轻量梯度提升点击树 LightGBM', [num], attrs={'输入': '253维', '配置': '31叶；112轮', '输出': 'N×1 概率'}, section='base_models', category='Tree')
multi = g.node('multi_dcn', '多时间尺度有界编号深度交叉网络', [emb,num], attrs={'输入': '312 + 325 = 637', '交叉 ∥ 深层': '两层全矩阵637 ∥ 637→128→64', '输出': '拼接701→1→Sigmoid'}, section='base_models')
context = g.node('base_context', '共享上下文轻量梯度提升点击树 LightGBM', [num], attrs={'输入': '349 = 24类别 + 325数值', '配置': '15叶；378轮'}, section='base_models', category='Tree')
oof = g.node('oof_dcn', '分组折外曝光有界编号深度交叉网络', [emb,num], attrs={'输入': '312 + 166 = 478', '配置': '独立训练权重；两层全矩阵交叉 ∥ 深层128→64'}, section='base_models')
tabm = g.node('tabm', '分段数值嵌入并行表格神经网络 TabM', [emb,num], attrs={'输入': '535×4 + 312 = 2452', '结构': '4成员；两层128；Dropout 0.15', '输出': '每成员Sigmoid后平均'}, section='tabular_parallel')
rich_gated = g.node('rich_gated', '丰富历史门控低秩多专家深度交叉网络', [emb,num], attrs={'输入': '312 + 500 = 812', '字段门控': '55→16→39→2×Sigmoid', '交叉 ∥ 深层': '2层×3专家；D→32→32→D ∥ D→128→64', '输出': '拼876→1→Sigmoid'}, section='gated')
day_state = g.node('day_state', '用户日无广告标签上下文', [aligned], ['G',598], attrs={'输入': '598维当日可观测上下文', '检查点': '参考、加权、排序三份检查点各有独立状态编码权重', '成员内部': '各加自己的8维用户嵌入，606→64→32'}, section='protected_day', category='Data')
protected = g.node('protected_ref', '日状态保护门控交叉参考网络', [emb,num,day_state], attrs={'广告主干': '478→交叉与深层→542→1', '日状态': '606→64→32；行头32→1→2tanh', '辅助训练': '仅训练；BCE权重0.02；辅助用户嵌入停止梯度', '输出': 'Sigmoid(广告主干Logit + 行状态增量)'}, section='protected_day')
weighted = g.node('protected_weighted', '正样本加权日状态保护门控交叉网络', [emb,num,day_state], attrs={'结构': '广告478与用户日606；独立检查点', '用途': '基础融合补充概率与排序迁移参照'}, section='protected_day')
rank = g.node('protected_rank', '成对排序微调日状态保护门控交叉网络', [emb,num,day_state], attrs={'结构': '广告478与用户日606；独立检查点', '用途': '将相对正样本加权网络的Logit差迁移到右路'}, section='protected_day')
allocated = g.node('allocated', '用户日总强度分配网络', [emb,num], attrs={'行编码': '312+32=344→128→64', '汇总': '4个32维学习槽→G×4×64', '双头': '256维行残差；193维日摘要', '输出': '组内Softmax分配λ；p=1−exp(−行强度)'}, section='allocation_encoder')
five = g.node('five', '五成员概率加权融合', [base,base_cat,base_lgb,multi,context], attrs={'权重': '0.24 / 0.24 / 0.12 / 0.20 / 0.20', '空间': '直接平均概率'}, section='base_fusion', category='Merge')
three = g.node('three', '折外曝光与并行表格概率补充', [five,oof,tabm], attrs={'公式': '0.675×五成员 + 0.225×折外曝光 + 0.10×TabM'}, section='base_fusion', category='Merge')
count = g.node('count_adjust', '相同预测字段曝光次数修正', [three], attrs={'分组': '33个完全相同预测字段计数 c', '公式': 'odds_new = odds × c^0.1；保留全部行'}, section='base_fusion', category='Transform')
reference = g.node('reference', '完整参考概率融合', [count,rich_gated,allocated,protected], attrs={'权重': '0.504 / 0.126 / 0.07 / 0.30'}, section='base_fusion', category='Merge')
baseprob = g.node('base_probability', '正样本加权参考概率融合', [reference,weighted], attrs={'公式': '0.75×完整参考 + 0.25×正样本加权网络'}, section='base_fusion', category='Merge')
cal = g.node('calibrated', '分组分数逻辑校准', [baseprob], attrs={'输入': '4组×5统计 + 常数 + 原Logit = 22', '分组': '用户 / 任务 / 用户广告位 / 用户应用', '输出': '两条用户日支路共同的 p_cal'}, section='calibration', category='Normalization')

# The left and right user-day branches share p_cal and meet only at p_join.
left = g.node('day_binary', '历史活跃度二元响应 LightGBM', [cal], attrs={'输入': 'G×615 = 598上下文 + 17历史', '配置': '15叶；354轮', '回写': '二元风险强度；power 0.25'}, section='day_branch', category='Tree')
left = g.node('day_pair', '用户日成对提升 LightGBM', [left], attrs={'输入': 'G×619 = 615 + 4分数统计', '配置': '7叶；48轮', '回写': '裁剪±3后广播加行Logit'}, section='day_branch', category='Tree')
left = g.node('personal', '个人上下文二元残差 LightGBM', [left], attrs={'输入': 'G×1883 = 685 + 1198个人变化', '配置': '7叶；105轮', '回写': '二元校正 power 0.75'}, section='day_branch', category='Tree')
left = g.node('p_day', '带用户身份的历史点击计数 CatBoost', [left], attrs={'输入': 'G×1552 = 1541数值 + 11类别', '配置': '深度6；752轮', '回写': '泊松校正 power 0.125；输出 p_day'}, section='day_branch', category='Tree')
opinions=[]
specs=[
 ('整体用户日二元响应','619','7叶71轮','二元 power0.5'),('整体用户日泊松计数','619','15叶47轮','泊松 power1'),
 ('整体用户日成对排序（温度0.5）','619','3叶65轮','成对 power1'),('整体用户日成对排序（温度1）','619','3叶116轮','成对 power1'),
 ('广告位子群二元响应','88','7叶132轮','二元 power0.75'),('广告位子群泊松计数','88','15叶107轮','泊松 power0.75'),
 ('应用子群二元响应','88','15叶195轮','二元 power0.5'),('应用子群泊松计数','88','15叶164轮','泊松 power0.5'),
 ('任务子群二元响应','88','7叶32轮','二元 power0.5'),('任务子群泊松计数','88','7叶29轮','泊松 power0.75'),
 ('扩展历史用户日二元响应','685','7叶79轮','二元 power0.5'),('扩展历史用户日泊松计数','685','15叶39轮','泊松 power1')]
for i,(title,width,config,operator) in enumerate(specs):
    opinions.append(g.node(f'opinion_{i+1}',title+' LightGBM',[cal], attrs={'输入维度':width, '树参数':config,'校正':operator,'输出':'相对 p_cal 的 N×1 Logit 残差'},section='twelve',category='Tree'))
right=g.node('meta', '十二残差意见逻辑元模型', opinions+[cal], attrs={'输入': '12标准化残差 + 常数 + 原Logit = 14', '处理': '残差裁剪±8；岭参数0.01'},section='merge',category='Merge')
right=g.node('catday','类别用户日点击计数残差 CatBoost',[right],attrs={'输入':'G×695 = 685数值 + 10类别','配置':'深度6；196轮；泊松power0.75'},section='merge',category='Tree')
right=g.node('news_pair','资讯语境用户日成对提升 LightGBM',[right],attrs={'输入':'G×2743 = 1883 + 860资讯及变化','配置':'15叶；40轮；温度0.5','回写':'裁剪±3后广播加行Logit'},section='merge',category='Tree')
prev=g.node('p_prev','前一日用户应用统计残差',[right],attrs={'回写':'Logit增量强度0.5；输出 p_prev'},section='merge',category='Transform')
ranked=g.node('p_ranked','成对排序神经差值迁移',[prev,rank,weighted],attrs={'公式':'p_ranked = S[L(p_prev)+L(p_rank)−L(p_weighted)]'},section='merge',category='Merge')
p=g.node('p_join','两条用户日并行路径汇合',[ranked,left,prev],attrs={'公式':'S[L(p_ranked)+0.5L(p_day)−0.5L(p_prev)]'},section='parallel_day',category='Merge')

def seq(key,title,attrs,section,category='Tree',parents=None):
    global p
    p=g.node(key,title,parents or [p],attrs=attrs,section=section,category=category)
    return p

seq('row291','全局行级点击 CatBoost',{'输入':'N×291 = 24类别+159多窗口+96日上下文+12历史','配置':'深度8；1200轮；正样本权重5','融合':'Logit 0.10'},'row_daily')
seq('prev_app','前一日用户应用统计残差',{'平滑':'alpha=3','融合':'Logit增量0.125'},'row_daily','Transform')
daily_parent=p
daily_binary=g.node('daily5051_binary','丰富用户日二元响应 CatBoost',[daily_parent],attrs={'输入':'G×5051 = 5040数值 + 11类别','配置':'深度6；1200轮；历史先验偏置','回写':'二元校正 power0.125'},section='row_daily',category='Tree')
daily_poisson=g.node('daily5051_poisson','丰富用户日泊松计数 CatBoost',[daily_parent],attrs={'输入':'G×5051','配置':'深度8；1200轮；历史先验偏置','回写':'泊松校正 power0.125'},section='row_daily',category='Tree')
seq('daily_mix','丰富日级双目标 Logit 融合',{'公式':'S[0.75L(p_binary)+0.25L(p_poisson)]'},'row_daily','Merge',[daily_binary,daily_poisson])
seq('row472','增强行级点击 CatBoost',{'输入':'N×472 = 291+166数值+15点击历史token','配置':'39类别；深度8；1200轮；正样本权重5','融合':'Logit 0.25'},'news_response')
seq('news_quality_response','完整资讯质量用户日二元响应 LightGBM',{'输入':'G×5389 = 5040+11类别+1先验mass+337质量','配置':'15叶；900轮；贝叶斯先验偏置','回写':'二元power0.125；替换旧响应位置'},'news_response')
seq('group_temperature','用户应用温度与风险总量保持',{'温度':'0.75；调整组内分数偏离','风险':'h=−log(1−p)；保持组总h'},'news_response','Normalization')
quality_parent=p
rich6=g.node('rich744_d6','丰富行级点击 CatBoost（深度6）',[quality_parent],attrs={'输入':'N×744 = 472+192精选日级+80资讯','配置':'深度6；2400轮；正样本权重5'},section='quality_transfer',category='Tree')
rich8=g.node('rich744_d8','丰富行级点击 CatBoost（深度8）',[quality_parent],attrs={'输入':'N×744','配置':'深度8；2400轮；正样本权重5'},section='quality_transfer',category='Tree')
quality_tree=g.node('quality1017','近期加权完整资讯质量行级 CatBoost',[quality_parent],attrs={'输入':'N×1017 = 744−64旧质量+337质量','配置':'深度8；3600轮；时间半衰期2天'},section='quality_transfer',category='Tree')
anchor=g.node('quality_anchor','丰富行级锚点对齐融合',[quality_parent,rich8],attrs={'融合':'标准差对齐 Logit 0.20；输出 p_anchor'},section='quality_transfer',category='Merge')
group=g.node('quality_group','用户应用组间与组内校正',[anchor,rich6,rich8],attrs={'参考':'两棵丰富行树的概率均值','迁移':'组间−0.10；组内0.25；得到 p_group'},section='quality_transfer',category='Transform')
quality=g.node('quality_delta','完整质量行树相对锚点融合',[anchor,quality_tree],attrs={'融合':'0.025；相对同一个 p_anchor 得 p_quality'},section='quality_transfer',category='Merge')
seq('quality_join','资讯质量差值迁移',{'公式':'S[L(p_group)+1.5L(p_quality)−1.5L(p_anchor)]'},'quality_transfer','Merge',[group,quality,anchor])
seq('quality_day_poisson','完整资讯质量用户日泊松 CatBoost',{'输入':'G×5389','配置':'深度5；1600轮；power0.125','迁移':'按任务：组间1.5；组内1'},'compact_dense')
dense=g.node('dense694','稠密上下文行级 LightGBM 增量',[anchor],attrs={'输入':'N×694；从744维固定选列','配置':'31叶；1400轮；分数对齐融合0.10','迁移':'相对质量锚点；按用户组间0、组内1.5'},section='compact_dense',category='Tree')
compact=g.node('compact540','紧凑质量用户日泊松 CatBoost 增量',[p],attrs={'输入':'G×540 = 192普通+337质量+11类别','配置':'深度7；3200轮；power0.125','迁移':'相对质量日结果；按任务组间1.5、组内1'},section='compact_dense',category='Tree')
seq('dense_compact_join','稠密行级与紧凑日级并行增量合成',{'回写':'当前Logit + 两种按各自分组迁移的增量'},'compact_dense','Merge',[p,dense,compact])
seq('lambdarank','用户内 LambdaRank 轻量梯度排序',{'输入':'N×694','配置':'31叶；600轮；LambdaRank','融合':'replace_within 0.20'},'compact_dense')
seq('within_pair','用户内 PairLogit 成对排序 CatBoost',{'输入':'N×1017','配置':'深度6；1600轮；PairLogit','融合':'add_within 0.05'},'ranking')
seq('binomial','近期加权用户日二项点击率 CatBoost',{'输入':'G×540','配置':'深度7；1600轮；半衰期2天','回写':'曝光加权二项目标；residual_add 0.05'},'ranking')
seq('global_pair','跨用户 PairLogit 成对排序 CatBoost',{'输入':'N×1017','配置':'深度8；1600轮；PairLogit','融合':'add_global 0.05'},'ranking')
seq('cold_task','冷任务专用点击 CatBoost',{'输入':'N×1017；指定候选身份列置0','配置':'深度8；1600轮','条件融合':'仅历史未见任务；replace_global 0.20'},'ranking')
seq('audience1193','任务受众点击率 CatBoost',{'输入':'T×1193；含6类别；337质量×3聚合','配置':'深度6；200轮','融合':'任务分数 replace_task 0.20；广播回N行'},'audience')
seq('compact423','紧凑全样本行级点击 CatBoost',{'输入':'N×423 = 39类别+384非类别；实际85受众列','配置':'深度6；2400轮','融合':'replace_global 0.05'},'audience')
seq('history495','无序历史增强点击 CatBoost',{'输入':'N×495 = 423+72快照','配置':'深度8；600轮','条件融合':'changed_click；分位映射后权重0.50'},'history')
seq('transition','历史任务转变平滑后验',{'输入':'日期、用户、任务、任务点击历史共4字段','证据':'0～4档；近期收缩5；强度5','回写':'证据≥1；仅正向Logit下界增量0.25'},'history','Transform')
seq('cold_family','冷任务类别提升四模型族',{'输入':'每棵N×1017','配置':'深度6/8 × 600/1600轮','融合':'每棵先分位映射再平均；changed_click 0.50'},'events')
seq('app_posterior','应用历史变化事件平滑后验',{'输入':'8字段；按用户日+候选应用+原历史集合分H组','事件':'1～4档；应用通道；平滑20','回写':'事件至少一次点击→条件边际；仅正向增量0.25'},'events','Transform')
context_parent=p
noncold=g.node('noncold_context','近期加权上下文 CatBoost（非冷任务）',[context_parent],attrs={'输入':'N×395 = 24类别+256数值+43曝光+72快照','配置':'深度8；2400轮；半衰期2天','条件融合':'noncold；replace_global 0.05'},section='context395',category='Tree')
cold=g.node('cold_context','常规上下文 CatBoost（冷任务）',[context_parent],attrs={'输入':'N×395；同一个父概率','配置':'深度6；2400轮','条件融合':'cold；rank_global 0.20'},section='context395',category='Tree')
seq('context_join','冷任务条件选择互斥两路',{'选择':'冷任务取常规分支；其余取近期分支','连接':'两树相对同一父结果计算'},'context395','Control',[noncold,cold])
seq('context_weighted','正样本加权上下文 CatBoost',{'输入':'N×395','配置':'深度6；3200轮；正样本权重25','融合':'无历史广告点击 add_within 0.10；其余0.025'},'context395')

# These are independent feature inputs, not successive compression stages.
teacher=g.node('frozen_teacher','冻结神经字典与候选历史关系', [aligned], ['N',138], attrs={'当前向量':'6×8=48；历史均值6×8=48','交互':'用户候选逐元素乘积4×8=32','点积':'6历史候选 + 4用户候选','总宽':'48+48+32+6+4=138；教师检查点字典冻结'}, section='frozen138')
context_features=g.node('context_features395','冻结选择器组装395维上下文底座',[aligned],['N',395],attrs={'输入':'24类别 + 256普通数值 + 43曝光 + 72快照','选择器':'256项从660候选数值中固定选择'},section='context395',category='Transform')
features533=g.node('features533','独立组装完整冻结表示树输入',[teacher,context_features],['N',533],attrs={'输入':'395上下文 + 138完整表示 = 533'},section='three_trees',category='Transform')
features489=g.node('features489','独立组装候选历史语义树输入',[teacher,context_features],['N',489],attrs={'输入':'395上下文 + 94语义 = 489','94维':'138−8用户向量−32乘积−4用户点积'},section='three_trees',category='Transform')
features401=g.node('features401','独立组装历史点积关系树输入',[teacher,context_features],['N',401],attrs={'输入':'395上下文 + 6历史候选点积 = 401'},section='three_trees',category='Transform')
seq('full_embedding','完整冻结神经表示 CatBoost',{'输入':'N×533','配置':'深度6；2400轮；正样本权重5','条件融合':'有历史广告点击用户；分位融合0.05'},'embedding_family',parents=[p,features533])
seq('compact_family','紧凑全样本类别提升四模型族',{'输入':'每棵N×423','配置':'深度6/8 × 1600/2400轮','条件融合':'每棵先分位映射再平均；仅cold 0.10'},'embedding_family')
seq('semantic','候选历史语义关系 CatBoost',{'输入':'N×489','配置':'深度6；2400轮；正样本权重5','条件融合':'仅无历史广告点击用户；add_within 0.10'},'final_cohorts',parents=[p,features489])
seq('dot_final','历史点积关系增强 CatBoost',{'输入':'N×401','配置':'深度6；2400轮；正样本权重5','条件':'预测日前任务曝光≤100','融合':'对齐树分数m̃；S[0.9L(parent)+0.1m̃]；其余原值'},'final_cohorts',parents=[p,features401])
seq('restore','原始行序恢复与提交输出',{'行数':'404,182；重复曝光保留','列':'id、pctr；id从0到404181','精度':'概率17位有效数字'},'training','Data')
g.output(p,'最终 pctr｜N×1')

DETAIL_SHAPES = {
 'embedding': [('B',27),('B',12,5),('B',39,8),('B',12,8),('B',312),('B','478/637/812')],
 'base_models': [('B','478/637'),('B','478/637'),('B',64),('B',1),('N',1),('N',1)],
 'gated': [('B','39×8+166/500'),('B',39),('B','478/812'),('B','478/812'),('B',64),('B',1)],
 'protected_day': [('B',1),('G',606),('G',32),('B',1),('G',1),('B',1),('N',1)],
 'tabular_parallel': [('B',535),('B',312),('B',2140),('B',4,2452),('B',4),('B',1)],
 'allocation_encoder': [('B',344),('B',64),('G',4,64),('G',64),('B',64),('G','193 / 行256')],
 'allocation_output': [('B','行64 / 日193'),('B',1),('G',1),('N',1),('N',1),('N',1)],
 'frozen138': [('词表',8),('N',48),('N',48),('N',32),('N',6),('N',4),('N',138)],
 'three_trees': [('N',395),('N',533),('N',489),('N',401),('N',1)],
 'base_fusion': [('N',1)]*5,
 'parallel_day': [('N',1),('N',1),('N',12)]+[('N',1)]*9,
 'day_branch': [('N',1)]*5,
 'twelve': [('N',1)]+[('N',2)]*6+[('N',12)],
 'merge': [('N',1)]*5,
 'row_daily': [('N',1),('N',1),('G',5051),('N',1),('N',1),('N',1)],
 'news_response': [('N',1),('G',5389),('N',1),('N',1),('N',1)],
 'quality_transfer': [('N','744 / 1017'),('N',2),('N',1),('N',1),('N',2),('N',1)],
 'compact_dense': [('N',1)]*5,
 'ranking': [('N',1)]*4,
 'audience': [('T',1193),('T',1),('N',1),('N',423),('N',1)],
 'history': [('N',72),('N',1),('N',1),('N',1),('N',1)],
 'events': [('N',4),('N',1),('H',1),('H',1),('N',1)],
 'context395': [('N',395)]+[('N',1)]*5,
 'embedding_family': [('N',1),('N',1),('N',4),('N',1),('N',1)],
 'final_cohorts': [('N',1)]*5,
}

def detail_graph(s):
    detail = Graph(s['title'], '\n\n'.join(s['paragraphs']))
    shapes = DETAIL_SHAPES.get(s['id'])
    # Original overview/feature-inventory diagrams do not define every edge as a
    # numeric tensor. Preserve the relation without inventing a tensor width.
    fallback = None
    if shapes:
        assert len(shapes) == len(s['nodes']), s['id']
    for i,n in enumerate(s['nodes']):
        detail.value(f'{s["id"]}_{i}',list(shapes[i]) if shapes else fallback)
    incoming=defaultdict(list)
    outgoing=defaultdict(list)
    for a,b in s['edges']:
        incoming[b].append(a);outgoing[a].append(b)
    for i,n in enumerate(s['nodes']):
        attrs={}
        for j,line in enumerate(n['body'].splitlines()):
            attrs[f'结构{j+1:02d}']=line
        attrs['说明']='\n\n'.join(s['paragraphs'])
        # Long prose belongs in the properties panel, not across the canvas.
        attrs.pop('说明')
        category='Tree' if any(w in n['title'] for w in ['树','CatBoost','LightGBM']) else 'Activation' if any(w in n['title'] for w in ['Sigmoid','概率','Logit']) else 'Layer'
        detail.node(str(i),n['title'],[f'{s["id"]}_{a}' for a in incoming[i]],
                    shapes[i] if shapes else fallback,attrs=attrs,category=category,out=f'{s["id"]}_{i}',description='\n\n'.join(s['paragraphs']))
    # Source and sink nodes stay as documented boxes. Graph I/O is only exposed
    # when its shape is explicit, to avoid adding false sequential connections.
    return detail.data

model = dict(signature='netron:integrated-ctr-structure',format='Integrated CTR Structure',producer='Verified Local Architecture Report',
             description='完整计算依赖图及35份模块详图。此文件是结构说明，不包含训练权重。',
             metadata=[dict(name='预测日',value='2019-11-17'),dict(name='规模',value='N=404182；G=8556；T=34510'),
                       dict(name='成绩来源',value='17日AUC 0.81635为用户平台反馈；15/16日反复用于开发'),
                       dict(name='训练截止',value='成员权重截止日不同；历史统计可更新至16日'),
                       dict(name='表示输入',value='533 / 489 / 401 三份独立输入；并非连续降维')],
             graphs=[g.data]+[detail_graph(s) for s in DATA['sections']])
target=ROOT/'integrated_model.netron'
target.write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf-8')

# Preserve the previous reading view; make the current URL the Netron canvas.
old=ROOT/'说明阅读.html'
if not old.exists():
    shutil.copy2(ROOT/'index.html',old)
shutil.copy2(ROOT/'netron_shell.html',ROOT/'index.html')
manifest=dict(viewer='Netron',graphs=len(model['graphs']),main_nodes=len(g.data['nodes']),
              detail_nodes=sum(len(s['nodes']) for s in DATA['sections']),
              detail_edges=sum(len(s['edges']) for s in DATA['sections']),
              graph_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              source='model_data.json / verified integrated architecture report',
              model_code_executed=False,weights_read=False,labels_read=False,
              patches=['named graphs and drill-down references','full Chinese names and attribute values','attributes visible and wheel zoom','attribute search','local telemetry/age-gate disabled','CSP-compatible sidebar close button'])
(ROOT/'netron_build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
