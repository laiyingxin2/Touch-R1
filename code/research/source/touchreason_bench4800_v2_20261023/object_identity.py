"""Conservative, inspectable alias/contact-part ontology; not physical ID truth.

Explicit rules join spelling variants and parts of a potentially identical
object, even when that may overmerge separate physical instances. No model
prediction or held-out score is used. Latin/digit material style IDs survive.
"""
import re
import unicodedata
from opencc import OpenCC
CC=OpenCC('t2s')
RULES=[
 ('充电宝',r'充电宝'),
 ('入耳式耳机',r'入耳式耳机|耳机按|耳机钮'),
 ('头戴式耳机',r'头戴式耳'),
 ('公交站结构',r'公交站(座椅|玻璃|金属框)'),
 ('公园长椅',r'公园(木质?长椅|塑料长椅|金属长椅)|公木质|长椅金属|长金属支架|^椅边缘$'),
 ('公园水泥步道',r'公园水(泥)?步道'),
 ('公园石板路面',r'公园石(材|板)路面'),
 ('公园鹅卵石路面',r'公园鹅卵石'),
 ('公园透水砖路面',r'^公砖路面$|公园透水砖'),
 ('路缘石',r'公园路缘石|公缘石'),
 ('共享单车',r'共享单车'),
 ('内六角扳手',r'内六角扳手|l型六角扳手|型六角扳手'),
 ('凹槽方形器件',r'^凹槽(方形器件)?$'),
 ('卷尺',r'卷尺'),
 ('吸汗腕带',r'吸汗[脘腕]带'),
 ('吸管',r'吸管'),
 ('圆孔电源连接器',r'dc圆孔插头|圆孔插头|dc电源插口|^电源插口$'),
 ('电源适配器',r'电源适配器'),
 ('塑料模型',r'塑料模型'),
 ('户外垃圾桶',r'户外?垃圾(桶)?|^户垃圾'),
 ('手机',r'^手机(?!支架)'),
 ('数据线',r'数据线|typec|type-c|^接头(塑料)?$'),
 ('积木与打印测试件',r'积木|方块|方塊|方向屋|^英文字母$|^字母$'),
 ('筷子',r'筷子|双筷'),
 ('易拉罐',r'易拉罐|易拉瓶底'),
 ('枯树枝',r'干枯树枝|枯枝'),
 ('树干',r'光滑树干|树干光滑|树皮粗糙|树粗糙'),
 ('橡皮擦',r'^橡皮(擦)?$'),
 ('橡胶减速带',r'橡胶减(速)?带'),
 ('水杯',r'^水杯|透明水杯|马克杯握把'),
 ('汽车',r'^汽车(?![挡档]轮器)'),
 ('汽车挡轮器',r'[挡档]轮器'),
 ('土块',r'^沙土|^泥土|干燥土块'),
 ('黑板擦',r'黑板擦'),
 ('玻璃墨水瓶',r'玻璃墨水瓶'),
 ('电梯按钮',r'电梯按[钮键]'),
 ('钥匙',r'钥匙|皮纹钥匙[柄炳]'),
 ('紫花油瓶',r'紫花油'),
 ('纸扇',r'纸扇'),
 ('线圈笔记本',r'笔记本线圈|线圈笔记[本办]线圈'),
 ('背包',r'背包'),
 ('自行车',r'^自行车'),
 ('草地',r'^草(地)?$'),
 ('花茎',r'^茎$|^花茎$'),
 ('蛋糕刀叉',r'蛋糕刀'),
 ('螺丝刀',r'螺丝刀[摩磨]砂'),
 ('透明亚克力板',r'透明亚克力[板版]'),
 ('遥控器',r'遥控器'),
 ('马克笔',r'马[克可]笔'),
 ('鼠标',r'鼠标'),
 ('防撞桶',r'防撞[桶筒]'),
 ('指示牌',r'指示牌金属?杆|提示牌|^金属杆$'),
 ('笔',r'^笔[杆秆帽]|^笔握持部'),
 ('树叶',r'^绿色(树)?叶|^叶片$'),
]

def canonical_object(row):
 text=CC.convert(unicodedata.normalize('NFKC',row['object_category'])).lower()
 text=re.sub(r'[\W_]+','',text)
 # Remove only contact-position suffixes; preserve lexical '中'/'上' elsewhere.
 previous=None
 while previous!=text:
  previous=text
  text=re.sub(r'(侧面|正面|背面|底面|顶面|上面|下面|表面|内面|外面|反面|底部|顶部|左上|左下|右上|右下|左侧|右侧|中心|中间|上部|下部|前部|后部|左|右|上|下|中)\d*$','',text)
  text=re.sub(r'\d+$','',text)
 for label,pattern in RULES:
  if re.search(pattern,text):return label
 return text or 'unknown_category'
