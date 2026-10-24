"""Named venue/table calibrations with an explicitly selected startup default."""
# 命名球房/球桌配置索引；路径由随机 ID 构造，显示名称不直接充当文件路径。
# 配置创建或默认值保存失败时回滚内存状态，避免 UI 状态与磁盘不一致。

import json
from copy import deepcopy
import re
from pathlib import Path
from uuid import uuid4
from src.calibration_state import write_json


class CalibrationPresets:
    # 读取并校验版本、ID、名称和默认引用；损坏索引记录 error，不覆盖原文件。
    def __init__(self,root):
        self.root=Path(root);self.index=self.root/'index.json';self.error=None
        self.data={'version':1,'profiles':[],'default':None}
        try:
            if self.index.exists():
                data=json.loads(self.index.read_text())
                if not isinstance(data,dict) or data.get('version')!=1 or not isinstance(data.get('profiles'),list):raise ValueError('配置目录格式错误')
                for profile in data['profiles']:
                    if not re.fullmatch('[a-f0-9]{32}',profile['id']) or not isinstance(profile['name'],str):raise ValueError('配置项目格式错误')
                if data.get('default') not in [None]+[p['id'] for p in data['profiles']]:raise ValueError('默认配置不存在')
                self.data=data
        except (OSError,ValueError,KeyError,TypeError) as exc:self.error=str(exc)

    # 只有索引可读时才保存；保留损坏原文件供人工恢复。
    def save(self):
        if self.error:raise ValueError('配置目录无法读取，已保留原文件：'+self.error)
        write_json(self.index,self.data)

    # 校验名称长度及大小写去重，生成随机 ID；首个配置自动成为默认。
    def create(self,name):
        name=name.strip()
        if not name or len(name)>80:raise ValueError('请输入 1–80 字的球房／球桌名称')
        if any(p['name'].casefold()==name.casefold() for p in self.data['profiles']):raise ValueError('配置名称已存在，请从列表选择')
        previous=deepcopy(self.data)
        identity=uuid4().hex
        self.data['profiles'].append({'id':identity,'name':name})
        if len(self.data['profiles'])==1:self.data['default']=identity
        try:self.save()
        except (OSError,ValueError):
            self.data=previous
            raise
        return identity

    # 仅接受现有 ID 或空默认；写入失败恢复旧默认值。
    def set_default(self,identity):
        if identity not in [None]+[p['id'] for p in self.data['profiles']]:raise ValueError('配置不存在')
        previous=self.data['default']
        self.data['default']=identity
        try:self.save()
        except (OSError,ValueError):
            self.data['default']=previous
            raise

    # 仅为已登记配置返回 manual.json 路径，拒绝未授权/未知标识。
    def path(self,identity):
        if identity not in [p['id'] for p in self.data['profiles']]:raise ValueError('配置不存在')
        return self.root/identity/'manual.json'
