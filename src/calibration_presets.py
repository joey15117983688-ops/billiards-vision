"""Named venue/table calibrations with an explicitly selected startup default."""
import json
from copy import deepcopy
import re
from pathlib import Path
from uuid import uuid4
from src.calibration_state import write_json


class CalibrationPresets:
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

    def save(self):
        if self.error:raise ValueError('配置目录无法读取，已保留原文件：'+self.error)
        write_json(self.index,self.data)

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

    def set_default(self,identity):
        if identity not in [None]+[p['id'] for p in self.data['profiles']]:raise ValueError('配置不存在')
        previous=self.data['default']
        self.data['default']=identity
        try:self.save()
        except (OSError,ValueError):
            self.data['default']=previous
            raise

    def path(self,identity):
        if identity not in [p['id'] for p in self.data['profiles']]:raise ValueError('配置不存在')
        return self.root/identity/'manual.json'
