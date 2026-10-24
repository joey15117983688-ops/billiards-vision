"""Typed results; confidence values are heuristic quality scores, not probabilities."""
# 袋口检测的数据结构和配置；坐标序列化为原图像素，normalized 字段另行标识。
# observed 代表局部图像支持，manual 代表人工确认，estimated 为几何推算，missing 为不可定位。

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PocketConfig:
    hsv_ranges: tuple = (((30, 35, 25), (95, 255, 255)), ((96, 150, 80), (125, 255, 255)))
    max_side: int = 1200
    min_table_area: float = .12
    min_aspect: float = 1.15
    max_aspect: float = 4.0
    canonical_width: int = 1000
    canonical_height: int = 500
    padding: int = 80
    search_radius_ratio: float = .12
    long_axis: str = 'auto'
    observed_quality_threshold: float = .60

    # 集中限制分辨率、长宽比、搜索半径、padding 和 HSV 范围，防止无效参数传入几何算法。
    def __post_init__(self):
        if not 320 <= self.max_side <= 2400:
            raise ValueError('max_side must be within 320..2400')
        if not (160 <= self.canonical_height <= 1200 and self.canonical_height <= self.canonical_width <= 2400):
            raise ValueError('Invalid canonical image size')
        if not (.01 <= self.min_table_area <= .8 and 1 <= self.min_aspect < self.max_aspect <= 6):
            raise ValueError('Invalid table area/aspect constraints')
        if not (.05 <= self.search_radius_ratio <= .2 and
                self.canonical_height*self.search_radius_ratio <= self.padding <= .4*self.canonical_height):
            raise ValueError('Padding must cover the configured pocket search radius')
        if not .4 <= self.observed_quality_threshold <= .95:
            raise ValueError('Invalid observed quality threshold')
        if self.long_axis not in ('auto','edge_01','edge_12'):
            raise ValueError('Unsupported long_axis configuration')
        if not self.hsv_ranges:
            raise ValueError('At least one cloth HSV range is required')
        for low, high in self.hsv_ranges:
            if len(low)!=3 or len(high)!=3 or any(not 0<=a<=b<=limit for a,b,limit in zip(low,high,(179,255,255))):
                raise ValueError('Invalid OpenCV HSV color range')


@dataclass
class TableResult:
    detected: bool = False
    complete: bool = False
    corners: list = field(default_factory=list)
    playable_polygon: list = field(default_factory=list)
    boundary_lines: list = field(default_factory=list)
    confidence: float = 0.0
    quality: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    cloth_mask: object = field(default=None, repr=False)
    scope_polygon: list = field(default_factory=list)

    # 只导出几何/质量/原因字段；不把用于调试的大图掩模写入公共结果。
    def to_dict(self):
        return {key: getattr(self, key) for key in (
            'detected', 'complete', 'corners', 'playable_polygon', 'scope_polygon', 'boundary_lines',
            'confidence', 'quality', 'reasons')}


# 把可选二维点序列转成 x/y 字典，保留 None 以表达未定位。
def _point(value):
    return None if value is None else {'x': float(value[0]), 'y': float(value[1])}


@dataclass
class PocketResult:
    id: str
    status: str = 'missing'
    candidate_center: object = None
    normalized_candidate: object = None
    center: object = None
    normalized_center: object = None
    entry_polygon: list = field(default_factory=list)
    confidence: float = 0.0
    quality: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    # 序列化中心、候选、入口及其状态和理由，估算结果不隐去来源标记。
    def to_dict(self):
        return {'id':self.id,'status':self.status,'candidate_center':_point(self.candidate_center),
                'normalized_candidate':_point(self.normalized_candidate), 'center':_point(self.center),
                'normalized_center':_point(self.normalized_center),'entry_polygon':self.entry_polygon,
                'confidence':float(self.confidence),'quality':self.quality,'reasons':self.reasons}


@dataclass
class PocketDetectionResult:
    image_width: int
    image_height: int
    table: TableResult
    pockets: list
    valid: bool = False
    geometry: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    elapsed_ms: float = 0.0
    table_mask: object = field(default=None, repr=False)
    pocket_mask: object = field(default=None, repr=False)
    warped: object = field(default=None, repr=False)
    source: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)

    # 打包同帧尺寸、台面、袋口、坐标矩阵与诊断信息，图像调试产物单独保存。
    def to_dict(self):
        return {'schema_version':1,'table_detected':self.table.detected,'coordinate_system':'image_pixels',
                'normalized_reference':'playable inner-cushion plane; physical pocket centers may lie slightly outside [0,1]',
                'image_width':self.image_width,'image_height':self.image_height,'table':self.table.to_dict(),
                'geometry':self.geometry,'pockets':[p.to_dict() for p in self.pockets], 'valid':self.valid,
                'failed_pockets':[p.id for p in self.pockets if p.status!='observed'],
                'confidence_kind':'heuristic_quality_score_not_calibrated_probability',
                'mask_semantics':{'table_mask':'filled visible inner-cushion polygon; partial when table.complete is false',
                                  'pocket_regions':'observed and estimated entry regions; consult per-pocket status before use'},
                'reasons':self.reasons,'elapsed_ms':self.elapsed_ms,'source':self.source,'artifacts':self.artifacts}
