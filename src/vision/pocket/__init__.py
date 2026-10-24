"""Classical table and pocket localization; independent of the ball detector."""
# 集中导出台面/袋口配置、结果和检测器；导入包不会执行检测或启动视频。

from .models import PocketConfig, TableResult
from .table_detector import TableDetector
from .pocket_detector import PocketDetector
from .models import PocketResult, PocketDetectionResult
from .table_geometry import TableGeometry
