"""Public inputs use one table plane; distances are fractions of its short side."""
from dataclasses import dataclass, field, asdict
from enum import Enum

BALLS=('black_ball','white_ball')

class State(str,Enum):
    TRACKING='TRACKING'
    APPROACHING_POCKET='APPROACHING_POCKET'
    POT_CANDIDATE='POT_CANDIDATE'
    POTTED='POTTED'
    NOT_POTTED='NOT_POTTED'
    UNKNOWN='UNKNOWN'

@dataclass(frozen=True)
class EventConfig:
    approach_radius: float=.16
    fallback_entry_radius: float=.035
    min_confidence: float=.4
    min_absence_seconds: float=1.0
    stable_seconds: float=.9
    max_observation_gap: float=1.5
    candidate_timeout: float=6.0
    motion_speed: float=.035
    min_track_seconds: float=.12

@dataclass(frozen=True)
class BallObservation:
    ball: str
    center: tuple
    radius: float=.015
    confidence: float=1.
    track_id: object=None

@dataclass
class BallState:
    state: State=State.UNKNOWN
    history: list=field(default_factory=list)
    pocket: object=None
    last_seen: object=None
    missing_since: object=None
    evidence: list=field(default_factory=list)
    reason: str='not_seen_before_shot'
    occlusion_seen: bool=False
    ambiguous: bool=False
    entry_seen: bool=False
    direction_seen: bool=False
    clear_since: object=None

@dataclass(frozen=True)
class PocketEvent:
    ball: str
    event: str
    pocket: object
    timestamp: object
    evidence: list
    reason: str
    confidence_status: str
    signal: str

    def to_dict(self):return asdict(self)
