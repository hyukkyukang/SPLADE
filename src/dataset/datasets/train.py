from dataclasses import dataclass


@dataclass
class TrainSample:
    query: str
    docs: list[str]
    pos_count: int
    teacher_scores: list[float]
