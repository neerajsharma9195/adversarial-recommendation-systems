import surprise
from typing import NewType, List, Tuple

Trainset = NewType('Trainset', surprise.Trainset)
Testset = NewType('Testset', List[Tuple[int, int, float]])
