"""Validated configuration independent of widgets and settings storage."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ResourceSettings:
    workers: int = 2
    memory_mb: int = 512

    @classmethod
    def from_values(cls, workers, memory_mb):
        parsed = []
        for name, value, minimum in (("workers",workers,1),("memory_mb",memory_mb,64)):
            number = float(value)
            if isinstance(value,bool) or not math.isfinite(number) or not number.is_integer() or number < minimum:
                raise ValueError(f"{name}: informe um inteiro maior ou igual a {minimum}.")
            parsed.append(int(number))
        return cls(*parsed)
