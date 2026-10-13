from typing import TypedDict


class SavingsBucketTotals(TypedDict):
    baseline: int
    served: int
    saved: int
    calls: int
