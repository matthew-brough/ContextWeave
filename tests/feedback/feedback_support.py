import random

from contextweave.feedback.fit import Sample
from contextweave.feedback.served import Features


def synthetic_samples(count: int = 300, seed: int = 7) -> list[Sample]:
    """Items whose use follows lexical_match only; answers_query never varies."""
    rng = random.Random(seed)
    samples: list[Sample] = []
    for _ in range(count):
        lexical = rng.random()
        features: Features = (
            0.0,
            rng.random(),
            1.0,
            lexical,
            rng.random(),
            rng.random(),
            rng.random(),
        )
        samples.append(Sample(features, lexical > 0.6))
    return samples
