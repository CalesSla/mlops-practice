from dataclasses import dataclass, field


@dataclass
class TrainRun:
    model: str
    epochs: int
    metrics: dict[str, float] = field(default_factory=dict)

    def best(self) -> float | None:
        if not self.metrics:
            return None
        return max(self.metrics.values())


my_train_run = TrainRun(
    model="resnet", epochs=10, metrics={"accuracy": 0.85, "loss": 0.42}
)

print(str(my_train_run.best()).upper())
