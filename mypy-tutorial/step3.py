metrics = {"accuracy": 0.94, "f1": 0.88}
batch_sizes = [16, 32, 64]
image_size = (128, 128)

reveal_type(metrics)
reveal_type(batch_sizes)

metrics["loss"] = 0.3
batch_sizes.append(128)

history: list[float] = []
