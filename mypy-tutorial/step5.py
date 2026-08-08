def get_lr(config: dict[str, float | list[float]]) -> None:
    lr = config["lr"]
    if isinstance(lr, list):
        lr = lr[0]
    print(lr / 2)
