def load_checkpoint(path: str) -> dict[str, float] | None:
    if path.endswith(".ckpt"):
        return {"loss": 0.42}
    return None


ckpt = load_checkpoint("model.pt")
reveal_type(ckpt)
if ckpt is not None:
    reveal_type(ckpt)
    print(ckpt["loss"])
