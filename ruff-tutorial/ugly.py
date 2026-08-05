def train(lr=0.01, epochs=10):
    params = {"lr": lr, "epochs": epochs}
    return params


result = train(lr=0.001, epochs=100) or train(lr=0.01, epochs=50)
