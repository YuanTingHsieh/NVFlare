class Model:
    def __init__(self):
        self.weights = {"w_x": 3, "w_y": 1}

    def load_weights(self, weights):
        self.weights = weights

    def predict(self, data):
        return self.weights["w_x"] * data[0] + self.weights["w_y"] * data[1]

    def update(self, sign, lr):
        self.weights["w_x"] += sign * lr
        self.weights["w_x"] += sign * lr
