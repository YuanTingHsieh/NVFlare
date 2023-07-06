from dataset import dataset, eval_dataset
from model import Model

# global weights
init_weights = {"w_x": 2, "w_y": 2}
pretrained_weights = init_weights


def train(total_epochs=5, lr=0.05):
    print("    Local: start train")
    print(f"    Local:    total_epochs: {total_epochs}")
    print(f"    Local:    lr: {lr}")
    print(f"    Local:    init weights: {init_weights}")

    model = Model()
    model.load_weights(init_weights)

    local_training_steps = 0
    for i in range(total_epochs):
        epoch_loss = 0
        for data in dataset:
            coord = data["coord"]
            label = data["label"]
            prediction = model.predict(coord)
            loss = prediction - label
            if loss > 0:
                model.update(-1, lr)
            else:
                model.update(1, lr)
            epoch_loss += loss
            local_training_steps += 1
        print(f"    Local:    epoch loss is {epoch_loss}")
    print(f"    Local:    trained weights: {model.weights}")
    print("    Local: end train")
    return model.weights


def evaluate():
    print("    Local: start evaluate")
    print(f"    Local:    pretrained weights: {pretrained_weights}")
    model = Model()
    model.load_weights(pretrained_weights)

    total_loss = 0
    for data in eval_dataset:
        coord = data["coord"]
        label = data["label"]
        prediction = model.predict(coord)
        loss = prediction - label
        total_loss += loss
        print(f"    Local:    total_loss is {total_loss}")
    print("    Local: end evaluate")


if __name__ == "__main__":
    trained_weights = train()
