import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from tqdm import tqdm

from util import load_device

from load_util import *



from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

def train():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train = True, download = True, transform = transform)
    test_dataset = datasets.MNIST('./data', train = False, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 1000, shuffle = False)






    device = load_device()
    model = load_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)




    ##########
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)

    test_metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(loss_fn)
    }

    train_evaluator = create_supervised_evaluator(model, metrics=test_metrics, device=device)
    test_evaluator = create_supervised_evaluator(model, metrics=test_metrics, device=device)

    log_interval = 100

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        test_evaluator.run(test_loader)
        metrics = test_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


    def score_function(engine):
        return engine.state.metrics["accuracy"]






    trainer.run(train_loader, max_epochs=2)

