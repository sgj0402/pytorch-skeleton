import torch.nn as nn
import torch.optim as optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver


from ignite.contrib.handlers import ProgressBar

from load_util import *
import global_config
import global_aim


#TODO implement Saving function, validation function
# TODO output transform

def track_training_loss(engine):

    run = global_aim.get_run()
    run.track(engine.state.output,
              name='loss',
              step=engine.state.iteration,
              epoch=engine.state.epoch,
              context={'mode': 'train', 'phase': 'train'})


def train():

    config = global_config.get_config()

    device = load_device()

    model = load_model(device)
    loss_fn = load_loss_fn()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
    


    train_loader = load_train_dataloader()
    test_loader = load_test_dataloader()


    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)
    
    if config.setting.train.is_resume:
        checkpoint_path = config.setting.train.checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True) 
        to_load = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
        
        print(f"Resuming from {checkpoint_path}")


    test_metrics = load_test_metrics()
    evaluator = create_supervised_evaluator(model, metrics=test_metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
        )


    to_save = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    checkpoint_dir = "checkpoints/"

    score_name = "accuracy"
    score_fn = lambda engine: engine.state.metrics[score_name]

    checkpoint = Checkpoint(
        to_save,
        save_handler=DiskSaver(checkpoint_dir, require_empty=False),
        score_name=score_name,
        score_function=score_fn,
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer)
    )

    
    evaluator.add_event_handler(Events.COMPLETED, checkpoint)




    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), track_training_loss)
    

    pbar = ProgressBar(persist=True, bar_format='')
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    trainer.run(train_loader, max_epochs=config.setting.train.epochs)



def track_validation_metrics(trainer, test_evaluator, test_loader):
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics
    
    run = global_aim.get_run()
    run.track(metrics['loss'],
              name='loss',
              step=trainer.state.iteration,
              epoch=trainer.state.epoch,
              context={'mode': 'train', 'phase': 'validation'})
    
    run.track(metrics['accuracy'],
              name='accuracy',
              step=trainer.state.iteration,
              epoch=trainer.state.epoch,
              context={'mode': 'train', 'phase': 'validation'})



def test(model, device):

    test_loader = load_test_dataloader()

    test_metrics = load_test_metrics()

    test_evaluator = create_supervised_evaluator(model, metrics=test_metrics, device=device)

    test_evaluator.run(test_loader)

    
def model_run():

    config = global_config.get_config()
    
    if config.setting.mode == 'train':
        train()
        
    elif config.setting.mode == 'test':
        NotImplementedError()
        
    else:
        raise ValueError(f"Invalid mode: {config.setting.mode}")