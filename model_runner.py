import os

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
    # Load config
    config = global_config.get_config()
    train_config = config.setting.train

    # Load model, optimizer, loss_fn, dataloaders
    device = load_device()
    model = load_model(device)
    loss_fn = load_loss_fn()
    optimizer = load_optimizer(model)
    train_loader, test_loader = load_dataloaders()
    
    # Create trainer and evaluator
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)
    evaluator = create_supervised_evaluator(model, metrics=load_test_metrics(), device=device)
    
    # Load checkpoint
    to_save_to_load = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    if train_config.is_resume:
        training_checkpoint_path = os.path.join(train_config.training_checkpoint_dir,
                                                train_config.training_checkpoint_name)
        training_checkpoint = torch.load(training_checkpoint_path,
                                         map_location=device, weights_only=True) 
        Checkpoint.load_objects(to_load=to_save_to_load, checkpoint=training_checkpoint)
        print(f"Resuming training from {training_checkpoint_path}")

    

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
        )


    


    score_name = "accuracy"
    score_fn = lambda engine: engine.state.metrics[score_name]

    checkpoint = Checkpoint(
        to_save_to_load,
        save_handler=DiskSaver(train_config.training_checkpoint_dir, require_empty=False),
        score_name=score_name,
        score_function=score_fn,
        n_saved=train_config.n_saved,
        global_step_transform=global_step_from_engine(trainer)
    )

    
    evaluator.add_event_handler(Events.COMPLETED, checkpoint)


    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), track_training_loss)
    

    pbar = ProgressBar(persist=True, bar_format='')
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    trainer.run(train_loader, max_epochs=train_config.epochs)



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