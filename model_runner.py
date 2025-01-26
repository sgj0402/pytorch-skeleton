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

def track_training_loss(trainer):

    run = global_aim.get_run()
    run.track(trainer.state.output,
              name='loss',
              step=trainer.state.iteration,
              epoch=trainer.state.epoch,
              context={'mode': 'train', 'type': 'training_loss'})
    

def track_metrics(trainer, evaluator, test_loader):

    evaluator.run(test_loader)
    
    run = global_aim.get_run()
    run.track(evaluator.state.metrics,
              step=trainer.state.iteration,
              epoch=trainer.state.epoch,
              context={'mode': 'train', 'type': 'metric'})


def train():
    # Load config
    config = global_config.get_config()
    train_config = config.setting.train

    # Load model, optimizer, loss_fn, dataloaders
    device = load_device()
    model = load_model(device)
    optimizer = load_optimizer(model)
    loss_fn = load_loss_fn()
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
    
    # save checkpoint settings
    # TODO
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

    # Attach handlers
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), track_training_loss)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, track_metrics, evaluator, test_loader)
    evaluator.add_event_handler(Events.COMPLETED, checkpoint)
    
    # Attach progress bar
    pbar = ProgressBar(persist=True, bar_format='')
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    # Run trainer
    trainer.run(train_loader, max_epochs=train_config.epochs)


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