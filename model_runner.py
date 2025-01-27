import os

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver
from ignite.contrib.handlers import ProgressBar

from load_util import *
import global_config
import global_aim


# TODO output transform

def track_training_loss(trainer):

    run = global_aim.get_run()
    run.track(trainer.state.output,
              name='training_loss',
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
    
    # If resuming, load checkpoint
    to_save_to_load = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    if train_config.is_resume:
        training_checkpoint_path = os.path.join(train_config.training_checkpoint_dir,
                                                train_config.training_checkpoint_name)
        training_checkpoint = torch.load(training_checkpoint_path,
                                         map_location=device, weights_only=True) 
        Checkpoint.load_objects(to_load=to_save_to_load, checkpoint=training_checkpoint)
        print(f"Resuming training from {training_checkpoint_path}")
    
    # Create checkpoint handler
    checkpoint_score_name = load_checkpoint_score_name()
    # TODO
    checkpoint_score_fn = lambda engine: engine.state.metrics[checkpoint_score_name]

    training_checkpoint_handler = Checkpoint(
        to_save=to_save_to_load,
        save_handler=DiskSaver(train_config.training_checkpoint_dir, require_empty=False),
        filename_prefix='training',
        score_function=checkpoint_score_fn,
        score_name=checkpoint_score_name,
        n_saved=train_config.n_saved,
        global_step_transform=global_step_from_engine(trainer)
    )

    model_checkpoint_handler = Checkpoint(
        to_save={'model': model},
        save_handler=DiskSaver(train_config.model_checkpoint_dir, require_empty=False),
        score_function=checkpoint_score_fn,
        score_name=checkpoint_score_name,
        n_saved=train_config.n_saved,
        global_step_transform=global_step_from_engine(trainer)
    )

    # Attach handlers
    trainer.add_event_handler(Events.ITERATION_COMPLETED, track_training_loss)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, track_metrics,
                              evaluator=evaluator, test_loader=test_loader)
    evaluator.add_event_handler(Events.COMPLETED, training_checkpoint_handler)
    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint_handler)
    
    # Attach progress bar
    pbar = ProgressBar(persist=True, bar_format='')
    pbar.attach(trainer)

    # Run trainer
    trainer.run(train_loader, max_epochs=train_config.epochs)


# TODO
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