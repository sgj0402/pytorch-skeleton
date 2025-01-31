import os
from pprint import pprint

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver
from ignite.contrib.handlers import ProgressBar

from load_util import *
from ignite_util import *
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
    

def track_training_metrics(trainer, evaluator, test_loader):

    evaluator.run(test_loader)
    
    run = global_aim.get_run()
    run.track(evaluator.state.metrics,
              step=trainer.state.iteration,
              epoch=trainer.state.epoch,
              context={'mode': 'train', 'type': 'metric'})
    

def track_test_metrics(evaluator):
    
    run = global_aim.get_run()
    run.track(evaluator.state.metrics,
              context={'mode': 'test', 'type': 'metric'})


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
    trainer = create_supervised_trainer(model=model,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        device=device,
                                        prepare_batch=get_prepare_batch(),
                                        model_transform=get_model_transform(),
                                        output_transform=get_trainer_output_transform(),
                                        model_fn=get_model_fn())
    
    evaluator = create_supervised_evaluator(model=model,
                                            metrics=load_test_metrics(),
                                            device=device,
                                            prepare_batch=get_prepare_batch(),
                                            model_transform=get_model_transform(),
                                            output_transform=get_evaluator_output_transform(),
                                            model_fn=get_model_fn())
    

    # Create checkpoint handler
    checkpoint_score_name = load_checkpoint_score_name()
    # TODO
    checkpoint_score_fn = lambda engine: engine.state.metrics[checkpoint_score_name]

    to_save_training = {'model': model, 'optimizer': optimizer, 'trainer': trainer}
    checkpoint_handler = Checkpoint(
        to_save=to_save_training,
        save_handler=DiskSaver(train_config.checkpoint_dir, require_empty=False),
        score_function=checkpoint_score_fn,
        score_name=checkpoint_score_name,
        n_saved=train_config.n_saved,
        global_step_transform=global_step_from_engine(trainer),
        include_self=True # save checkpointer itself as key 'checkpointer'
    )

    # If resuming, load training checkpoint
    to_load_training = {'model': model,
                        'optimizer': optimizer,
                        'trainer': trainer,
                        'checkpointer': checkpoint_handler}
    if train_config.is_resume:
        training_checkpoint_path = os.path.join(train_config.checkpoint_dir,
                                                train_config.checkpoint_name)
        training_checkpoint = torch.load(training_checkpoint_path,
                                         map_location=device,
                                         weights_only=True) 
        Checkpoint.load_objects(to_load=to_load_training, checkpoint=training_checkpoint)
        print(f"Resuming training from {training_checkpoint_path}")
    
    # Attach handlers
    trainer.add_event_handler(Events.ITERATION_COMPLETED, track_training_loss)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, track_training_metrics,
                              evaluator=evaluator, test_loader=test_loader)
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)
    
    # Attach progress bar
    pbar = ProgressBar(bar_format='')
    pbar.attach(trainer)

    # Run trainer
    print(f"Training started.")

    trainer.run(train_loader, max_epochs=train_config.epochs)

    print(f"Training completed.")
    print(f"Best training saved at {checkpoint_handler.last_checkpoint}")


def test():
    # Load config
    config = global_config.get_config()
    test_config = config.setting.test

    # Load model, test dataloader
    device = load_device()
    model = load_model(device)
    test_loader = load_test_dataloader()

    # Load model checkpoint
    model_checkpoint_path = os.path.join(test_config.checkpoint_dir, test_config.checkpoint_name)
    model_checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=True)
    Checkpoint.load_objects(to_load={'model': model}, checkpoint=model_checkpoint)

    # Create evaluator
    evaluator = create_supervised_evaluator(model, metrics=load_test_metrics(), device=device)

    # Attach handlers
    evaluator.add_event_handler(Events.COMPLETED, track_test_metrics)

    # Attach progress bar
    pbar = ProgressBar(bar_format='')
    pbar.attach(evaluator)

    # Run evaluator
    print(f"Testing started.")

    evaluator.run(test_loader)

    print(f"Testing completed.")
    pprint(f"{evaluator.state.metrics}")


    
def model_run():

    config = global_config.get_config()
    
    if config.setting.mode == 'train':
        train()
        
    elif config.setting.mode == 'test':
        test()
        
    else:
        raise ValueError(f"Invalid mode: {config.setting.mode}")