from ignite.utils import convert_tensor

"""
def example_evaluate_step(engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            output = model_fn(model, x)
            y_pred = model_transform(output)
            return output_transform(x, y, y_pred)
"""

# https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html#ignite.engine.create_supervised_trainer

def get_prepare_batch():
    """
    function that receives batch, device, non_blocking
    and outputs tuple of tensors (batch_x, batch_y).
    """

    # Default implementation, you can uncomment and create your own
    def _prepare_batch(batch, device = None, non_blocking = False):
        '''Prepare batch for training or evaluation: pass to a device with options.'''
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
        )

    return _prepare_batch


def get_model_fn():
    """
    the model function that receives model and x,
    and returns y_pred.
    """

    # Default implementation, you can uncomment and create your own
    def _model_fn(model, x):
        return model(x)
    
    return _model_fn


def get_model_transform():
    """
    function that receives the output from the model
    and convert it into the form as required by the loss function
    """

    # Default implementation, you can uncomment and create your own
    def _model_transform(output):
        return output
    
    return _model_transform


def get_trainer_output_transform():
    """
    function that receives ‘x’, ‘y’, ‘y_pred’, ‘loss’ 
    and returns value to be assigned to engine’s state.output after each iteration. 
    Default is returning loss.item().
    """

    # Default implementation, you can uncomment and create your own
    def _output_transform(x, y, y_pred, loss):
        return loss.item()
    
    return _output_transform


def get_evaluator_output_transform():
    """
    function that receives ‘x’, ‘y’, ‘y_pred’
    and returns value to be assigned to engine’s state.output after each iteration.
    Default is returning (y_pred, y,) which fits output expected by metrics.
    If you change it you should use output_transform in metrics.
    """
    # Default implementation, you can uncomment and create your own
    def _output_transform(x, y, y_pred):
        return y_pred, y
    
    return _output_transform