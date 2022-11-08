import os, sys
import torch
# sys.path.insert(0, 'external_code')
# import PyTorch_CIFAR10
import helpers


def load_model(model_restore_path,
               model_type,
               model_source,
               learning_rate,
               weight_decay,
               n_epochs):
    '''
    Return the model with restored weights

    Arg(s):
        model_restore_path : str
            path to model checkpoint
        model_type : str
            tells what model architecture to create
        model_source : str
            what repository model comes from ["pretrained"]

    Returns:
        model with restored weights (nn.module)
    '''

    if model_source == "pretrained":
        # args.classifier = model_type
        model = helpers.CIFAR10Module(
            model_type=model_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs)
        checkpoint = torch.load(model_restore_path)
        model.model.load_state_dict(checkpoint)
        return model
    else:
        raise ValueError("Model type {} from {} is not supported".format(
            model_type, model_source
        ))
