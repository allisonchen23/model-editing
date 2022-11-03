import os, sys
import pytorch_lightning as pl
import torch
from torchmetrics.functional import accuracy

sys.path.insert(0, os.path.join('external_code', 'PyTorch_CIFAR10'))
from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}
    
def get_model_module(model_type):
    '''
    Return module for model type
    
    Arg(s):
        model_type : str
            type of model name
    Returns:
        model module
    '''
    if model_type in all_classifiers.keys():
        return all_classifiers[model_type]
    else:
        raise ValueError("Unrecognized model type {}".format(model_type))
    
class CIFAR10Module(pl.LightningModule):
    def __init__(self, 
                 model_type,
                 loss_func=None,
                 learning_rate=0.01,
                 weight_decay=0.001,
                 n_epochs=100):
        super().__init__()
        # self.hparams = hparams

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = accuracy

        self.model = all_classifiers[model_type]
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.n_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console
    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
               o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')
