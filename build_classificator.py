from abc import ABC,abstractmethod
from torch.utils.data import DataLoader
import torchvision
torchvision.datasets.ImageFolder
from build_dataset import BalanceTrainingFrames,ValidateFrames,TestFrames,TrainFrames,InputProcessor
from torchvision import transforms
from enum import IntEnum
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch
import numpy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import (ResNet,VGG, AlexNet,DenseNet,EfficientNet,MNASNet,MobileNetV2, MobileNetV3,RegNet,ShuffleNetV2,SqueezeNet,GoogLeNet, Inception3,VisionTransformer, SwinTransformer,ConvNeXt, MaxVit)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from overrides import override
import argparse
from enum import Enum
import warnings
import traceback
import sys
import warnings

warnings.filterwarnings("ignore")

#######################################################################################################################

# SOURCES

    # https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html

    # https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/
    # https://www.codegenes.net/blog/pytorch-detach-cpu-numpy/
    # https://numpy.org/doc/stable/reference/generated/numpy.trace.html

    # https://www.geeksforgeeks.org/python/enum-intenum-in-python/#:~:text=With%20the%20help%20of%20enum.IntEnum%28%29%20method%2C%20we%20can,Return%20%3A%20IntEnum%20doesn%27t%20have%20a%20written%20type.

    # https://pypi.org/project/overrides/
    # https://stackoverflow.com/questions/75440/how-do-i-get-the-string-with-name-of-a-classom

    # https://www.geeksforgeeks.org/python/line-chart-in-matplotlib-python/
    # https://www.geeksforgeeks.org/python/how-to-draw-a-circle-using-matplotlib-in-python
    # https://www.geeksforgeeks.org/python/matplotlib-pyplot-xticks-in-python
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
    # https://docs.ray.io/en/latest/train/examples/pytorch/pytorch_resnet_finetune.html
    # https://docs.pytorch.org/docs/stable/generated/torch.cat.html
    # https://www.datacamp.com/tutorial/tqdm-python
    # https://tqdm.github.io/docs/tqdm/
    # https://docs.pytorch.org/vision/stable/models.html
    # https://docs.pytorch.org/docs/stable/optim.html

    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    # https://docs.pytorch.org/docs/stable/optim.html



#######################################################################################################################

# INPUT

# class for holding input parameters holding
class TrainModes(Enum):
    TYPE_MODEL = "type_model",
    TYPE_OPTIMIZER = "type_optimizer",
    LR_MOMENTUM = "lr_momentum",
    WEIGHT_DECAY = "weight_decay"
    NESTEROV = "nesterov",
    FULL_FIT = "full_fit"

parser = argparse.ArgumentParser()
parser.add_argument("--train_mode", help="Choose what type of training do you want:",default = TrainModes.TYPE_MODEL)


#######################################################################################################################

# PREPARE DATASET


# Dataset for training, where we can augmented training data
class Dataset(torch.utils.data.Dataset):

    def __init__(self,image_folder_name,transforms,copies):

        self.transforms = transforms
        self.copies = copies
        self.image_dataset = torchvision.datasets.ImageFolder(root=image_folder_name)

    def __len__(self):
        return self.copies * len(self.image_dataset)

    def __getitem__(self,index):

        index_img = index % len(self.image_dataset)

        img, label = self.image_dataset[index_img]

        img_transformed = self.transforms(img)

        return img_transformed, label

# Transforms for training and validating,testing
class Transforms:
        TRAIN_TRANSFORMS = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        EVAL_TRANSFORMS = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


#######################################################################################################################

# TRAININIG,VALIDATING AND TESTING EPOCHS EVALUATORS

# holder of phase results during epochs
class EpochsEvaluator:

    def __init__(self):
        self.losses = []
        self.accuracies = []
    
    # adding losss to list
    def add_loss(self,loss):
        self.losses.append(loss)
    
    # adding and counting accuracies
    def add_accuracies(self,correct_labels,predicted_labels):
        correct_labels_numpy = correct_labels.detach().cpu().numpy()
        predicted_labels_numpy = predicted_labels.detach().cpu().numpy()

        confusion_matrix_ = confusion_matrix(correct_labels_numpy,predicted_labels_numpy)
        accuracy = round(numpy.trace(confusion_matrix_) / numpy.sum(confusion_matrix_),2)

        self.accuracies.append(accuracy)


#######################################################################################################################

# LOADING DATA

class PhaseTag(IntEnum):

    TRAIN = 0
    VALIDATE = 1
    TEST = 2

class Data(ABC):

    """all phases -> training,validating,testing has this properties"""

    def __init__(self):
        super().__init__()

        self.dataset = None
        self.dataloader = None
        self.epochsEvaluator = EpochsEvaluator()
    
    # setting dataloader
    def set_dataloader(self):
        self.dataloader = DataLoader(self.dataset,batch_size=8,num_workers=8,pin_memory=True ,shuffle=True)

class NotAugmentedData(Data):

    """is parent class for validating and testing"""

    def __init__(self,name,transforms):
        super().__init__()

        self.dataset = torchvision.datasets.ImageFolder(root=name,transform = transforms)

        self.set_dataloader()

class Train(Data):

    """holder of testing data"""

    def __init__(self,parameter,TRANSFORMS = Transforms.TRAIN_TRANSFORMS,NAME = TrainFrames.NAME):
        super().__init__()

        self.dataset = Dataset(NAME,TRANSFORMS,parameter)

        self.set_dataloader()

class Validate(NotAugmentedData):
    def __init__(self,TRANSFORM = Transforms.EVAL_TRANSFORMS,NAME = ValidateFrames.NAME):
        super().__init__(NAME,TRANSFORM)

class Test(NotAugmentedData):

    """holder of testing data"""

    def __init__(self,TRANSFORM = Transforms.EVAL_TRANSFORMS,NAME = TestFrames.NAME):
        super().__init__(NAME,TRANSFORM)

#######################################################################################################################

# SOLVING PARAMETERS

class Parameter(ABC):

    """abstract class for model parameters as type model, type optimizer...."""
    COUNT_CLASSES = 1

    def __init__(self,values):
        super().__init__()

        self.values = values
        self.accuracies = {}
    
    # method for searching parameters and finding the best one
    @abstractmethod
    def searching(self,count_classes):
        pass

class TypeModel(Parameter):

    """finding best type model for this classification task"""

    BEST_TYPE_MODEL = models.maxvit_t(weights="DEFAULT") # set it handly, else None

    def __init__(self):

        type_models = [models.alexnet(weights="DEFAULT"),
                       models.convnext_tiny(weights="DEFAULT"),
                       models.densenet121(weights="DEFAULT"),
                       models.efficientnet_b0(weights="DEFAULT"),
                       models.efficientnet_v2_s(weights="DEFAULT"),
                       models.googlenet(weights="DEFAULT"),
                       models.maxvit_t(weights="DEFAULT"),
                       models.mnasnet1_0(weights="DEFAULT"),
                       models.mobilenet_v2(weights="DEFAULT"),
                       models.mobilenet_v3_small(weights="DEFAULT"),
                       models.regnet_y_400mf(weights="DEFAULT"),
                       models.resnet50(weights="DEFAULT"),
                       models.resnext50_32x4d(weights="DEFAULT"),
                       models.shufflenet_v2_x1_0(weights="DEFAULT"),
                       models.squeezenet1_0(weights="DEFAULT"),
                       models.swin_t(weights="DEFAULT"),
                       models.vgg16(weights="DEFAULT"),
                       models.vit_b_16(weights="DEFAULT"),
                       models.wide_resnet50_2(weights="DEFAULT")
                    ]

        self.default_optimizer = (optim.SGD, {"lr": 0.001,"momentum": 0.9})

        super().__init__(type_models)
    
    # searching best model type 
    @override
    def searching(self,count_classes):

        for type_model in self.values:
            print(self.accuracies)
            model = Model(type_model,self.default_optimizer,count_classes)
            model.fit(Copies.DEFAULT_COPIES,self.COUNT_CLASSES)
            type_model_name = type_model.__class__.__name__
            self.accuracies[type_model_name] = model.best_validation_accuracy
    

class TypeOptimizer(Parameter):

    """finding best optimizer type fot this classification task"""

    BEST_TYPE_OPTIMIZER = (optim.SGD, {"lr":0.01, "momentum":0, "weight_decay":0, "dampening":0, "nesterov":False}) # set it handly, else None

    def __init__(self):

        type_optimizers = [
            (optim.Adadelta, {"lr":1.0, "rho":0.9, "eps":1e-6, "weight_decay":0}),
            (optim.Adagrad, {"lr":0.01, "lr_decay":0, "weight_decay":0, "initial_accumulator_value":0, "eps":1e-10}),
            (optim.Adam, {"lr":0.001, "betas":(0.9, 0.999), "eps":1e-8, "weight_decay":0, "amsgrad":False}),
            (optim.AdamW, {"lr":0.001, "betas":(0.9, 0.999), "eps":1e-8, "weight_decay":0, "amsgrad":False}),
            (optim.Adamax, {"lr":0.002, "betas":(0.9, 0.999), "eps":1e-8, "weight_decay":0}),
            (optim.ASGD, {"lr":0.01, "lambd":0.0001, "alpha":0.75, "t0":1e6, "weight_decay":0}),
            (optim.LBFGS, {"lr":1.0, "max_iter":20, "max_eval":None, "tolerance_grad":1e-7, "tolerance_change":1e-9, "history_size":100, "line_search_fn":None}),
            (optim.NAdam, {"lr":0.001, "betas":(0.9, 0.999), "eps":1e-8, "weight_decay":0, "momentum_decay":0.004}),
            (optim.RAdam, {"lr":0.001, "betas":(0.9, 0.999), "eps":1e-8, "weight_decay":0}),
            (optim.RMSprop, {"lr":0.01, "alpha":0.99, "eps":1e-8, "weight_decay":0, "momentum":0, "centered":False}),
            (optim.Rprop, {"lr":0.01, "etas":(0.5, 1.2), "step_sizes":(1e-6, 50)}),
            (optim.SGD, {"lr":0.01, "momentum":0, "weight_decay":0, "dampening":0, "nesterov":False})
        ]
        
        super().__init__(type_optimizers)
    
    # searching for best classification optimiizer task
    @override
    def searching(self,count_classes):

        for type_optimizer in self.values:
            model = Model(TypeModel.BEST_TYPE_MODEL,type_optimizer,count_classes)
            model.fit(Copies.DEFAULT_COPIES,self.COUNT_CLASSES)
            type_optimizer_name = type_optimizer.__class__.__name__
            self.accuracies[type_optimizer_name] = model.best_validation_accuracy
        
        print(self.accuracies)

class lrMomentum(Parameter):

    BEST_LR_MOMENTUM = (0.01,0.6) # set it handly, else None

    def __init__(self):
        lr_momentum_values = [(0.001, 0.0),
                              (0.001, 0.9),
                              (0.001, 0.2),
                              (0.001, 0.4),
                              (0.001, 0.6),
                              (0.001, 0.8),
                              (0.01, 0.0),
                              (0.01, 0.2),
                              (0.01, 0.4),
                              (0.01, 0.6),
                              (0.01, 0.8),
                              (0.01, 0.9),
                              (0.1, 0.0),
                              (0.1, 0.2),
                              (0.1, 0.4),
                              (0.1, 0.6),
                              (0.1, 0.8),
                              (0.1, 0.9),
                              (0.2, 0.0),
                              (0.2, 0.2),
                              (0.2, 0.4),
                              (0.2, 0.6),
                              (0.2, 0.8),
                              (0.2, 0.9),
                              (0.3, 0.0),
                              (0.3, 0.2),
                              (0.3, 0.4),
                              (0.3, 0.6),
                              (0.3, 0.8),
                              (0.3, 0.9)
                            ]

        super().__init__(lr_momentum_values)
    
    # returns optmizer with best lr,momentum values
    @classmethod
    def get_best_lr_momentum_optimizer(self):
        best_type_optimizer_lr_momentum = TypeOptimizer.BEST_TYPE_OPTIMIZER
        best_type_optimizer_lr_momentum[1]["lr"] = lrMomentum.BEST_LR_MOMENTUM[0]
        best_type_optimizer_lr_momentum[1]["momentum"] = lrMomentum.BEST_LR_MOMENTUM[1]

        return best_type_optimizer_lr_momentum
    
    # searching for best lr_momentum values
    @override
    def searching(self,count_classes):

        for lr_momentum in self.values:

            model = Model(TypeModel.BEST_TYPE_MODEL,TypeOptimizer.BEST_TYPE_OPTIMIZER,count_classes)
            model.fit(Copies.DEFAULT_COPIES,self.COUNT_CLASSES)
            lr_momentum_name = str(lr_momentum[0]) + " " + str(lr_momentum[1])
            self.accuracies[lr_momentum_name] = model.best_validation_accuracy
        
        print(self.accuracies)


class WeightDecay(Parameter):

    """finding best weifht_decay for this classification task"""

    BEST_WEIGHT_DECAY_VALUES = 0

    def __init__(self):
        weight_decay_values =  [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,0]

        super().__init__(weight_decay_values)
    
    # returns optimizer with updated lr,momentum and weight_decay parameters
    def get_best_lr_momentum_weight_optimizer(self):
        best_type_optimizer_lr_momentum = lrMomentum.get_best_lr_momentum_optimizer()
        best_type_optimizer_lr_momentum[1]["weight_decay"] = self.BEST_WEIGHT_DECAY_VALUES

        return best_type_optimizer_lr_momentum
    
    # searching for best weight_decay parameter
    @override
    def searching(self, count_classes):

        best_type_optimizer_lr_momentum = lrMomentum.get_best_lr_momentum_optimizer()

        for weight_decay in self.values:
            best_type_optimizer_lr_momentum[1]["weight_decay"] = weight_decay

            model = Model(TypeModel.BEST_TYPE_MODEL,best_type_optimizer_lr_momentum,count_classes)
            model.fit(Copies.DEFAULT_COPIES,self.COUNT_CLASSES)
            weight_decay_name = str(weight_decay)
            self.accuracies[weight_decay_name] = model.best_validation_accuracy

        print(self.accuracies)

class Nesterov(Parameter):

    """fidning best nesterov value for this classification task"""

    BEST_NESTEROV_VALUE = False

    def __init__(self):
        nesterov_values = [True,False]

        super().__init__(nesterov_values)
    
    # returns optmizer with updated lr,momentum,weight_decay and nesterov parameters
    def get_best_lr_momentum_weight_nesterov_optimizer(self):
        best_type_optimizer_lr_momentum_weight = lrMomentum.get_best_lr_momentum_optimizer()
        best_type_optimizer_lr_momentum_weight[1]["nesterov"] = self.BEST_NESTEROV_VALUE

        return best_type_optimizer_lr_momentum_weight
    
    # searching for best nesterov parameter
    @override
    def searching(self,count_classes):
        
        best_type_optimizer_lr_momentum_weight = WeightDecay.get_best_lr_momentum_weight_optimizer()

        for nesterov_value in self.values:

            best_type_optimizer_lr_momentum_weight[1]["nesterov"] = nesterov_value
            model = Model(TypeModel.BEST_TYPE_MODEL,best_type_optimizer_lr_momentum_weight,count_classes)
            model.fit(Copies.DEFAULT_COPIES,self.COUNT_CLASSES)
            nesterov_value = str(nesterov_value)
            self.accuracies[nesterov_value] = model.best_validation_accuracy
        
        print(self.accuracies)

class Copies(Parameter):

    """finding best count of copies for this classification task"""

    DEFAULT_COPIES = 1
    BEST_COPIES = 2

    def __init__(self):
        copies_values = [1,3,5,8,10]

        super().__init__(copies_values)

    # returns optimizer with updated lr,momentum,weight_decay and nesterov parameter
    @override
    def searching(self,count_classes):

        best_lr_momentum_weight_nesterov = Nesterov.get_best_lr_momentum_weight_nesterov_optimizer()

        for copies in self.values:
            model = Model(TypeModel.BEST_TYPE_MODEL,best_lr_momentum_weight_nesterov,count_classes)
            model.fit(copies,self.COUNT_CLASSES)
            copies_name = str(copies)
            self.accuracies[copies_name] = model.best_validation_accuracy
        
        print(self.accuracies)

#######################################################################################################################

# MODEL TRAINING VISUALISATION

# visualisator of model performans during epochs
class Visualisator:
    
    @staticmethod
    def plot(train_values,validate_values,test_values,title,epochs):
        x = [epoch for epoch in range(1,epochs+1)]

        plt.plot(x, train_values, label="Train",marker='o')
        plt.plot(x,validate_values, label="Validate",marker='o')
        plt.plot(x,test_values, label="Test",marker='o')
        plt.xlabel("Epochs")
        plt.legend()
        plt.title(title)
        plt.show()

#######################################################################################################################

# FITTING MODEL


# Class which representing model
class Model(nn.Module):
    

    MODEL_NAME = "museum_model.pth"
    DEVICE = "cuda"

    def __init__(self,model,optimizer,count_classes):
        super().__init__()

        self.device = self.DEVICE
        self.model = model.to(self.device)
        self.best_validation_accuracy = float("-inf")

        for param in self.model.parameters():
            param.requires_grad = False

        # Získané jednotlivé architektury jednotlivých modelů byly získány pomocí chatGPT. Tady se začíná            ## 
        if isinstance(model, (VGG, AlexNet)):                                                                        ##                                                                  
            in_features = model.classifier[6].in_features                                                            ##
            model.classifier[6] = nn.Linear(in_features, count_classes).to(self.device)                              ##
                                                                                                                     ##
        elif isinstance(model, DenseNet):                                                                            ##
            in_features = model.classifier.in_features                                                               ##
            model.classifier = nn.Linear(in_features,count_classes).to(self.device)                                  ##
                                                                                                                     ##
        elif isinstance(model, (MobileNetV2, MobileNetV3)):                                                          ##                                                    
            for i in reversed(range(len(model.classifier))):                                                         ##                                                
                if isinstance(model.classifier[i], nn.Linear):                                                       ##
                    in_features = model.classifier[i].in_features                                                    ##
                    model.classifier[i] = nn.Linear(in_features, count_classes).to(self.device)                      ##
                    break                                                                                            ##
                                                                                                                     ##
        elif isinstance(model, RegNet):                                                                              ##
            in_features = model.fc.in_features                                                                       ##
            model.fc = nn.Linear(in_features,count_classes).to(self.device)                                          ##
                                                                                                                     ##
        elif isinstance(model, ShuffleNetV2):                                                                        ##
            in_features = model.fc.in_features                                                                       ##
            model.fc = nn.Linear(in_features,count_classes).to(self.device)                                          ##
                                                                                                                     ##
        elif isinstance(model, SqueezeNet):                                                                          ##
            model.classifier[1] = nn.Conv2d(512,count_classes, kernel_size=(1, 1), stride=(1, 1)).to(self.device)    ##
            model.num_classes = count_classes                                                                        ##
                                                                                                                     ##
        elif isinstance(model, (GoogLeNet, Inception3)):                                                             ##
            in_features = model.fc.in_features                                                                       ##
            model.fc = nn.Linear(in_features,count_classes).to(self.device)                                          ##
                                                                                                                     ##
        elif isinstance(model, VisionTransformer):                                                                   ##
            in_features = model.heads.head.in_features                                                               ##
            model.heads.head = nn.Linear(in_features,count_classes).to(self.device)                                  ##
                                                                                                                     ##
        elif isinstance(model, SwinTransformer):                                                                     ##
            in_features = model.head.in_features                                                                     ##
            model.head = nn.Linear(in_features,count_classes).to(self.device)                                        ##
                                                                                                                     ##
        elif isinstance(model, ConvNeXt):                                                                            ##
            in_features = model.classifier[2].in_features                                                            ##
            model.classifier[2] = nn.Linear(in_features, count_classes).to(self.device)                              ##
                                                                                                                     ##
        elif isinstance(model, MaxVit):                                                                              ##
            for layer in reversed(model.classifier):                                                                 ##
                if isinstance(layer, nn.Linear):                                                                     ##
                    in_features = layer.in_features                                                                  ##
                    break                                                                                            ##
                                                                                                                     ##
                                                                                                                     ##
            model.classifier[-1] = nn.Linear(in_features, count_classes).to(self.device)                                  ##
                                                                                                                     ## Tady generovaná neozdrojovaná část končí. Zpetám se vedoucího, jestli ok.

        for param in self.model.parameters():
                param.requires_grad = True
        
        self.optimizer = optimizer[0](self.model.parameters(), **optimizer[1])

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def classify_img(img,model):
            tranformed_img = Transforms.EVAL_TRANSFORMS(img).unsqueeze(0).to(Model.DEVICE)

            with torch.no_grad():
                outputs = model(tranformed_img)
                _, prediction = outputs.max(1)
            
            label = str(prediction.item())

            return label
    
    # for loop for training model
    def train_epoch(self,dataloader,loss_fn,progress_bar):

        for batch in dataloader: 
            inputs,labels = batch
            inputs,labels = inputs.to(self.device),labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            progress_bar.update(1)
    
    # for lopp for validating and testing model
    def eval_epoch(self,data:Data,loss_fn,progress_bar):
        current_loss = 0
        all_correct_labels = []
        all_predicted_Labels = []

        with torch.no_grad():
            for batch in data.dataloader:
                inputs,labels = batch

                inputs,labels = inputs.to(self.device),labels.to(self.device)

                outputs = self.model(inputs)
                predicted_labels = outputs.argmax(dim=1) 
                loss = loss_fn(outputs,labels)
                current_loss += loss.item()

                all_correct_labels.append(labels)
                all_predicted_Labels.append(predicted_labels)
                progress_bar.update(1)
        
        all_correct_labels = torch.cat(all_correct_labels)
        all_predicted_Labels = torch.cat(all_predicted_Labels)

        data.epochsEvaluator.add_loss(current_loss)
        data.epochsEvaluator.add_accuracies(all_correct_labels,all_predicted_Labels)

    # fitting model
    def fit(self,copies_parameter,epochs_count,MAX_NOT_BETTER_EPOCH = 10):
        loss_fn = torch.nn.CrossEntropyLoss()
        data = [Train(copies_parameter),Validate(),Test()]
        count_batches = sum(len(data_instance.dataloader) for data_instance in data) + len(data[0].dataloader)
        not_better_epoch = 0
        epochs = 0

        for _ in range(epochs_count):
            progress_bar = tqdm(total=count_batches)

            for i in range(len(PhaseTag)):
                current_data = None

                if i == PhaseTag.TRAIN:
                    current_data = data[0]
                    self.model.train()
                    self.train_epoch(current_data.dataloader,loss_fn,progress_bar)
                    self.model.eval()
                
                elif i == PhaseTag.VALIDATE:
                    current_data = data[1]
                    self.model.eval()
                
                elif i == PhaseTag.TEST:
                    current_data = data[2]
                    self.model.eval()
                
                self.eval_epoch(current_data,loss_fn,progress_bar)

                if i == PhaseTag.VALIDATE and data[1].epochsEvaluator.accuracies[-1] > self.best_validation_accuracy:
                    self.best_validation_accuracy = data[1].epochsEvaluator.accuracies[-1]
                    not_better_epoch = 0
                    # self.save()
                
                elif  i == PhaseTag.VALIDATE and data[1].epochsEvaluator.accuracies[-1] <= self.best_validation_accuracy:
                    not_better_epoch += 1
                
            epochs += 1
                
            progress_bar.set_postfix(train_accuracy=data[0].epochsEvaluator.accuracies[-1],validae_accuracy = data[1].epochsEvaluator.accuracies[-1], test_accuracy = data[2].epochsEvaluator.accuracies[-1])

            if not_better_epoch == MAX_NOT_BETTER_EPOCH:
                break
        
        Visualisator.plot(data[0].epochsEvaluator.accuracies,data[1].epochsEvaluator.accuracies,data[2].epochsEvaluator.accuracies,"Accuracies",epochs)
        Visualisator.plot(data[0].epochsEvaluator.losses,data[1].epochsEvaluator.losses,data[2].epochsEvaluator.losses,"Losses",epochs)
        
    def save(self):
        torch.save(self.model,self.MODEL_NAME)


#######################################################################################################################

# INPUT PROCESSING

class InputProcessorTraining(InputProcessor):

    """process input parameter"""

    def __init__(self):
        super().__init__()

        dataset = torchvision.datasets.ImageFolder(root=TrainFrames.NAME, transform=Transforms.TRAIN_TRANSFORMS)
        self.count_classes = len(dataset.classes)
    
    # main method for type of finding parameters or full model classification
    @override
    def process(self, input_argument,epochs_count = 20):

        match input_argument:

            case TrainModes.TYPE_MODEL:
                TypeModel().searching(self.count_classes)

            case TrainModes.TYPE_OPTIMIZER:
                TypeOptimizer().searching(self.count_classes)

            case TrainModes.LR_MOMENTUM:
                lrMomentum().searching(self.count_classes)

            case TrainModes.WEIGHT_DECAY:
                WeightDecay().searching(self.count_classes)

            case TrainModes.NESTEROV:
                Nesterov().searching(self.count_classes)

            case TrainModes.FULL_FIT:
                optimizer = TypeOptimizer.BEST_TYPE_OPTIMIZER
                optimizer[1]["lr"] =  lrMomentum.BEST_LR_MOMENTUM[0]
                optimizer[1]["momentum"] =  lrMomentum.BEST_LR_MOMENTUM[1]
                optimizer[1]["weight_decay"] = WeightDecay.BEST_WEIGHT_DECAY_VALUES
                optimizer[1]["nesterov"] = Nesterov.BEST_NESTEROV_VALUE

                model = Model(TypeModel.BEST_TYPE_MODEL,optimizer,self.count_classes)
                model.fit(Copies.BEST_COPIES,epochs_count)

            case _:
                pass


# main method for creating classificator for all classes
def build():
    args = parser.parse_args()
    InputProcessorTraining().process(args.train_mode)

if __name__ == "__main__":
    build()