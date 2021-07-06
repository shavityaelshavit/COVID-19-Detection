#!/usr/bin/env python
# coding: utf-8

# # drive.mount()

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# used directories:
data_dir = '/content/drive/MyDrive/DL/project_ariel_yael/pneumonia/dataset'
checkpoint_dir = '/content/drive/MyDrive/DL/project_ariel_yael/pneumonia/Checkpoints_Pneumonia'


data_dir_covid  = '/content/drive/MyDrive/DL/project_ariel_yael/Covid19/dataset/dataset_covid'
checkpoint_dir_covid = '/content/drive/MyDrive/DL/project_ariel_yael/Covid19/Checkpoints_Covid'


# # Imports

# In[ ]:


get_ipython().system('pip install imutils')


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from imutils import paths
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import random
import shutil
import cv2
import os
import time
from datetime import datetime

# efficientNet:
get_ipython().system('pip install efficientnet_pytorch')
from efficientnet_pytorch import EfficientNet

# set random seed so we get the same sampling every time for reproducibility
random_seed = 2020
torch.manual_seed(random_seed);


# # Data Preparations - Utils

# In[ ]:


# Function for plotting samples
def plot_samples(f_samples):  
    """
    plot out multiple samples.
    Args:   
        f_samples - samples created with random.sample() from data diretory
    """    
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,8))
    for i in range(len(f_samples)):
        image = cv2.cvtColor(imread(f_samples[i]), cv2.COLOR_BGR2RGB)
        ax[i//5][i%5].imshow(image)
        if i<5:
            ax[i//5][i%5].set_title("Normal", fontsize=20)
        else:
            ax[i//5][i%5].set_title("Pneumonia", fontsize=20)
        ax[i//5][i%5].axis('off')


# In[ ]:


# function for plotting out some images with their labels. 
get_ipython().magic('matplotlib inline')
def show_example(f_img, f_label, f_dataset):
    """ 
    plot out some images with their labels. 
    Args:  
        f_img, f_label - some example from dataset
        f_dataset - dataset created with ImageFolder
    """

    print('Label: ', f_dataset.classes[f_label], "("+str(f_label)+")")
    plt.imshow(f_img.permute(1, 2, 0))


# In[ ]:


# visualize a batch of data in a grid.
#       using the make_grid function from torchvision, and using the .permute method on the tensor 
#       to move the channels to the last dimension, as expected by matplotlib.
def show_batch(f_dl):
    """
    visualize a batch of data in a grid.
    args: 
        f_dl - dataloader, we visualize a batch from this dataloader. 
    """
    for images, labels in f_dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:60], nrow=10).permute(1, 2, 0))
        break


# # Data Preparation
# 
# >This dataset is well structured for most deep learning frameworks. It is organized into train, val, and test folders, which contains subfolders of the classes of pneumonia (normal, pneumonia), each containing the respective jpeg images.
# 
# 
# 

# In[ ]:


data_dir = '/content/drive/MyDrive/DL/project_ariel_yael/pneumonia/dataset'

print("directories in data_dir:", os.listdir(data_dir))

print("classes in data_dir_train:", os.listdir(data_dir + "/train"))
print()
pneumonia_files_train = os.listdir(data_dir + "/train/PNEUMONIA")
print('No. of training examples for Pneumonia:', len(pneumonia_files_train))
normal_files_train = os.listdir(data_dir + "/train/NORMAL")
print('No. of training examples for Normal:', len(normal_files_train))

# lets take a look on the train and how the folder orgenized:
print("\n",'lets take a look on the train/PNEUMONIA and how the folder orgenized:')
print(pneumonia_files_train[:5])
print("\n",'lets take a look on the train/NORMAL and how the folder orgenized:')
print(normal_files_train[:5])


# ## Create train dataset and add Data Augmentation
# >  
# During training, this image might be transformed to one form, but in another epoch, this image will receive another random transformation, and this DIVERSIFIES the dataset, since the model is not receiving each image in the same 'form' every epoch.

# In[ ]:


# define pre-processing steps on images
data_transorm_train = tt.Compose([tt.Resize(255),
                              tt.CenterCrop(224),
                              tt.RandomHorizontalFlip(),
                              tt.RandomRotation(10),
                              tt.RandomGrayscale(),
                              tt.RandomAffine(translate=(0.05,0.05), degrees=0),
                              tt.ToTensor()
                              #tt.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] ,inplace=True)
                              ])

# create train dataset:
dataset_train = ImageFolder(data_dir + "/train", data_transorm_train)
print("classes in train dataset:",dataset_train.classes,"\n")
print("train dataset:","\n")
print(dataset_train)


# ## statisics and visualizations

# In[ ]:


# Let's look at an example of an image in its tensor form. Note that after being loaded, 
# each image is now [3, 224, 224], with 3 being the RGB channels
img, label = dataset_train[0]
print(img.shape, label)
img


# In[ ]:


# plot out some images with their labels.

show_example(*dataset_train[4], dataset_train) 


# In[ ]:


# number of images for each class


train_samplesize = pd.DataFrame.from_dict(
    {'Normal': [len([os.path.join(data_dir+'/train/NORMAL', filename) 
                     for filename in os.listdir(data_dir+'/train/NORMAL')])], 
     'Pneumonia': [len([os.path.join(data_dir+'/train/PNEUMONIA', filename) 
                        for filename in os.listdir(data_dir+'/train/PNEUMONIA')])]})


sns.barplot(data=train_samplesize).set_title('Training Set Data Inbalance', fontsize=20)
plt.show()


# In[ ]:


## Plot training samples



rand_samples_train = random.sample([os.path.join(data_dir+'/train/NORMAL', filename) 
                              for filename in os.listdir(data_dir+'/train/NORMAL')], 5) + \
    random.sample([os.path.join(data_dir+'/train/PNEUMONIA', filename) 
                   for filename in os.listdir(data_dir+'/train/PNEUMONIA')], 5)

plot_samples(rand_samples_train)
plt.suptitle('Training Set Samples', fontsize=30)
plt.show()


# ## Split into Train & Validation Data
# > Split training dataset into (80%)train and (20%)validation.
# * **Training set** - used to train the model i.e. compute the loss and adjust the weights of the model using gradient descent.
# * **Validation set** - used to evaluate the model while training, adjust hyperparameters (learning rate etc.) and pick the best version of the model.
# * **Test set** - used only to assess the performance of the model.
#     * the test set will be defined later

# In[ ]:


# Split training dataset to train and validation
train_size = round(len(dataset_train)*0.8) # 80%
val_size = len(dataset_train) - train_size # 20%

train_ds, val_ds = random_split(dataset_train, [train_size, val_size])

print("number of samples in dataset:\n", "train_ds:", len(train_ds), "val_ds:", len(val_ds))


# ## Create dataloaders
# > **batch size:** Here we select a batch size of 32 to perform mini_batch gradient descent/or other optimizers. This is a hyperparameter that can be tuned. The batch size means that the 4173 train images will be divided into batches of 32 images and gradient descent will be performed on each of this 32 images in one epoch (1 runthrough of the whole data).
# 
# 

# In[ ]:


# hyperparameter: batch zise
batch_size_train=32

# create dataloaders
train_dataloader = DataLoader(train_ds, batch_size_train, shuffle=True, num_workers=2, pin_memory=True) 
val_dataloader = DataLoader(val_ds, batch_size_train*2,shuffle=False, num_workers=2, pin_memory=True) 


# # device

# In[ ]:


# device - cpu or gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# # Utils

# ## printTime()

# In[ ]:


def printTime(f_startTime) : 
    """
    Print time elapsed since f_startTime
    """
    f_epochIter_time = time.time() - f_startTime
    print("time : {:.2f} secs".format(f_epochIter_time))


# ## getDateTimeString()

# In[ ]:


def getDateTimeString() :
    """
    get the date and time as a string, use for checkpoints.
    """
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y-%m-%d_%H:%M")
    return "time:" + dt_string


# ## set_parameter_requires_grad()
# 
# 
# > The following helper function sets the .requires_grad attribute of the parameters in the model to False when we are feature extracting.
# By default, when we load a pretrained model all of the parameters have .requires_grad=True, which is fine if we are **training from scratch or finetuning**.
# However, if we are **feature extracting** and only want to compute gradients for the newly initialized layer then we want all of the other parameters to not require gradients.

# In[ ]:


def set_parameter_requires_grad(f_model, f_feature_extracting=False):
    """
    Set Model Parameters’ .requires_grad attribute
    Args:
        f_model - the pretrained model
        f_feature_extracting - False when we are feature extracting
                             - True when we are training from scratch or finetuning.
    """
    if f_feature_extracting:
        for param in f_model.parameters():
            param.requires_grad = False
    else:
        for param in f_model.parameters():
            param.requires_grad = True


# ## gather_params_to_update()
# 
# > The following helper function Gather the parameters to be optimized/updated in this run. If we are fine-tuning we will be updating all parameters. if we are doing feature extract method, we will only update the parameters that we have just initialized, i.e. the parameters with requires_grad is True.
# 

# In[ ]:


def gather_params_to_update(f_model, f_feature_extracting):
  """ 
  Gather the parameters to be optimized/updated in this run
  Args:
        f_model - the pretrained model
        f_feature_extracting - False when we are feature extracting
                             - True when we are training from scratch or finetuning.

  return: list of parameters to be optimized.
        (when feature extracting this list should be short and only include 
        the weights and biases of the reshaped layers.)
  """
  f_params_to_update = f_model.parameters()
  print("Params to learn:")
  if f_feature_extracting:
      f_params_to_update = []
      for name,param in f_model.named_parameters():
          if param.requires_grad == True:
              f_params_to_update.append(param)
              print("\t",name)
  else:
      for name,param in f_model.named_parameters():
          if param.requires_grad == True:
              print("\t",name)
  return f_params_to_update 


# ## plot_acc_and_loss_for_train_and_val()

# In[ ]:


# Plot Accuracy and Loss according to the history
def plot_acc_and_loss_for_train_and_val( f_history, f_header= ""):
    """
    plots of the accuracy and loss for the training and validation data. 
    This gives us an idea of how our model is performing (e.g., underfitting, overfitting).

    Args:
        f_history - {'train_loss','train_acc','val_loss','val_acc'}
        f_header - header to print
    """
    f_number_of_epochs = len(f_history['train_acc'])

    f, (f_ax1, f_ax2) = plt.subplots(1, 2, figsize=(25, 5))
    t = f.suptitle('Performance'+"\n"+f_header, fontsize=12)
    f.subplots_adjust(top=0.8, wspace=0.3)

    f_epoch_list = list(range(1,f_number_of_epochs+1))
    # plot train and validation accuracy on the same subplot
    f_ax1.plot(f_epoch_list, f_history['train_acc'], label='Train Accuracy')
    f_ax1.plot(f_epoch_list, f_history['val_acc'], label='Validation Accuracy')
    f_ax1.set_xticks(np.arange(0, f_number_of_epochs+1, 5))
    f_ax1.set_ylabel('Accuracy Value')
    f_ax1.set_xlabel('Epoch')
    f_ax1.set_title('Accuracy')
    l1 = f_ax1.legend(loc="best")

     # plot train and validation loss on the same subplot
    f_ax2.plot(f_epoch_list, f_history['train_loss'], label='Train Loss')
    f_ax2.plot(f_epoch_list, f_history['val_loss'], label='Validation Loss')
    f_ax2.set_xticks(np.arange(0, f_number_of_epochs+1, 5))
    f_ax2.set_ylabel('Loss Value')
    f_ax2.set_xlabel('Epoch')
    f_ax2.set_title('Loss')
    l2 = f_ax2.legend(loc="best")


# # Model Functions

# ## Intialize the model- functions:

# ### initialize_model_old()

# > the following function used to compare between the pretrained models :
# * efficientnet-b0
# * resnet
# * alexnet
# * vgg
# * squeezenet
# * densenet
# 
# > We load each of the models, setting the pretrained parameter to true to load the pretrained weights. We then freeze the starting layers of the network (except the fully connected layer) and replaced the last layer (fc) with our own as our output has only 2 classes, where the pretained model was trained to output more classes (most of them 1000 classes).
# 
# 

# In[ ]:


def initialize_model_old(f_model_name, f_use_pretrained, f_feature_extract):
    print("initialize_model parameters:")
    print("         model_name:", f_model_name , "use_pretrained:", f_use_pretrained, "feature_extract:", f_feature_extract)

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    f_model_ft = None
    f_num_classes = 2 # normal. pneumonia
    f_input_size = 0  # image size, e.g. (3, 224, 224)

    if f_model_name == 'efficientnet-b0':
        """ Efficientnet-b0
        """
        f_model_ft = EfficientNet.from_pretrained(f_model_name)
        set_parameter_requires_grad(f_model_ft, f_feature_extract)
        num_ftrs = f_model_ft._fc.in_features 
        f_model_ft._fc = nn.Linear(num_ftrs, f_num_classes) # replace the last FC layer
        f_input_size = 224 # for b0: input_size = 224
        

    elif f_model_name == "resnet":
        """ Resnet18
        """
        f_model_ft = models.resnet18(pretrained=f_use_pretrained)
        set_parameter_requires_grad(f_model_ft, f_feature_extract)
        num_ftrs = f_model_ft.fc.in_features
        f_model_ft.fc = nn.Linear(num_ftrs, f_num_classes) # replace the last FC layer
        f_input_size = 224

    elif f_model_name == "alexnet":
        """ Alexnet
        """
        f_model_ft = models.alexnet(pretrained=f_use_pretrained)
        set_parameter_requires_grad(f_model_ft, f_feature_extract)
        num_ftrs = f_model_ft.classifier[6].in_features
        f_model_ft.classifier[6] = nn.Linear(num_ftrs, f_num_classes)
        f_input_size = 224

    elif f_model_name == "vgg":
        """ VGG16
        """
        f_model_ft = models.vgg16(pretrained=f_use_pretrained)
        set_parameter_requires_grad(f_model_ft, f_feature_extract)
        num_ftrs = f_model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, f_num_classes)
        f_input_size = 224

    elif f_model_name == "squeezenet":
        """ Squeezenet
        """
        f_model_ft = models.squeezenet1_0(pretrained=f_use_pretrained)
        set_parameter_requires_grad(f_model_ft, f_feature_extract)
        f_model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        f_model_ft.f_num_classes = f_num_classes
        f_input_size = 224

    elif f_model_name == "densenet":
        """ Densenet
        """
        f_model_ft = models.densenet121(pretrained=f_use_pretrained)
        set_parameter_requires_grad(f_model_ft, f_feature_extract)
        num_ftrs = f_model_ft.classifier.in_features
        f_model_ft.classifier = nn.Linear(num_ftrs, f_num_classes)
        f_input_size = 224

    else:
        raise NotImplementedError

    return f_model_ft, f_input_size


# ### initialize_model()

# the following function initialize the model with the chosen pretrained net.<br/>
# after choosing Efficientnet-b0 as the pretrained model, 
# * we tested 5 versions for the classifier we train - the layers replacing the last fc of the pretrained model. 
#     * The versions differ by their depth (the number of fully connected layers)
# * We tried to use different activation functions and chosen to work with LeakyReLU, with a slope of 0.1
# * We tested the effect of batch normalization and decided to use it.

# In[ ]:


def initialize_model(f_model_name, f_use_pretrained, f_feature_extract, f_chooseLong = 0):
 
    print("initialize_model parameters:")
    print("model_name:", f_model_name , "use_pretrained:", f_use_pretrained, "feature_extract:", f_feature_extract, "chooseLong:",f_chooseLong)

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    f_model_ft = None
    f_num_classes = 2 # normal. pneumonia
    f_input_size = 0  # image size, e.g. (3, 224, 224)

    if f_model_name == 'efficientnet-b0':
        """ Efficientnet-b0
        """
        f_model_ft = EfficientNet.from_pretrained(f_model_name)
        set_parameter_requires_grad(f_model_ft, f_feature_extract)
        # replace model classifier
        num_ftrs = f_model_ft._fc.in_features 

        
        if f_chooseLong == 0 :
            f_model_ft._fc = nn.Linear(num_ftrs, f_num_classes) # replace the last FC layer

        elif f_chooseLong == 1 :
            f_model_ft._fc = nn.Sequential(nn.Linear(num_ftrs, 256),   
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.25),   
                                        nn.Linear(256, 64),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.1),  
                                        nn.Linear(64, f_num_classes)) # replace the last FC layer
        elif f_chooseLong == 2 :
            f_model_ft._fc = nn.Sequential(nn.Linear(num_ftrs, 512),   
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.25),   
                                        nn.Linear(512, 128), 
                                        nn.BatchNorm1d(128), 
                                        nn.LeakyReLU(0.1),  
                                        nn.Linear(128, 64),
                                        nn.BatchNorm1d(64), 
                                        nn.LeakyReLU(0.1),                                          
                                        nn.Linear(64,f_num_classes))         
        elif f_chooseLong == 3 :
            f_model_ft._fc = nn.Sequential(nn.Linear(num_ftrs, 512),   
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.25),   
                                        nn.Linear(512, 128), 
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(0.1),  
                                        nn.Linear(128, 64), 
                                        nn.LeakyReLU(0.1),                                        
                                        nn.Linear(64,32),
                                        nn.BatchNorm1d(32), 
                                        nn.LeakyReLU(0.1),     
                                        nn.Linear(32,f_num_classes))             
        elif f_chooseLong == 4 :
            f_model_ft._fc = nn.Sequential(nn.Linear(num_ftrs, 512),     
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(0.25),   
                                        nn.Linear(512, 256),
                                        nn.BatchNorm1d(256), 
                                        nn.LeakyReLU(0.1),  
                                        nn.Linear(256, 128), 
                                        nn.LeakyReLU(0.1),    
                                        nn.Linear(128, 64),
                                        nn.BatchNorm1d(64),     
                                        nn.LeakyReLU(0.1),                              
                                        nn.Dropout(0.1), 
                                        nn.Linear(64,32), 
                                        nn.LeakyReLU(0.1),
                                        nn.Linear(32,f_num_classes))
    else:
        raise NotImplementedError

    return f_model_ft, f_input_size


# ## val_phase()

# In[ ]:


# torch.no_grad() impacts the autograd engine and deactivate it. 
#       It will reduce memory usage and speed up computations but we won’t be able to backprop 
#       (which we don’t want in an evaluation phase).
@torch.no_grad()
def val_phase(f_model, f_dataloader, f_criterion, f_device):
    """
    NOTE: use @torch.no_grad() as a decorator
    Args: 
        f_model - the model we validate
        f_dataloader - dataloader for valdation dataset
        f_device - device to use

    RETURN:
        epoch_loss
        epoch_acc
        PerformanceMetrics = {'precision', 'recall', 'f1'}
    """
    # model.eval() set specific layers like dropout and batchnorm to evaluation mode 
    #       while model.train() will enable them.  

    f_model.eval()                            # Set model to training mode

    num_classes = 2
    num_samples = len(f_dataloader.dataset)     

    running_loss = 0.0
    running_corrects = 0
    confusion_matrix = np.zeros([num_classes,num_classes], int)

    # Iterate over data.
    for inputs, labels in f_dataloader:       # get the data
        inputs = inputs.to(f_device)          # send to device
        labels = labels.to(f_device)          # send to device  
        # forward
        # with torch.set_grad_enabled(phase == 'train'): #(allready use @torch.no_grad() as a decorator)
            # Get model outputs and calculate loss
        outputs = f_model(inputs)         
        loss = f_criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)


        running_loss += loss.item() * inputs.size(0)    # loss.item() = (losses for entire batch)/ batchSize
        running_corrects += torch.sum(preds == labels.data)

        for i, l in enumerate(labels):
            confusion_matrix[l.item(), preds[i].item()] += 1 

    # Compute regular Performance
    epoch_loss = running_loss / len(f_dataloader.dataset) 
    epoch_acc = running_corrects.double() / len(f_dataloader.dataset)       

    # Compute Performance Metrics    
    tn, fp, fn, tp = confusion_matrix.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))   
    PerformanceMetrics = {'precision': precision, 'recall': recall, 'f1': f1}    

    return epoch_loss, epoch_acc, PerformanceMetrics


# ## train_model()

# In[ ]:


"""
Training function
"""
def train_model(f_model, f_train_dl, f_val_dl, f_criterion, f_optimizer, f_scheduler, f_num_epochs, f_device):
    """ 
    train_model
    Args:
        f_model - the model we train 
        f_train_dl - train dataloader
        f_val_dl - validation dataloader
        f_criterion - loss function
        f_optimizer - Optimizer for weight updates
        f_scheduler -  method to adjust the learning rate
        f_num_epochs - number of efochs for the train

    this is the main function to train the model.
    Each epoch has a training and validation phase.
    Save the model when the loss improve (smaller)

    Returns:    
        f_model - the trained model, when the model achive the best loss (the smaller)
        f_history - {'train_loss','train_acc','val_loss','val_acc'}
    """
    # intializations:
    since = time.time()

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    history = {'train_loss': train_loss_history,'train_acc': train_acc_history,
               'val_loss': val_loss_history,'val_acc': val_acc_history}

    best_model_wts = copy.deepcopy(f_model.state_dict())
    best_acc = 0.0
    best_loss = 1.0                                     # use for saving the best model.

    for epoch in range(f_num_epochs):
        epochSince = time.time()
        logEpoch = 'Epoch {}/{} |'.format(epoch, f_num_epochs - 1)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # model.eval() set specific layers like dropout and batchnorm to evaluation mode 
                #       (dropout won’t drop activations, batchnorm will use running estimates instead of batch statistics) 
                #       while model.train() will enable them.
                f_model.train()                         # Set model to training mode
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for inputs, labels in f_train_dl:       # get the data
                    inputs = inputs.to(f_device)          # send to device
                    labels = labels.to(f_device)          # send to device  
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = f_model(inputs)         
                        loss = f_criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        # backward + optimize only if in training phase
                        # zero the parameter gradients
                        f_optimizer.zero_grad()           
                        loss.backward()
                        f_optimizer.step()
                    # collect statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # epoch statistics
                epoch_loss_train = running_loss / len(f_train_dl.dataset)
                epoch_acc_train = running_corrects.double() / len(f_train_dl.dataset)  
                # log
                logTrain = "Train Loss: {:.4f} | Train Acc: {:.4f} |".format(epoch_loss_train, epoch_acc_train)
                # append statistics to history for later use:
                history['train_loss'].append(epoch_loss_train)
                history['train_acc'].append(epoch_acc_train)               

            if phase == 'val':
                epoch_loss_val, epoch_acc_val, Performance = val_phase(f_model, f_val_dl, f_criterion, f_device)
                # log
                logVal = "Val Loss: {:.4f} | Val Acc: {:.4f} |".format(epoch_loss_val, epoch_acc_val)
                logValPerformance = " recall: {:.3f} | precision: {:.3f} | f1: {:.3f} |".format(Performance['recall'],Performance['precision'],Performance['f1'])
                # append statistics to history for later use:
                history['val_loss'].append(epoch_loss_val)
                history['val_acc'].append(epoch_acc_val)

                
                # deep copy the model
                if ( epoch_acc_val > best_acc):         #TODO: maybe change condition to fromm acc to loss            
                    best_acc = epoch_acc_val
                    val_when_best_acc = epoch_loss_val
                    epoch_when_best_acc = epoch
                    best_model_wts = copy.deepcopy(f_model.state_dict())
                """
                # deep copy the model
                if ( epoch_loss_val < best_loss):         #TODO: maybe change condition to fromm acc to loss            
                    best_loss = epoch_loss_val
                    acc_when_best_loss = epoch_acc_val
                    epoch_when_best_loss = epoch
                    best_model_wts = copy.deepcopy(f_model.state_dict())
                """    

                   
        # log
        epoch_time_elapsed = time.time() - epochSince
        logTime = 'Epoch Time: {:}'.format(epoch_time_elapsed)
        log = logEpoch + logTrain + logVal + logValPerformance + logTime
        print(log)

        f_scheduler.step(epoch_acc_val)  # loss is used to track down a plateau

        ####  finish epoch  ####
    ####  finish train ####
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f} where val loss: {:4f} at epoch: {:} '.format(best_acc,val_when_best_acc,epoch_when_best_acc))
    #print('Best val loss: {:4f} where val acc: {:4f} at epoch: {:} '.format(best_loss,acc_when_best_loss,epoch_when_best_loss))


    # load best model weights
    f_model.load_state_dict(best_model_wts)
    return f_model, history


# ## test_model()

# In[ ]:


@torch.no_grad()
def test_model(f_model, f_dataloader, f_device, showCM = False):
    """
    NOTE: use @torch.no_grad() as a decorator
    Args: 
        f_model - the model we test 
        f_dataloader - dataloader for test dataset
        f_device - device to use
        showCM - choose to plot confusion_matrix, default False
    RETURN: PerformanceMetrics = {'accuracy','precision', 'recall', 'f1'}
    """
    # model.eval() set specific layers like dropout and batchnorm to evaluation mode 
    #       while model.train() will enable them.  
    f_model.eval() # put in evaluation mode 

    num_classes = 2
    num_samples = len(f_dataloader.dataset)
    # print("DEBUG:test_model->number_of_samples= ", num_samples)
    running_corrects = 0.0                                  
    confusion_matrix = np.zeros([num_classes,num_classes], int)
    # Iterate over data.
    for inputs, labels in f_dataloader:       # get the data
        inputs = inputs.to(f_device)          # send to device
        labels = labels.to(f_device)          # send to device  
        # forward
        outputs = f_model(inputs)                    
        _, preds = torch.max(outputs.data, 1)    
        # collect statistics
        running_corrects += (preds == labels).sum().item() 
        for i, l in enumerate(labels):
            confusion_matrix[l.item(), preds[i].item()] += 1 
    
    # Compute regular Performance
    epoch_acc = running_corrects / num_samples         
    # Compute Performance Metrics    
    tn, fp, fn, tp = confusion_matrix.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))   
    # Performance:
    PerformanceMetrics = {'accuracy': epoch_acc, 'precision': precision, 'recall': recall, 'f1': f1}

    if showCM == True :
        # plot confusion matrix
        plt.figure()
        plot_confusion_matrix(confusion_matrix,figsize=(12,8),cmap=plt.cm.Blues)
        plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
        plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
        plt.xlabel('Predicted Label',fontsize=18)
        plt.ylabel('True Label',fontsize=18)
        plt.show()

    return PerformanceMetrics


# # Train and Evaluate Model

# ## hyperparameters

# In[ ]:


"""parameters for the model intialization
"""
idx = 0
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet]
#       used when we tested different models.
#       model_name_options = ['efficientnet-b0', 'resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet']
model_name = 'efficientnet-b0'

# Use pretrained model
use_pretrained=True

# Flag for feature extracting. When False, we fine-tune the whole model,
#       when True we only update the reshaped layer params
feature_extract = True

# depth of model for efficientNets
#       used when we tested different versions.
versionLongefficientNets_options = [0,1,2,3,4]
versionLong = 3

"""parameters for model training:
"""
# Number of epochs to train for
num_epochs = 40

# learning rate:
#       used when we tested different learning rates.
#       learning_rate_options = [0.01, 0.001, 0.0001]
learning_rate = 0.001

# Setup the loss fn
#       weighted loss for data class imbalance
class_weight = torch.FloatTensor([3876/(1342+3876), 1342/(1342+3876)])
criterion_ft = nn.CrossEntropyLoss(weight=class_weight).to(device)


# ## Initialize the model

# In[ ]:


"""Initialize the model for this run
"""
model_ft, _ = initialize_model(model_name, use_pretrained, feature_extract, versionLong)

# Print the model we just instantiated
# print(model_ft) # shutdown whhile working
model_ft = model_ft.to(device)


# ## Create the Optimizer

# In[ ]:


"""Create the Optimizer
"""
# list of parameters to be optimized. check out the printed parameters to learn!
params_to_update = gather_params_to_update(model_ft, feature_extract)
# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(params_to_update, lr=learning_rate)
# INSTANTIATE STEP LEARNING SCHEDULER CLASS:
#   lr = lr * factor 
#   factor = decaying factor
#   mode:   if mode='max': look for the maximum validation accuracy to track
#           if mode='min': look for the minimum validation loss to track
#   patience: number of epochs - 1 where loss plateaus before decreasing LR
#           if patience = 0, after 1 bad epoch, reduce LR
factor_ft=0.7
patience_ft=4
scheduler_ft = ReduceLROnPlateau(optimizer_ft, mode='max', factor=factor_ft, patience=patience_ft, verbose=True)


# In[ ]:


# log:
logParametersForTrain = "idx:{} model:{} versionLong:{} lr:{}".format(idx,model_name,versionLong,learning_rate)
print(logParametersForTrain)


# ## Train and evaluate

# In[ ]:


""" ~~~~~~~~~~~~~ Train and Evaluate -Pneumonia ~~~~~~~~~~~~~~~~~~~~~~~~ """
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
# train and evaluate:
model_ft, history_ft = train_model(model_ft, train_dataloader, val_dataloader, criterion_ft, optimizer_ft, scheduler_ft, num_epochs, device)


# ### Save training

# In[ ]:


"""save training
"""
checkpoint_dir = '/content/drive/MyDrive/DL/project_ariel_yael/pneumonia/Checkpoints_Pneumonia'
checkpoint_model_path = checkpoint_dir + '/' + logParametersForTrain + '__ckpt.pth'
checkpoint_hist_path = checkpoint_dir + '/' + logParametersForTrain + '__hist.pth'
print('==> Saving model ...')
print("checkpoint_dir:", checkpoint_dir)
print("checkpoint_model_path:", checkpoint_model_path)
print("checkpoint_hist_path:" , checkpoint_hist_path)

state_save = {
    'net': model_ft.state_dict(),
}
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
torch.save(state_save, checkpoint_model_path)
torch.save(history_ft, checkpoint_hist_path)
print('==> SAVING DONE')


# ### Plot training

# In[ ]:


"""Plot
"""
# Plot Accuracy and Loss 
print(logParametersForTrain)
plot_acc_and_loss_for_train_and_val(history_ft)


# # Test model

# ## Test Data Preparations
# > preparing the test dataset, create test dataloader, and print some information

# In[ ]:


""" preparing the test dataset, create test dataloader, and print some information
""" 
data_dir = '/content/drive/MyDrive/DL/project_ariel_yael/pneumonia/dataset'

pneumonia_files_test = os.listdir(data_dir + "/test" + "/PNEUMONIA")
print('No. of test examples for Pneumonia:', len(pneumonia_files_test))
normal_files_test = os.listdir(data_dir + "/test" + "/NORMAL")
print('No. of test examples for Normal:', len(normal_files_test))

# define pre-processing steps on images
data_transorm_test = tt.Compose([tt.Resize(255),
                              tt.CenterCrop(224),                                                              
                              tt.ToTensor() 
                              ])

# create test dataset:
dataset_test = ImageFolder(data_dir + "/test", data_transorm_test)
print("classes in test dataset:",dataset_test.classes,"\n")
print("test dataset:","\n")
print(dataset_test)

# number of images for each class
test_samplesize = pd.DataFrame.from_dict(
    {'Normal': [len([os.path.join(data_dir+'/test/NORMAL', filename) 
                     for filename in os.listdir(data_dir+'/test/NORMAL')])], 
     'Pneumonia': [len([os.path.join(data_dir+'/test/PNEUMONIA', filename) 
                        for filename in os.listdir(data_dir+'/test/PNEUMONIA')])]})

sns.barplot(data=test_samplesize).set_title('Testing Set Data Inbalance', fontsize=20)
plt.show()


# create dataloaders
batch_size_test=128
test_ds = dataset_test                     # just for consistency with the train,val name
print()
print("number of samples in dataset:")
print("\t", "train_ds:", len(train_ds), "val_ds:", len(val_ds)) 
print("\t", "test_ds:", len(test_ds))

test_dataloader = DataLoader(test_ds, batch_size=batch_size_test, shuffle=False)


# ## test

# In[ ]:


""" ~~~~~~~~~~~~~ Test - Pneumonia ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
# Initialize the model for this run
model_for_test, _ = initialize_model(model_name, use_pretrained, feature_extract, versionLong)
# load weights from the training
model_for_test = model_for_test.to(device)
state_load = torch.load(checkpoint_model_path, map_location=device)
model_for_test.load_state_dict(state_load['net'])
print()

# Test
Performance_test = test_model(model_for_test, test_dataloader, device, True)
print()

# Print statisitcs
print("test accuracy: {:.3f}".format(Performance_test['accuracy']))
print("Recall of the model is {:.3f}".format(Performance_test['recall']))
print("Precision of the model is {:.3f}".format(Performance_test['precision']))
print("F1 Score of the model is {:.3f}".format(Performance_test['f1']))


# ## save test resultes

# In[ ]:


"""save testing
"""
checkpoint_dir = '/content/drive/MyDrive/DL/project_ariel_yael/pneumonia/Checkpoints_Pneumonia'
checkpoint_test_path = checkpoint_dir + '/' + logParametersForTrain  + '__test_performance.pth'
print('==> Saving test results ...')
print("checkpoint_dir:", checkpoint_dir)
print("checkpoint_test_path:", checkpoint_test_path)

torch.save(Performance_test, checkpoint_test_path)


# # COVID

# ## Covid Data preparetions

# In[ ]:


####################################################
# lets look on the data dir:
data_dir_covid = '/content/drive/MyDrive/DL/project_ariel_yael/Covid19/dataset/dataset_covid'

print("\n","directories in data_dir:", os.listdir(data_dir_covid))

classes = os.listdir(data_dir_covid + "/train")
print("classes in data:", classes)

covid_files_train_covid = os.listdir(data_dir_covid + "/train" + "/COVID")
print('No. of training examples for Covid:', len(covid_files_train_covid))
normal_files_train_covid = os.listdir(data_dir_covid + "/train" + "/NORMAL")
print('No. of training examples for Normal:', len(normal_files_train_covid))

covid_files_test_covid = os.listdir(data_dir_covid + "/test" + "/COVID")
print('No. of test examples for Covid:', len(covid_files_test_covid))
normal_files_test_covid = os.listdir(data_dir_covid + "/test" + "/NORMAL")
print('No. of test examples for Normal:', len(normal_files_test_covid))

# lets take a look on the train and how the folder orgenized:
print("\n",'lets take a look on the train/COVID and how the folder orgenized:')
print(covid_files_train_covid[:5])
print("\n",'lets take a look on the train/NORMAL and how the folder orgenized:')
print(normal_files_train_covid[:5])


# ### Covid - create dataset

# In[ ]:


# define pre-processing steps on images
data_transorm_train_covid = tt.Compose([tt.Resize(255),
                              tt.CenterCrop(224),
                              tt.RandomHorizontalFlip(),
                              tt.RandomRotation(10),
                              tt.RandomGrayscale(),
                              tt.RandomAffine(translate=(0.05,0.05), degrees=0),
                              tt.ToTensor()
                              #tt.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] ,inplace=True)
                              ])

data_transorm_test_covid = tt.Compose([tt.Resize(255),
                              tt.CenterCrop(224),                                                              
                              tt.ToTensor() 
                              ])


# In[ ]:


# create train dataset:
dataset_train_covid = ImageFolder(data_dir_covid + "/train", data_transorm_train_covid)
print("classes in train dataset:",dataset_train_covid.classes,"\n")
print("train dataset:","\n")
print(dataset_train_covid)

# create test dataset:
dataset_test_covid = ImageFolder(data_dir_covid + "/test", data_transorm_test_covid)
print("classes in test dataset:",dataset_test_covid.classes,"\n")
print("test dataset:","\n")
print(dataset_test_covid)


# ### Covid - statisics and visualizations

# In[ ]:


# Let's look at an example of an image in its tensor form. Note that after being loaded, 
# each image is now [3, 224, 224], with 3 being the RGB channels



img, label = dataset_train_covid[0]
print(img.shape, label)
img



# In[ ]:


# plot out some images with their labels.


show_example(*dataset_train_covid[4], dataset_train_covid) 


# In[ ]:


# number of images for each class

train_samplesize_covid = pd.DataFrame.from_dict(
    {'Normal': [len([os.path.join(data_dir_covid+'/train/NORMAL', filename) 
                     for filename in os.listdir(data_dir_covid+'/train/NORMAL')])], 
     'Pneumonia': [len([os.path.join(data_dir_covid+'/train/COVID', filename) 
                        for filename in os.listdir(data_dir_covid+'/train/COVID')])]})


sns.barplot(data=train_samplesize_covid).set_title('Covcid - Training Set Data Is Balanced', fontsize=20)
plt.show()


# In[ ]:


# Function for plotting samples
def plot_samples_covid(f_samples):  
    """
    plot out multiple samples.
    Args:   
        f_samples - samples created with random.sample() from data diretory
    """    
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,8))
    for i in range(len(f_samples)):
        image = cv2.cvtColor(imread(f_samples[i]), cv2.COLOR_BGR2RGB)
        ax[i//5][i%5].imshow(image)
        if i<5:
            ax[i//5][i%5].set_title("Normal", fontsize=20)
        else:
            ax[i//5][i%5].set_title("Covid", fontsize=20)
        ax[i//5][i%5].axis('off')


# In[ ]:


## Plot training samples

#### shotdown while working

rand_samples_train_covid = random.sample([os.path.join(data_dir_covid+'/train/NORMAL', filename) 
                              for filename in os.listdir(data_dir_covid+'/train/NORMAL')], 5) + \
    random.sample([os.path.join(data_dir_covid+'/train/COVID', filename) 
                   for filename in os.listdir(data_dir_covid+'/train/COVID')], 5)

plot_samples_covid(rand_samples_train_covid)
plt.suptitle('Covcid - Training Set Samples', fontsize=30)
plt.show()


# ### Covid - split dataset

# In[ ]:


# Split training dataset to train and validation
train_size_covid = round(len(dataset_train_covid)*0.8) # 80%
val_size_covid = len(dataset_train_covid) - train_size_covid # 20%

train_ds_covid, val_ds_covid = random_split(dataset_train_covid, [train_size_covid, val_size_covid])
test_ds_covid = dataset_test_covid                     # just for consistency with the train,val name

print("number of samples in dataset for covid:\n", "train_ds:", len(train_ds_covid), "val_ds:", len(val_ds_covid), "test_ds:", len(test_ds_covid))


# ### Covid - create dataloaders

# In[ ]:


# hyperparameter: batch zise
batch_size_covid=32

# create dataloaders
train_dataloader_covid = DataLoader(train_ds_covid, batch_size_covid, shuffle=True, num_workers=2, pin_memory=True) 
val_dataloader_covid = DataLoader(val_ds_covid, batch_size_covid*2,shuffle=False, num_workers=2, pin_memory=True) 
test_dataloader_covid = DataLoader(test_ds_covid, batch_size=64, shuffle=False)


# ## Covid - device

# In[ ]:


# device - cpu or gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ## Covid - prepare for train

# > Now we will take pretrained model we trained for pneumonia. Then we will do fine-tuning only on the last layers - the layers we trained in the pneomonia model.

# In[ ]:


###
### Take the checkpoint dir and path

###

pretrained_pneuminia_dir = '/content/drive/MyDrive/DL/project_ariel_yael/pneumonia/Checkpoints_Pneumonia'
checkpoint_dir_files = os.listdir(pretrained_pneuminia_dir)
for file_name in checkpoint_dir_files:
  if '__ckpt.pth' in file_name:
    pretrained_path = pretrained_pneuminia_dir + "/" +file_name
    print(file_name)
    print(pretrained_path)
    print()

pretrained_pneominia_checkpoint_path = pretrained_path
preTreined_moedl_name = 'efficientnet-b0'
preTreined_versionLong = 3


# ## Covid - initialize the model

# > The hyperparameters for the model initialization must be the same as the pretrained model

# In[ ]:


""" ~~~~~~~~~~~~~ Set Up - COVID ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
idx = 0
""" parameters for the model intialization """
# the model must be the same model as the pretrained model
model_name_covid = preTreined_moedl_name
# the depth version must be the same depth version as the pretrained model
versionLong_covid = preTreined_versionLong
# Use pretrained model
use_pretrained_covid=True
# we fine-tune only the last layers ( the layers replaced the last fc).
#       when True we only update the reshaped layer params as we need
feature_extract_covid = True

""" parameters for model training - covid:"""
# Number of epochs to train for
num_epochs_covid = 70
# learning rate:
learning_rate_covid = 0.001
# Setup the loss fn
criterion_ft_covid = nn.CrossEntropyLoss()

""" Initialize the model for this run - covid: """
# Initialize the model for this run
model_ft_covid, _ = initialize_model(model_name_covid, use_pretrained_covid, feature_extract_covid, versionLong_covid)   
# load_pretrained weights from pneumonia
model_ft_covid = model_ft_covid.to(device)
state_pretrained_pneominia = torch.load(pretrained_pneominia_checkpoint_path, map_location=device)
model_ft_covid.load_state_dict(state_pretrained_pneominia['net'])
print()

""" Create the Optimizer - covid: """
# list of parameters to be optimized. check out the printed parameters to learn!
params_to_update_covid = gather_params_to_update(model_ft_covid, feature_extract_covid)
# Observe that all parameters are being optimized
optimizer_ft_covid = torch.optim.Adam(params_to_update_covid, lr=learning_rate_covid)
# INSTANTIATE STEP LEARNING SCHEDULER CLASS:
scheduler_ft_covid = ReduceLROnPlateau(optimizer_ft_covid, mode='min', factor=0.5, patience=4, verbose=True)
# log:
logParametersForTrain_covid = "idx:{} model:{} versionLong:{} lr:{}".format(idx,model_name_covid,versionLong_covid,learning_rate_covid)
print(logParametersForTrain_covid)
print()


# 
# ## Covid - train & evaluate

# In[ ]:


""" ~~~~~~~~~~~~~ Train and Evaluate - COVID ~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
# train and evaluate:
model_ft_covid, history_ft_covid = train_model(model_ft_covid, train_dataloader_covid, val_dataloader_covid, criterion_ft_covid, optimizer_ft_covid, scheduler_ft_covid,num_epochs_covid, device)
print()



# In[ ]:


""" save training - covid:"""
checkpoint_dir_covid = '/content/drive/MyDrive/DL/project_ariel_yael/Covid19/Checkpoints_Covid'

checkpoint_model_path_covid = checkpoint_dir_covid + '/' + logParametersForTrain_covid  + '__ckpt.pth'
checkpoint_hist_path_covid = checkpoint_dir_covid + '/' + logParametersForTrain_covid + '__hist.pth'
print('==> Saving model covid...')
print("checkpoint_dir_covid:", checkpoint_dir_covid)
print("checkpoint_model_path_covid:", checkpoint_model_path_covid)
print("checkpoint_hist_path_covid:" , checkpoint_hist_path_covid)
state_save_covid = {
    'net': model_ft_covid.state_dict(),
}
if not os.path.isdir(checkpoint_dir_covid):
    os.mkdir(checkpoint_dir_covid)
torch.save(state_save_covid, checkpoint_model_path_covid)
torch.save(history_ft_covid, checkpoint_hist_path_covid)
print('==> SAVING DONE')
print()
""" Plot - covid: """
# Plot Accuracy and Loss 
print(logParametersForTrain_covid)
plot_acc_and_loss_for_train_and_val(history_ft_covid, 'Covid Classification')
print()


# ## Covid - test 

# In[ ]:


@torch.no_grad()
def test_model_Covid(f_model, f_dataloader, f_device, showCM = False):
    """
    NOTE: use @torch.no_grad() as a decorator
    Args: 
        f_model - the model we test 
        f_dataloader - dataloader for test dataset
        f_device - device to use
        showCM - choose to plot confusion_matrix, default False
    RETURN: PerformanceMetrics = {'accuracy','precision', 'recall', 'f1'}
    """
    # model.eval() set specific layers like dropout and batchnorm to evaluation mode 
    #       while model.train() will enable them.  
    f_model.eval() # put in evaluation mode 

    num_classes = 2
    num_samples = len(f_dataloader.dataset)
    # print("DEBUG:test_model->number_of_samples= ", num_samples)
    running_corrects = 0.0                                  
    confusion_matrix = np.zeros([num_classes,num_classes], int)
    # Iterate over data.
    for inputs, labels in f_dataloader:       # get the data
        inputs = inputs.to(f_device)          # send to device
        labels = labels.to(f_device)          # send to device  
        # forward
        outputs = f_model(inputs)                    
        _, preds = torch.max(outputs.data, 1)    
        # collect statistics
        running_corrects += (preds == labels).sum().item() 
        for i, l in enumerate(labels):
            confusion_matrix[l.item(), preds[i].item()] += 1 
    
    # Compute regular Performance
    epoch_acc = running_corrects / num_samples         
    # Compute Performance Metrics    
    tn, fp, fn, tp = confusion_matrix.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))   
    # Performance:
    PerformanceMetrics = {'accuracy': epoch_acc, 'precision': precision, 'recall': recall, 'f1': f1}

    if showCM == True :
        # plot confusion matrix
        plt.figure()
        plot_confusion_matrix(confusion_matrix,figsize=(12,8),cmap=plt.cm.Blues)
        plt.xticks(range(2), ['Normal', 'Covid'], fontsize=16)
        plt.yticks(range(2), ['Normal', 'Covid'], fontsize=16)
        plt.xlabel('Predicted Label',fontsize=18)
        plt.ylabel('True Label',fontsize=18)
        plt.show()

    return PerformanceMetrics


# ###covid test - plot and performance

# In[ ]:


""" ~~~~~~~~~~~~~ Test - COVID ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
# Initialize the model for this run
model_for_test_covid, _ = initialize_model(model_name_covid, use_pretrained_covid, feature_extract_covid, versionLong_covid)
print()
model_for_test_covid = model_for_test_covid.to(device)
state_load_covid = torch.load(checkpoint_model_path_covid, map_location=device)
model_for_test_covid.load_state_dict(state_load_covid['net'])
print()
# Test
Performance_test_covid = test_model_Covid(model_for_test_covid, test_dataloader_covid, device, True)
print()


# In[ ]:


# Print statisitcs
print("test accuracy: {:.3f}".format(Performance_test_covid['accuracy']))
print("Recall of the model is {:.3f}".format(Performance_test_covid['recall']))
print("Precision of the model is {:.3f}".format(Performance_test_covid['precision']))
print("F1 Score of the model is {:.3f}".format(Performance_test_covid['f1']))
print()

""" save testing """
checkpoint_dir_covid = '/content/drive/MyDrive/DL/project_ariel_yael/Covid19/Checkpoints_Covid'
checkpoint_test_path_covid = checkpoint_dir_covid + '/' + logParametersForTrain_covid  + '__test_performance.pth'
print('==> Saving test results ...')
print("checkpoint_dir_covid:", checkpoint_dir_covid)
print("checkpoint_test_path_covid:", checkpoint_test_path_covid)
torch.save(Performance_test_covid, checkpoint_test_path_covid)
print('==> SAVING DONE')
print()

