# COVID-19-Detection
In this project we depeloped a  Convolutional Neural Network algorithm to predict Covid-19 from chest X-Ray images. First we trained the model to predict Pneumonia from Pneumonia chest X-Ray dataset, and then fine-tuned the model on the Covid data.


- [COVID-19-Detection](#COVID-19-Detection)
  * [Background](#background)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [API (`ls_dqn_main.py --help`)](#api---ls-dqn-mainpy---help--)
  * [Playing](#playing)
  * [Training](#training)
  * [Playing Atari on Windows](#playing-atari-on-windows)
  * [TensorBoard](#tensorboard)
  * [References](#references)

## Background
This project goal: Given an input chest X-ray image, the algorithm must detect whether the person has been infected with Covid-19 or not.
The first idea was to use Transfer learning and levarage the already existing labeled data of some related task or domain, in our case image classification.
We notinced that unlike availability of a small amount of X-ray images of COVID-19 patients, we found bigger dataset of X-ray images of Pneumonia patients.
Hence we decided to devide our mission into 2 parts:

* The first part 
  * Given an input chest X-ray image, the algorithm must detect whether the person has been infected with Pneumonia or not.
The algorothm for this part use transfer learning. We take some familier ConvNet and use it as fixed feature extractor. we freeze the weights for all of the network except that of the final fully connected layer. This last fully connected layer is replaced with a sevral of new ones, with random weights and only the new layers are trained.
 
* The second part: 
  * Given an input chest X-ray image, the algorithm must detect whether the person has been infected with Covid-19 or not.
The algorithm is a form of fine-tuning method. We take the same model we developed in the first part as a preatrained net. instead of random initializaion, we initialize the network with the weights trained in the first part. next we will do fine-tuning not on the whole Net, but on the layers we added at the first part.


## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`Creating_the_Covid_19_Dataset.ipynb`| code for fetchin only PA X-ray images from the covid dataset|
|`pong_ls_dqn.py`| main application tailored for Atari's Pong|
|`boxing_ls_dqn.py`| main application tailored for Atari's Boxing|
|`dqn_play.py`| sample code for playing a game, also in `ls_dqn_main.py`|
|`actions.py`| classes for actions selection (argmax, epsilon greedy)|
|`agent.py`| agent class, holds the network, action selector and current state|
|`dqn_model.py`| DQN classes, neural networks structures|
|`experience.py`| Replay Buffer classes|
|`hyperparameters.py`| hyperparameters for several Atari games, used as a baseline|
|`srl_algorithms.py`| Shallow RL algorithms, LS-UPDATE|
|`utils.py`| utility functions|
|`wrappers.py`| DeepMind's wrappers for the Atari environments|
|`*.pth`| Checkpoint files for the Agents (playing/continual learning)|
|`Deep_RL_Shallow_Updates_for_Deep_Reinforcement_Learning.pdf`| Writeup - theory and results|


## API (`ls_dqn_main.py --help`)


## Changing the model
there is an option to change the depth of the classifier (version 0,1,2,3,4) and you can add more if you want.
you can also play with the pretrained model. we choose efficientNet, but you can try other.


## References
* Pneumonia Dataset source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
* Cobid-19 Dataset source: https://www.kaggle.com/bachrr/covid-chest-xray
  * Note: we wanted to take only PA images, and also to split the data to train/test folders (in order to use inagefolder). 
    * we created the dataset using the code in the file: 
    * we split manualy and in random way the dataset into train and test.


![image](https://user-images.githubusercontent.com/65540180/124578077-8c758d80-de56-11eb-949d-7c68d9293f9e.png)

