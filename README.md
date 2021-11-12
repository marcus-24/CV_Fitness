# CV_Fitness

## Objective
The purpose of this project is to use pre-trained computer vision-based models to generate joint kinematics during a lifting workout. Then use a physics model to estimate the joint kinetics (muscle torque).

## Setup
This section focuses on setting up the libraries and downloading pre-trained models to run this code. All of these commands need to be executed in the `CV_Fitness` folder. 

### Installing Python Environment
To install the python environment for this project, use the following command (in command prompt for Windows and bash terminal for Linux):

`conda env create -f environment.yml`

### Downloading pre-trained Model Parameters

The pre-trained computer vision model can be installed from the Carneige Mellon University (CMU) page using one of the commands below:

#### For Windows (in command prompt)

`getModels.bat`

#### For Linux (in bash terminal)

`sudo chmod a+x getModels.sh`

`./getModels.sh`


