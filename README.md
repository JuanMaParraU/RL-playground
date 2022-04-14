# RL-playground
Tests with different RL approaches to solve some enviroments from [OpenAI-gym](https://gym.openai.com/). 

1. Check python version: 
```python 
python --version  
```
2. Download anaconda (check python 2 or 3 and select the appropriate Anaconda 2 or 3): 
```python 
cd /tmp 

curl -O https://repo.anaconda.com/archive/AnacondaX-2019.03-Linux-x86_64.sh 

bash AnacondaX-2019.03-Linux-x86_64.sh 
```
3. Check installation:   
```python
 conda -h
```
 
4. Create and activate anaconda environment (“ML_env”) the environment should be created with Python 3.X as interpreter: 
```python
conda create --name ML_env python=3.X 

conda activate ML_env 
```
5. Install numpy 
```python 
conda install -c anaconda numpy 
```
6. Install pandas 
```python
conda install -c anaconda pandas 
```
7. Install torch 
```python
conda install -c pytorch pytorch 
```
8. Install matplotlib 
```python 
conda install -c conda-forge matplotlib 
```
9. Install paho-mqtt. Check docmuentation about MQTT [here](https://www.eclipse.org/paho/index.php?page=clients/python/docs/index.php)
```python 
conda install -c conda-forge paho-mqtt
```
For GPU
11. Check your device specs
```python
nvidia-smi
```
12. Install the corresponding [GPU drivers](https://askubuntu.com/questions/1362970/problem-installing-nvidia-driver-on-ubuntu-20-04)
13. Install tensorFlow
```python
conda install -c anaconda tensorflow-gpu
```
14. (Optional) upgrade your cudatoolkit and cudnn if needed. For Nvidia SMI 470 and Cuda version 11.4: 
```python
conda install cudatoolkit=11.3.1 -c conda-forge
conda install -c conda-forge cudnn=8.2.*
```
