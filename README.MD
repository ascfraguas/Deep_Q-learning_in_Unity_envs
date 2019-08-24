ENVIRONMENT SETUP

To set up your environment to run this project, follow the below commands from your Anaconda prompt - assuming
you are running on a Windows 10 machine with Anaconda already installed.

**NOTE**: Version conflicts found when pip installing from available wheels created issues when pip installing
the environment from a requirements.txt file. Below commands solve this issue.

conda create -n banana_env python==3.6
activate banana_env 
conda install pytorch=1.0.1
pip install unityagents==0.4.0

After this you can launch your Jupyter kernel with `jupyter notebook`, and ready to go!


