# Odor Analysis
This repository consist of the data analysis done for Odor Tracking experiment. All the data is stored locally in the home directory in a folder named **~/data** .

## Setup Environment

Install <a href = "https://docs.python-guide.org/dev/virtualenvs/"> Virtualenv </a>: ```pip install virtualenv```<br/>

Install Anaconda from <a href = "https://docs.anaconda.com/anaconda/install/linux/">here. </a>



1. Create a Conda Environment:  

   ```bash
   conda create -n FlyDataAnalysis python=3.6  
   ```
2. Create the virtualenv:

    ```
   virtualenv -p /usr/bin/python3.6 dataEnv  
   ```
  
3. Install Packages:

   ```
   pip install pandas
   pip install h5py
   pip install numpy
   pip install matplotlib
   pip install tables
   conda install --channel conda-forge cartopy  
   ``` 

4. To Install Jupyter Dark Theme (optional - better not to):

   ```bash
   conda install -c conda-forge jupyterthemes 
   ```


### Bash_Aliases Setup
```bash
alias venv="source dataEnv/bin/activate"
alias denv="deactivate"
alias condaenv="conda activate FlyDataAnalysis"
alias dconda="conda deactivate"
alias jread="jupyter notebook"
alias rosenv="source rosenv/bin/activate"
alias start="venv && condaenv"
alias stop="denv && dconda"
```