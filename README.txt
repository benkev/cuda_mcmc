To start using CUDA MCMC software, first enter this directory and issue the
command

$ make

It will create the directories ~/lib64/python, and ~/bin,  and setup the
environment for them by adding to the end of the ~/.bashrc file the commands

export PATH=$PATH:~/bin
export PYTHONPATH=$PYTHONPATH:~/lib64/python

After make finishes, please issue the command

$ source ~/.bashrc

Or simply open a new console window.


In ./doc/ directory:
gpu_mcmc_soft_r.*        vector of parameters is 'r'
gpu_mcmc_soft_th.*       vector of parameters is '\theta'
gpu_mcmc_soft.*          vector of parameters is 'p'

gpu_mcmc_soft_r.tex    is being worked on. Others are abandoned (so far).



