#!/bin/bash
set -ex

# install pyflakes to do code error checking
echo "pip3 install pyflakes --cache-dir $HOME/.pip-cache"
pip3 install pyflakes --cache-dir $HOME/.pip-cache

# Conda Installation
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

echo "conda create -n fedml python=3.7.4"
conda create -n fedml python=3.7.4

echo "conda activate fedml"
conda activate fedml

# Install PyTorch (please visit pytorch.org to check your version according to your physical machines
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install pytorch torchvision cudatoolkit=11.2 -c pytorch
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html(这个就可以)

# Install MPI
conda install -c anaconda mpi4py

conda install wandb

# Install other required package

conda install numpy
conda install h5py
conda install setproctitle
conda install networkx
conda install pandas
conda install matplotlib
conda install tqdm
pip install sklearn
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchcluster
pip install adder
pip install -r requirements.txt

test:
自动查杀僵死进程指令:
ps -A -o stat,ppid,pid,cmd | grep -e '^[Zz]' | awk '{print $2}' | xargs kill -9
查看指定进程
ps -ef |grep Zeno
conda activate fedml
cd Alvin/fedml/
cd Alvin/fedml/fedml_experiments/standalone/fedavg
sh run_Fisher_standalone_pytorch.sh 0 200 200 20 5 mnist ./../../../data/mnist lr homo 120 1 0.01 sgd 0 fedavg ABCD 1 0.02 3 model_attack arg_and_update
sh run_Fisher_standalone_pytorch.sh 2 60 60 10 10 cifar10 ./../../../data/cifar10 resnet20 homo 100 1 0.2 sgd 0 fedbt BCD 10 0.01 0.02 model_attack arg_and_update
sh run_Fisher_standalone_pytorch.sh 0 60 60 6 5 femnist ./../../../data/FederatedEMNIST/datasets cnn hetero 60 3 0.05 sgd 0 fedbt ABCD 5 0.0 0.0025 model_attack arg_and_update
sh run_Fisher_standalone_pytorch.sh 1 60 60 6 120 shakespeare ./../../../data/shakespeare rnn hetero 500 1 0.8 sgd 0 fedbt ABCD 120 0.00 0.004 model_attack arg_and_update
# install the dataset
# 1. MNIST
cd ./data/MNIST
sh download_and_unzip.sh
cd ../../

# 2. FederatedEMNIST
cd ./data/FederatedEMNIST
sh download_federatedEMNIST.sh
cd ../../

# 3. shakespeare
cd ./data/shakespeare
sh download_shakespeare.sh
cd ../../


# 4. fed_shakespeare
cd ./data/fed_shakespeare
sh download_shakespeare.sh
cd ../../

# 5. fed_cifar100
cd ./data/fed_cifar100
sh download_fedcifar100.sh
cd ../../

# 6. stackoverflow
# cd ./data/stackoverflow
# sh download_stackoverflow.sh
# cd ../../

# 7. CIFAR10
cd ./data/cifar10
sh download_cifar10.sh
cd ../../

# 8. CIFAR100
cd ./data/cifar100
sh download_cifar100.sh
cd ../../

# 9. CINIC10
cd ./data/cinic10
sh download_cinic10.sh > cinic10_downloading_log.txt
cd ../../
