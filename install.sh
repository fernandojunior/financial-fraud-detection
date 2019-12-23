#!/bin/bash
 
# installing java 
add-apt-repository ppa:linuxuprising/java
sudo apt-get update
sudo apt-get install oracle-java13-installer
sudo apt-get install oracle-java13-set-default
java -version

# installing anaconda 
mkdir anacondainstaller
cd anacondainstaller
wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
chmod +x Anaconda3-5.1.0-Linux-x86_64.sh
./Anaconda3-5.1.0-Linux-x86_64.sh
echo ". /home/ubuntu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
conda activate
python --version

# installing spark
cd ~
wget https://www-eu.apache.org/dist/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz
mkdir ~/spark
tar -xvf spark-2.4.4-bin-hadoop2.7.tgz -C ~/spark
echo 'export SPARK_HOME=~/spark/spark-2.4.4-bin-hadoop2.7/' >> ~/.bashrc
echo 'export PATH=$PATH:~/spark/spark-2.4.4-bin-hadoop2.7/bin' >> ~/.bashrc
source ~/.bashrc

# installing pip
conda activate
sudo apt install python3-pip
pip install --upgrade pip
pip install -r requirements.txt
pyspark
