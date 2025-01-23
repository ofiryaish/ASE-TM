pip install -r requirements.txt
cd mamba_install
pip install .
pip install rir-generator
pip install pyroomacoustics


tensorboard --logdir=data/model --port=8080
nohup python train.py &> train.log &