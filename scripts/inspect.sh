# !/bin/bash

cd /root/autodl-pvt/BeaconNet
python -m src.activation.inspect \
  -m models/llama3.1-8B-Instruct \
  -a models/aligned/llama3.1-8B-Instruct-dpo
