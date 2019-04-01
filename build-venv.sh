#!/usr/bin/env bash
python3 -m venv venv
echo "alias dcgan='./dcgan'" >> venv/bin/activate
echo "alias data='./dcgan data'" >> venv/bin/activate
echo "alias generate='./dcgan generate'" >> venv/bin/activate
echo "alias train='./dcgan train'" >> venv/bin/activate
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt