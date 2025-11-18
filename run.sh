#!/bin/bash 

# Assume running on Ubuntu with bash shell
#pip install -r requirements.txt
#get requirement from system
#pip freeze > requirements.txt

# Create virtual environment
python3 -m venv my_env

# Activate it
source my_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Stat the main program
python3 src/main.py
