#pip install -r requirements.txt
#get requirement from system
#pip freeze > requirements.txt

# 1. Create virtual environment
python3 -m venv my_env

# 2. Activate it
source my_env/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Work on your project
python3 src/main.py
