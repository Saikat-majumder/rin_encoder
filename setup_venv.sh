# Step 1: Create the virtual environment named '.venv'
python3 -m venv .venv

# Step 2: Activate the virtual environment
# (Change based on OS: use '.venv\Scripts\activate' on Windows CMD)
source .venv/bin/activate

# Step 3: Upgrade pip to ensure smooth installations
pip install --upgrade pip

# Step 4: Install all dependencies listed in requirements.txt
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "All dependencies installed successfully!"
else
    echo "Error: requirements.txt not found in this directory."
fi
