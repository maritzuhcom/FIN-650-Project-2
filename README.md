# Financial Analysis Project (FIN 650)

This project contains financial analysis tools and scripts using Python with various data science libraries.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Setup Instructions

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd fin_650

# Or simply navigate to the project directory
cd /path/to/fin_650
```

### 2. Create and Activate Virtual Environment

The project already includes a virtual environment. To activate it:

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
# Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

This will install:
- pandas
- numpy
- yfinance
- statsmodels
- matplotlib

### 4. Run the Main Script

To run the main.py file:

```bash
python main.py
```

## Project Structure

```
fin_650/
├── venv/                 # Virtual environment directory
├── requirements.txt      # Python dependencies
├── main.py              # Main script to run
└── README.md            # This file
```

## Virtual Environment Commands

### Activate Virtual Environment
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Check if Virtual Environment is Active
When the virtual environment is active, you should see `(venv)` at the beginning of your command prompt.

### Install New Packages
```bash
# Install a new package
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt
```

## Troubleshooting

### If you get "command not found" errors:
- Make sure you're in the correct directory
- Ensure the virtual environment is activated
- Check that Python is installed correctly

### If you get import errors:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify the virtual environment is activated

### If the virtual environment doesn't exist:
Create a new one:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Notes

- Always activate the virtual environment before running any Python scripts
- The virtual environment isolates this project's dependencies from your system Python
- Remember to deactivate the virtual environment when you're done working on the project
