# How to Run a Django Project

## Prerequisites
- Python 3.x
- pip (Python package installer)
- virtualenv

## Steps

### 1. Create a Virtual Environment
First, navigate to your project directory and create a virtual environment:
```bash
cd /d:/document/Hackaton/MLH GWH/GWH AI
python -m venv env
```

### 2. Activate the Virtual Environment
Activate the virtual environment:
- On Windows:
    ```bash
    .\env\Scripts\activate
    ```
- On macOS/Linux:
    ```bash
    source env/bin/activate
    ```

### 3. Install Requirements
Create a `requirements.txt` file with the following content:
```
django
python-dotenv
```
Install the requirements using pip:
```bash
pip install -r requirements.txt
```

### 4. Create a New Django Project
Create a new Django project:
```bash
django-admin startproject myproject
cd myproject
```

### 5. Set Up Environment Variables
Create a `.env` file in the root of your project and add your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

### 6. Load Environment Variables in Django
Modify `myproject/settings.py` to load the environment variables:
```python
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
```

### 7. Run Migrations
Apply migrations to set up your database:
```bash
python manage.py migrate
```

### 8. Run the Development Server
Start the Django development server:
```bash
python manage.py runserver
```

Your Django project should now be running at `http://127.0.0.1:8000/`.
