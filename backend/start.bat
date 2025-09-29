@echo off

:: Install dependencies
pip install -r requirements.txt

:: Run migrations or any other setup
python -c "from app.db import init_db; init_db()"

:: Start the application
gunicorn -c gunicorn.conf.py app.main:app
