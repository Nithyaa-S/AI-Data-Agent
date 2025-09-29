#!/bin/bash
set -e

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize database
echo "Initializing database..."
python -c "from app.db import get_engine, init_db; engine = get_engine(); init_db(engine)"

# Run database migrations if using Alembic
# Uncomment if you set up Alembic
# echo "Running database migrations..."
# alembic upgrade head

# Start the application
echo "Starting application..."
exec gunicorn -c gunicorn.conf.py app.main:app
