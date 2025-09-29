#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Ensure gunicorn is installed
pip install gunicorn==21.2.0 uvicorn[standard]==0.30.6

# Create a wrapper script for gunicorn
echo '#!/bin/bash
python -m gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:10000' > /opt/render/project/backend/start.sh
chmod +x /opt/render/project/backend/start.sh
