# Gunicorn configuration file
import multiprocessing

workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'uvicorn.workers.UvicornWorker'
bind = '0.0.0.0:8000'
keepalive = 120
timeout = 120
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
