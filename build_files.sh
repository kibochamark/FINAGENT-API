#!/usr/bin/env bash

echo "Building project packages ...."
python3 -m pip install -r requirement.txt



echo "Collecting static files..."
python3 manage.py collectstatic --noinput