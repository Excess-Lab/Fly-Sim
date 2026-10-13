#!/bin/sh
# Start the web app at once; the public connectome data (~1.2 GB) downloads in the background on
# first start (progress: data/download.log, also shown on the home page).
cd /app
mkdir -p data
( scripts/download_data.sh > data/download.log 2>&1 && echo "DATA READY" >> data/download.log ) &
exec python -m flysim.live --host 0.0.0.0 --port 8800
