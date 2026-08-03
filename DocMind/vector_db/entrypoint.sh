#!/bin/sh
set -e

cleanup() {
    echo "Termination signal received — deleting docmind-index..."
    python delete_index.py
    exit 0
}
trap cleanup TERM INT

python create_index.py

echo "docmind-index ready. Holding container open to catch shutdown signal..."
sleep infinity &
wait $!
