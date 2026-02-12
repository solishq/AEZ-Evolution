#!/bin/bash
# Run the AEZ Evolution server

cd "$(dirname "$0")"

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run server
echo "Starting AEZ Evolution API on http://localhost:8000"
python3 -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
