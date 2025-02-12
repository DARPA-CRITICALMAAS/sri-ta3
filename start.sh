#!/bin/bash

### Description: Start script for the SRI MAPER server that handles initialization and cleanup of server and its connection to CDR
###

# Start a new tmux session named 'server' and run the Python script
tmux new-session -d -s server "python /workspace/sri_maper/src/server.py"

# Function to send Ctrl+C to the tmux session when the container is stopped
function handle_sigterm {
    echo "Received SIGTERM. Sending Ctrl+C to tmux session..."
    tmux send-keys -t server C-c
}

# Trap the SIGTERM signal and call the handle_sigterm function
trap handle_sigterm SIGTERM

# Keep the container running so the tmux session persists
exec tail -f /dev/null
