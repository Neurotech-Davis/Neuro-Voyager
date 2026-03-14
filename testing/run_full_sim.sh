#!/bin/bash
# run_full_sim.sh
# Automates the virtual serial connection and runs the BCI pipeline.

# Get the project root (one level up from this script)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# 1. Start socat to create virtual serial ports
socat -d -d pty,raw,echo=0 pty,raw,echo=0 2> testing/socat_output.txt &
SOCAT_PID=$!
sleep 2

# 2. Extract port names from socat output
PORT1=$(grep -o "/dev/ttys[0-9]*" testing/socat_output.txt | head -n 1)
PORT2=$(grep -o "/dev/ttys[0-9]*" testing/socat_output.txt | tail -n 1)

if [ -z "$PORT1" ] || [ -z "$PORT2" ]; then
    echo "Failed to create virtual ports. Make sure 'socat' is installed."
    kill $SOCAT_PID
    exit 1
fi

echo "Virtual Ports Created: $PORT1 <-> $PORT2"

PYTHON_CMD="/opt/homebrew/Caskroom/miniconda/base/envs/NeuralVoyager/bin/python"

# 3. Start Serial Simulator on PORT2
PYTHONUNBUFFERED=1 $PYTHON_CMD testing/serial_simulator.py $PORT2 &
SIM_PID=$!

# 4. Start SSVEP Pipeline on PORT1
echo "Starting Pipeline simulation..."
# Now we are in root, so main_ssvep.py can find data/Trial1.xdf
PYTHONUNBUFFERED=1 PYTHONPATH=rover_pipeline $PYTHON_CMD rover_pipeline/main_ssvep.py $PORT1

# 5. Cleanup
echo "Cleaning up..."
kill $SIM_PID $SOCAT_PID 2>/dev/null
echo "Simulation Finished."
