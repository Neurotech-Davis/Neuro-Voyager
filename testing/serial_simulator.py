"""Phase 4: Serial Simulator (The C++ Receiver equivalent in Python)."""

import serial
import sys
import time

def run_simulator(port):
    """
    Listens to the virtual serial port and prints motor commands.
    """
    try:
        ser = serial.Serial(port, 9600, timeout=1)
        print(f"Simulator listening on {port}...")
        
        while True:
            if ser.in_waiting > 0:
                data = ser.read().decode('utf-8')
                if data == 'F':
                    print(">>> Motor State: DRIVING FORWARD")
                elif data == 'L':
                    print(">>> Motor State: TURNING LEFT")
                elif data == 'R':
                    print(">>> Motor State: TURNING RIGHT")
                elif data == 'B':
                    print(">>> Motor State: DRIVING BACKWARD")
                elif data == 'S':
                    print(">>> Motor State: STOPPED")
                else:
                    print(f">>> Received Unknown Command: {data}")
            time.sleep(0.1)
    except serial.SerialException as e:
        print(f"Simulator Error: {e}")
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python serial_simulator.py <port>")
        sys.exit(1)
    run_simulator(sys.argv[1])
