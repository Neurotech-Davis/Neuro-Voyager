"""Standalone Arduino connection test.

Finds USB ports, connects, sends each command, verifies echo-back.
Run: python testing/test_arduino_connection.py [port]
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.serial_utils import ArduinoController, find_usb_ports


def main():
    # Find or use specified port
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        ports = find_usb_ports()
        if not ports:
            print("No USB serial ports found.")
            print("Check: Arduino plugged in? Drivers installed?")
            print("Expected: /dev/tty.usb* or /dev/cu.usb*")
            return
        print(f"Found USB ports: {ports}")
        port = ports[0]

    print(f"\nConnecting to {port} at 9600 baud...")
    ctrl = ArduinoController()
    ok, msg = ctrl.connect(port)
    print(f"  {msg}")
    if not ok:
        return

    # Test each command
    commands = [
        ('e', 'Stop'),
        ('w', 'Forward'),
        ('a', 'Left'),
        ('d', 'Right'),
        ('s', 'Backward'),
        ('e', 'Stop'),
    ]

    print(f"\nSending test commands:")
    print(f"{'Command':>10} {'Name':>10} {'Write':>7}  Arduino Response")
    print("-" * 60)

    any_echo = False
    write_ok = True
    for cmd, name in commands:
        ctrl.conn.reset_input_buffer()
        try:
            ctrl.conn.write(cmd.encode('utf-8'))
            w_status = "OK"
        except Exception as e:
            w_status = "FAIL"
            write_ok = False

        time.sleep(0.3)
        resp = ctrl.read_response(timeout=0.5)
        if resp:
            any_echo = True
        resp_str = resp if resp else "(no echo)"

        print(f"{cmd:>10} {name:>10}   {w_status:<7} {resp_str}")
        time.sleep(0.5)

    ctrl.disconnect()

    print(f"\n{'='*60}")
    if write_ok:
        print("ALL COMMANDS SENT SUCCESSFULLY — serial write OK")
        if any_echo:
            print("Arduino firmware echoed back responses")
        else:
            print("No echo-back (firmware may not have Serial.print)")
            print("Commands still sent — rover should respond if wired")
    else:
        print("WRITE FAILURES — check port and connection")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
