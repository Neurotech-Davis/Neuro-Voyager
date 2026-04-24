"""Serial port utilities for Arduino communication."""

import glob
import sys
import time
import serial
import serial.tools.list_ports


def find_usb_ports():
    """Auto-detect USB serial ports. Works on macOS, Linux, and Windows."""
    if sys.platform == 'win32':
        # Windows: use pyserial's port enumeration (COM ports)
        return [p.device for p in serial.tools.list_ports.comports()
                if p.vid is not None]  # filter to USB devices (have vendor ID)
    elif sys.platform == 'linux':
        # Linux: /dev/ttyUSB* (FTDI/CH340) and /dev/ttyACM* (native USB)
        patterns = ['/dev/ttyUSB*', '/dev/ttyACM*']
        ports = []
        for pat in patterns:
            ports.extend(glob.glob(pat))
        return sorted(set(ports))
    else:
        # macOS
        patterns = ['/dev/tty.usb*', '/dev/cu.usb*']
        ports = []
        for pat in patterns:
            ports.extend(glob.glob(pat))
        return sorted(set(ports))


class ArduinoController:
    """Serial communication with Arduino. Sends single-char commands."""

    def __init__(self, baud=9600):
        self.baud = baud
        self.conn = None
        self.current_state = None

    @property
    def connected(self):
        return self.conn is not None and self.conn.is_open

    def connect(self, port):
        """Open serial connection. Returns (success, message).

        Waits 2s after opening for Arduino to reset (DTR toggles reset).
        """
        try:
            self.conn = serial.Serial(port, self.baud, timeout=1)
            time.sleep(2)  # Arduino resets on serial open, wait for bootloader
            self.conn.reset_input_buffer()
            return True, f"Connected to {port}"
        except serial.SerialException as e:
            self.conn = None
            return False, str(e)

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None
        self.current_state = None

    def send(self, command_char, force=False):
        """Send single character command. Only sends on state change unless force=True.

        Returns: (sent, message)
        """
        if not force and command_char == self.current_state:
            return False, f"Already in state {command_char}"

        self.current_state = command_char
        if self.connected:
            try:
                self.conn.write(command_char.encode('utf-8'))
                return True, f"Sent: {command_char}"
            except Exception as e:
                return False, f"Send error: {e}"
        return True, f"[No serial] {command_char}"

    def read_response(self, timeout=1.0):
        """Read any available response from Arduino.

        The Arduino firmware echoes "I received: <char>" for each command.
        Returns the response string, or empty string if nothing available.
        """
        if not self.connected:
            return ""
        self.conn.timeout = timeout
        try:
            lines = []
            while self.conn.in_waiting > 0 or not lines:
                line = self.conn.readline().decode('utf-8', errors='replace').strip()
                if line:
                    lines.append(line)
                else:
                    break
            return "\n".join(lines)
        except Exception:
            return ""

    def test_connection(self):
        """Send stop command and check connection is alive.

        Verifies port is open and writable. If Arduino firmware echoes back
        ("I received: ..."), reports that too.

        Returns: (success, message)
        """
        if not self.connected:
            return False, "Not connected"

        try:
            self.conn.reset_input_buffer()
            self.conn.write(b'e')  # stop command
            time.sleep(0.3)

            response = self.read_response(timeout=0.5)
            if "I received" in response:
                return True, f"Verified — Arduino echoed: {response}"
            elif response:
                return True, f"Connected, Arduino sent: {response}"
            else:
                return True, "Connected — stop command sent (no echo-back from firmware)"
        except Exception as e:
            return False, f"Write failed: {e}"
