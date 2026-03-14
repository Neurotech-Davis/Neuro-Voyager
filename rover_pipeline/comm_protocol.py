"""Phase 3: Communication Protocol with state management."""

import serial
import time
from config import TARGETS, SERIAL_PORT, SERIAL_BAUD

class MotorController:
    """Manages serial communication with the Arduino."""
    
    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD):
        self.port = port
        self.baud = baud
        self.serial_conn = None
        self.current_state = None
        
    def connect(self):
        """Try to open serial port connection."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=1)
            print(f"Connected to serial port: {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Serial Error: {e}")
            return False
            
    def disconnect(self):
        """Close serial port."""
        if self.serial_conn:
            self.serial_conn.close()
            print("Serial connection closed.")
            
    def send_command(self, freq):
        """
        Sends the mapped character byte to Arduino only if state has changed.
        
        Args:
            freq (float): Winning frequency from SSVEP classification.
        """
        command = TARGETS.get(freq, 'S') # Default to Stop
        
        # State Management: Only send if state changes
        if command != self.current_state:
            print(f"State Change: {self.current_state} -> {command} (Freq: {freq} Hz)")
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    self.serial_conn.write(command.encode('utf-8'))
                    self.current_state = command
                except Exception as e:
                    print(f"Failed to send command: {e}")
            else:
                # Still update current_state even if not connected for local testing
                self.current_state = command
                print("(Virtual/Disconnected) Command sent.")
        else:
            # Command already sent, no need to flood
            pass
