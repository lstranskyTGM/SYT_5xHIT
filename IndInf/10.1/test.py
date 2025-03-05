#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

# Use BCM numbering for GPIO
GPIO.setmode(GPIO.BCM)

# Define pins based on your configuration
ENA = 17
FORWARD = 15
BACKWARD = 18

# Setup pins as outputs
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(FORWARD, GPIO.OUT)
GPIO.setup(BACKWARD, GPIO.OUT)

# Enable the motor driver and set the motor to run forward continuously.
GPIO.output(ENA, True)      # Enable the driver
GPIO.output(FORWARD, True)  # Set motor forward
GPIO.output(BACKWARD, False)  # Ensure reverse is off

print("Motor running forward continuously...")

try:
    # Run indefinitely
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping motor and cleaning up GPIO.")
finally:
    GPIO.cleanup()
