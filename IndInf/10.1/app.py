from flask import Flask, Response, render_template, request
from flask_socketio import SocketIO
#from picamera2 import Picamera2
#import cv2
import RPi.GPIO as GPIO
import time
import math


# Version 1.2

app = Flask(__name__)
socketio = SocketIO(app)

# # camera = cv2.VideoCapture('/dev/media4', cv2.CAP_V4L2)  # Assuming your USB camera is at index 0
# camera = cv2.VideoCapture(0)

# # Check if the camera opened correctly
# if not camera.isOpened():
#     # raise RuntimeError("Could not start camera.")
#     print("Could not start camera.")
# camera.set(cv2.CAP_PROP_FPS, 30)

# def generate_frames():
#     while True:
#         ret, frame = camera.read()  # Capture frame from the USB camera
#         if not ret:
#             break
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Motor configuration
GPIO.setmode(GPIO.BCM)
motor_pins = {"forwards": 15, "backwards": 18, "ENA": 17}

# Servo configuration (MG996R)
servo_pin = 12
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)

# Function to set servo angle
def setServoAngle(angle):
    assert angle >= 30 and angle <= 150
    pwm.start(8)
    dutyCycle = angle / 18. + 3.
    pwm.ChangeDutyCycle(dutyCycle)
    time.sleep(0.3)
    pwm.stop()

# Initialize PWM on servo_pin at 50Hz (typical for servos)
#servo_pwm = GPIO.PWM(servo_pin, 50)
# Start at center position (duty cycle ~7.5; adjust as necessary)
setServoAngle(90)

# Virtual car state for tracking purposes
car_state = {
    'x': 300,    # initial x on canvas
    'y': 300,    # initial y on canvas
    'angle': 0   # initial angle (0 = facing right)
}

for pin in motor_pins.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# Control functions
def move_car(action):
    konstante = 90
    if action == "backward":
        GPIO.output(motor_pins["ENA"], True)
        GPIO.output(motor_pins["backwards"], True)
#        GPIO.output(motor_pins["forwards"], False)
    elif action == "forward":
        GPIO.output(motor_pins["ENA"], True)
        GPIO.output(motor_pins["forwards"], True)
#        GPIO.output(motor_pins["backwards"], False)
    elif action == "right":
        # Turn servo to the right.
        # Adjust the duty cycle (e.g., 10) for your desired right angle.
        setServoAngle(konstante + 80)
        # Optional: allow time for servo movement then stop sending PWM signal to avoid jitter
    elif action == "left":
        # Turn servo to the left.
        # Adjust the duty cycle (e.g., 5) for your desired left angle.
        setServoAngle(konstante - 80)
    elif action == "stop":
        for pin in motor_pins.values():
            GPIO.output(pin, False)
        # Optionally, reset servo to center
        #servo_pwm.ChangeDutyCycle(7.5)
        time.sleep(0.3)
        setServoAngle(konstante)
        
def update_virtual_state(action):
    """Update the virtual position based on the command.
       For simplicity, assume:
         - 'forward' moves 10 units,
         - 'backward' moves -10 units,
         - 'left' rotates -15°,
         - 'right' rotates +15°.
    """
    distance = 0
    angle_change = 0
    if action == "forward":
        distance = 10
    elif action == "backward":
        distance = -10
    elif action == "left":
        angle_change = -15
    elif action == "right":
        angle_change = 15

    # Update heading
    car_state['angle'] += angle_change
    # Calculate displacement based on new heading
    rad = math.radians(car_state['angle'])
    car_state['x'] += distance * math.cos(rad)
    car_state['y'] += distance * math.sin(rad)        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/track')
def track():
    return render_template('track.html')

@app.route('/controlpanel')
def controlpanel():
    return render_template('controlpanel.html')

@app.route('/control', methods=['POST'])
def control():
    action = request.form['action']
    move_car(action)
    return 'OK'

if __name__ == "__main__":
    try:
        #app.run(host='0.0.0.0', port=5000, debug=True)
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        servo_pwm.stop()
        GPIO.cleanup()
