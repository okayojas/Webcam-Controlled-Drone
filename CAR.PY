import airsim
import time
from HandAnalogStick import VirtualAnalogStick
import os

analog_stick = VirtualAnalogStick()
analog_stick.start()

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

def drive_forward(throttle, steering):
    car_controls.is_manual_gear = False
    car_controls.throttle = throttle
    car_controls.steering = steering
    car_controls.brake = 0
    client.setCarControls(car_controls)

def drive_backward(throttle, steering):
    car_controls.is_manual_gear = True
    car_controls.manual_gear = -1
    car_controls.throttle = throttle
    car_controls.steering = steering
    car_controls.brake = 0
    client.setCarControls(car_controls)

def apply_brake():
    car_controls.brake = 1
    car_controls.throttle = 0
    car_controls.steering = 0
    client.setCarControls(car_controls)

while analog_stick.is_opened():
    analog_stick.update()
    # get state of the car
    car_state = client.getCarState()
    
    # set the controls for car
    if analog_stick.y < 0:
        drive_backward(-analog_stick.y, analog_stick.x)
    elif analog_stick.y > 0:
        drive_forward(analog_stick.y, analog_stick.x)
    else:
        apply_brake()


    os.system('cls' if os.name == 'nt' else 'clear')
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
    print("X: %.2f, Y: %.2f" % (analog_stick.x, analog_stick.y))
