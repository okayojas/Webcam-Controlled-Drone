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

while analog_stick.is_opened():
    analog_stick.update()
    # get state of the car
    car_state = client.getCarState()
    
    # set the controls for car
    # if analog_stick.y < 1:
    #  car_controls.throttle = analog_stick.y
    #     car_controls.brake = 0
    # else:
    #     car_controls.brake = analog_stick.y
    #     car_controls.throttle = 0
    car_controls.throttle = analog_stick.y     
    car_controls.steering = analog_stick.x
    
    client.setCarControls(car_controls)


    os.system('cls' if os.name == 'nt' else 'clear')
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
    print("X: %.2f, Y: %.2f" % (analog_stick.x, analog_stick.y))

    # # let car drive a bit
    # time.sleep(1)

    # # get camera images from the car
    # responses = client.simGetImages([
    #     airsim.ImageRequest(0, airsim.ImageType.DepthVis),
    #     airsim.ImageRequest(1, airsim.ImageType.DepthPlanar, True)])
    # print('Retrieved images: %d', len(responses))

    # # do something with images
    # for response in responses:
    #     if response.pixels_as_float:
    #         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
    #         airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
    #     else:
    #         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
    #         airsim.write_file('py1.png', response.image_data_uint8)