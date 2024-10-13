import airsim
import time
from HandAnalogStick import VirtualAnalogStick
import numpy as np

# Initialize the analog stick
analog_stick = VirtualAnalogStick()
analog_stick.start()

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Define maximum pitch and yaw rate
MAX_PITCH_ANGLE = 0.2  # radians (~11 degrees)
MAX_YAW_RATE = 1.0     # radians per second (~57 degrees per second)
MAX_THROTTLE = 0.75    # Maximum throttle value
MIN_THROTTLE = 0.5     # Minimum throttle to maintain hover
LOOP_SLEEP_TIME = 0.05  # 50 milliseconds

def drive_drone(pitch, yaw_rate, throttle):
    client.moveByRollPitchYawrateThrottleAsync(
        roll=0,
        pitch=pitch,
        yaw_rate=yaw_rate,
        throttle=throttle,
        duration=LOOP_SLEEP_TIME
    )

try:
    while analog_stick.is_opened():
        analog_stick.update()
        
        # Get the analog stick inputs
        pitch = analog_stick.y * MAX_PITCH_ANGLE  # Forward/backward tilt
        yaw_rate = analog_stick.x * MAX_YAW_RATE  # Left/right rotation

        # Get throttle from index finger percentage
        index_percent = analog_stick.get_percent('index')  # Should return a value between 0 and 100
        throttle = MIN_THROTTLE + (index_percent / 100) * (MAX_THROTTLE - MIN_THROTTLE)
        throttle = np.clip(throttle, 0.0, 1.0)  # Ensure throttle is within valid range

        # Control the drone with the analog stick inputs
        drive_drone(pitch, yaw_rate, throttle)

        # Print out current state for debugging
        drone_state = client.getMultirotorState()
        print(f"Pitch: {pitch:.2f} rad, Yaw Rate: {yaw_rate:.2f} rad/s, Throttle: {throttle:.2f}")
        print(f"Speed X: {drone_state.kinematics_estimated.linear_velocity.x_val:.2f} m/s, Altitude: {-drone_state.kinematics_estimated.position.z_val:.2f} m")
        
        time.sleep(LOOP_SLEEP_TIME)
except KeyboardInterrupt:
    print("Stopping drone control...")
finally:
    # Ensure the drone is safely landed and disarmed if the loop exits
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    analog_stick.close()
