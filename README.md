# MediaPipe Drones
Using MediaPipe, a computer vision framework offered by Google, I am controlling a drone using hand pose detection as a controller for a virtual drone using AirSim in Unreal Engine.

## Directions

1. Download an environment from the link: https://github.com/microsoft/AirSim/releases/tag/v1.8.1-windows. I used Coastline and Africa. For example, for Coastline, unzip the folder and open 'Coastline\WindowsNoEditor\Coastline.exe'. Once prompted, click YES for the car and click NO for the drone.

2. Run the according python script (For Car, car.py. For Drone, drone.py). Make sure you have a camera connected, as the virtual analog stick overlay will appear as a seperate window.

3. The virtual analog stick has two axes, a horizontal and vertical. Think of a console controller, but only one stick. The analog stick uses hand position relative to where it is in the camera frame. It also finger tracking to emulate button. For example, the throttle of the car is dependent on the angle of your index finger with your palm. Additionally, the analog stick itself is controlled by a point on your wrist.

## Controls

Car:

Up - Foward
Down - Backward
Right - Turn Right
Left - Turn Left
Center - Brake

Drone:

Up - Forward
Down - Backward
Right - Turn Right
Left - Turn Left
Index Finger Up - Move Up Vertically
Index Finger Down - Move Down Vertically

## To do
- Make hand gestures
    - Acceleration?
    - Braking?
    - Z-axis for drone?
- Create diff control schemes for cars and drones (make drone control)
- Clean HandTracking to fit personal needs and improve readability
    - Frame width is 2x what it actually is in HandTracking due to "predicted"
    being displayed.
    - Update HandTracking to not use "predicted" display since it's unneeded.
- Make the analog stick more modular
    - You can feed in any image and you can just tell it where on the image you want to draw the analog stick
    - Make control more general, more applicable to other games, vehicles, etc
    - Make method measurements relative rather than absolute
        - Instead of pixels, use percentages
- Figure out how to map analog x and y onto drone
- Figure out acceleration scheme
    - Slider?
    - Polar Coords?    

- Fix the color scheme
