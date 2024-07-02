# MediaPipe Drones
description...
# To do
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
