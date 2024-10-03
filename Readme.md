# Virtual Mouse Control Using Hand Gestures

This project implements a virtual mouse control system using hand gestures captured through a webcam. It utilizes computer vision techniques to detect hand movements and translates them into mouse actions on your computer.

## Features

1. Mouse Cursor Movement
2. Left Click
3. Right Click
4. Double Click
5. Click and Drag
6. Scrolling
7. Zooming

## Installation

1. Clone this repository or download the source code.
2. Install the required packages:
   ```
   pip install -r requirments.txt
   ```
3. Ensure you have the `hand.py` file in the same directory as the main script.
4. Place a `cursor.png` file in a `Cursor` folder in the same directory (optional, for custom cursor overlay).

## Usage

Run the script using Python:

```
python virtual_mouse.py
```

Hold your hand up to the webcam and use the following gestures to control the mouse:

### Mouse Cursor Movement
- Hold up your index finger (fingers: 0,1,0,0,0)
- Move your hand to control the cursor position

### Left Click
- Hold up your index and middle fingers, forming an L-shape (fingers: 0,1,1,0,0)
- Bend your index finger slightly to perform a left click

### Right Click
- Hold up your index, middle, and ring fingers (fingers: 0,1,1,1,0)

### Double Click
- Hold up all fingers except the thumb (fingers: 0,1,1,1,1)

### Click and Drag
- Pinch your index finger and thumb together
- Move your hand to drag items on the screen

### Scrolling
- Hold up your thumb and index finger in an L-shape (fingers: 1,1,0,0,0)
- Move your hand up or down to scroll

### Zooming
- Zoom In: Hold up your thumb, index finger, and pinky (fingers: 1,1,0,0,1)
- Zoom Out: Hold up all fingers (fingers: 1,1,1,1,1)

### Exit the Program
- Close all fingers (make a fist) to exit the program
- Alternatively, press 'q' on your keyboard while the webcam window is active

## Customization

You can adjust various parameters in the script to fine-tune the behavior:

- `CURSOR_WIDTH` and `CURSOR_HEIGHT`: Size of the cursor overlay
- `SCROLL_THRESHOLD`: Sensitivity of scrolling
- `PINCH_THRESHOLD`: Distance threshold for pinch detection
- `ANGLE_THRESHOLD_MIN` and `ANGLE_THRESHOLD_MAX`: Angle range for left click detection
- `L_ANGLE_MIN` and `L_ANGLE_MAX`: Angle range for scroll mode detection

## Troubleshooting

1. If the cursor movement is too sensitive or not sensitive enough, adjust the `smoothing` and `speed_multiplier` variables in the `move_cursor` function.

2. If gesture detection is unreliable, try adjusting the lighting conditions or the `detectionCon` and `trackCon` parameters when initializing the `handDetector`.

## Limitations

- The system is designed for single-hand use only.
- Performance may vary depending on lighting conditions and webcam quality.
- Some gestures might require practice to perform consistently.

## Contributing

Feel free to fork this project and submit pull requests with improvements or additional features.

## License

This project is open-source and available under the MIT License.# virtual-mouse
