# WLED Ambient Lighting Controller

*Transform your space with the dynamic interplay of light and color, seamlessly extending your screen into the environment.*

## Overview

The **WLED Ambient Lighting Controller** captures the essence of motion and light, bringing an immersive ambient lighting experience to your surroundings. By mirroring your screen's colors onto a WLED-enabled LED matrix, it creates a harmonious fusion of digital content and physical space, enhancing visual storytelling and entertainment.

Inspired by the elegance and simplicity of design philosophies akin to Apple, this project embodies meticulous attention to detail, combining sophisticated algorithms with an intuitive interface to deliver a captivating ambient lighting solution.

## Features

- **Real-Time Screen Capture**: Effortlessly captures your screen to reflect live color dynamics.
- **Advanced Color Processing**: Utilizes Gaussian blurring, spatial smoothing, and temporal filtering for seamless color transitions and reduced bleeding.
- **Comprehensive Calibration**: Offers a guided calibration process to ensure accurate color representation between your display and LED matrix.
- **Dynamic Color Flow**: Implements fluid-like motion effects to enhance the natural movement of light across your LEDs.
- **High Performance**: Optimized for responsiveness with minimal latency, maintaining synchronization with on-screen content.
- **Customizable Settings**: Adjustable parameters for matrix dimensions, update rates, and visual effects to suit your preferences.

## Installation

### Prerequisites

Ensure you have the following installed:

- **Python 3.x**
- **Required Python Libraries**: Install via pip.

### Dependency Installation

```sh
pip install numpy mss opencv-python requests scipy pillow
```

## Usage

Run the ambient lighting controller with your WLED device's IP address:

```sh
python wled.py --ip <WLED_DEVICE_IP>
```

Replace `<WLED_DEVICE_IP>` with the IP address of your WLED-enabled LED matrix.

### Calibration

For optimal color accuracy, initiate the calibration process:

```sh
python wled.py --ip <WLED_DEVICE_IP> --calibrate
```

Follow the on-screen prompts. The system will display a series of colors to fine-tune the color correction factors, ensuring a precise match between your display and ambient lighting.

## Configuration

Customize your setup via the `wled_config.json` file:

```json
{
  "orientation": {
    "flip_horizontal": true,
    "flip_vertical": true,
    "rotate": 270,
    "start_corner": "top_left",
    "serpentine": true
  },
  "matrix_width": 12,
  "matrix_height": 6
}
```

- **Orientation Settings**: Adjust to align with your LED matrix's physical layout.
- **Matrix Dimensions**: Modify 

matrix_width

 and 

matrix_height

 to match your LED setup.

## How It Works

### Screen Capture

Utilizes `mss` to perform efficient screen capturing, focusing on minimal performance overhead while capturing high-resolution imagery from your display.

### Image Processing Pipeline

1. **Downsampling with Blur**: Reduces the captured image to the LED matrix size, applying Gaussian blur to smooth out fine details and prevent flickering.
2. **Spatial Smoothing**: Employs advanced smoothing techniques to reduce color bleeding, ensuring each LED represents the average color of its corresponding screen area.
3. **Color Enhancement**: Enhances colors using calibrated correction factors and boosts saturation to make colors more vivid and true-to-life.
4. **Dynamic Color Flow**: Introduces momentum-based adjustments to simulate the fluid motion of light, adding depth and dynamism to the lighting effects.
5. **Temporal Filtering**: Applies temporal smoothing to create seamless transitions over time, eliminating abrupt changes and enhancing the ambient experience.

### Communication with WLED

Sends the processed color data to the WLED device using HTTP POST requests with JSON payloads, leveraging WLED's API for real-time updates.

```py
requests.post(
    f"http://{self.wled_ip}/json/state",
    json=payload,
    timeout=0.05
)
```

## Advanced Calibration

The calibration process adjusts for discrepancies in color representation, accounting for variations in LED output and environmental factors.

- **Calibration Samples**: Collects samples by displaying known colors and measuring the actual output.
- **Color Correction Calculation**: Computes correction factors using least squares to minimize color errors.
- **Persistence**: Saves calibration data to `calibration_cache.json` for future sessions.

## Performance Optimization

- **Efficient Algorithms**: Ensures high frame rates by optimizing image processing tasks.
- **Multithreading**: Leverages threading to maintain responsiveness and real-time performance.
- **Adaptive Updates**: Adjusts processing based on system capabilities to maintain a balance between quality and performance.

## Troubleshooting

- **No Light Output**: Verify the WLED device IP address and network connectivity.
- **Inaccurate Colors**: Re-run the calibration process to update color correction factors.
- **High CPU Usage**: Lower the `update_rate` or reduce the `matrix_width` and `matrix_height` settings in the configuration file to decrease the processing load.

## Future Enhancements

- **Graphical User Interface**: Develop a GUI for easier control and customization.
- **Multi-Device Support**: Enable synchronization across multiple WLED devices.
- **Effect Library**: Introduce a library of visual effects and transitions.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- **WLED Community**: For creating an incredible platform for LED control.
- **Open-Source Libraries**: Special thanks to the developers of NumPy, OpenCV, MSS, and other libraries that made this project possible.