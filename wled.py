import numpy as np
import mss
import time
import requests
from PIL import Image
import threading
import cv2
from scipy.ndimage import gaussian_filter
import json
import os
import argparse

class WLEDAmbientController:
    def __init__(self, wled_ip, matrix_width=12, matrix_height=6, update_rate=60):
        self.wled_ip = wled_ip
        self.matrix_width = matrix_width
        self.matrix_height = matrix_height
        self.update_interval = 1.0 / update_rate
        self.running = False
        self.sct = mss.mss()
        
        # Initialize screen properties
        self.monitor = self.sct.monitors[0]
        self.screen_width = self.monitor["width"]
        self.screen_height = self.monitor["height"]
        
        # Enhanced smoothing parameters
        self.prev_matrix = np.zeros((matrix_height, matrix_width, 3))
        self.temporal_smooth = 0.9
        self.saturation_boost = 1.8
        
        # Initialize color correction with calibration support
        self.color_correction = self.load_calibration()
        if self.color_correction is None:
            self.color_correction = np.array([1.0, 1.0, 1.0])
        self.calibration_samples = []
        self.calibration_mode = False
        self.calibration_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 255, 255) # White
        ]
        self.current_calibration_index = 0
        
        # Color flow system
        self.color_momentum = np.zeros((matrix_height, matrix_width, 3))
        self.momentum_decay = 0.85
        self.flow_strength = 0.25

    def load_calibration(self):
        """Load calibration values from cache file"""
        cache_file = 'calibration_cache.json'
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return np.array(data['color_correction'])
            except:
                return None
        return None

    def save_calibration(self):
        """Save calibration values to cache file"""
        cache_file = 'calibration_cache.json'
        data = {
            'color_correction': self.color_correction.tolist()
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def start_calibration(self):
        """Start the color calibration process"""
        print("Starting color calibration...")
        self.calibration_mode = True
        self.calibration_samples = []
        self.current_calibration_index = 0
        self.display_calibration_color()

    def display_calibration_color(self):
        """Display current calibration color on WLED"""
        if self.current_calibration_index < len(self.calibration_colors):
            color = self.calibration_colors[self.current_calibration_index]
            matrix = np.full((self.matrix_height, self.matrix_width, 3), color)
            self.update_wled(matrix)
            print(f"Displaying calibration color {self.current_calibration_index + 1}/{len(self.calibration_colors)}")
            time.sleep(2)  # Allow WLED to stabilize

    def capture_calibration_sample(self):
        """Capture and store a calibration sample"""
        if not self.calibration_mode:
            return

        # Capture screen
        screen = np.array(self.sct.grab(self.monitor))
        screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
        
        # Process to matrix size
        small = cv2.resize(screen_rgb, (self.matrix_width * 2, self.matrix_height * 2))
        blurred = cv2.GaussianBlur(small, (3, 3), 0)
        matrix = cv2.resize(blurred, (self.matrix_width, self.matrix_height))
        
        # Store sample
        target_color = self.calibration_colors[self.current_calibration_index]
        actual_color = np.mean(matrix, axis=(0, 1))
        self.calibration_samples.append((target_color, actual_color))
        
        print(f"Captured sample {self.current_calibration_index + 1}:")
        print(f"Target color: {target_color}")
        print(f"Actual color: {actual_color}")
        
        self.current_calibration_index += 1
        
        if self.current_calibration_index < len(self.calibration_colors):
            self.display_calibration_color()
        else:
            self.complete_calibration()

    def complete_calibration(self):
        """Calculate and apply color correction based on calibration samples"""
        print("\nCompleting calibration...")
        
        # Convert samples to numpy arrays
        target_colors = np.array([t for t, _ in self.calibration_samples])
        actual_colors = np.array([a for _, a in self.calibration_samples])
        
        # Calculate correction factors using least squares
        correction_factors = np.zeros(3)
        for i in range(3):  # For each RGB channel
            # Avoid division by zero
            mask = actual_colors[:, i] != 0
            if np.any(mask):
                correction_factors[i] = np.mean(target_colors[mask, i] / actual_colors[mask, i])
            else:
                correction_factors[i] = 1.0
        
        # Apply some constraints to avoid extreme corrections
        correction_factors = np.clip(correction_factors, 0.5, 2.0)
        
        print("Calculated correction factors:", correction_factors)
        
        # Update color correction
        self.color_correction = correction_factors
        
        # Save calibration
        self.save_calibration()
        
        # Exit calibration mode
        self.calibration_mode = False
        print("Calibration complete!")

    def enhance_colors(self, matrix):
        """Enhance colors with calibrated color correction"""
        # Apply calibrated color correction
        matrix = matrix.astype(float)
        matrix *= self.color_correction.reshape(1, 1, 3)
        
        # Color enhancement pipeline
        matrix = matrix.reshape(-1, 3)
        matrix = np.clip(matrix, 0, 255).astype(np.uint8)
        
        hsv = cv2.cvtColor(matrix.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
        hsv = hsv.reshape(-1, 3)
        
        # Boost saturation
        hsv[:, 1] = np.clip(hsv[:, 1] * self.saturation_boost, 0, 255)
        
        enhanced = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB)
        return enhanced.reshape(self.matrix_height, self.matrix_width, 3)

    def apply_color_flow(self, matrix):
        """Apply fluid-like color flow effect with improved dynamics"""
        # Update color momentum with weighted color difference
        color_diff = matrix - self.prev_matrix
        self.color_momentum = (self.color_momentum * self.momentum_decay + 
                             color_diff * self.flow_strength)
        
        # Apply momentum with color correction
        matrix = matrix + self.color_momentum
        
        # Ensure values stay in valid range while preserving color ratios
        return np.clip(matrix, 0, 255)
    
    def spatial_smooth(self, matrix):
        """Apply spatial smoothing with reduced color bleeding"""
        # Apply gaussian smoothing separately for each color channel
        smoothed = np.zeros_like(matrix)
        for i in range(3):
            smoothed[:, :, i] = gaussian_filter(matrix[:, :, i], sigma=1.0)
        
        # Create a larger virtual matrix for better edge blending
        padded = np.pad(smoothed, ((1, 1), (1, 1), (0, 0)), mode='edge')
        
        # Apply additional neighbor blending with adjusted weights
        for y in range(self.matrix_height):
            for x in range(self.matrix_width):
                neighbors = padded[y:y+3, x:x+3]
                weights = np.array([[0.3, 0.7, 0.3],
                                  [0.7, 2.0, 0.7],
                                  [0.3, 0.7, 0.3]])
                weights = weights[:, :, np.newaxis] / np.sum(weights)
                matrix[y, x] = np.sum(neighbors * weights, axis=(0, 1))
        
        return matrix

    def capture_and_process(self):
        """Capture screen and process colors"""
        # Capture screen
        screen = np.array(self.sct.grab(self.monitor))
        screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
        
        # Downsample with adjusted initial blur
        small = cv2.resize(screen_rgb, (self.matrix_width * 2, self.matrix_height * 2))
        blurred = cv2.GaussianBlur(small, (3, 3), 0)
        
        # Further downsample to matrix size
        matrix = cv2.resize(blurred, (self.matrix_width, self.matrix_height))
        
        # Apply our pipeline
        matrix = self.enhance_colors(matrix)
        matrix = self.spatial_smooth(matrix)
        matrix = self.apply_color_flow(matrix)
        
        # Temporal smoothing with adjusted weight
        matrix = self.prev_matrix * self.temporal_smooth + matrix * (1 - self.temporal_smooth)
        self.prev_matrix = matrix.copy()
        
        return matrix

    def update_wled(self, matrix):
        """Send update to WLED device"""
        led_data = matrix.reshape(-1, 3).astype(int).tolist()
        
        # Send only color data, letting WLED control brightness
        payload = {
            "seg": {
                "i": led_data
            }
        }
        
        try:
            requests.post(
                f"http://{self.wled_ip}/json/state",
                json=payload,
                timeout=0.05
            )
        except requests.exceptions.RequestException:
            pass
    
    def run(self):
        """Main loop for ambient lighting control"""
        self.running = True
        
        while self.running:
            start_time = time.time()
            
            matrix = self.capture_and_process()
            self.update_wled(matrix)
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def start(self):
        """Start the ambient lighting controller"""
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the ambient lighting controller"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WLED Ambient Lighting Controller')
    parser.add_argument('--ip', default='192.168.1.9', help='WLED IP address')
    parser.add_argument('--calibrate', action='store_true', help='Force recalibration')
    args = parser.parse_args()
    
    controller = WLEDAmbientController(args.ip)
    
    try:
        # Only calibrate if forced or no cached values exist
        if args.calibrate or controller.color_correction is None:
            print("Starting calibration...")
            controller.start_calibration()
            while controller.calibration_mode:
                controller.capture_calibration_sample()
                time.sleep(3)
        
        print("\nStarting enhanced ambient lighting controller...")
        print("Use --calibrate flag to recalibrate")
        controller.start()
        
        print("\nController running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping ambient lighting controller...")
        controller.stop()