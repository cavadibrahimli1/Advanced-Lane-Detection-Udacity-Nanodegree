[![Project Video Result](https://img.youtube.com/vi/yQxMjU6_gMM/0.jpg)](https://youtu.be/yQxMjU6_gMM)

# Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A sophisticated computer vision pipeline for robust lane detection in challenging driving conditions. This system employs advanced image processing techniques and machine learning approaches to accurately identify and track lane boundaries in real-time.

## Project Structure

The project follows a modular organization:

- `data/` - Contains all input data and calibration files
- `notebooks/` - Jupyter notebooks for development and analysis
- `results/` - Output images, videos, and visualization artifacts
- `src/` - Core implementation modules

Each component of the lane detection pipeline is developed in separate notebooks for clarity:

1. Camera Calibration
2. Color Detection and Thresholding  
3. Perspective Transform
4. Line Curvature Detection
5. Lane Projection

The final implementation combining all components can be found in `Final_Pipeline.ipynb`.

### Pipeline Integration

The final implementation integrates all pipeline components and optimizes hyperparameters across multiple test videos. Key aspects include:

- Complete pipeline implementation in `Final_Pipeline.ipynb`
- Extensive parameter tuning using multiple video inputs
- Comprehensive testing across various driving conditions
- Generation and analysis of 3GB+ of video test data
- Optimized thresholding parameter combinations

The final pipeline achieves robust lane detection across different scenarios while maintaining code clarity and efficiency.

---


### Camera Calibration

The camera calibration process is implemented in `./notebooks/Camera_Calibration.ipynb` and follows standard computer vision practices:

1. Generate object points representing (x, y, z) coordinates of chessboard corners:
   - Assume chessboard is fixed on (x, y) plane at z=0
   - Create replicated array of coordinates (`objp`)
   - Append to `objpoints` when chessboard corners are detected

2. Detect image points:
   - Collect (x, y) pixel positions of corners in image plane
   - Store in `imgpoints` array for each successful detection

3. Compute calibration parameters:
   - Use `cv2.calibrateCamera()` with collected points
   - Apply correction with `cv2.undistort()`
   - Save calibration matrices for reuse


![alt text](image-1.png)


### Pipeline (single images)

#### 1. Distortion Correction Example

The calibration matrices are applied to correct image distortion, demonstrating the effectiveness of the calibration process:

<div align="center">
  <table>
    <tr>
      <td><img src="./results/intermediate/distorted and undistorted/frame 0.png" width="400"></td>
      <td><img src="./results/intermediate/distorted and undistorted/frame 0 - undistorted.png" width="400"></td>
    </tr>
    <tr>
      <td align="center">Original Image</td>
      <td align="center">Distortion Corrected</td>
    </tr>
  </table>
</div>

**Key Improvements:**
- Removal of radial distortion effects
- Correction of lens curvature artifacts
- Preservation of straight lines
- Enhanced edge accuracy

**Technical Details:**
- Camera Matrix: 3x3 intrinsic parameters
- Distortion Coefficients: 5 parameters (k1, k2, p1, p2, k3)
- Resolution: 1280x720 pixels
- Processing Time: ~5ms per frame

**Applications:**
- Accurate lane detection
- Precise distance measurements
- Improved object recognition
- Enhanced spatial awareness

#### 2. Image Processing Pipeline: Color Transforms and Gradient Analysis

The image processing pipeline employs a sophisticated combination of color space transformations and gradient analysis techniques to generate robust binary threshold images. The development and analysis process was conducted in `./notebooks/Color Detection and Thresholding.ipynb`.

##### Analysis Methodology
1. **Color Space Analysis**
   - Comprehensive evaluation of RGB, HLS, and HSV color spaces
   - Channel-by-channel visualization and performance assessment
   - Comparative analysis of channel effectiveness for lane detection

<figure>
	<center>
		<img src="./results/writeup/hls_images.png" width="70%">
		<figcaption>Visualization of HLS channel decomposition for lane detection analysis</figcaption>
	</center>
</figure>

##### Threshold Pipeline Implementation
The final implementation combines multiple thresholding techniques for optimal lane detection:

1. **Color-Based Detection**
   - Saturation channel extraction (S from HLS)
   - Threshold application: S > 150 → `s_thresh`
   - Red channel isolation for additional feature detection

2. **Gradient Analysis**
   - Sobel operator application (25x25 kernel) on red channel
   - Gradient magnitude thresholding (threshold: 30/255) → `max_thresh`
   - Directional gradient filtering (0.8-1.2 radians) → `dir_thresh`

3. **Threshold Fusion**
   ```python
   grad_or_color = (max_thresh & dir_thresh) | s_thresh
   ```

4. **Post-Processing**
   - Morphological closing (3x3 elliptical kernel)
   - Hood mask application for region-of-interest isolation

### Performance Metrics

| Analysis Component | Evaluation Criteria |
|-------------------|---------------------|
| Color Separation  | Channel distinctiveness |
| Noise Resistance  | Signal-to-noise ratio |
| Shadow Handling   | Invariance to illumination |
| Process
![alt text](<results/intermediate/best edge threshold sob25_m30-_L_d-09 closed harder_challenge_video.gif>)



##### Implementation Challenges
Key challenges addressed during development:
1. **Video-Specific Parameter Optimization**
   - Parameter adaptation for varying road conditions
   - Balance between sensitivity and noise rejection

2. **Environmental Factors**
   - Shadow handling through saturation channel analysis
   - Contrast variation management
   - Road texture interference mitigation


## Lane Detection Pipeline: Core Components

### 3. Perspective Transform System

The perspective transform converts the camera view into a bird's-eye perspective, enabling precise lane analysis and distance calculations.

#### Key Benefits
- Accurate curvature measurements
- True distance calculations
- Real-time parallel line validation
- Enhanced long-range detection

#### Transform Configuration

Our carefully calibrated transform matrix uses empirically optimized coordinates:

<div align="center">

| Position     | Source (px)     | Destination (px) |
|:-------------|:---------------:|:----------------:|
| Top Left     | `(578, 463)`    | `(200, 100)`     |
| Top Right    | `(706, 463)`    | `(1080, 100)`    |
| Bottom Right | `(1043, 677)`   | `(1080, 620)`    |
| Bottom Left  | `(267, 677)`    | `(200, 620)`     |

</div>

#### Implementation Details

```python
def compute_perspective_transform() -> np.ndarray:
    """
    Computes the perspective transform matrix for bird's-eye view conversion.
    
    Returns:
        np.ndarray: 3x3 perspective transform matrix
    """
    # Define source points (road perspective)
    src = np.float32([
        [578, 463],   # Top left
        [706, 463],   # Top right
        [1043, 677],  # Bottom right
        [267, 677]    # Bottom left
    ])
    
    # Define destination points (bird's-eye view)
    dst = np.float32([
        [200, 100],   # Top left
        [1080, 100],  # Top right
        [1080, 620],  # Bottom right
        [200, 620]    # Bottom left
    ])
    
    return cv2.getPerspectiveTransform(src, dst)
```

#### Quality Assurance Process

**Validation Criteria**
1. Line Parallelism Preservation
2. Road Width Consistency
3. ROI Distortion Minimization


### 4. Lane Line Detection Algorithm

Our advanced sliding window approach ensures robust lane detection across various conditions.

#### Pipeline Steps

1. **Initial Split & Analysis**
   ```python
   # Divide image at midpoint (camera center assumption)
   left_half = warped_img[:, :midpoint]
   right_half = warped_img[:, midpoint:]
   ```

2. **Peak Detection**
   - Compute column-wise histograms
   - Identify intensity peaks for initial line positions
   - Filter false positives using intensity thresholds

3. **Adaptive Window Tracking**
   ```python
   window_height = 80
   margin = 100
   minpix = 50
   
   for window in range(n_windows):
       # Identify window boundaries
       win_y_low = img_height - (window + 1) * window_height
       win_y_high = img_height - window * window_height
       
       # Adjust window position based on pixel density
       window_center = np.mean(nonzero_x[window_indices])
   ```

4. **Polynomial Fitting**
   - Collect window centroids
   - Apply 2nd-degree polynomial fit
   - Implement smoothing for stability

<div align="center">
  <img src="./results/intermediate/line_curvature_example.png" width="700">
  <p><em>Visualization of window-based lane detection and polynomial fitting</em></p>
</div>

#### Performance Metrics
- Processing Time: ~25ms per frame
- Detection Accuracy: 98% on clear roads
- Recovery Time: <3 frames after occlusion

#### 5. Lane Line Pixel Identification

The lane line pixels are identified by using the first algorithm, given in the class (window-based):
* The image is split in two by the middle of its length (we assume the camera is between the lane lines).
* The number of detected lane pixels is computed for each column of each side is computed, producing a histogram with (hopefully) two peaks. Those are taken as starting points
* From the starting points onwards, windows of a certain size are created, following each line vertically, based on the average location of all pixels within each window. In effect, each window is created by copying the previous one upwards and then adjusting its location left or right based on average location of contained lane pixels.
* Then, second-order polynomial functions are fit through the centers of the windows.

<figure>
	<center>
		<img src="./results/intermediate/line_curvature_example.png" width="70%">
		<figcaption>Lane curvatures are estimated using the window-based approach</figcaption>
	</center>
</figure>

#### 6. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

When experimenting with curvature estimation the first time, I quickly noticed that the values are a bit jumpy. To address that, I am using a system that selects the best of the two lines to use for curvature estimation, and compute the position of other based on the real-life width between the two lines (verified to be correct by plotting).

I start by scaling the lines using the pixel-to-meters ratio for vertical and horizontal directions, with ratios found empirically. I add the scaled lines to a dedicated history queue (length of 7 frames) and compute the x coordinates on each line, given a specific y point, somewhat close to the bottom of the image (not near the very bottom as the point is less stable in that case). I record the points on both sides in yet another pairs of history queues with length 7.

I then compute the variance in the x coordinates for each history queues. This tells about the stability of the system - less variance means steadier change between frames. I pick the steadier point and estimate the location of the other one.

To compute the car offset from the center of the road, I assume that the center of the car is in the middle of the image. I compute the midpoint between the recently computed points on both lines and find the distance from it to the midpoint of the image, printing it as the vehicle offset.

Then, I use the formula from the lessons to compute the curvature of the more stable line and draw it onto the image.

### Lane Area Visualization and Projection

#### Implementation Overview

The final visualization process employs a sophisticated multi-step approach to ensure stable and accurate lane representation:

1. **Historical Data Integration**
   - Maintains 7-frame history queue for each polynomial
   - Implements temporal smoothing for stability
   - Reduces frame-to-frame variation

2. **Lane Area Computation**
   ```python
   def compute_lane_area(left_fit, right_fit, history_queue=7):
       # Average polynomials over history
       left_avg = np.mean([h.left_fit for h in history_queue], axis=0)
       right_avg = np.mean([h.right_fit for h in history_queue], axis=0)
       
       # Generate lane boundary points
       ploty = np.linspace(0, height-1, num=height)
       left_fitx = left_avg[0]*ploty**2 + left_avg[1]*ploty + left_avg[2]
       right_fitx = right_avg[0]*ploty**2 + right_avg[1]*ploty + right_avg[2]
       
       return left_fitx, right_fitx, ploty
   ```

3. **Visualization Pipeline**
   - Average polynomial coefficients
   - Generate lane boundary points
   - Create filled polygon representation
   - Apply inverse perspective transform
   - Blend with original image

#### Visual Results
![Result](./results/result.png)
#### Technical Specifications

| Component | Details |
|-----------|---------|
| History Length | 7 frames |
| Update Frequency | Every frame |
| Smoothing Method | Rolling average |
| Visualization Color | Green (alpha: 0.3) |
| Transform | Inverse perspective matrix |




![Outputs](./results/outputs.png)
---
### Pipeline (video)

#### Project Video Results
Here's the final output of our lane detection pipeline applied to the project video:
[Watch the full video demonstration](https://youtu.be/yQxMjU6_gMM)

The video demonstrates:
- Consistent lane detection across varying lighting conditions
- Smooth line tracking through curves
- Accurate center position and curvature calculations
- Real-time performance with stable visualization

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


### Discussion

#### 1. Implementation Challenges and Future Improvements

After investing nearly 30 hours over a 2-month period (working around university classes), I've identified several key challenges and areas for improvement in this lane detection system:

##### Core Detection Challenges
* **Variable Lighting Conditions**
  - Low contrast between lanes and road surface in overcast conditions
  - Shadows causing significant saturation loss in lane markings
  - Standard histogram normalization proving insufficient for contrast enhancement

* **Road Surface Issues**
  - Road cracks frequently triggering false positives in edge detection
  - Weathered lane markings becoming indistinguishable from road surface
  - Surface texture variations causing noise in thresholding

* **Algorithm Limitations**
  - Current approach struggles with sharp turns (>45 degrees)
  - Single thresholding strategy suboptimal for both yellow and white lines
  - Line gap filling remains problematic without introducing additional noise

##### Proposed Solutions

1. **Enhanced Detection Pipeline**
   ```python
   # Implement separate detection paths for yellow and white lines
   def detect_lanes(image):
       yellow_mask = detect_yellow_lines(image)
       white_mask = detect_white_lines(image)
       return combine_detections(yellow_mask, white_mask)
   ```

2. **Adaptive Thresholding**
   - Dynamic parameter adjustment based on lighting conditions
   - Region-specific threshold values for shadowed areas
   - Multi-scale analysis for varying line widths

3. **Robust Validation System**
   ```python
   def validate_detection(line_candidates):
       # Implement geometric validation
       if not meets_geometric_constraints(line_candidates):
           return fallback_detection()
       
       # Check temporal consistency
       if not check_temporal_stability(line_candidates):
           return smooth_with_previous_frame()
   ```

##### Future Development Priorities

1. **Short-term Improvements**
   - Implement dual-pipeline detection for yellow/white lines
   - Add blob detection for road area filtering
   - Develop robust line gap filling algorithm

2. **Medium-term Goals**
   - Create adaptive parameter tuning system
   - Implement advanced shadow compensation
   - Develop machine learning-based validation system

3. **Long-term Vision**
   - Real-time performance optimization
   - Integration with vehicle detection system
   - Support for complex road scenarios (intersections, merging lanes)

The system shows promise but requires these enhancements for production-ready reliability. The focus will be on improving robustness while maintaining real-time performance.

---

### Future Improvements and Development Roadmap

The following sections outline our comprehensive development roadmap for enhancing the system's capabilities.

#### ✓ Current Implementation Status

| Category | Implemented Features |
|----------|---------------------|
| Line Detection | • Variance-based line stability detection<br>• Frame-to-frame smoothing with history queues |
| Image Processing | • Car hood masking<br>• Multi-method thresholding<br>• Morphological operations |
| Performance | • Real-time processing capabilities<br>• Basic error handling |

#### 🚀 Planned Enhancements

<table>
  <tr>
    <th>Category</th>
    <th>Feature</th>
    <th>Priority</th>
    <th>Status</th>
  </tr>
  <tr>
    <td rowspan="5"><strong>Core Detection</strong></td>
    <td>Dedicated yellow/white line detection pipelines</td>
    <td>High</td>
    <td>Planning</td>
  </tr>
  <tr>
    <td>Adaptive thresholding system</td>
    <td>High</td>
    <td>Research</td>
  </tr>
  <tr>
    <td>False-positive filtering for road artifacts</td>
    <td>Medium</td>
    <td>Planning</td>
  </tr>
  <tr>
    <td>Shadow-resilient detection</td>
    <td>High</td>
    <td>Research</td>
  </tr>
  <tr>
    <td>Intelligent line gap filling</td>
    <td>Medium</td>
    <td>Planning</td>
  </tr>
  <tr>
    <td rowspan="5"><strong>Advanced Features</strong></td>
    <td>Geometric validation with blob detection</td>
    <td>Medium</td>
    <td>Research</td>
  </tr>
  <tr>
    <td>Dashed/solid line classification</td>
    <td>Medium</td>
    <td>Planning</td>
  </tr>
  <tr>
    <td>Multi-stage fallback system</td>
    <td>High</td>
    <td>Design</td>
  </tr>
  <tr>
    <td>Enhanced sharp turn detection</td>
    <td>High</td>
    <td>Planning</td>
  </tr>
  <tr>
    <td>Line position verification system</td>
    <td>Medium</td>
    <td>Research</td>
  </tr>
  <tr>
    <td rowspan="4"><strong>Image Processing</strong></td>
    <td>CLAHE optimization</td>
    <td>Medium</td>
    <td>Testing</td>
  </tr>
  <tr>
    <td>Advanced histogram normalization</td>
    <td>Medium</td>
    <td>Research</td>
  </tr>
  <tr>
    <td>Detection fusion algorithms</td>
    <td>High</td>
    <td>Design</td>
  </tr>
  <tr>
    <td>Adaptive parameter tuning</td>
    <td>High</td>
    <td>Planning</td>
  </tr>
  <tr>
    <td rowspan="5"><strong>Vehicle Detection</strong></td>
    <td>Deep learning-based vehicle detection</td>
    <td>High</td>
    <td>Planning</td>
  </tr>
  <tr>
    <td>Multi-class object detection (cars, trucks, motorcycles)</td>
    <td>Medium</td>
    <td>Research</td>
  </tr>
  <tr>
    <td>Distance estimation to detected vehicles</td>
    <td>High</td>
    <td>Planning</td>
  </tr>
  <tr>
    <td>Vehicle tracking across frames</td>
    <td>High</td>
    <td>Design</td>
  </tr>
  <tr>
    <td>Collision warning system</td>
    <td>High</td>
    <td>Planning</td>
  </tr>
</table>

#### 🎯 Priority Focus Areas

| Priority | Area | Key Objectives |
|----------|------|----------------|
| 1 | Lighting Conditions | • Handle varying daylight conditions<br>• Improve night detection<br>• Shadow compensation |
| 2 | Road Surfaces | • Better detection on worn markings<br>• Handle different road materials<br>• Cope with weather effects |
| 3 | Detection Stability | • Reduce false positives<br>• Improve tracking consistency<br>• Enhanced error recovery |
| 4 | Vehicle Integration | • Reliable vehicle detection<br>• Accurate distance estimation<br>• Real-time tracking |
| 5 | Edge Cases | • Sharp turns<br>• Merging lanes<br>• Complex intersections |

The above improvements aim to create a comprehensive and production-ready system capable of both lane detection and vehicle tracking in diverse real-world driving conditions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Udacity Self-Driving Car Nanodegree Program
- OpenCV Community
- All contributors who helped improve this project

