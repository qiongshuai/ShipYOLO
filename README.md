# ShipYOLO
**Version:** 1.0.0  
**Authors:** Qiongshuai Lyu, Jiaojiao Zhang, Zhipeng Wang, Mengyuan Liu  
**License:** MIT  
**Repository/Website:** https://github.com/qiongshuai/ShipYOLO/

## Overview
ShipYOLO is a lightweight graphical software for small-target vessel detection in maritime scenarios, developed based on YOLOv10. It supports multi-source file detection, customizable confidence thresholds, automatic result storage, and real-time Excel export. With pure-CPU operability and an intuitive interface, it lowers deployment and usage thresholds, providing a practical engineering solution for maritime supervision and related applications.

---

## Installation Instructions

### Hardware Prerequisites
- **Minimum**: Intel Core i5-10th generation CPU, 8GB RAM, 20GB free disk space
- **Recommended (Development & Test Environment)**: Desktop computer equipped with an Intel Core i7-12700H @ 2.70GHz central processing unit, 16GB DDR4 physical memory, and 512GB NVMe SSD storage device

### Software Prerequisites
- Operating System: Windows 11 Professional 64-bit (fully compatible with Windows 10 64-bit and Ubuntu 20.04/22.04 LTS)
- Core Runtime: Python 3.9.13

### Dependencies
- All required dependencies with exact versions are listed below:  
-- Deep learning framework: PyTorch 2.0.1  
-- Object detection inference framework: Ultralytics 8.2.0 (with native YOLOv10 support)  
-- Image and video processing: OpenCV-Python 4.7.0, Pillow 9.2.0  
-- Graphical user interface: PyQt5 5.15.7  
-- Structured data processing: Pandas 2.0.3  
-- Excel export functionality: XlsxWriter 3.1.2  

### Installation Steps
- Open a terminal and navigate to the directory of the current project folder.  
- Install all required dependencies via pip:    
   pip install -r requirements.txt  

### Usage
- Key Features  
-- Supports single image detection, batch detection of all images in a folder, and video file detection  
-- Allows real-time adjustment of confidence thresholds through an intuitive slider control  
-- Displays detection results in real time with bounding boxes, category labels and confidence scores  
-- Automatically records comprehensive detection information: image/video name, entry time, detection result, target count, and processing time  
-- Exports all detection records to Excel files automatically with standardized formatting  
-- Features a user-friendly PyQt5-based graphical interface with a clear left-right layout  

### How to run
- Complete the installation steps as described above.
- Execute the main program:  
  python main.py

### Run Detection
- Single image detection: Click "Select Image File" to choose an image file, adjust the confidence threshold as needed, then click "Start Detection". A completion prompt will pop up when finished, and detection results will be displayed in the preview window and recorded in the result table.


- Folder batch detection: Click "Select Image Folder" to choose a folder containing multiple images, adjust the confidence threshold, then click "Start Detection". The system will process all images sequentially, and a prompt will indicate that detection records have been saved to Excel after completion.


- Video file detection: Click "Select Video File" to choose a video file, adjust the confidence threshold, then click "Start Detection". The system will process video frames sequentially, display detection results in real time, and record the target count and processing time of each frame in the table.


- To stop detection at any time, click the "Stop Detection" button.

### License
- The code is licensed under the MIT License.

### Screenshots
The interface during the operation of ShipYOLO.(images/show.png)
