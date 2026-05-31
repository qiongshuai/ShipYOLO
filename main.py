# -*- coding: utf-8 -*-
import os
import sys
import threading
import cv2
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QHBoxLayout, QGroupBox, QFormLayout, QSpinBox, QMessageBox
)

import logging
logging.basicConfig(level=logging.DEBUG)


class ImageProcessingThread(QThread):
    update_frame = pyqtSignal(QImage, dict, str, float)  # Modified signal, added time parameter. Define a signal for updating image frame, detection info and source path

    def __init__(self, source_path, conf_threshold, main_window):
        super().__init__()
        self.source_path = source_path  # Set source path
        self.conf_threshold = conf_threshold  # Set confidence threshold
        self.running = True  # Set thread running flag
        self.model = YOLO('yolov10s.pt')  # Load YOLO model
        self._lock = threading.Lock()  # Add thread lock
        self.main_window = main_window  # Save main window reference

    def run(self):
        try:
            logging.info(f"Start processing: {self.source_path}")
            if os.path.isdir(self.source_path):
                self.process_folder()
            elif self.source_path.lower().endswith(('.mp4', '.avi')):
                self.process_video()
            else:
                self.process_single_image(self.source_path)
        except Exception as e:
            logging.error(f"Processing exception: {str(e)}")

    def process_folder(self):
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = [f for f in os.listdir(self.source_path)
                       if f.lower().endswith(valid_ext)]

        for filename in image_files:
            if not self.running:
                break
            img_path = os.path.join(self.source_path, filename)
            self.process_image(img_path, filename)

    def process_video(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                logging.error(f"Cannot open video file: {self.source_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = max(int(1000 / (fps if fps > 0 else 30)), 1)
            output_path = self.auto_save_video_path()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (int(cap.get(3)), int(cap.get(4))))

            frame_count = 0
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Force refresh
                QApplication.processEvents()

                frame_start_time = time.time()  # Frame timing start
                processed_frame, qt_image, info = self.process_frame(frame)
                if processed_frame is not None:
                    processing_time = (time.time() - frame_start_time) * 1000  # Calculate per-frame processing time
                    out.write(processed_frame)
                    # Don't save individual frame time during video processing, pass 0
                    self.update_frame.emit(qt_image, info, self.source_path, 0)  # Video doesn't show single frame time
                    self.msleep(delay)
                    frame_count += 1

            out.release()
            logging.info(f"Video auto-saved to: {output_path}")
            logging.info(f"Video processing complete, {frame_count} frames processed")

        finally:
            if cap and cap.isOpened():
                cap.release()

    def process_image(self, img_path, filename):
        start_time = time.time()  # Add timing start
        frame = cv2.imread(img_path)
        if frame is not None:
            processed_frame, qt_image, info = self.process_frame(frame)
            if processed_frame is not None:
                processing_time = (time.time() - start_time) * 1000  # Calculate processing time (ms)
                info['filename'] = filename  # Add filename to info
                # Emit update signal with processing time
                self.update_frame.emit(qt_image, info, img_path, processing_time)  # Modified signal, added time parameter
                with self._lock:  # Add thread lock
                    output_path = self.auto_save_image_path(filename)
                    try:
                        # Handle path issues
                        ext = os.path.splitext(output_path)[1]
                        temp_file = "temp" + ext
                        cv2.imwrite(temp_file, processed_frame)
                        os.rename(temp_file, output_path)
                        logging.info(f"Image auto-saved to: {output_path}")
                    except Exception as e:
                        logging.error(f"Save failed: {str(e)}")

    def process_frame(self, frame):
        try:
            results = self.model(frame, conf=self.conf_threshold)
            annotated = results[0].plot()

            rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)

            info = self.parse_results(results)
            return annotated, qt_image.copy(), info

        except Exception as e:
            logging.error(f"Frame processing exception: {str(e)}")
            return None, None, None

    def parse_results(self, results):  # Parse YOLO detection results
        info = {
            'class': [],  # Store detected classes
            'confidence': [],  # Store detection confidence
            'xmin': [],  # Store top-left x coordinate of detection box
            'ymin': [],  # Store top-left y coordinate of detection box
            'xmax': [],  # Store bottom-right x coordinate of detection box
            'ymax': []  # Store bottom-right y coordinate of detection box
        }
        # Iterate all detection boxes
        for box in results[0].boxes:
            info['class'].append(results[0].names[int(box.cls)])  # Add class name
            info['confidence'].append(f"{float(box.conf):.2%}")  # Add confidence
            info['xmin'].append(int(box.xyxy[0][0]))  # Add top-left x coordinate
            info['ymin'].append(int(box.xyxy[0][1]))  # Add top-left y coordinate
            info['xmax'].append(int(box.xyxy[0][2]))  # Add bottom-right x coordinate
            info['ymax'].append(int(box.xyxy[0][3]))  # Add bottom-right y coordinate
        return info

    def auto_save_image_path(self, filename):
        output_dir = os.path.join("results", "images")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"processed_{filename}")

    def auto_save_video_path(self):
        output_dir = os.path.join("results", "videos")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(output_dir, f"processed_{timestamp}.mp4")

    def stop(self):
        with self._lock:
            self.running = False
        self.quit()  # Ensure thread exit

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()  # Call parent class init method
        self.setWindowTitle("ShipYOLO")  # Set window title
        self.setGeometry(100, 100, 800, 600)  # Set window size and position
        self.initUI()  # Initialize user interface
        self.video_thread = None  # Initialize video thread to None
        self.df = pd.DataFrame(
            columns=["No.", "Image Name", "Entry Time", "Detection Result", "Target Count", "Time", "Save Path"])  # Initialize data table
        self.row_count = 0  # Initialize row count
        self.selected_video_file = None  # Initialize selected video file path
        self.image_folder = None  # Initialize selected image folder path
        self.selected_image_file = None  # Initialize selected single image path
        self.processing_thread = None  # Initialize processing thread to None
        self.last_result_image = None  # Last processed image
        self.processed_video_frames = []  # List of processed video frames

        # Create Excel file (at program startup)
        self.excel_file_path = self.create_excel_file()  # New

        self.setStyleSheet("""
               ...
           """)

        self.setStyleSheet("""
            QMainWindow {
                background-image: url(background.jpg);
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-size: cover;
            }
            QGroupBox {
                background-color: rgba(255, 255, 255, 200);
                border-radius: 8px;
                padding: 10px;
            }
            QLabel, QPushButton, QTableWidget {
                background-color: rgba(255, 255, 255, 150);
            }
        """)

    def create_excel_file(self):
        """Create Excel file (at program startup)"""
        try:
            # Create results directory
            os.makedirs("results", exist_ok=True)

            # Generate Excel filename (with timestamp)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = os.path.join("results", f"detection_records_{timestamp}.xlsx")

            # Create empty DataFrame and save
            columns = ["No.", "Image Name", "Entry Time", "Detection Result", "Target Count", "Time", "Save Path"]
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_excel(excel_path, index=False, engine='openpyxl')

            logging.info(f"Created Excel file: {excel_path}")
            return excel_path
        except Exception as e:
            logging.error(f"Failed to create Excel file: {str(e)}")
            return None

    # Initialize user interface
    def initUI(self):
        main_widget = QWidget()  # Create main window widget
        main_widget.setObjectName("centralwidget")  # Add object name
        main_widget.setAttribute(Qt.WA_TranslucentBackground)  # Enable transparent background
        self.setCentralWidget(main_widget)  # Set main window widget
        main_layout = QHBoxLayout(main_widget)  # Set main layout to horizontal layout

        # Left control panel
        control_panel = QGroupBox("System Control")  # Create control panel
        control_panel.setFixedWidth(220)  # Fixed width to prevent content from expanding panel
        control_layout = QVBoxLayout()  # Set control panel layout to vertical layout

        # File selection
        btn_style = "QPushButton {padding: 10px; font-size: 14px;}"
        self.btn_single_image = QPushButton("Select Image File")  # Create select image file button
        self.btn_image_folder = QPushButton("Select Image Folder")
        self.btn_video = QPushButton("Select Video File")
        self.btn_run = QPushButton("Start Detection")
        self.btn_quit = QPushButton("Exit System")

        # Set button colors
        self.btn_single_image.setStyleSheet("background-color: #9C27B0; color: white;")
        self.btn_image_folder.setStyleSheet("background-color: #FF9800; color: white;")
        self.btn_video.setStyleSheet("background-color: #607D8B; color: white;")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_quit.setStyleSheet("background-color: #f44336; color: white;")

        # Detection settings
        confidence_group = QGroupBox("Detection Settings")  # Create detection settings panel
        confidence_layout = QFormLayout()  # Set detection settings panel layout to form layout
        self.conf_spinbox = QSpinBox()  # Create confidence threshold spinbox
        self.conf_spinbox.setRange(0, 100)  # Set confidence threshold range
        self.conf_spinbox.setValue(50)  # Set default confidence threshold
        confidence_layout.addRow("Confidence (%):", self.conf_spinbox)  # Add confidence threshold settings to layout
        confidence_group.setLayout(confidence_layout)

        # Add buttons to layout
        for btn in [self.btn_single_image, self.btn_image_folder, self.btn_video,
                    self.btn_run, self.btn_quit]:
            btn.setStyleSheet(btn_style)  # Set button style
            control_layout.addWidget(btn)  # Add button to layout

        # Right display area
        display_panel = QWidget()  # Create display area widget
        display_layout = QVBoxLayout()  # Set display area layout to vertical layout

        # Video display
        self.video_label = QLabel()  # Create video display label
        self.video_label.setAlignment(Qt.AlignCenter)  # Set label content centered
        self.video_label.setMinimumSize(600, 400)  # Set video display area size
        self.video_label.setStyleSheet("border: 2px solid #999;")  # Set video display area border style

        # Detection info
        info_group = QGroupBox("Detection Info")  # Create detection info panel
        info_layout = QFormLayout()  # Set detection info panel layout to form layout
        self.class_label = QLabel("N/A")  # Create class label
        self.class_label.setWordWrap(True)  # Word wrap to prevent text from expanding panel
        self.pos_label = QLabel("N/A")  # Create position label
        self.pos_label.setWordWrap(True)  # Word wrap to prevent text from expanding panel
        info_layout.addRow("Class:", self.class_label)  # Add class label to layout
        info_layout.addRow("Position:", self.pos_label)  # Add position label to layout
        info_group.setLayout(info_layout)  # Set detection info panel layout

        control_layout.addWidget(confidence_group)  # Add detection settings panel to layout
        control_layout.addWidget(info_group)  # Add detection info
        control_panel.setLayout(control_layout)  # Set control panel layout

        # Results table
        self.table = QTableWidget()  # Create results table
        self.table.setColumnCount(7)  # Set table column count
        self.table.setHorizontalHeaderLabels(
            ["No.", "Image Name", "Entry Time", "Detection Result", "Target Count", "Time", "Save Path"])  # Set table column headers
        self.table.horizontalHeader().setStretchLastSection(True)  # Set last column auto-stretch

        display_layout.addWidget(self.video_label)  # Add video display label to layout
        display_layout.addWidget(self.table)  # Add results table to layout
        display_panel.setLayout(display_layout)  # Set display area layout

        main_layout.addWidget(control_panel)  # Add control panel to main layout
        main_layout.addWidget(display_panel)  # Add display area to main layout

        # Connect button signals to slot functions
        self.btn_single_image.clicked.connect(self.select_single_image)  # Connect select image file button to slot
        self.btn_image_folder.clicked.connect(self.select_image_folder)  # Connect select image folder button to slot
        self.btn_video.clicked.connect(self.select_video)  # Connect select video button to slot
        self.btn_run.clicked.connect(self.toggle_detection)  # Connect start detection button to slot
        self.btn_quit.clicked.connect(self.close)  # Connect exit system button to slot

    def select_single_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file:
            print(f"Selected image file: {file}")
            self.selected_image_file = file
            self.image_folder = None
            self.selected_video_file = None
            self.preview_image(file)

    def select_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            print(f"Selected image folder: {folder}")
            self.image_folder = folder
            self.selected_image_file = None
            self.selected_video_file = None

            # Find and preview the first image
            valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
            image_files = [f for f in os.listdir(folder)
                           if f.lower().endswith(valid_ext)]

            if image_files:
                first_image = os.path.join(folder, image_files[0])
                self.preview_image(first_image)
            else:
                QMessageBox.warning(self, "Warning", "No valid images in the folder!")

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if file:
            print(f"Selected video file: {file}")
            self.selected_video_file = file
            self.selected_image_file = None
            self.image_folder = None

            # Extract first frame preview from video
            cap = cv2.VideoCapture(file)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Convert and display first frame
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.set_image(qt_image)
                cap.release()
            else:
                QMessageBox.warning(self, "Error", "Cannot read video file!")

    def preview_image(self, image_path):
        """Preview selected image"""
        frame = cv2.imread(image_path)  # Read image using OpenCV
        if frame is not None:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
            h, w, ch = rgb_image.shape  # Get image height, width and channels
            bytes_per_line = ch * w  # Calculate bytes per line
            convert_to_qt = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)  # Convert to QImage format
            self.set_image(convert_to_qt)  # Update UI to display image

    def toggle_detection(self):
        """Toggle detection state (Start/Stop)"""
        if self.btn_run.text() == "Start Detection":
            # Check if detection content is selected
            if not (self.selected_image_file or self.image_folder or self.selected_video_file):
                # Show info popup
                QMessageBox.information(self, "Info", "Please select detection content first (image file, image folder or video file)")
                return  # Return directly without changing button state
            self.btn_run.setText("Stop Detection")
            self.start_detection()  # Start detection
        else:
            self.btn_run.setText("Start Detection")
            self.stop_detection()  # Stop detection

    def start_detection(self):
        """Start corresponding detection task based on user selection"""
        if self.selected_image_file:
            self.process_single_image()  # Process single image
        elif self.image_folder:
            self.start_folder_processing()  # Start folder processing thread
        elif self.selected_video_file:
            self.start_video_processing()  # Start video processing thread
        else:
            # This won't execute as it's already checked in toggle_detection
            QMessageBox.warning(self, "Warning", "Please select detection content first")
            self.btn_run.setText("Start Detection")  # Restore button state

    def process_single_image(self):
        """Process single image"""
        logging.debug("Starting single image processing...")
        try:
            if not self.selected_image_file:
                return

            start_time = time.time()  # Start timing (seconds)
            frame = cv2.imread(self.selected_image_file)
            if frame is None:
                raise ValueError("Cannot read image file")

            yolo_model = YOLO('last.pt')
            results = yolo_model(frame, conf=self.conf_spinbox.value() / 100.0)

            annotated = results[0].plot()
            info = self.parse_results(results)
            info['filename'] = os.path.basename(self.selected_image_file)

            rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.update_interface(
                convert_to_qt,
                info,
                self.selected_image_file,
                processing_time
            )

            QMessageBox.information(self, "Done", "Image detection completed!")
            logging.debug("Finished single image processing successfully.")

            # Save processed image
            output_path = self.auto_generate_save_path(info['filename'])
            cv2.imwrite(output_path, annotated)
            logging.info(f"Image saved to: {output_path}")

        except Exception as e:
            logging.error(f"Error during single image processing: {e}")
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
        finally:
            logging.debug("Resetting state after single image processing.")
            self.btn_run.setText("Start Detection")  # Ensure button state is reset

    def start_folder_processing(self):
        """Start folder processing thread"""
        if self.image_folder:
            self.btn_run.setText("Stop Detection")
            self.processing_thread = ImageProcessingThread(
                self.image_folder,
                self.conf_spinbox.value() / 100.0,
                self  # Pass main window instance
            )
            # Modify signal connection to receive time parameter
            self.processing_thread.update_frame.connect(
                lambda qt_img, info, path, proc_time: self.update_interface(qt_img, info, path, proc_time),
                Qt.QueuedConnection)
            self.processing_thread.finished.connect(self.on_processing_finished)  # Connect finished signal
            self.processing_thread.start()  # Start thread

    def start_video_processing(self):
        try:
            if not self.selected_video_file:
                return

            # Terminate existing thread
            if self.processing_thread:
                self.stop_detection()

            # Create new thread
            self.processing_thread = ImageProcessingThread(
                self.selected_video_file,
                self.conf_spinbox.value() / 100.0,
                self
            )

            # Connect signal to receive time parameter
            self.processing_thread.update_frame.connect(
                lambda qt_img, info, path, proc_time: self.update_interface(qt_img, info, path, proc_time),
                Qt.QueuedConnection  # Critical: use queued connection for thread safety
            )
            self.processing_thread.finished.connect(
                lambda: [
                    self.on_processing_finished(),
                    self.processing_thread.deleteLater()
                ]
            )

            # Start thread
            self.processing_thread.start()
            logging.info("Video processing thread started")

        except Exception as e:
            logging.error(f"Failed to start video thread: {str(e)}")
            QMessageBox.critical(self, "Error", f"Start failed: {str(e)}")

    def update_interface(self, qt_image, info, source_path, processing_time=0.0):
        """Enhanced interface update"""
        # Force interface refresh
        QApplication.processEvents()
        self.class_label.setText(" | ".join(info.get('class', ['N/A'])))
        try:
            if not qt_image.isNull():
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))

            # Update class info
            self.class_label.setText(", ".join(info.get('class', [])) or "N/A")

            # Update position info
            positions = [
                f"({x},{y})-{w}x{h}"
                for x, y, w, h in zip(
                    info.get('xmin', []),
                    info.get('ymin', []),
                    info.get('xmax', []),
                    info.get('ymax', [])
                )
            ]
            self.pos_label.setText("\n".join(positions) or "N/A")

            # Ensure 'filename' exists in info, otherwise use source_path filename
            filename = info.get('filename', os.path.basename(source_path) if source_path else "N/A")
            self.update_table(info, filename, processing_time)

            # Cache video frame (if video processing)
            if source_path.lower().endswith(('.mp4', '.avi')):
                # Video processing does not cache
                pass

        except Exception as e:
            logging.error(f"Interface update error: {str(e)}")

    def update_table(self, info, filename, processing_time):
        """Auto-generate save path when updating table"""
        try:
            save_path = self.auto_generate_save_path(filename)
            # Ensure all data is valid
            class_list = info.get('class', []) or ["No target detected"]
            # Time display: show milliseconds if >=1, otherwise show seconds
            if processing_time >= 1:
                time_display = f"{processing_time:.2f}ms"
            else:
                time_display = f"{processing_time * 1000:.2f}ms"

            items = [
                str(self.row_count + 1),  # Serial number starts from 1
                filename,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ", ".join(class_list),
                str(len(info.get('class', []))),
                time_display,  # Use millisecond unit
                save_path
            ]

            # Insert into table
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, text in enumerate(items):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)

            self.row_count += 1
            self.table.scrollToBottom()

            # Auto-save to Excel after each table update
            self.save_to_excel()

        except Exception as e:
            logging.error(f"Failed to update table: {str(e)}")

    def stop_detection(self):
        if self.processing_thread:
            try:
                # Stop first then disconnect
                self.processing_thread.stop()
                self.processing_thread.disconnect()
                self.processing_thread = None
                logging.info("Video thread stopped safely")
            except Exception as e:
                logging.error(f"Error stopping thread: {str(e)}")
        self.btn_run.setText("Start Detection")
        # Save Excel on stop too
        self.save_to_excel()

    def on_processing_finished(self):
        """Callback after processing complete"""
        self.btn_run.setText("Start Detection")
        # Auto-save Excel after processing complete
        self.save_to_excel()
        QMessageBox.information(self, "Done", "Processing complete! Detection records saved to Excel file.")

    def set_image(self, image):
        """Set image display"""
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio))

    def _save_processed_video(self, output_path):
        """Save processed video"""
        if not self.processed_video_frames:
            return

        # Get video parameters
        height, width = self.processed_video_frames[0].shape[:2]
        fps = 30  # Default frame rate, can be changed to get from source video

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in self.processed_video_frames:
            out.write(frame)
        out.release()

    def parse_results(self, results):
        """Parse YOLO detection results"""
        info = {
            'class': [],
            'confidence': [],
            'xmin': [],
            'ymin': [],
            'xmax': [],
            'ymax': []
        }
        for box in results[0].boxes:
            info['class'].append(results[0].names[int(box.cls)])
            info['confidence'].append(f"{float(box.conf):.2%}")
            info['xmin'].append(int(box.xyxy[0][0]))
            info['ymin'].append(int(box.xyxy[0][1]))
            info['xmax'].append(int(box.xyxy[0][2]))
            info['ymax'].append(int(box.xyxy[0][3]))
        return info

    def auto_generate_save_path(self, filename):
        """Auto-generate save path based on file type"""
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            return os.path.join("results", "images", f"processed_{filename}")
        elif filename.lower().endswith(('.mp4', '.avi')):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return os.path.join("results", "videos", f"processed_{timestamp}.mp4")
        return "Auto-save path"

    def save_to_excel(self):
        """Append table data to Excel file"""
        try:
            if not self.excel_file_path:
                return

            # Prepare Excel data
            excel_data = []
            for row in range(self.table.rowCount()):
                row_data = []
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    row_data.append(item.text() if item else "")
                excel_data.append(row_data)

            # Create DataFrame
            columns = ["No.", "Image Name", "Entry Time", "Detection Result", "Target Count", "Time", "Save Path"]
            df = pd.DataFrame(excel_data, columns=columns)

            # Save to Excel (overwrite, as table already contains all historical data)
            df.to_excel(self.excel_file_path, index=False, engine='openpyxl')
            logging.info(f"Detection records saved to: {self.excel_file_path}")

        except Exception as e:
            logging.error(f"Failed to save Excel: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
