import os
import sys
import fitz  # PyMuPDF
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTreeWidget,
    QTreeWidgetItem, QHeaderView, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_date = datetime.now().strftime('%Y-%m-%d')
log_file = os.path.join(log_dir, f'pdfcounter_{current_date}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging started. Log file: {log_file}")

class PDFCounterThread(QThread):
    progress = pyqtSignal(int, int)  # current, total
    result = pyqtSignal(list)  # list of (file_path, page_count) tuples
    error = pyqtSignal(str)
    file_skipped = pyqtSignal(str, str)  # file_path, error_message

    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.is_running = True

    def run(self):
        try:
            results = []
            skipped_files = []
            
            # Get all PDF files
            pdf_files = []
            for root, _, files in os.walk(self.directory):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))

            total_files = len(pdf_files)
            logger.info(f"Found {total_files} PDF files to process in {self.directory}")
            
            for i, file_path in enumerate(pdf_files):
                if not self.is_running:
                    break
                
                try:
                    logger.info(f"Processing file: {file_path}")
                    doc = fitz.open(file_path)
                    page_count = len(doc)
                    results.append((file_path, page_count))
                    doc.close()
                    logger.info(f"Successfully processed {file_path}: {page_count} pages")
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    logger.error(error_msg)
                    self.file_skipped.emit(file_path, str(e))
                    skipped_files.append((file_path, str(e)))
                
                self.progress.emit(i + 1, total_files)

            if skipped_files:
                logger.warning(f"Skipped {len(skipped_files)} files due to errors")
                for file_path, error in skipped_files:
                    logger.warning(f"Skipped {file_path}: {error}")

            self.result.emit(results)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)

    def stop(self):
        self.is_running = False
        logger.info("Counting process stopped by user")

class PDFCounterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Page Counter")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select directory containing PDF files...")
        dir_layout.addWidget(self.dir_input)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_button)
        
        layout.addLayout(dir_layout)
        
        # Start button
        self.start_button = QPushButton("Start Counting")
        self.start_button.clicked.connect(self.start_counting)
        layout.addWidget(self.start_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results tree
        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["File Path", "Page Count", "Status"])
        self.results_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.results_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(self.results_tree)
        
        # Summary label
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)
        
        # Initialize variables
        self.counter_thread = None
        self.total_pages = 0
        self.total_files = 0
        self.skipped_files = 0

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)
            logger.info(f"Selected directory: {directory}")

    def start_counting(self):
        directory = self.dir_input.text()
        if not directory or not os.path.isdir(directory):
            QMessageBox.warning(self, "Error", "Please select a valid directory")
            return
        
        # Clear previous results
        self.results_tree.clear()
        self.summary_label.clear()
        self.total_pages = 0
        self.total_files = 0
        self.skipped_files = 0
        
        # Disable UI elements
        self.start_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start counting thread
        self.counter_thread = PDFCounterThread(directory)
        self.counter_thread.progress.connect(self.update_progress)
        self.counter_thread.result.connect(self.show_results)
        self.counter_thread.error.connect(self.show_error)
        self.counter_thread.file_skipped.connect(self.handle_skipped_file)
        self.counter_thread.start()
        
        logger.info("Started counting process")

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def handle_skipped_file(self, file_path, error_message):
        self.skipped_files += 1
        item = QTreeWidgetItem(self.results_tree)
        item.setText(0, file_path)
        item.setText(1, "0")
        item.setText(2, f"Error: {error_message}")
        item.setForeground(2, Qt.red)

    def show_results(self, results):
        # Sort results by file path
        results.sort(key=lambda x: x[0])
        
        # Add results to tree
        for file_path, page_count in results:
            item = QTreeWidgetItem(self.results_tree)
            item.setText(0, file_path)
            item.setText(1, str(page_count))
            item.setText(2, "Success")
            item.setForeground(2, Qt.darkGreen)
            self.total_pages += page_count
            self.total_files += 1
        
        # Update summary
        summary_text = (
            f"Total PDF files: {self.total_files}\n"
            f"Total pages: {self.total_pages}\n"
            f"Skipped files: {self.skipped_files}"
        )
        self.summary_label.setText(summary_text)
        
        # Re-enable UI
        self.start_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        logger.info(f"Counting completed. Processed {self.total_files} files, {self.total_pages} pages, skipped {self.skipped_files} files")

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.start_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        logger.error(f"Error in counting process: {error_message}")

    def closeEvent(self, event):
        if self.counter_thread and self.counter_thread.isRunning():
            self.counter_thread.stop()
            self.counter_thread.wait()
        event.accept()
        logger.info("Application closed")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PDFCounterApp()
    window.show()
    sys.exit(app.exec_())
