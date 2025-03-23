import sys
import fitz  # PyMuPDF
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                            QScrollArea, QSizePolicy, QSplitter, QTextEdit, QComboBox,
                            QDialog, QLineEdit, QDialogButtonBox, QMessageBox,
                            QStatusBar)
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import os
import openai  # Revert to standard import
from dotenv import load_dotenv
import time
import json
import keyring  # For secure storage of API key
from fpdf import FPDF  # Replace with fpdf2 for better Unicode support
import logging
from datetime import datetime

# Load environment variables for API keys
load_dotenv()

# Set OpenAI API key the old way
openai.api_key = os.getenv('OPENAI_API_KEY')

# Try to determine which version of the OpenAI package we're using
try:
    # Use importlib.metadata instead of pkg_resources
    import importlib.metadata
    openai_version = importlib.metadata.version("openai")
    is_new_version = int(openai_version.split('.')[0]) >= 1
except:
    # If we can't determine version, assume old version
    is_new_version = False

def setup_logging():
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Use date in filename to create a new log file each day
    log_filename = f"logs/pdf_translator_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_filename,
        filemode='a'  # Append mode
    )
    
    logging.info("=== Application Started ===")
    return logging.getLogger('PDFTranslator')

# Create a worker thread for background translation
class TranslationWorker(QThread):
    # Signal to emit when translation is complete
    translationComplete = pyqtSignal(str, int, str)
    
    def __init__(self, text, page_num, input_language, output_language, output_language_name, model="gpt-3.5-turbo"):
        super().__init__()
        self.text = text
        self.page_num = page_num
        self.input_language = input_language
        self.output_language = output_language
        self.output_language_name = output_language_name
        self.model = model
        self.is_running = True
        
    def run(self):
        try:
            # Check if thread should still be running
            if not self.is_running:
                return
                
            # Check if we have an API key
            if not os.getenv('OPENAI_API_KEY'):
                self.translationComplete.emit("Translation error: No API key available", self.page_num, self.output_language_name)
                return
            
            # Prepare the messages for GPT
            messages = [
                {"role": "system", "content": f"You are a professional Korean translator. Translate the following {self.input_language} text into fluent and natural {self.output_language_name}. Use appropriate academic tone and accurate terminology."},
                {"role": "user", "content": f"Translate the following text to {self.output_language_name}:\n\n{self.text}"}
            ]
            
            # Log the API request (truncate text if too long to avoid huge log files)
            display_text = self.text[:200] + "..." if len(self.text) > 200 else self.text
            logger.info(f"API Request - Page {self.page_num+1} - {self.input_language} to {self.output_language_name} - Model: {self.model}")
            logger.debug(f"Full request: {messages}")
            
            # Call OpenAI API
            if is_new_version:
                # For openai >= 1.0.0
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                translated = response.choices[0].message.content.strip()
            else:
                # For openai < 1.0.0 (older versions)
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                translated = response.choices[0].message['content'].strip()
            
            # Log the API response (truncate if too long)
            display_translated = translated[:200] + "..." if len(translated) > 200 else translated
            logger.info(f"API Response - Page {self.page_num+1} - Success - Length: {len(translated)} chars")
            logger.debug(f"Full response: {display_translated}")
            
            # Emit signal with translation results
            self.translationComplete.emit(translated, self.page_num, self.output_language_name)
            
        except Exception as e:
            # Log the error
            logger.error(f"Translation error on page {self.page_num+1}: {str(e)}")
            
            if self.is_running:  # Only emit signal if thread hasn't been terminated
                self.translationComplete.emit(f"Translation error: {str(e)}", self.page_num, self.output_language_name)
    
    def stop(self):
        """Signal the thread to stop"""
        self.is_running = False

class ApiKeyDialog(QDialog):
    """Dialog to request API key from user"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenAI API Key Required")
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add explanation
        layout.addWidget(QLabel("An OpenAI API key is required for translation functionality."))
        layout.addWidget(QLabel("Your key will be stored securely using system keyring."))
        layout.addWidget(QLabel("You can get an API key from: https://platform.openai.com/api-keys"))
        
        # Add input field
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Enter your OpenAI API key (sk-...)")
        self.key_input.setEchoMode(QLineEdit.Password)  # Mask the input
        layout.addWidget(self.key_input)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

class TranslateAllDialog(QDialog):
    """Dialog to show cost estimation and confirm full document translation"""
    def __init__(self, parent=None, page_count=0, token_estimate=0, cost_estimate=0):
        super().__init__(parent)
        self.setWindowTitle("Translate Entire Document")
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add explanation and cost estimation
        layout.addWidget(QLabel(f"You are about to translate all {page_count} pages of this document."))
        layout.addWidget(QLabel(f"Estimated token count: {token_estimate:,}"))
        layout.addWidget(QLabel(f"Estimated cost: ${cost_estimate:.2f}"))
        layout.addWidget(QLabel("Note: Actual token usage and cost may vary based on text content."))
        
        # Add progress information
        layout.addWidget(QLabel("The translation will run in the background. You can continue browsing the document."))
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Ok).setText("Translate All")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

class TranslationProgressBar(QWidget):
    """Visual indicator showing translated and untranslated pages as colored blocks"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Reduce the minimum height for a more minimal appearance
        self.setMinimumHeight(15)  # Reduced from 30
        self.setMaximumHeight(15)  # Add maximum height to keep it compact
        self.setMinimumWidth(200)
        
        # Initialize with no pages
        self.total_pages = 0
        self.translated_pages = set()  # Set of translated page indices
        self.translating_pages = set()  # Set of pages currently being translated
        self.current_page = 0
        self.aggregation_factor = 1  # How many pages are represented by one pixel
        
        # Set background to a light gray
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.lightGray)
        self.setPalette(p)
        
    def set_document_info(self, total_pages):
        """Update the total number of pages"""
        self.total_pages = total_pages
        self.translated_pages = set()
        self.translating_pages = set()
        self.current_page = 0
        
        # Calculate aggregation factor based on widget width and total pages
        if total_pages > 0:
            # Estimate available pixels (width - 2 for borders)
            available_pixels = self.width() - 2
            # If we have more pages than pixels, we need to aggregate
            if total_pages > available_pixels:
                self.aggregation_factor = max(1, (total_pages + available_pixels - 1) // available_pixels)
            else:
                self.aggregation_factor = 1
                
        self.update()
        
    def mark_page_translated(self, page_num):
        """Mark a page as translated"""
        self.translated_pages.add(page_num)
        # Remove from translating pages if it was there
        if page_num in self.translating_pages:
            self.translating_pages.remove(page_num)
        self.update()
        
    def mark_page_translating(self, page_num):
        """Mark a page as currently being translated"""
        if page_num not in self.translated_pages:  # Only mark as translating if not already translated
            self.translating_pages.add(page_num)
            self.update()
    
    def unmark_page_translating(self, page_num):
        """Remove a page from the translating set"""
        if page_num in self.translating_pages:
            self.translating_pages.remove(page_num)
            self.update()
        
    def set_current_page(self, page_num):
        """Set the current page indicator"""
        self.current_page = page_num
        self.update()
        
    def clear(self):
        """Reset the progress bar"""
        self.total_pages = 0
        self.translated_pages = set()
        self.translating_pages = set()
        self.current_page = 0
        self.aggregation_factor = 1
        self.update()
        
    def get_page_group_status(self, group_start, group_end):
        """Determine the status of a group of pages (for aggregation)"""
        # Count pages in each state within this group
        translated_count = 0
        translating_count = 0
        total_count = min(group_end - group_start, self.total_pages - group_start)
        
        for i in range(group_start, min(group_end, self.total_pages)):
            if i in self.translated_pages:
                translated_count += 1
            elif i in self.translating_pages:
                translating_count += 1
        
        # Determine dominant status based on thresholds
        # If more than 50% of pages in group are translated, consider the group translated
        if translated_count > total_count * 0.5:
            return "translated"
        # If more than 25% of pages are being translated (and not already considered translated), mark as translating
        elif translating_count > total_count * 0.25:
            return "translating"
        # Otherwise, the group is predominantly untranslated
        else:
            return "untranslated"
    
    def resizeEvent(self, event):
        """Handle resize events to recalculate aggregation"""
        super().resizeEvent(event)
        # Recalculate aggregation factor when the widget is resized
        if self.total_pages > 0:
            available_pixels = self.width() - 2
            if self.total_pages > available_pixels:
                self.aggregation_factor = max(1, (self.total_pages + available_pixels - 1) // available_pixels)
            else:
                self.aggregation_factor = 1
        self.update()
        
    def paintEvent(self, event):
        """Draw the blocks representing pages"""
        if self.total_pages == 0:
            return
            
        painter = QPainter(self)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Draw the background first (outer frame)
        painter.setPen(Qt.darkGray)
        painter.drawRect(0, 0, width - 1, height - 1)
        
        # Available width for blocks (accounting for borders)
        available_width = width - 2
        
        # If aggregation is needed (more pages than pixels)
        if self.aggregation_factor > 1:
            # Calculate how many "groups" of pages we'll display
            num_groups = (self.total_pages + self.aggregation_factor - 1) // self.aggregation_factor
            block_width = available_width / num_groups
            
            # Draw each group
            painter.setPen(Qt.NoPen)
            for g in range(num_groups):
                group_start = g * self.aggregation_factor
                group_end = min((g + 1) * self.aggregation_factor, self.total_pages)
                
                # Get the status of this group
                group_status = self.get_page_group_status(group_start, group_end)
                
                # Determine color based on status
                if group_status == "translated":
                    color = QColor(0, 180, 0)  # Green
                elif group_status == "translating":
                    color = QColor(255, 215, 0)  # Gold/Yellow
                else:
                    color = QColor(220, 0, 0)  # Red
                
                # Calculate x position and draw the block
                x = 1 + (g * block_width)
                painter.fillRect(int(x), 1, int(block_width) + 1, height - 2, color)
                
            # Show current page indicator
            if 0 <= self.current_page < self.total_pages:
                current_group = self.current_page // self.aggregation_factor
                x = 1 + (current_group * block_width)
                painter.setPen(QPen(QColor(0, 0, 255), 1))  # Blue, 1px width
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(int(x), 1, int(block_width), height - 2)
                
        else:
            # Original drawing logic for when we have enough pixels for each page
            block_width = available_width / self.total_pages
            
            # Draw blocks for all pages based on their status
            painter.setPen(Qt.NoPen)
            
            # First draw all untranslated pages in red
            for i in range(self.total_pages):
                if i not in self.translated_pages and i not in self.translating_pages:
                    x = 1 + (i * block_width)
                    painter.fillRect(int(x), 1, int(block_width) + 1, height - 2, QColor(220, 0, 0))  # Red
            
            # Then draw all translated pages in green
            for i in range(self.total_pages):
                if i in self.translated_pages:
                    x = 1 + (i * block_width)
                    painter.fillRect(int(x), 1, int(block_width) + 1, height - 2, QColor(0, 180, 0))  # Green
            
            # Then draw all currently translating pages in yellow
            for i in range(self.total_pages):
                if i in self.translating_pages:
                    x = 1 + (i * block_width)
                    painter.fillRect(int(x), 1, int(block_width) + 1, height - 2, QColor(255, 215, 0))  # Gold/Yellow
            
            # Draw the current page indicator with a blue border
            if 0 <= self.current_page < self.total_pages:
                x = 1 + (self.current_page * block_width)
                painter.setPen(QPen(QColor(0, 0, 255), 1))  # Blue, 1px width
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(int(x), 1, int(block_width), height - 2)

class PDFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up logging
        global logger
        logger = setup_logging()
        
        self.current_page = 0
        self.doc = None
        self.extracted_text = ""
        self.translation_cache = {}  # Dictionary to store translated text: {(page_num, language): translated_text}
        self.look_ahead_worker = None  # Keep reference to worker thread
        self.active_workers = []  # Keep track of all active worker threads
        
        # Check for API key on startup
        self.check_api_key()
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('PDF Translator')
        self.setGeometry(100, 100, 1200, 800)  # Wider window for two panes
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create buttons
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton('Open PDF')
        self.open_btn.clicked.connect(self.open_pdf)
        
        self.prev_btn = QPushButton('Previous Page')
        self.prev_btn.clicked.connect(self.prev_page)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton('Next Page')
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        
        self.page_label = QLabel('Page:')
        self.page_input = QLineEdit()
        self.page_input.setFixedWidth(50)  # Keep it reasonably sized
        self.page_input.setAlignment(Qt.AlignCenter)
        self.page_total = QLabel('/ 0')

        # Create a layout for the page navigation
        page_layout = QHBoxLayout()
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(5)
        page_layout.addWidget(self.page_label)
        page_layout.addWidget(self.page_input)
        page_layout.addWidget(self.page_total)

        # Connect the enter/return pressed signal
        self.page_input.returnPressed.connect(self.go_to_page)

        # Add input language selection combo box with English as default
        self.input_language_combo = QComboBox()
        self.input_language_combo.addItem("English", "en")  # Move English to first position (default)
        self.input_language_combo.addItem("Auto-detect", "auto")
        self.input_language_combo.addItem("Spanish", "es")
        self.input_language_combo.addItem("French", "fr")
        self.input_language_combo.addItem("German", "de")
        self.input_language_combo.addItem("Italian", "it")
        self.input_language_combo.addItem("Portuguese", "pt")
        self.input_language_combo.addItem("Russian", "ru")
        self.input_language_combo.addItem("Japanese", "ja")
        self.input_language_combo.addItem("Korean", "ko")
        self.input_language_combo.addItem("Chinese", "zh")
        
        # Add output language selection combo box
        self.output_language_combo = QComboBox()
        self.output_language_combo.addItem("Korean", "ko")
        self.output_language_combo.addItem("Japanese", "ja")
        self.output_language_combo.addItem("Chinese", "zh")
        self.output_language_combo.addItem("Spanish", "es")
        self.output_language_combo.addItem("French", "fr")
        
        # Connect language change signal
        self.output_language_combo.currentTextChanged.connect(self.on_output_language_changed)
        
        # Add model selection combo box with GPT-3.5 and GPT-4 options
        self.model_combo = QComboBox()
        self.model_combo.addItem("GPT-3.5 Turbo", "gpt-3.5-turbo")
        self.model_combo.addItem("GPT-4 Turbo", "gpt-4-turbo")
        self.model_combo.setToolTip("Select the AI model for translation")
        
        # Add debug level selection combo box
        self.debug_level_combo = QComboBox()
        self.debug_level_combo.addItem("Error", logging.ERROR)
        self.debug_level_combo.addItem("Warning", logging.WARNING)
        self.debug_level_combo.addItem("Info", logging.INFO)
        self.debug_level_combo.addItem("Debug", logging.DEBUG)
        self.debug_level_combo.setToolTip("Set logging verbosity level")
        
        # Connect debug level change signal
        self.debug_level_combo.currentIndexChanged.connect(self.on_debug_level_changed)
        
        # Add translate button
        self.translate_btn = QPushButton('Translate')
        self.translate_btn.clicked.connect(lambda: self.translate_text(True))  # Force new translation when button clicked
        self.translate_btn.setEnabled(False)
        
        # Add translate all button
        self.translate_all_btn = QPushButton('Translate All')
        self.translate_all_btn.clicked.connect(self.translate_all)
        self.translate_all_btn.setEnabled(False)
        
        # Add export button
        self.export_btn = QPushButton('Export')
        self.export_btn.clicked.connect(self.export_translation)
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("Export translated document")
        
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addLayout(page_layout)  # Use the new layout instead of page_label
        btn_layout.addWidget(QLabel("From:"))
        btn_layout.addWidget(self.input_language_combo)
        btn_layout.addWidget(QLabel("To:"))
        btn_layout.addWidget(self.output_language_combo)
        btn_layout.addWidget(QLabel("Model:"))
        btn_layout.addWidget(self.model_combo)
        btn_layout.addWidget(QLabel("Debug:"))
        btn_layout.addWidget(self.debug_level_combo)
        btn_layout.addWidget(self.translate_btn)
        btn_layout.addWidget(self.translate_all_btn)
        btn_layout.addWidget(self.export_btn)
        
        # Create a widget for the status and progress bar at the bottom
        status_layout = QHBoxLayout()
        
        # Create a status bar widget (rename to avoid conflict with Qt's statusBar() method)
        self.status_label = QStatusBar()
        
        # Create the translation progress widget
        self.translation_progress = TranslationProgressBar()
        
        # Add widgets to the status layout
        # Give the progress bar less horizontal space (20% instead of 30%)
        status_layout.addWidget(self.status_label, 8)  # Status bar takes 80% of width
        status_layout.addWidget(self.translation_progress, 2)  # Progress bar takes 20% of width
        
        # Create splitter for the two panes
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left pane - PDF View
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        
        # Create scroll area for PDF display
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Create label to display PDF
        self.pdf_label = QLabel()
        self.pdf_label.setAlignment(Qt.AlignCenter)
        self.pdf_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.scroll_area.setWidget(self.pdf_label)
        
        left_layout.addWidget(self.scroll_area)
        
        # Right pane - Translation
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        
        # Text display for extracted text
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        self.original_text.setPlaceholderText("Original text will appear here")
        
        # Text display for translated text
        self.translated_text = QTextEdit()
        self.translated_text.setReadOnly(True)
        self.translated_text.setPlaceholderText("Translated text will appear here")
        
        # Create a splitter for the text widgets to allow user resizing
        text_splitter = QSplitter(Qt.Vertical)
        
        # Create containers for the text widgets
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.addWidget(QLabel("Original Text:"))
        original_layout.addWidget(self.original_text)
        
        translated_container = QWidget()
        translated_layout = QVBoxLayout(translated_container)
        translated_layout.addWidget(QLabel("Translated Text:"))
        translated_layout.addWidget(self.translated_text)
        
        # Add containers to the splitter
        text_splitter.addWidget(original_container)
        text_splitter.addWidget(translated_container)
        
        # Set the initial sizes - 1:3 ratio (25% original, 75% translated)
        text_splitter.setSizes([200, 600])
        
        # Add the splitter to the right layout
        right_layout.addWidget(text_splitter)
        
        # Add widgets to main splitter
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        self.splitter.setSizes([600, 600])  # Equal initial sizes
        
        # Add everything to main layout
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.splitter)
        main_layout.addLayout(status_layout)  # Add status/progress bar at bottom
        
    def open_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        
        if file_path:
            # Log the file opening
            logger.info(f"Opening PDF: {os.path.basename(file_path)}")
            
            # Clean up any running threads before opening new document
            self.cleanup_threads()
            
            # Ensure normal cursor at the start
            self.ensure_normal_cursor()
            
            # Close previous document if one is open
            if self.doc:
                self.doc.close()
                
            # Reset the progress bar immediately
            self.translation_progress.clear()
                
            # Open new document
            self.doc = fitz.open(file_path)
            self.current_page = 0
            
            # Clear translation cache when opening a new document
            self.translation_cache = {}
            
            # Reset detected language when opening a new PDF
            self.detected_language = None
            
            # Update UI
            self.update_page_display()
            self.update_buttons()
            self.extract_text()
            self.translate_btn.setEnabled(True)
            self.translate_all_btn.setEnabled(True)  # Enable the Translate All button
            self.export_btn.setEnabled(True)  # Enable the Export button
            
            # Clear translated text area
            self.translated_text.clear()
            
            # Check if there's any text to translate
            if not self.extracted_text or self.extracted_text.strip() == "":
                self.translated_text.setText("No text to translate on this page.")
                return
            
            # If API key is available, automatically translate the first page
            if os.getenv('OPENAI_API_KEY'):
                # Show wait cursor and translation starting message
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.translated_text.setText("Translating first page...")
                QApplication.processEvents()  # Update UI immediately
                
                # Get input and output languages
                input_lang_code = self.input_language_combo.currentData()
                input_lang_name = self.input_language_combo.currentText()
                
                # If auto-detect, we need to detect the language
                if input_lang_code == "auto":
                    self.status_label.showMessage("Detecting language...")
                    QApplication.processEvents()  # Update UI immediately
                    self.detected_language = self.detect_language(self.extracted_text)
                    input_lang_code = self.detected_language
                    
                    # Update input language combo box
                    for i in range(self.input_language_combo.count()):
                        if self.input_language_combo.itemData(i) == input_lang_code:
                            input_lang_name = self.input_language_combo.itemText(i)
                            break
                    
                    self.status_label.showMessage(f"Detected language: {input_lang_name}", 3000)
                
                output_lang_name = self.output_language_combo.currentText()
                output_lang_code = self.output_language_combo.currentData()
                
                # Create worker thread for first page
                worker = TranslationWorker(
                    self.extracted_text,
                    self.current_page,
                    input_lang_code,
                    output_lang_code,
                    output_lang_name,
                    self.model_combo.currentData()
                )
                
                # Connect signals before starting
                worker.translationComplete.connect(self.on_translation_complete)
                
                # Add to active workers
                self.active_workers.append(worker)
                
                # Connect finished signal to remove from active workers
                worker.finished.connect(
                    lambda: self.active_workers.remove(worker) if worker in self.active_workers else None
                )
                
                # Start the worker
                worker.start()
            else:
                # Inform user they need an API key for translation
                self.translated_text.setText("Translation requires an API key. Click 'Translate' to provide one.")
            
            # Start look-ahead translation for next page if available
            self.start_look_ahead_translation()
            
            # Now set up the progress bar for the new document
            self.translation_progress.set_document_info(len(self.doc))
            self.translation_progress.set_current_page(self.current_page)
    
    def update_page_display(self):
        if not self.doc:
            return
            
        # Get the current page
        page = self.doc.load_page(self.current_page)
        
        # Get the view area size
        view_width = self.scroll_area.width() - 20  # Subtract some padding
        view_height = self.scroll_area.height() - 20
        
        # Calculate scaling factors to fit width and height
        page_rect = page.rect
        width_scale = view_width / page_rect.width
        height_scale = view_height / page_rect.height
        
        # Use the smaller scale to ensure the whole page fits
        scale = min(width_scale, height_scale)
        
        # Use scaled matrix for rendering
        matrix = fitz.Matrix(scale, scale)
        
        # Render page to an image
        pix = page.get_pixmap(matrix=matrix)
        
        # Convert to QImage
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(img)
        self.pdf_label.setPixmap(pixmap)
        
        # Update page display with 1-based page numbers
        self.page_input.setText(str(self.current_page + 1))
        self.page_total.setText(f"/ {len(self.doc)}")
        
        # Request repaint to ensure proper display
        self.scroll_area.viewport().update()
    
    def update_buttons(self):
        if not self.doc:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
            
        # Enable/disable previous button
        self.prev_btn.setEnabled(self.current_page > 0)
        
        # Enable/disable next button
        self.next_btn.setEnabled(self.current_page < len(self.doc) - 1)
    
    def next_page(self):
        if self.doc and self.current_page < len(self.doc) - 1:
            # Get the next page number
            next_page_num = self.current_page + 1
            language_name = self.output_language_combo.currentText()
            cache_key = (next_page_num, language_name)
            
            # Only clean up unneeded threads, preserving any that are translating our target page
            self.cleanup_threads()
            
            # Move to next page immediately
            self.current_page = next_page_num
            self.update_page_display()
            self.update_buttons()
            self.extract_text()
            
            # Clear translated text area before checking cache
            self.translated_text.clear()
            QApplication.processEvents()  # Update UI immediately
            
            # Check if there's any text to translate
            if not self.extracted_text or self.extracted_text.strip() == "":
                self.translated_text.setText("No text to translate on this page.")
                # Ensure normal cursor
                self.ensure_normal_cursor()
                return
            
            # Check if translation exists in cache
            if cache_key in self.translation_cache:
                # If we have a cached translation, display it and ensure normal cursor
                self.translated_text.setText(self.translation_cache[cache_key])
                # Make sure to restore cursor if it was waiting
                if QApplication.overrideCursor() is not None:
                    QApplication.restoreOverrideCursor()
            else:
                # Only show "Waiting for translation" if we have an API key
                if os.getenv('OPENAI_API_KEY'):
                    # First restore any existing override cursor to avoid stacking
                    while QApplication.overrideCursor() is not None:
                        QApplication.restoreOverrideCursor()
                        
                    # Then set wait cursor since we're actively waiting for translation
                    self.translated_text.setText("Waiting for translation...")
                    QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                else:
                    # No API key - inform the user
                    self.translated_text.setText("Translation requires an API key. Click 'Translate' to provide one.")
                    # Make sure cursor is normal
                    if QApplication.overrideCursor() is not None:
                        QApplication.restoreOverrideCursor()
            
            # Start look-ahead translation for next page if available
            # (This won't restart an already ongoing translation)
            self.start_look_ahead_translation()
            
            # Update current page in progress bar
            self.translation_progress.set_current_page(self.current_page)
    
    def prev_page(self):
        if self.doc and self.current_page > 0:
            # Get the previous page number
            prev_page_num = self.current_page - 1
            language_name = self.output_language_combo.currentText()
            cache_key = (prev_page_num, language_name)
            
            # Only clean up unneeded threads
            self.cleanup_threads()
            
            # Move to previous page immediately
            self.current_page = prev_page_num
            self.update_page_display()
            self.update_buttons()
            self.extract_text()
            
            # Clear translated text area before checking cache
            self.translated_text.clear()
            QApplication.processEvents()  # Update UI immediately
            
            # Check if there's any text to translate
            if not self.extracted_text or self.extracted_text.strip() == "":
                self.translated_text.setText("No text to translate on this page.")
                # Ensure normal cursor
                self.ensure_normal_cursor()
                return
            
            # Check if translation exists in cache
            if cache_key in self.translation_cache:
                # If we have a cached translation, display it and ensure normal cursor
                self.translated_text.setText(self.translation_cache[cache_key])
                # Make sure to restore cursor if it was waiting
                if QApplication.overrideCursor() is not None:
                    QApplication.restoreOverrideCursor()
            else:
                # Only show "Waiting for translation" if we have an API key
                if os.getenv('OPENAI_API_KEY'):
                    # First restore any existing override cursor to avoid stacking
                    while QApplication.overrideCursor() is not None:
                        QApplication.restoreOverrideCursor()
                        
                    # Then set wait cursor since we're actively waiting for translation
                    self.translated_text.setText("Waiting for translation...")
                    QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                else:
                    # No API key - inform the user
                    self.translated_text.setText("Translation requires an API key. Click 'Translate' to provide one.")
                    # Make sure cursor is normal
                    if QApplication.overrideCursor() is not None:
                        QApplication.restoreOverrideCursor()
            
            # Start look-ahead translation for next page if available
            self.start_look_ahead_translation()
            
            # Update current page in progress bar
            self.translation_progress.set_current_page(self.current_page)
    
    def extract_text(self):
        """Extract text from the current PDF page"""
        if not self.doc:
            return
            
        page = self.doc.load_page(self.current_page)
        text = page.get_text()
        
        # Check if the text ends with an incomplete sentence
        if self.current_page < len(self.doc) - 1 and not self._ends_with_sentence_terminator(text):
            # Get the first few lines from the next page to complete the context
            next_page = self.doc.load_page(self.current_page + 1)
            next_page_text = next_page.get_text()
            
            # Get the first few lines from the next page (up to 200 characters or first period)
            continuation_text = self._extract_continuation(next_page_text, 200)
            
            # Add a visual separator
            text += "\n[...continuation from next page...]\n" + continuation_text
        
        self.extracted_text = text
        self.original_text.setText(self.extracted_text)
        
        # Auto-detect language if set to auto-detect and we have text
        if self.input_language_combo.currentData() == "auto" and text.strip():
            if not hasattr(self, 'detected_language') or self.detected_language is None:
                # Show status message
                self.status_label.showMessage("Detecting language...")
                QApplication.processEvents()  # Update UI immediately
                
                # Detect language
                self.detected_language = self.detect_language(text)
                
                # Update input language combo box
                for i in range(self.input_language_combo.count()):
                    if self.input_language_combo.itemData(i) == self.detected_language:
                        self.input_language_combo.setCurrentIndex(i)
                        break
                
                # Clear status message
                self.status_label.showMessage(f"Detected language: {self.detected_language}", 3000)
    
    def _ends_with_sentence_terminator(self, text):
        """Check if text ends with proper sentence termination"""
        # Strip whitespace from the end
        text = text.rstrip()
        
        # If empty string or no text, return True (no need for continuation)
        if not text:
            return True
        
        # Check if the text ends with common sentence terminators
        terminators = ['.', '!', '?', '."', '!"', '?"', '.")', '!")]', '?")]', ':', ';']
        
        # Also check for some common abbreviations that might end a sentence
        abbreviations = [' etc.', ' et al.', ' e.g.', ' i.e.']
        
        for term in terminators:
            if text.endswith(term):
                # Check that it's not a common abbreviation when ending with a period
                if term == '.':
                    for abbr in abbreviations:
                        if text.endswith(abbr):
                            return True
                return True
        
        return False
    
    def _extract_continuation(self, text, max_chars=200):
        """Extract the first part of text up to a logical break point"""
        # Get the text up to max_chars
        continuation = text[:max_chars]
        
        # Find the last sentence terminator in the continuation
        last_period = max(
            continuation.rfind('.'), 
            continuation.rfind('!'), 
            continuation.rfind('?')
        )
        
        # If we found a terminator, cut the text there
        if last_period > 0 and last_period < len(continuation) - 1:
            # Check that it's not part of an abbreviation like "Dr."
            if not continuation[last_period-2:last_period+1] in [' Dr.', ' Mr.', ' Ms.', 'Mrs.']:
                return continuation[:last_period+1]
        
        # If no good break point found or it's too early in the text, 
        # try to find the last complete line
        last_newline = continuation.rfind('\n')
        if last_newline > 0:
            return continuation[:last_newline]
        
        return continuation
    
    def check_translation_cache(self):
        """Check if a translation for the current page and language exists in cache"""
        output_lang_name = self.output_language_combo.currentText()
        cache_key = (self.current_page, output_lang_name)
        
        if cache_key in self.translation_cache:
            # Translation exists in cache, display it
            self.translated_text.setText(self.translation_cache[cache_key])
        else:
            # No translation in cache, clear the translated text area
            self.translated_text.clear()
    
    def start_look_ahead_translation(self):
        """Start background translation of the next page if it exists and isn't translated yet"""
        # Check if we have an API key before proceeding
        if not os.getenv('OPENAI_API_KEY'):
            # No API key available, don't attempt look-ahead translation
            return
        
        # Only look ahead if we have a document and there's a next page
        if not self.doc or self.current_page >= len(self.doc) - 1:
            return
        
        next_page_num = self.current_page + 1
        language_name = self.output_language_combo.currentText()
        cache_key = (next_page_num, language_name)
        
        # Check if next page is already translated
        if cache_key in self.translation_cache:
            return  # Already translated, nothing to do
        
        # Check if this translation is already in progress
        for worker in self.active_workers:
            if worker.page_num == next_page_num and worker.output_language_name == language_name:
                return  # Translation already in progress
        
        # Get text from next page
        next_page = self.doc.load_page(next_page_num)
        next_page_text = next_page.get_text()
        
        # Check if there's any text to translate
        if not next_page_text or next_page_text.strip() == "":
            return  # No text to translate
        
        # Get input and output languages
        input_lang_code = self.input_language_combo.currentData()
        if input_lang_code == "auto" and hasattr(self, 'detected_language') and self.detected_language:
            input_lang_code = self.detected_language
        
        output_lang_name = self.output_language_combo.currentText()
        output_lang_code = self.output_language_combo.currentData()
        
        # Mark the next page as translating in the progress bar
        self.translation_progress.mark_page_translating(next_page_num)
        
        # Create and start worker thread for translation
        self.look_ahead_worker = TranslationWorker(
            next_page_text, 
            next_page_num,
            input_lang_code,
            output_lang_code,
            output_lang_name,
            self.model_combo.currentData()
        )
        self.look_ahead_worker.translationComplete.connect(self.on_translation_complete)
        
        # Add to active workers list
        self.active_workers.append(self.look_ahead_worker)
        
        # Connect finished signal to remove from active workers
        self.look_ahead_worker.finished.connect(
            lambda: self.active_workers.remove(self.look_ahead_worker) if self.look_ahead_worker in self.active_workers else None
        )
        
        self.look_ahead_worker.start()
        
        # Add a small status indicator that look-ahead translation is in progress
        self.status_label.showMessage("Pre-translating next page...")
    
    def on_translation_complete(self, translated_text, page_num, language_name):
        """Callback when background translation completes"""
        try:
            # Log successful translation completion
            logger.info(f"Translation completed - Page {page_num+1} - Language: {language_name}")
            
            # Store in cache
            cache_key = (page_num, language_name)
            self.translation_cache[cache_key] = translated_text
            
            # Update progress bar if this is for the current output language
            if language_name == self.output_language_combo.currentText():
                # Unmark as translating and mark as translated
                self.translation_progress.unmark_page_translating(page_num)
                self.translation_progress.mark_page_translated(page_num)
            
            # Clear status message if this is the look-ahead translation
            if page_num == self.current_page + 1:
                self.status_label.clearMessage()
            
            # If this is for the current page, update the display
            if page_num == self.current_page and language_name == self.output_language_combo.currentText():
                self.translated_text.setText(translated_text)
        except Exception as e:
            # Log error
            logger.error(f"Error handling translation result: {str(e)}")
            print(f"Error in on_translation_complete: {e}")
        finally:
            # Always attempt to restore cursor when a translation completes for the current page
            if page_num == self.current_page:
                self.ensure_normal_cursor()
    
    def check_api_key(self):
        """Check if API key is available, and if not, request it from user"""
        # First check environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        
        # If not in environment, check keyring
        if not api_key:
            api_key = keyring.get_password("pdf_translator", "openai_api_key")
            
            # If found in keyring, set it in the environment
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
                openai.api_key = api_key
        
    def request_api_key(self):
        """Show dialog to request API key from user"""
        logger.info("API key requested from user")
        
        dialog = ApiKeyDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            api_key = dialog.key_input.text().strip()
            
            if api_key and api_key.startswith("sk-"):
                logger.info("Valid API key provided")
                # Store in keyring
                keyring.set_password("pdf_translator", "openai_api_key", api_key)
                
                # Set in environment and for OpenAI
                os.environ['OPENAI_API_KEY'] = api_key
                openai.api_key = api_key
                
                return True
            else:
                logger.warning("Invalid API key provided")
                QMessageBox.warning(self, "Invalid API Key", 
                                  "The API key doesn't appear to be valid. It should start with 'sk-'.")
        
        return False
    
    def translate_text(self, force_new_translation=True):
        """Translate the extracted text using OpenAI's GPT API"""
        # Check if there is any text to translate
        if not self.extracted_text or self.extracted_text.strip() == "":
            self.translated_text.setText("No text to translate.")
            return
        
        # Log translation request
        logger.info(f"Manual translation requested - Page {self.current_page+1} - Force new: {force_new_translation}")
        
        language_name = self.output_language_combo.currentText()
        cache_key = (self.current_page, language_name)
        
        # Check if translation is already in cache and use it if not forcing new translation
        if not force_new_translation and cache_key in self.translation_cache:
            self.translated_text.setText(self.translation_cache[cache_key])
            # Ensure cursor is restored in case this was called from automatic translation
            QApplication.restoreOverrideCursor()
            return
            
        # Always set cursor to wait cursor, regardless of previous state
        # Store current cursor override state
        cursor_already_waiting = QApplication.overrideCursor() is not None
        
        # Only change cursor if it's not already in waiting state
        if not cursor_already_waiting:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        
        # Update status message
        self.translated_text.setText("Translating...")
        QApplication.processEvents()  # Update UI immediately
        
        output_lang_name = self.output_language_combo.currentText()
        output_lang_code = self.output_language_combo.currentData()
        
        input_lang_code = self.input_language_combo.currentData()
        input_lang_name = self.input_language_combo.currentText()  # Get the current input language name
        
        # If auto-detect, we need to detect the language
        if input_lang_code == "auto":
            self.status_label.showMessage("Detecting language...")
            QApplication.processEvents()  # Update UI immediately
            self.detected_language = self.detect_language(self.extracted_text)
            input_lang_code = self.detected_language
            
            # Update input language combo box and get the language name
            for i in range(self.input_language_combo.count()):
                if self.input_language_combo.itemData(i) == input_lang_code:
                    input_lang_name = self.input_language_combo.itemText(i)
                    self.input_language_combo.setCurrentIndex(i)
                    break
            
            self.status_label.showMessage(f"Detected language: {input_lang_name}", 3000)
        
        try:
            # Check for API key
            if not os.getenv('OPENAI_API_KEY'):
                # Restore cursor before showing dialog
                if not cursor_already_waiting:
                    QApplication.restoreOverrideCursor()
                
                # Request API key from user
                if not self.request_api_key():
                    self.translated_text.setText("Translation canceled: No API key provided.")
                    return
                
                # Re-apply wait cursor if we're continuing with translation
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            
            # Get actual text to translate (remove the continuation marker if present)
            text_to_translate = self.extracted_text
            continuation_marker = "\n[...continuation from next page...]\n"
            
            # Handle whether we need to format the translated text specially
            has_continuation = continuation_marker in text_to_translate
            
            if has_continuation:
                # Split the text at the continuation marker
                parts = text_to_translate.split(continuation_marker)
                
                # Add a special instruction for the model to handle the continuation correctly
                instruction = ("The text below contains a main part followed by a continuation from the next page. "
                             "Translate it naturally as a continuous piece, but use [[[...]]] to mark where the page break occurs.")
                
                text_to_translate = text_to_translate.replace(continuation_marker, " ")  # Replace with a space for translation
            
            # Prepare the messages for GPT
            messages = [
                {"role": "system", "content": f"You are a professional translator. Translate the provided text from {input_lang_name} to {output_lang_name} accurately while maintaining the original meaning and context."}
            ]
            
            # Add continuation handling instruction if needed
            if has_continuation:
                messages[0]["content"] += " " + instruction
            
            messages.append({"role": "user", "content": f"Translate the following text to {output_lang_name}:\n\n{text_to_translate}"})
            
            # Get the selected model
            model = self.model_combo.currentData()
            
            # Mark the current page as translating in the progress bar
            self.translation_progress.mark_page_translating(self.current_page)
            
            # Call OpenAI API with version-appropriate syntax
            if is_new_version:
                # For openai >= 1.0.0
                response = openai.chat.completions.create(
                    model=model,  # Use the selected model
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                translated = response.choices[0].message.content.strip()
            else:
                # For openai < 1.0.0 (older versions)
                response = openai.ChatCompletion.create(
                    model=model,  # Use the selected model
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                translated = response.choices[0].message['content'].strip()
            
            # Store translation in cache
            self.translation_cache[cache_key] = translated
            
            # Display the translation
            self.translated_text.setText(translated)
            
            # Start look-ahead translation for next page after completing this one
            self.start_look_ahead_translation()
                
        except Exception as e:
            self.translated_text.setText(f"Translation error: {str(e)}\n\nTip: Try running 'pip install openai==0.28' to downgrade to a compatible version.")
            
        finally:
            # Always restore the cursor if we set it
            if not cursor_already_waiting:
                QApplication.restoreOverrideCursor()

    def resizeEvent(self, event):
        """Handle window resize events to update PDF display"""
        super().resizeEvent(event)
        if hasattr(self, 'doc') and self.doc:
            # Update the PDF display when window is resized
            self.update_page_display()

    def cleanup_threads(self):
        """Properly terminate all running worker threads"""
        # Keep threads for the next page if they exist, but clean up others
        keep_workers = []
        next_page_num = self.current_page + 1
        
        # Identify workers we want to keep (those translating the target page)
        for worker in self.active_workers:
            if worker.page_num == next_page_num:
                keep_workers.append(worker)
        
        # Clean up all workers that aren't translating the next page
        for worker in self.active_workers:
            if worker not in keep_workers and worker.isRunning():
                worker.wait()
        
        # Update active workers list to only include the workers we're keeping
        self.active_workers = keep_workers

    def closeEvent(self, event):
        """Handle application close event to clean up threads"""
        logger.info("=== Application Closing ===")
        self.ensure_normal_cursor()
        self.cleanup_threads()
        super().closeEvent(event)

    def ensure_normal_cursor(self):
        """Make sure the cursor is restored to normal"""
        # Restore cursor state if it's been overridden
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    def detect_language(self, text):
        """
        Detect the language of the extracted text using OpenAI API
        Returns the language code
        """
        if not text or text.strip() == "":
            return "en"  # Default to English if no text
        
        # Check if we have an API key
        if not os.getenv('OPENAI_API_KEY'):
            return "en"  # Default to English if no API key
        
        try:
            # Sample the text (first 500 characters should be enough for detection)
            sample_text = text[:500]
            
            # Prepare the messages for GPT
            messages = [
                {"role": "system", "content": "You are a language detection system. Analyze the provided text and respond only with the ISO 639-1 language code (e.g., 'en', 'es', 'fr', etc.). Do not include any other text in your response."},
                {"role": "user", "content": f"Detect the language of this text:\n\n{sample_text}"}
            ]
            
            # Call OpenAI API with version-appropriate syntax
            if is_new_version:
                # For openai >= 1.0.0
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",  # Faster, cheaper model for detection
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=5  # Very short response needed
                )
                detected_lang = response.choices[0].message.content.strip().lower()
            else:
                # For openai < 1.0.0 (older versions)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=5
                )
                detected_lang = response.choices[0].message['content'].strip().lower()
            
            # Clean up response to ensure it's just the language code
            # Remove any quotes, periods, or other characters
            detected_lang = detected_lang.replace('"', '').replace('.', '').replace("'", "")
            
            # Verify it's a valid language code (basic check)
            valid_codes = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
            
            if detected_lang in valid_codes:
                return detected_lang
            else:
                # If not a valid code, default to English
                return "en"
            
        except Exception as e:
            print(f"Language detection error: {str(e)}")
            return "en"  # Default to English on error

    def estimate_document_tokens(self):
        """Estimate the total number of tokens in the document for translation"""
        if not self.doc:
            return 0
        
        total_tokens = 0
        sample_pages = min(len(self.doc), 5)  # Sample up to 5 pages for estimation
        sample_text = ""
        
        # Get sample text from the first few pages
        for i in range(sample_pages):
            page = self.doc.load_page(i)
            sample_text += page.get_text()
        
        # Rough estimate: 1 token ≈ 4 characters for English text
        avg_tokens_per_page = len(sample_text) / 4 / sample_pages
        
        # Multiply by total pages and add a 30% buffer for safety
        total_tokens = int(avg_tokens_per_page * len(self.doc) * 1.3)
        
        # Double for input + output tokens (source text + translation)
        return total_tokens * 2

    def estimate_translation_cost(self, token_count):
        """Estimate the cost of translation based on token count and selected model"""
        # Get the currently selected model
        model = self.model_combo.currentData()
        
        # Set rates based on selected model
        if model == "gpt-3.5-turbo":
            # GPT-3.5 Turbo pricing
            input_cost_per_1k = 0.0005  # $0.0005 per 1K tokens for input
            output_cost_per_1k = 0.0015  # $0.0015 per 1K tokens for output
        elif model == "gpt-4-turbo":
            # GPT-4 Turbo pricing
            input_cost_per_1k = 0.01  # $0.01 per 1K tokens for input
            output_cost_per_1k = 0.03  # $0.03 per 1K tokens for output
        else:
            # Fallback for any other models
            input_cost_per_1k = 0.01  # Default to higher pricing to be safe
            output_cost_per_1k = 0.03
        
        # Assume half the tokens are input and half are output
        input_tokens = token_count / 2
        output_tokens = token_count / 2
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost

    def translate_all(self):
        """Translate the entire document"""
        # Check if we have a document
        if not self.doc:
            return
        
        # Log bulk translation request
        logger.info(f"Bulk translation requested - Total pages: {len(self.doc)}")
        
        # Check for API key
        if not os.getenv('OPENAI_API_KEY'):
            # Request API key from user
            if not self.request_api_key():
                QMessageBox.warning(self, "Translation Canceled", "An API key is required for translation.")
                return
        
        # Estimate tokens and cost
        token_estimate = self.estimate_document_tokens()
        cost_estimate = self.estimate_translation_cost(token_estimate)
        
        # Show confirmation dialog with cost estimate
        dialog = TranslateAllDialog(
            self, 
            page_count=len(self.doc), 
            token_estimate=token_estimate, 
            cost_estimate=cost_estimate
        )
        
        if dialog.exec_() != QDialog.Accepted:
            return  # User canceled
        
        # Start translating all pages
        self.translate_all_pages()

    def translate_all_pages(self):
        """Process and translate all pages in the document"""
        if not self.doc:
            return
        
        # Get input and output languages
        input_lang_code = self.input_language_combo.currentData()
        input_lang_name = self.input_language_combo.currentText()
        
        # If auto-detect, we'll need to detect for each page, but use English as fallback
        if input_lang_code == "auto":
            input_lang_code = "en"
        
        output_lang_name = self.output_language_combo.currentText()
        output_lang_code = self.output_language_combo.currentData()
        model = self.model_combo.currentData()
        
        # Translate the current page if not already in cache
        current_cache_key = (self.current_page, output_lang_name)
        if current_cache_key not in self.translation_cache:
            self.translate_text(False)  # Translate current page without forcing
        
        # Start background translation for all other pages
        pending_pages = []
        for page_num in range(len(self.doc)):
            # Skip current page as it's already being translated
            if page_num == self.current_page:
                continue
            
            # Skip pages that are already translated
            cache_key = (page_num, output_lang_name)
            if cache_key in self.translation_cache:
                continue
            
            # Check if this page is already being translated
            is_in_progress = False
            for worker in self.active_workers:
                if worker.page_num == page_num and worker.output_language_name == output_lang_name:
                    is_in_progress = True
                    break
                
            if is_in_progress:
                continue
            
            # Add page to list of pending pages
            pending_pages.append(page_num)
        
        # If we have pending pages, start translating them
        if pending_pages:
            # Update status to show we're starting bulk translation
            self.status_label.showMessage(f"Starting translation of {len(pending_pages)} pages...", 3000)
            
            # Start translation for each page, but limit to first 3 to avoid overwhelming API
            for page_num in pending_pages[:min(3, len(pending_pages))]:
                self.start_page_translation(page_num, input_lang_code, output_lang_code, output_lang_name, model)
            
            # If a timer already exists, stop it first
            if hasattr(self, 'bulk_translation_timer') and self.bulk_translation_timer is not None:
                self.bulk_translation_timer.stop()
            
            # Create a new timer to check progress and start new translations as workers finish
            self.bulk_translation_timer = QTimer()
            self.bulk_translation_timer.timeout.connect(lambda: self.check_bulk_translation_progress(
                pending_pages, input_lang_code, output_lang_code, output_lang_name, model
            ))
            self.bulk_translation_timer.start(1000)  # Check every second
        else:
            # All pages are either translated or in progress
            self.status_label.showMessage("All pages are either translated or currently being translated.", 3000)

    def start_page_translation(self, page_num, input_lang_code, output_lang_code, output_lang_name, model):
        """Start translation for a specific page"""
        # Extract text from the page
        page = self.doc.load_page(page_num)
        page_text = page.get_text()
        
        # Skip if no text to translate
        if not page_text or page_text.strip() == "":
            return
        
        # Get the input language name based on its code
        input_lang_name = "English"  # Default fallback
        for i in range(self.input_language_combo.count()):
            if self.input_language_combo.itemData(i) == input_lang_code:
                input_lang_name = self.input_language_combo.itemText(i)
                break
        
        # Create worker thread for translation
        worker = TranslationWorker(
            page_text,
            page_num,
            input_lang_code,
            output_lang_code,
            output_lang_name,
            model
        )
        
        # Connect signals
        worker.translationComplete.connect(self.on_translation_complete)
        
        # Add to active workers
        self.active_workers.append(worker)
        
        # Mark the page as currently being translated
        if output_lang_name == self.output_language_combo.currentText():
            self.translation_progress.mark_page_translating(page_num)
        
        # Connect finished signal to remove from active workers
        worker.finished.connect(
            lambda: self.active_workers.remove(worker) if worker in self.active_workers else None
        )
        
        # Start the worker
        worker.start()

    def check_bulk_translation_progress(self, pending_pages, input_lang_code, output_lang_code, output_lang_name, model):
        """Check the progress of bulk translation and start new translations as workers finish"""
        # Count active workers
        active_count = len(self.active_workers)
        
        # Calculate how many pages are still pending (excluding those in progress)
        remaining_pages = []
        in_progress_pages = []
        
        # First identify which pages are currently being processed
        for worker in self.active_workers:
            if worker.output_language_name == output_lang_name:
                in_progress_pages.append(worker.page_num)
        
        # Now check which pages still need translation
        for page_num in pending_pages:
            cache_key = (page_num, output_lang_name)
            # If it's not in cache and not currently being translated, add to remaining
            if cache_key not in self.translation_cache and page_num not in in_progress_pages:
                remaining_pages.append(page_num)
        
        # Count all translated pages for the current language (not just the ones from this batch)
        total_translated_pages = 0
        for i in range(len(self.doc)):
            if (i, output_lang_name) in self.translation_cache:
                total_translated_pages += 1
        
        # Total document pages
        total_doc_pages = len(self.doc)
        
        # Pages currently in progress
        pages_in_progress = len(in_progress_pages)
        
        # Update status message based on progress
        if remaining_pages or in_progress_pages:
            self.status_label.showMessage(
                f"Translating document: {total_translated_pages}/{total_doc_pages} pages translated, {pages_in_progress} in progress"
            )
        else:
            # All pages in this batch are translated
            if total_translated_pages == total_doc_pages:
                self.status_label.showMessage(f"Document translation complete: All {total_doc_pages} pages translated", 5000)
            else:
                self.status_label.showMessage(
                    f"Batch translation complete: {total_translated_pages}/{total_doc_pages} pages translated total", 5000
                )
            
            # Stop the timer only if we're truly done
            if hasattr(self, 'bulk_translation_timer'):
                self.bulk_translation_timer.stop()
                # Remove the timer attribute to allow future translate all operations
                delattr(self, 'bulk_translation_timer')
        
        # If we have capacity and remaining pages, start new translations
        if active_count < 3 and remaining_pages:  # Limit to 3 concurrent translations
            # Calculate how many new translations we can start
            can_start = min(3 - active_count, len(remaining_pages))
            
            # Start new translations
            for page_num in remaining_pages[:can_start]:
                self.start_page_translation(page_num, input_lang_code, output_lang_code, output_lang_name, model)

    def on_output_language_changed(self, language_name):
        """Handle output language selection change"""
        if not self.doc:
            return
        
        # Refresh the progress bar based on the new language
        self.refresh_progress_bar()
        
        # Check if current page is already translated in the new language
        cache_key = (self.current_page, language_name)
        
        if cache_key in self.translation_cache:
            # Display existing translation
            self.translated_text.setText(self.translation_cache[cache_key])
        else:
            # Clear and indicate translation is needed
            self.translated_text.setText("Select 'Translate' to translate to " + language_name)

    def refresh_progress_bar(self):
        """Refresh the progress bar to show translations for current output language"""
        if not self.doc:
            return
        
        # Reset the progress bar
        self.translation_progress.set_document_info(len(self.doc))
        self.translation_progress.set_current_page(self.current_page)
        
        # Mark translated pages for the current language
        current_language = self.output_language_combo.currentText()
        for i in range(len(self.doc)):
            if (i, current_language) in self.translation_cache:
                self.translation_progress.mark_page_translated(i)

    def export_translation(self):
        """Export the translated document as a text or PDF file"""
        # Check if we have a document
        if not self.doc:
            return
        
        # Check if we have any translations in the cache
        current_language = self.output_language_combo.currentText()
        has_translations = False
        for key in self.translation_cache:
            if key[1] == current_language:
                has_translations = True
                break
        
        if not has_translations:
            QMessageBox.warning(
                self, 
                "Export Error", 
                f"No translations available to export for language: {current_language}. Please translate at least one page first."
            )
            return
        
        # Ask user for export format with clearer options
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Choose Export Format")
        dialog.setText("Select the format to export the translated document:")
        dialog.setIcon(QMessageBox.Question)
        
        # Create specific buttons for each format
        txt_button = dialog.addButton("Export as Text (.txt)", QMessageBox.ActionRole)
        pdf_button = dialog.addButton("Export as PDF (.pdf)", QMessageBox.ActionRole)
        cancel_button = dialog.addButton(QMessageBox.Cancel)
        
        dialog.exec_()
        
        # Check which button was clicked
        clicked_button = dialog.clickedButton()
        
        if clicked_button == txt_button:
            self.export_as_text()
        elif clicked_button == pdf_button:
            self.export_as_pdf()
        # If cancel was clicked, do nothing

    def export_as_text(self):
        """Export the translated document as a text file, skipping untranslated pages"""
        # Log export attempt
        logger.info("Text export initiated")
        
        # Show file save dialog for the text file
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Translation as Text", 
            "", 
            "Text Files (*.txt)"
        )
        
        if not file_path:
            return  # User canceled
        
        # Add .txt extension if not already present
        if not file_path.lower().endswith('.txt'):
            file_path += '.txt'
        
        try:
            # Show progress message
            self.status_label.showMessage(f"Exporting translations to {file_path}...")
            QApplication.processEvents()  # Update UI immediately
            
            # Get current language
            current_language = self.output_language_combo.currentText()
            
            # Calculate how many pages are translated and get their numbers
            translated_pages = 0
            translated_page_numbers = []
            for i in range(len(self.doc)):
                if (i, current_language) in self.translation_cache:
                    translated_pages += 1
                    translated_page_numbers.append(i + 1)  # Store 1-based page numbers
            
            # Create text file and write all available translations in page order
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write header with document info
                f.write(f"=== TRANSLATED DOCUMENT ===\n")
                f.write(f"Language: {current_language}\n")
                f.write(f"Total Document Pages: {len(self.doc)}\n")
                f.write(f"Translated Pages: {translated_pages}\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Translated using: {self.model_combo.currentText()}\n")
                f.write("="*40 + "\n\n")
                
                # Write included page numbers
                f.write(f"Included pages: {', '.join(map(str, translated_page_numbers))}\n\n")
                f.write("="*40 + "\n\n")
                
                # Write only translated pages
                for i in range(len(self.doc)):
                    cache_key = (i, current_language)
                    
                    # Skip untranslated pages
                    if cache_key not in self.translation_cache:
                        continue
                    
                    # Write translated page with header
                    f.write(f"\n\n--- PAGE {i+1} ---\n\n")
                    f.write(self.translation_cache[cache_key])
                    f.write("\n\n")
                    f.write("-"*40 + "\n")
            
            # Show success message
            self.status_label.showMessage(f"Export complete: {file_path}", 5000)
            
            # Ask if user wants to open the exported file
            self.open_exported_file(file_path)
            
            # Add log at the end of successful export
            logger.info(f"Text export completed: {os.path.basename(file_path)}")
            
        except Exception as e:
            # Show error message if export fails
            QMessageBox.critical(
                self, 
                "Export Error", 
                f"Failed to export translation:\n{str(e)}"
            )
            self.status_label.showMessage(f"Export failed: {str(e)}", 5000)

    def export_as_pdf(self):
        """Export the translated document as a PDF file with page structure preserved"""
        # Log export attempt
        logger.info("PDF export initiated")
        
        # Show file save dialog for the PDF file
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Translation as PDF", 
            "", 
            "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return  # User canceled
        
        # Add .pdf extension if not already present
        if not file_path.lower().endswith('.pdf'):
            file_path += '.pdf'
        
        try:
            # Show progress message
            self.status_label.showMessage(f"Exporting translations to PDF...")
            QApplication.processEvents()  # Update UI immediately
            
            # Get current language
            current_language = self.output_language_combo.currentText()
            
            # Use PyMuPDF (fitz) for PDF creation which has better Unicode support
            doc = fitz.open()  # Create a new empty PDF
            
            # Add a cover page with document information
            page = doc.new_page()  # Add first page (A4 format is the default)
            
            # Calculate how many pages are translated
            translated_pages = 0
            translated_page_numbers = []
            for i in range(len(self.doc)):
                if (i, current_language) in self.translation_cache:
                    translated_pages += 1
                    translated_page_numbers.append(i + 1)  # Store 1-based page numbers
            
            # Create cover page content
            cover_text = f"""Translated Document

Language: {current_language}
Total Document Pages: {len(self.doc)}
Translated Pages: {translated_pages}
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Translated using: {self.model_combo.currentText()}

Included pages: {', '.join(map(str, translated_page_numbers))}
"""
            
            # Insert cover text with proper spacing
            text_lines = cover_text.split('\n')
            y_pos = 72
            for line in text_lines:
                page.insert_text((72, y_pos), line, fontsize=12)
                y_pos += 20  # Advance position for next line
            
            # Add horizontal line
            page.draw_line((72, 200), (page.rect.width - 72, 200))
            
            # For each page in the document
            for i in range(len(self.doc)):
                # Skip pages that haven't been translated
                cache_key = (i, current_language)
                if cache_key not in self.translation_cache:
                    continue
                
                # Get the translated content
                content = self.translation_cache[cache_key]
                
                # Create a new page for this translated content
                page = doc.new_page()
                
                # Add page header
                page.insert_text((72, 72), f"Original Page {i+1}", fontsize=14)
                
                # Add horizontal line
                page.draw_line((72, 100), (page.rect.width - 72, 100))
                
                # Create a rectangle for the text area with proper margins
                # Left: 72 points (1 inch), Right: 72 points from page edge
                # Top: 120 points, Bottom: 72 points from page edge
                text_rect = fitz.Rect(72, 120, page.rect.width - 72, page.rect.height - 72)
                
                try:
                    # Try to use malgunGothic for better CJK support if available
                    font_path = "C:/Windows/Fonts/malgun.ttf"
                    fontname = page.insert_font(fontname="MalgunGothic", fontfile=font_path)
                    page.insert_textbox(
                        text_rect,
                        content,
                        fontname="MalgunGothic",
                        fontsize=11,
                        align=0
                    )
                except Exception as font_error:
                    # Fall back to default font if NanumGothic is not available
                    page.insert_textbox(
                        text_rect,
                        content,
                        fontsize=11,
                        align=0
                    )
            
            # Save the PDF
            doc.save(file_path)
            doc.close()
            
            # Show success message
            self.status_label.showMessage(f"PDF export complete: {file_path}", 5000)
            
            # Ask if user wants to open the exported file
            self.open_exported_file(file_path)
            
            # Add log at the end of successful export
            logger.info(f"PDF export completed: {os.path.basename(file_path)}")
            
        except Exception as e:
            # Show error message if export fails
            error_msg = f"Failed to export PDF: {str(e)}\n\n"
            error_msg += "Try exporting as text (.txt) instead."
            
            QMessageBox.critical(
                self, 
                "Export Error", 
                error_msg
            )
            self.status_label.showMessage(f"Export failed: {str(e)}", 5000)

    def open_exported_file(self, file_path):
        """Open the exported file with the default system application"""
        reply = QMessageBox.question(
            self, 
            "Export Complete", 
            f"Export completed successfully to:\n{file_path}\n\nWould you like to open this file now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Open the file with the default system application
            import platform
            
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                import subprocess
                subprocess.call(('open', file_path))
            else:  # Linux
                import subprocess
                subprocess.call(('xdg-open', file_path))

    def go_to_page(self):
        """Navigate to the specific page entered by the user and translate it"""
        if not self.doc:
            return
        
        try:
            # Get the page number from the input (subtract 1 for zero-based index)
            entered_page = int(self.page_input.text())
            
            # Validate the page number
            if entered_page < 1:
                entered_page = 1
            elif entered_page > len(self.doc):
                entered_page = len(self.doc)
            
            # Convert to zero-based index
            page_num = entered_page - 1
            
            # Don't do anything if we're already on this page
            if page_num == self.current_page:
                return
            
            # Get the language name for cache lookup
            language_name = self.output_language_combo.currentText()
            cache_key = (page_num, language_name)
            
            # Clean up unneeded threads
            self.cleanup_threads()
            
            # Move to the specified page
            self.current_page = page_num
            self.update_page_display()
            self.update_buttons()
            self.extract_text()
            
            # Clear translated text area before checking cache
            self.translated_text.clear()
            QApplication.processEvents()  # Update UI immediately
            
            # Check if there's any text to translate
            if not self.extracted_text or self.extracted_text.strip() == "":
                self.translated_text.setText("No text to translate on this page.")
                self.ensure_normal_cursor()
                return
            
            # Set wait cursor while we check for translations or initiate new ones
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            
            # Check if API key is available for translation
            if not os.getenv('OPENAI_API_KEY'):
                self.ensure_normal_cursor()
                if self.request_api_key():
                    # If user provided an API key, proceed with translation
                    QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                else:
                    # User canceled API key dialog
                    self.translated_text.setText("Translation requires an API key. Click 'Translate' to provide one.")
                    return
            
            # Check if translation exists in cache
            if cache_key in self.translation_cache:
                # If we have a cached translation, display it
                self.translated_text.setText(self.translation_cache[cache_key])
            else:
                # No cached translation, so translate the page
                self.translated_text.setText("Translating...")
                QApplication.processEvents()  # Update UI immediately
                
                # Get input and output languages
                input_lang_code = self.input_language_combo.currentData()
                input_lang_name = self.input_language_combo.currentText()
                
                # If auto-detect, we need to detect the language
                if input_lang_code == "auto":
                    self.status_label.showMessage("Detecting language...")
                    QApplication.processEvents()
                    self.detected_language = self.detect_language(self.extracted_text)
                    input_lang_code = self.detected_language
                    
                    # Update input language combo box
                    for i in range(self.input_language_combo.count()):
                        if self.input_language_combo.itemData(i) == input_lang_code:
                            input_lang_name = self.input_language_combo.itemText(i)
                            self.input_language_combo.setCurrentIndex(i)
                            break
                    
                    self.status_label.showMessage(f"Detected language: {input_lang_name}", 3000)
                
                output_lang_code = self.output_language_combo.currentData()
                
                # Create worker thread for translation
                worker = TranslationWorker(
                    self.extracted_text,
                    self.current_page,
                    input_lang_code,
                    output_lang_code,
                    language_name,
                    self.model_combo.currentData()
                )
                
                # Connect signals
                worker.translationComplete.connect(self.on_translation_complete)
                
                # Mark the page as translating
                self.translation_progress.mark_page_translating(self.current_page)
                
                # Add to active workers
                self.active_workers.append(worker)
                
                # Connect finished signal to remove from active workers
                worker.finished.connect(
                    lambda: self.active_workers.remove(worker) if worker in self.active_workers else None
                )
                
                # Start the worker
                worker.start()
            
            # Start look-ahead translation for next page
            self.start_look_ahead_translation()
            
            # Update current page in progress bar
            self.translation_progress.set_current_page(self.current_page)
            
            # Restore cursor if we displayed cached translation
            if cache_key in self.translation_cache:
                self.ensure_normal_cursor()
        
        except ValueError:
            # Reset to current page if the input is not a valid number
            self.update_page_display()

    def on_debug_level_changed(self):
        """Handle debug level selection change"""
        selected_level = self.debug_level_combo.currentData()
        logger.setLevel(selected_level)
        
        # Show a brief status message
        level_name = self.debug_level_combo.currentText()
        self.status_label.showMessage(f"Logging level set to: {level_name}", 3000)
        
        # Log the level change
        logger.info(f"Logging level changed to: {level_name}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = PDFViewer()
    viewer.show()
    sys.exit(app.exec_())


'''
Requirements:
- PyQt5
- PyMuPDF (fitz)
- openai
- python-dotenv
- keyring

Build command:
pyinstaller --name "PDFTranslator_v0.0.2.exe" --onefile --noconsole PDFTranslator.py
'''