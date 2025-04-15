import sys
import os
import re
import fitz
import openai
import time
import datetime
import json
import logging
import keyring
import httpx
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QFileDialog, QComboBox, QCheckBox, QStatusBar, 
    QSplitter, QScrollArea, QLineEdit, QMessageBox, QDialog, QDialogButtonBox,
    QFormLayout, QTabWidget, QRadioButton, QButtonGroup, QSpinBox, QGroupBox,
    QProgressBar, QSizePolicy, QMenu, QAction, QDoubleSpinBox
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QColor, QTextCursor, QPen, QBrush, QImage, QFont, 
    QFontMetrics, QCursor, QTextFormat, QTextCharFormat, QPalette, QPolygonF,
    QTextBlockFormat, QTextDocument
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QRectF, QSize, QTimer, QRect, QPoint, QSettings, QPointF
)

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')
log_file = os.path.join('logs', f'pdf_translator_{datetime.datetime.now().strftime("%Y-%m-%d")}.log')

# Load environment variables for API keys
load_dotenv()

COMPANY_NAME = "PALEOBYTES"
PROGRAM_NAME = "PDFTranslator"
PROGRAM_VERSION = "0.0.3"

# Set OpenAI API key the old way
openai.api_key = os.getenv('OPENAI_API_KEY')
system_prompt = """
You are a professional translator. 
Translate the provided text from English to Korean accurately while maintaining the original meaning and context. 
Preserve the XML-like tags in the output exactly as they appear in the input.
존대어를 사용하지 말고 문어체 평서문으로 번역해줘.
"""
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
    log_filename = f"logs/pdf_translator_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_filename,
        filemode='a'  # Append mode
    )
    
    logging.info("=== Application Started ===")
    return logging.getLogger('PDFTranslator')

class UpstageClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.ssl_verification_disabled = False
        self.client = self._create_client()

    def _create_client(self):
        if self.ssl_verification_disabled:
            return httpx.Client(verify=False, follow_redirects=True)  # Add follow_redirects=True
        return httpx.Client(follow_redirects=True)  # Add follow_redirects=True

    def post(self, url, files=None, data=None):
        try:
            # Log request details
            logger.debug("=== Request Details ===")
            logger.debug("URL: %s", url)
            logger.debug("Headers: %s", {k: '***' if k == 'Authorization' else v for k, v in self.headers.items()})
            if files:
                logger.debug("Files: %s", {k: f"<file: {v.name}>" for k, v in files.items()})
            if data:
                logger.debug("Data: %s", data)
            
            response = self.client.post(url, headers=self.headers, files=files, data=data)
            response.raise_for_status()
            
            # Log response details
            logger.debug("=== Response Details ===")
            logger.debug("Status Code: %d", response.status_code)
            logger.debug("Headers: %s", dict(response.headers))
            response_data = response.json()
            logger.debug("Response Data: %s", json.dumps(response_data, indent=2))
            
            return response_data
        except httpx.TransportError as e:
            logger.warning(f"Transport error occurred: {e}. Disabling SSL verification.")
            self.ssl_verification_disabled = True
            self.client = self._create_client()
            return self.post(url, files=files, data=data)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            if e.response:
                logger.error("Error Response: %s", e.response.text)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

    def get(self, url):
        try:
            # Log request details
            logger.debug("=== Request Details ===")
            logger.debug("URL: %s", url)
            logger.debug("Headers: %s", {k: '***' if k == 'Authorization' else v for k, v in self.headers.items()})
            
            response = self.client.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Log response details
            logger.debug("=== Response Details ===")
            logger.debug("Status Code: %d", response.status_code)
            logger.debug("Headers: %s", dict(response.headers))
            response_data = response.json()
            logger.debug("Response Data: %s", json.dumps(response_data, indent=2))
            
            return response_data
        except httpx.TransportError as e:
            logger.warning(f"Transport error occurred: {e}. Disabling SSL verification.")
            self.ssl_verification_disabled = True
            self.client = self._create_client()
            return self.get(url)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            if e.response:
                logger.error("Error Response: %s", e.response.text)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

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
        self._is_running = True
        logger.debug(f"Created TranslationWorker for page {page_num} ({input_language} -> {output_language_name})")
        
    def run(self):
        try:
            logger.info(f"=== Starting translation for page {self.page_num} ===")
            logger.debug(f"Thread ID: {int(QThread.currentThreadId())}")
            logger.debug(f"Model: {self.model}")
            logger.debug(f"Languages: {self.input_language} -> {self.output_language_name}")
            
            # Check if the text is structured
            is_structured = any(tag in self.text for tag in ['<header>', '<heading1>', '<paragraph>', '<footnote>', '<footer>', '<caption>'])
            logger.debug(f"Text is structured: {is_structured}")
            
            if is_structured:
                logger.debug("Preparing structured translation messages")
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate the provided text from {self.input_language} to {self.output_language_name} accurately while maintaining the original meaning and context. Preserve the XML-like tags in the output exactly as they appear in the input."
                    }
                ]
                
                # Add the text to translate
                messages.append({
                    "role": "user",
                    "content": f"Translate the following structured text to {self.output_language_name}:\n\n{self.text}"
                })
            else:
                logger.debug("Preparing regular translation messages")
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate the provided text from {self.input_language} to {self.output_language_name} accurately while maintaining the original meaning and context."
                    }
                ]
                
                messages.append({
                    "role": "user",
                    "content": f"Translate the following text to {self.output_language_name}:\n\n{self.text}"
                })
            
            logger.debug("Calling OpenAI API for translation...")
            # Call OpenAI API
            if is_new_version:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2000
                )
                translated = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2000
                )
                translated = response.choices[0].message['content'].strip()
            
            logger.debug("Translation API call completed successfully")
            
            # Emit the result if thread is still running
            if self._is_running:
                logger.info(f"=== Translation completed for page {self.page_num} ===")
                self.translationComplete.emit(translated, self.page_num, self.output_language_name)
            else:
                logger.warning(f"Thread was stopped before translation could be emitted for page {self.page_num}")
            
        except Exception as e:
            logger.error(f"Error in TranslationWorker for page {self.page_num}: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            if self._is_running:
                self.translationComplete.emit(f"Translation error: {str(e)}", self.page_num, self.output_language_name)
    
    def stop(self):
        """Stop the thread gracefully"""
        logger.info(f"=== Stopping translation thread for page {self.page_num} ===")
        logger.debug(f"Thread ID: {int(QThread.currentThreadId())}")
        self._is_running = False
        if not self.wait(1000):  # Wait up to 1 second for thread to finish
            logger.warning(f"Thread for page {self.page_num} did not finish in time, forcing termination...")
            self.terminate()  # Force terminate if it doesn't finish in time
            self.wait()  # Wait for termination to complete
            logger.info(f"Thread for page {self.page_num} terminated")
        else:
            logger.info(f"Thread for page {self.page_num} stopped gracefully")

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
        logger.debug(f"Marked page {page_num} as translated")
        # Remove from translating pages if it was there
        if page_num in self.translating_pages:
            self.translating_pages.remove(page_num)
        self.update()
        
    def mark_page_structure(self, page_num):
        """Mark a page as translated"""
        self.structure_pages.add(page_num)
        logger.debug(f"Marked page {page_num} as structure analyzed")
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
        
    def set_total_pages(self, total_pages):
        """Set the current page indicator"""
        self.total_pages = total_pages
        self.update()

    def clear(self):
        """Reset the progress bar"""
        self.total_pages = 0
        self.translated_pages = set()
        self.structure_pages = set()
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
        logger.debug(f"Painting event for page {self.current_page} of {self.total_pages}")
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
            
            # Then draw all structure analyzed pages in blue
            for i in range(self.total_pages):
                if i in self.structure_pages:
                    x = 1 + (i * block_width)
                    painter.fillRect(int(x), 1, int(block_width) + 1, height - 2, QColor(255, 215, 0))  # Yellow
            
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
        
        global logger
        logger = setup_logging()
        # Initialize settings
        self.settings = QSettings("PDFTranslator", "Settings")
        
        # Initialize variables
        self.doc = None
        self.current_page = 0
        self.current_file = None
        self.extracted_text = ""
        self.document_data = {
            'page_structures': {},
            'translations': {},
            'metadata': {}
        }
        self.active_workers = []
        self.detected_language = None
        
        # Initialize UI
        self.init_ui()
        
        # Apply settings
        self.apply_settings()
        
    def init_ui(self):
        #self.setWindowTitle('PDF Translator')

        # Set window title
        self.setWindowTitle(f"{PROGRAM_NAME} v{PROGRAM_VERSION}")

        self.setGeometry(100, 100, 1200, 800)  # Wider window for two panes
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        

        # Create PDF display widget
        self.pdf_display = PDFDisplayWidget()
        

        # Create top button layout
        top_btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton('Open PDF')
        self.open_btn.clicked.connect(self.open_pdf)
        
        self.prev_btn = QPushButton('Previous Page')
        self.prev_btn.clicked.connect(self.prev_page)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton('Next Page')
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        
        self.page_input = QLineEdit()
        self.page_input.setFixedWidth(50)
        self.page_input.setAlignment(Qt.AlignCenter)
        self.page_total = QLabel('/ 0')

        # Create page navigation layout
        page_layout = QHBoxLayout()
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(5)
        page_layout.addWidget(self.page_input)
        page_layout.addWidget(self.page_total)

        # Connect the enter/return pressed signal
        self.page_input.returnPressed.connect(self.go_to_page)

        # Add analyze button
        self.analyze_btn = QPushButton('Analyze')
        self.analyze_btn.clicked.connect(lambda: self.analyze_page(True))
        self.analyze_btn.setEnabled(False)

        self.analyze_all_btn = QPushButton('Analyze All')
        self.analyze_all_btn.clicked.connect(lambda: self.analyze_all(True))
        self.analyze_all_btn.setEnabled(False)

        # Add checkbox for showing bounding boxes
        self.show_boxes_checkbox = QCheckBox("Show Bounding Boxes")
        self.show_boxes_checkbox.setChecked(True)
        self.show_boxes_checkbox.toggled.connect(self.pdf_display.toggle_bounding_boxes)

        # Add text scale factor control
        self.text_scale_label = QLabel("Text Scale:")
        self.text_scale_spin = QDoubleSpinBox()
        self.text_scale_spin.setRange(0.1, 2.0)
        self.text_scale_spin.setSingleStep(0.1)
        self.text_scale_spin.setValue(0.8)  # Default to 80%
        self.text_scale_spin.setToolTip("Scale factor for translated text size (0.1 to 2.0)")
        self.text_scale_spin.valueChanged.connect(self.on_text_scale_changed)

        # Add line spacing control
        self.line_spacing_label = QLabel("Line Spacing:")
        self.line_spacing_spin = QDoubleSpinBox()
        self.line_spacing_spin.setRange(0.5, 3.0)
        self.line_spacing_spin.setSingleStep(0.1)
        self.line_spacing_spin.setValue(self.settings.value("line_spacing", 1.0, type=float))  # Load saved value
        self.line_spacing_spin.setToolTip("Line spacing multiplier (0.5 to 3.0)")
        self.line_spacing_spin.valueChanged.connect(self.on_line_spacing_changed)

        # Add translate button
        self.translate_btn = QPushButton('Translate')
        self.translate_btn.clicked.connect(lambda: self.translate_text(True))
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
        
        # Create session management buttons
        self.save_session_btn = QPushButton('Save Session')
        self.save_session_btn.clicked.connect(self.save_session)
        self.save_session_btn.setEnabled(False)
        self.save_session_btn.setToolTip("Save current session state")
        
        self.load_session_btn = QPushButton('Load Session')
        self.load_session_btn.clicked.connect(self.load_session)
        self.load_session_btn.setToolTip("Load a previously saved session")
        
        # Add session buttons to a layout
        session_layout = QHBoxLayout()
        session_layout.addWidget(self.save_session_btn)
        session_layout.addWidget(self.load_session_btn)
        
        # Add widgets to top button layout
        top_btn_layout.addWidget(self.open_btn)
        top_btn_layout.addWidget(self.prev_btn)
        top_btn_layout.addWidget(self.next_btn)
        top_btn_layout.addLayout(page_layout)
        top_btn_layout.addWidget(self.analyze_btn)
        top_btn_layout.addWidget(self.analyze_all_btn)
        top_btn_layout.addWidget(self.show_boxes_checkbox)
        top_btn_layout.addWidget(self.text_scale_label)
        top_btn_layout.addWidget(self.text_scale_spin)
        top_btn_layout.addWidget(self.line_spacing_label)
        top_btn_layout.addWidget(self.line_spacing_spin)
        top_btn_layout.addWidget(self.translate_btn)
        top_btn_layout.addWidget(self.translate_all_btn)
        top_btn_layout.addWidget(self.export_btn)
        
        # Create a widget for the status and progress bar at the bottom
        status_layout = QHBoxLayout()
        
        # Create a status bar widget
        self.status_label = QStatusBar()
        
        # Create the translation progress widget
        self.translation_progress = TranslationProgressBar()
        
        # Add widgets to the status layout
        status_layout.addWidget(self.status_label, 8)
        status_layout.addWidget(self.translation_progress, 2)
        
        # Create text display areas
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        
        self.translated_text = QTextEdit()
        self.translated_text.setReadOnly(True)
        
        self.structure_text = QTextEdit()
        self.structure_text.setReadOnly(True)

        # Create translated display widget
        self.translated_display = TranslatedPageDisplayWidget()

        # Create tab widget for original text and structure
        left_tabs = QTabWidget()
        left_tabs.addTab(self.original_text, "Original Text")
        left_tabs.addTab(self.structure_text, "Structure")
        left_tabs.addTab(self.translated_text, "Translated Text")
        left_tabs.addTab(self.translated_display, "Translated Page")        
        
        # Create a splitter for left and right panes
        display_splitter = QSplitter(Qt.Horizontal)
        display_splitter.addWidget(self.pdf_display)
        
        # Add the text splitter to the display splitter
        display_splitter.addWidget(left_tabs)
        display_splitter.setSizes([600, 600])  # Set initial sizes
        
        # Make splitter the central widget of the main splitter
        self.splitter = display_splitter
        
        
        # Add the status bar
        self.statusBar().addPermanentWidget(self.show_boxes_checkbox)
        
        # Create menu bar
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open PDF', self)
        open_action.triggered.connect(self.open_pdf)
        file_menu.addAction(open_action)
        
        save_session_action = QAction('Save Session', self)
        save_session_action.triggered.connect(self.save_session)
        file_menu.addAction(save_session_action)
        
        load_session_action = QAction('Load Session', self)
        load_session_action.triggered.connect(self.load_session)
        file_menu.addAction(load_session_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('Export Translation', self)
        export_action.triggered.connect(self.export_translation)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        analyze_action = QAction('Analyze Page', self)
        analyze_action.triggered.connect(lambda: self.analyze_page(True))
        tools_menu.addAction(analyze_action)
        
        translate_action = QAction('Translate Page', self)
        translate_action.triggered.connect(lambda: self.translate_text(True))
        tools_menu.addAction(translate_action)
        
        translate_all_action = QAction('Translate All Pages', self)
        translate_all_action.triggered.connect(self.translate_all)
        tools_menu.addAction(translate_all_action)
        
        # Settings menu
        settings_menu = menubar.addMenu('Settings')
        
        preferences_action = QAction('Preferences', self)
        preferences_action.triggered.connect(self.show_preferences)
        settings_menu.addAction(preferences_action)
        
        # Add everything to main layout
        main_layout.addLayout(top_btn_layout)
        #main_layout.addLayout(session_layout)
        main_layout.addWidget(self.splitter)
        main_layout.addLayout(status_layout)
        
        # Set focus to the main window
        self.setFocus()
        
        # Initialize variables for translation worker
        self.worker = None
        
        # Log the UI initialization
        logger.info("UI initialized")
        
    def show_preferences(self):
        """Show the preferences dialog"""
        dialog = PreferencesDialog(self)
        if dialog.exec_():
            self.apply_settings()
            
    def apply_settings(self):
        """Apply settings from QSettings"""
        # Update debug level
        debug_level = self.settings.value("debug_level", logging.INFO, int)
        logger.setLevel(debug_level)
        logger.debug(f"Debug level set to: {debug_level}")
        
        # Update line spacing control
        line_spacing = self.settings.value("line_spacing", 1.0, type=float)
        self.line_spacing_spin.setValue(line_spacing)
        
        # If we have an open document, update the interface
        if self.doc:
            # Check if the current language has changed, and if so refresh translations
            output_language_name = self.get_output_language_name()
            translation = self.get_translation(self.current_page, output_language_name)
            if translation:
                self.display_translation(translation)
            else:
                # No translation available for this language, clear and show message
                self.translated_text.setText(f"Select 'Translate' to translate to {output_language_name}")
                
            # Update the progress bar
            self.refresh_progress_bar()
            
            # If auto-translate is enabled, start translation
            if self.is_auto_translate_enabled() and not self.get_translation(self.current_page, output_language_name):
                self.translate_text(False)  # Don't force new translation
                
            # If look-ahead is enabled, check next page
            if self.is_look_ahead_enabled() and self.current_page < len(self.doc) - 1:
                self.start_look_ahead_translation()
        
    def on_settings_changed(self):
        """Handle when settings have changed from the preferences dialog"""
        self.apply_settings()
        # If there's an open document, refresh the current page translation if possible
        if self.doc:
            output_language_name = self.settings.value("output_language_name", "Korean", str)
            translation = self.get_translation(self.current_page, output_language_name)
            if translation:
                self.display_translation(translation)
            self.refresh_progress_bar()

    def get_input_language(self):
        """Get input language from settings"""
        return self.settings.value("input_language", "en", str)
        
    def get_input_language_name(self):
        """Get input language name from settings"""
        return self.settings.value("input_language_name", "English", str)
        
    def get_output_language(self):
        """Get output language from settings"""
        return self.settings.value("output_language", "ko", str)
        
    def get_output_language_name(self):
        """Get output language name from settings"""
        return self.settings.value("output_language_name", "Korean", str)
        
    def get_model(self):
        """Get model from settings"""
        return self.settings.value("model", "gpt-4o-mini", str)
        
    def get_model_name(self):
        """Get model name from settings"""
        return self.settings.value("model_name", "GPT-4o mini", str)
        
    def is_auto_translate_enabled(self):
        """Check if auto-translate is enabled"""
        return self.settings.value("auto_translate", False, bool)
        
    def is_look_ahead_enabled(self):
        """Check if look-ahead translation is enabled"""
        return self.settings.value("look_ahead", False, bool)
    
    def set_page_structure(self, page_num, structure):
        """Set the page structure for a specific page"""
        logger.debug(f"Setting structure for page {page_num}")
        self.document_data['page_structures'][page_num] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'structure': structure
        }
        logger.debug(f"Stored structure: {self.document_data['page_structures'][page_num]}")
        self.update_page_display()

    def get_page_structure(self, page_num):
        """Get the page structure for a specific page"""
        return self.document_data['page_structures'].get(str(page_num))

    def set_translation(self, page_num, language, translation):
        """Set the translation for a specific page and language"""
        try:
            if not hasattr(self, 'document_data'):
                self.document_data = {}
            if 'translations' not in self.document_data:
                self.document_data['translations'] = {}
                
            cache_key = f"{page_num}_{language}"
            self.document_data['translations'][cache_key] = translation
            logger.debug(f"Set translation for page {page_num} in {language}")
            
        except Exception as e:
            logger.error(f"Error setting translation: {str(e)}")
            logger.error("Full error details:", exc_info=True)

    def get_translation(self, page_num, language):
        """Get translation for a specific page and language"""
        try:
            logger.debug(f"Getting translation for page {page_num} and language {language}")
            if not hasattr(self, 'document_data') or not self.document_data:
                logger.debug("No document data available")
                return None
                
            translations = self.document_data.get('translations', {})
            cache_key = f"{page_num}_{language}"
            logger.debug(f"Cache key: {cache_key}")
            logger.debug(f"Available translations: {translations}")
            
            if cache_key in translations:
                translation_data = translations[cache_key]
                logger.debug(f"Found translation: {translation_data}")
                return translation_data
            else:
                logger.debug(f"No translation found for page {page_num} in {language}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting translation: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            return None
        

    def save_session(self):
        """Save current session data to a file"""
        try:
            if not hasattr(self, 'current_file') or not self.current_file:
                logger.debug("No current file to save session for")
                return
                
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            session_file = os.path.join(self.pdf_directory, f'{base_name}.json')
            
            # Convert tuple keys to strings in translations
            translations_for_json = {}
            for key, translation in self.document_data['translations'].items():
                translations_for_json[key] = translation
            
            # Prepare session data
            session_data = {
                'filename': self.current_file,
                'current_page': self.current_page,
                'document_data': {
                    'page_structures': self.document_data['page_structures'],
                    'translations': translations_for_json,
                    'metadata': self.document_data['metadata'],
                    'page_dimensions': self.document_data.get('page_dimensions', {})
                },
                'auto_translate': self.is_auto_translate_enabled(),
                'look_ahead': self.is_look_ahead_enabled(),
                'timestamp': datetime.datetime.now().isoformat(),
                'total_pages': len(self.doc) if self.doc else 0,
                'session_info': {
                    'analyzed_pages': len(set(self.document_data['page_structures'].keys())),
                    'translated_pages': len(self.document_data['translations']),
                    'app_version': '0.0.3'
                }
            }
            
            # Save session data as JSON
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Session saved to {session_file}")
            logger.debug(f"Saved {len(self.document_data['page_structures'])} page structures and {len(self.document_data['translations'])} translations")
            
        except Exception as e:
            logger.error(f"Error saving session: {str(e)}")
            logger.error("Full error details:", exc_info=True)


    def load_session(self, session_file=None):
        """Load a previously saved session"""
        try:
            if not session_file:
                session_file, _ = QFileDialog.getOpenFileName(
                    self, "Load Session", "", "Session Files (*.json)"
                )
                if not session_file:
                    return

            logger.info(f"Loading session from {session_file}")
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                
            # Load document data
            self.document_data = session_data.get('document_data', {})
            logger.debug(f"Loaded document data: {self.document_data.keys()}")
            
            # Load page structures - create a copy before iteration
            page_structures = dict(self.document_data.get('page_structures', {}))
            logger.debug(f"Loaded page structures: {page_structures}")
            for page_num, structure in list(page_structures.items()):
                page_num = int(page_num)  # Convert string key to int
                self.pdf_display.set_page_structure(page_num, structure)
                logger.debug(f"Set structure for page {page_num}")
            
            # Load translations - create a copy before iteration
            translations = dict(self.document_data.get('translations', {}))
            logger.debug(f"Loaded translations: {translations}")
            
            # Process translations
            for key, translation_data in list(translations.items()):
                try:
                    # Handle string format keys (page_num_language)
                    if isinstance(key, str) and '_' in key:
                        parts = key.split('_')
                        if len(parts) >= 2:
                            page_num = int(parts[0])
                            language = parts[1]
                            self.set_translation(page_num, language, translation_data)
                            logger.debug(f"Set translation for page {page_num} in {language}")
                    else:
                        logger.warning(f"Skipping invalid translation key format: {key}")
                except Exception as e:
                    logger.error(f"Error processing translation key {key}: {str(e)}")
                    continue
            
            # Load PDF file with fallback options
            pdf_path = session_data.get('filename')  # Using 'filename' instead of 'pdf_path'
            if not pdf_path:
                logger.error("No PDF filename found in session file")
                QMessageBox.warning(self, "Error", "No PDF filename found in session file")
                return
                
            # Try to find the PDF file
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found at original path: {pdf_path}")
                
                # Try to find PDF in the same directory as the session file
                session_dir = os.path.dirname(session_file)
                pdf_filename = os.path.basename(pdf_path)
                local_pdf_path = os.path.join(session_dir, pdf_filename)
                
                if os.path.exists(local_pdf_path):
                    pdf_path = local_pdf_path
                    logger.info(f"Found PDF file in session directory: {pdf_path}")
                else:
                    # Ask user to locate the PDF file
                    pdf_path, _ = QFileDialog.getOpenFileName(
                    self,
                        "Locate PDF File", 
                        session_dir, 
                        "PDF Files (*.pdf)"
                    )
                    if not pdf_path:
                        logger.error("User cancelled PDF file selection")
                        QMessageBox.warning(self, "Error", "PDF file is required to load the session")
                        return
                    logger.info(f"User selected PDF file: {pdf_path}")
            
            # Open the PDF file
            try:
                self.pdf_path = pdf_path
                self.pdf_document = fitz.open(pdf_path)
                self.total_pages = len(self.pdf_document)
                self.current_page = self.document_data.get('current_page', 0)
                #self.total_pages = self.document_data.get('total_pages', self.total_pages)
                
                # Update UI
                self.update_page_display()
                self.update_buttons()

                self.translation_progress.set_total_pages(self.total_pages)

                # Load page structures - create a copy before iteration
                page_structures = dict(self.document_data.get('page_structures', {}))
                for page_num, structure in list(page_structures.items()):
                    page_num = int(page_num)  # Convert string key to int
                    self.translation_progress.mark_page_structure(page_num)
                    logger.debug(f"Set structure for page {page_num}")

                # Load translations - create a copy before iteration
                translations = dict(self.document_data.get('translations', {}))
                for key, translation_data in list(translations.items()):
                    try:
                        # Handle string format keys (page_num_language)
                        if isinstance(key, str) and '_' in key:
                            parts = key.split('_')
                            if len(parts) >= 2:
                                page_num = int(parts[0])
                                language = parts[1]
                                self.translation_progress.mark_page_translated(page_num)
                        else:
                            logger.warning(f"Skipping invalid translation key format: {key}")
                    except Exception as e:
                        logger.error(f"Error processing translation key {key}: {str(e)}")
                        continue
                
                self.translation_progress.set_current_page(self.current_page)
                self.translation_progress.update()

                # Enable bounding boxes if any page has structure
                if page_structures:
                    self.show_boxes_checkbox.setChecked(True)
                    self.pdf_display.toggle_bounding_boxes(True)
                
                # Display translation for current page if available
                #current_language = self.output_language_combo.currentText()
                current_language = self.get_output_language_name()
                translation = self.get_translation(self.current_page, current_language)
                if translation:
                    logger.debug(f"Found translation for page {self.current_page} in {current_language}")
                    self.display_translation(translation)
                else:
                    logger.debug(f"No translation found for page {self.current_page} in {current_language}")
                    self.translated_text.clear()  # Changed from translation_text to translated_text_edit
                
                logger.info(f"Successfully loaded session with {self.total_pages} pages")
                self.status_label.showMessage("Session loaded successfully", 3000)
                
            except Exception as e:
                logger.error(f"Error opening PDF file: {str(e)}")
                QMessageBox.warning(self, "Error", f"Failed to open PDF file: {str(e)}")
                return
                
        except Exception as e:
            logger.error(f"Error loading session: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load session: {str(e)}")
    
    def update_page_display(self):
        """Update the PDF display and text areas with the current page content"""
        logger.debug(f"Updating page display: current page={self.current_page}")
        try:
            if self.doc is None:
                return
            
            # Update PDF display
            page = self.doc.load_page(self.current_page)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(4, 4))  # 2x zoom for better quality
            image = QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride, QImage.Format_RGB888)
            self.pdf_display.set_page(self.current_page, QPixmap.fromImage(image))
            
            # Get the page structure if it exists
            if str(self.current_page) in self.document_data['page_structures']:
                structure = self.document_data['page_structures'][str(self.current_page)]
                self.pdf_display.set_page_structure(self.current_page, structure)
                logger.debug(f"Setting page structure for page {self.current_page}: {structure}")
            else:
                self.pdf_display.set_page_structure(self.current_page, None)
                logger.debug(f"No structure found for page {self.current_page}, {self.document_data['page_structures']}")
            
            # Update page number display
            self.page_input.setText(str(self.current_page + 1))
            self.page_total.setText(f"/ {len(self.doc)}")
            
            # Extract and display text
            self.extract_text()
            
            # Check for cached translation
            current_language = self.get_output_language_name()
            translation = self.get_translation(self.current_page, current_language)
            
            if translation:
                # Display the translation
                self.display_translation(translation)
            else:
                self.translated_text.clear()
                self.translated_text.setText("Select 'Translate' to translate to " + current_language)
            
            # Update structure text if available
            structure = self.get_page_structure(self.current_page)
            if structure:
                # Display structure in the structure text area
                self.show_page_structure(structure)
            else:
                self.structure_text.setText("Select 'Analyze' to analyze page structure")
            
            # Also update the translated page display if in that view

            if structure and translation:
                logger.debug(f"Setting page to translated_display: {self.current_page} with structure and translation {structure} {translation}")
                self.translated_display.set_page(self.current_page, structure, translation)
            else:
                self.translated_display.set_page(self.current_page, None, None)
                logger.debug(f"No structure or translation found for page {self.current_page} {structure} {translation}")
            
            # Update button states
            self.update_buttons()
            
            # Check if we should auto-translate
            if self.is_auto_translate_enabled() and not self.get_translation(self.current_page, current_language):
                logger.debug("Auto-translate is enabled, translating page...")
                self.translate_text(False)  # Don't force new translation
                
            # Check if we should pre-translate next page
            if self.is_look_ahead_enabled() and self.current_page < len(self.doc) - 1:
                logger.debug("Look-ahead translation is enabled, checking next page...")
                self.start_look_ahead_translation()
                
            # Ensure translated text is visible if it exists
            #if translation:
            #    self.translated_text.setVisible(True)
            #    self.translated_text.raise_()
            
        except Exception as e:
            logger.error(f"Error updating page display: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Error updating display: {str(e)}", 3000)
    
    def update_buttons(self):
        if not self.doc:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
            
        # Enable/disable previous button
        self.prev_btn.setEnabled(self.current_page > 0)
        
        # Enable/disable next button
        self.next_btn.setEnabled(self.current_page < len(self.doc) - 1)

    def update_progress_bar(self):
        """Update the progress bar based on the current page and total pages"""
        if self.doc and self.document_data:
            self.translation_progress.set_current_page(self.current_page)
            #self.translation_progress.set_total_pages(len(self.doc))
            self.translation_progress.update()
    
    def next_page(self):
        """Go to next page"""
        if self.doc and self.current_page < len(self.doc) - 1:
            self.current_page += 1
            self.update_page_display()
            self.update_buttons()
            self.translation_progress.set_current_page(self.current_page)
            self.translation_progress.update()
            
            # Log state after page change
            logger.debug(f"Moved to next page: {self.current_page}")
            logger.debug(f"Available structures: {list(self.document_data['page_structures'].keys())}")
            
            # Only translate if auto-translate is enabled
            #if self.auto_translate_checkbox.isChecked():
            if self.is_auto_translate_enabled():
                self.translate_text()
                
            # Only do look-ahead if both auto-translate and look-ahead are enabled
            #if self.look_ahead_checkbox.isChecked():
            if self.is_look_ahead_enabled():
                self.start_look_ahead_translation()
    
    def prev_page(self):
        """Go to previous page"""
        if self.doc and self.current_page > 0:
            self.current_page -= 1
            self.update_page_display()
            self.update_buttons()
            self.translation_progress.set_current_page(self.current_page)
            self.translation_progress.update()

            # Log state after page change
            logger.debug(f"Moved to previous page: {self.current_page}")
            logger.debug(f"Available structures: {list(self.document_data['page_structures'].keys())}")

            # Only translate if auto-translate is enabled
            #if self.auto_translate_checkbox.isChecked():
            if self.is_auto_translate_enabled():
                self.translate_text()
                
            # Only do look-ahead if both auto-translate and look-ahead are enabled
            if self.is_look_ahead_enabled():
                self.start_look_ahead_translation()

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
        if self.get_input_language() == "auto" and text.strip():
            if not hasattr(self, 'detected_language') or self.detected_language is None:
                # Show status message
                self.status_label.showMessage("Detecting language...")
                QApplication.processEvents()  # Update UI immediately
                
                # Detect language
                self.detected_language = self.detect_language(text)
                
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
        terminators = ['.', '!', '?', '."', '!"', '?"', '.)', '!")]', '?")]', ':', ';']
        
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
        """Check if translation exists in cache for current page and language"""
        try:
            #current_language = self.output_language_combo.currentText()
            current_language = self.get_output_language_name()
            cache_key = f"{self.current_page}_{current_language}"
            
            if not hasattr(self, 'document_data'):
                return None
                
            translations = self.document_data.get('translations', {})
            return translations.get(cache_key)
            
        except Exception as e:
            logger.error(f"Error checking translation cache: {str(e)}")
            return None
    
    def start_look_ahead_translation(self):
        """Start background translation of the next page"""
        try:
            # Check if we already have a look-ahead worker running
            if hasattr(self, 'look_ahead_worker') and self.look_ahead_worker and self.look_ahead_worker.isRunning():
                # Worker is still running, don't start another one
                logger.debug("Look-ahead translation already in progress")
                return
        
            # Calculate the next page number
            next_page = self.current_page + 1
            if next_page >= len(self.doc):
                # No next page
                return
        
            # Get language settings from preferences
            input_lang_code = self.get_input_language()
            output_lang_code = self.get_output_language()
            output_lang_name = self.get_output_language_name()
            
            # Create cache key for this translation
            cache_key = (next_page, output_lang_name)
            
            # Check if we already have a translation for this page
            if cache_key in self.document_data['translations']:
                # Translation already exists
                logger.debug(f"Translation for next page {next_page} already exists")
                return
                
            # Check if the next page has a structure
            if next_page not in self.document_data['page_structures']:
                # No structure for the next page, analyze it first
                logger.debug(f"No structure found for next page {next_page}, analyzing...")
                
                # Create a worker to analyze the next page
                page = self.doc.load_page(next_page)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                
                # TODO: Implement analysis worker
                # For now, just skip look-ahead translation if no structure
                logger.debug(f"Skipping look-ahead translation for page {next_page} due to missing structure")
                return
                
            # Get the structure for the next page
            structure = self.document_data['page_structures'][next_page]
            if not isinstance(structure, dict) or 'structure' not in structure:
                logger.warning(f"Invalid structure format for next page {next_page}")
                return
                
            structure_data = structure['structure']
            if 'elements' not in structure_data:
                logger.warning(f"No elements found in structure for next page {next_page}")
                return
                
            # Prepare structured text for translation
            structured_text = []
            for element in structure_data['elements']:
                category = element.get('category', 'unknown')
                content = element.get('content', {})
                html = content.get('html', '')
                if html:
                    # Remove HTML tags and convert to plain text
                    text = re.sub(r'<[^>]+>', '', html)
                    if text.strip():  # Only add if there's actual text
                        structured_text.append(f"[{category}]\n{text}")
                    
            if not structured_text:
                logger.warning(f"No text found in structure elements for next page {next_page}")
                return
                
            text_to_translate = "\n\n".join(structured_text)
            
            # Get the selected model from preferences
            model = self.get_model()
            
            # Create a worker to translate the next page
            self.look_ahead_worker = TranslationWorker(
                text_to_translate,
                next_page,
            input_lang_code,
            output_lang_code,
            output_lang_name,
                model
            )
            
            # Connect signals
            self.look_ahead_worker.translationComplete.connect(self.on_translation_complete)
        
            # Start the worker
            logger.debug(f"Starting look-ahead translation for page {next_page}")
            self.look_ahead_worker.start()
        
        except Exception as e:
            logger.error(f"Error in look-ahead translation: {str(e)}")
            logger.error("Full error details:", exc_info=True)
    
    def on_translation_complete(self, translated_text, page_num, language_name):
        """Handle completed translation and store in structured format"""
        try:
            logger.info(f"Processing translation for page {page_num}")
            
            # Create cache key using string format
            cache_key = f"{page_num}_{language_name}"
            
            # Get the original page structure
            logger.debug(f"Document data: {self.document_data}")
            logger.debug(f"Page structures: {self.document_data['page_structures']}")
            # check if page number is integer or string and debugprint it
            logger.debug(f"Page number type: {type(page_num)}")
            #if isinstance(page_num, int):
            #    page_num = str(page_num)

            original_structure = self.document_data['page_structures'].get(str(page_num))
            if not original_structure or 'structure' not in original_structure:
                logger.error(f"No valid structure found for page {page_num}")
                return
                
            # Create translation object with timestamp
            translation = {
                "timestamp": datetime.datetime.now().isoformat(),
                "structure": {
                    "elements": []
                }
            }
            
            # Process the translated text based on the original structure
            original_elements = original_structure['structure']['elements']
            translated_lines = translated_text.split('\n\n')
            current_line = 0

            # iterate original elements and translated line together
            for element, translated_line in zip(original_elements, translated_lines):
                # [paragraph]\ntext 
                # separate paragraph name and text
                paragraph_name = element['category'].lower()
                # remove [paragraph] from text
                #separate line by \n and remove only the first line
                # get first line and check if it is [paragraph]
                first_line = translated_line.split('\n')[0].strip().lower()
                if first_line == paragraph_name:
                    text = '\n'.join(translated_line.split('\n')[1:])
                else:
                    text = translated_line
                # remove [paragraph] from text
                text = text.replace(f'[{paragraph_name}]', '').strip()
                # create a new element with the same structure as original
                translated_element = {
                    "category": element['category'],
                    "content": {
                        "html": text,
                        "text": text
                    },
                    "id": element['id'],
                    "page": page_num,
                    "coordinates": element['coordinates']
                }

                translation['structure']['elements'].append(translated_element)
            
            # Store the translation
            if 'translations' not in self.document_data:
                self.document_data['translations'] = {}
            self.document_data['translations'][cache_key] = translation
            
            # Update UI
            self.display_translation(translation)
            self.translation_progress.mark_page_translated(page_num)
            self.status_label.showMessage(f"Translation completed for page {page_num + 1}", 3000)
            self.translate_btn.setEnabled(True)
            
            # Save session after successful translation
            self.auto_save_session()
            
        except Exception as e:
            logger.error(f"Error processing translation: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Error processing translation: {str(e)}", 3000)
            self.translate_btn.setEnabled(True)
    
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
    
    def analyze_all(self, force_new_analysis=True):
        """Analyze all pages using Upstage API"""
        logger.info("Analyzing all pages")
        # wait cursor

        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        for page_num in range(len(self.doc)):
            if self.document_data['page_structures'].get(str(page_num+1)):
                logger.info(f"Using cached analysis for page {page_num+1}")
                continue
            else:
                logger.info(f"Analyzing page {page_num+1}")
                self.go_to_page(page_num+1)
                self.update()
                self.analyze_page(True)
        
        # restore cursor
        QApplication.restoreOverrideCursor()

    
    def analyze_page(self, force_new_analysis=True):
        """Analyze the current page using Upstage API"""
        # wait cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            # Check if analysis already exists and force_new_analysis is False
            if not force_new_analysis and self.current_page in self.document_data['page_structures']:
                logger.info(f"Using cached analysis for page {self.current_page + 1}")
                result = self.document_data['page_structures'][self.current_page]['structure']
                self.show_page_structure(result)
                # Enable bounding boxes
                self.pdf_display.show_bounding_boxes = True
                self.show_boxes_checkbox.setChecked(True)
                self.pdf_display.update()  # Force repaint
                # restore cursor
                QApplication.restoreOverrideCursor()
                return result

            if not hasattr(self, 'upstage_client'):
                self.initialize_upstage_client()
                
            # Create temporary file for the current page
            temp_path = f"temp_page_{self.current_page}.png"
            page = self.doc[self.current_page]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            pix.save(temp_path)
            
            try:
                url = "https://api.upstage.ai/v1/document-ai/document-parse"
                with open(temp_path, "rb") as img_file:
                    files = {"document": img_file}
                    result = self.upstage_client.post(url, files=files)
                    
                    # Get page dimensions in points and mm
                    page_dims = self.document_data['page_dimensions'][str(self.current_page)]
                    
                    # Calculate conversion factors
                    points_to_mm = 0.352778  # 1 point = 0.352778 mm
                    page_width_mm = page_dims['points']['width'] * points_to_mm
                    page_height_mm = page_dims['points']['height'] * points_to_mm
                    
                    # Calculate actual DPI based on page dimensions and image size
                    # DPI = (pixels * 25.4) / (mm * 2)  # 2 is the zoom factor we used
                    dpi_x = (pix.width * 25.4) / (page_width_mm * 2)
                    dpi_y = (pix.height * 25.4) / (page_height_mm * 2)
                    # Use average DPI
                    dpi = (dpi_x + dpi_y) / 2
                    logger.debug(f"Calculated DPI: {dpi:.2f} (x: {dpi_x:.2f}, y: {dpi_y:.2f})")
                    
                    # Process each text element to calculate relative sizes and extract point size
                    if 'elements' in result:
                        for element in result['elements']:
                            if 'bbox' in element:
                                # Get pixel dimensions from analysis
                                px_width = element['bbox'][2] - element['bbox'][0]
                                px_height = element['bbox'][3] - element['bbox'][1]
                                
                                # Calculate relative size in mm
                                # Assuming the analysis image has the same aspect ratio as the PDF page
                                rel_width_mm = (px_width / pix.width) * page_width_mm
                                rel_height_mm = (px_height / pix.height) * page_height_mm
                                
                                # Extract point size from HTML attributes if available
                                point_size = None
                                if 'attributes' in element and 'style' in element['attributes']:
                                    style = element['attributes']['style']
                                    # Look for font-size in style attribute
                                    import re
                                    font_size_match = re.search(r'font-size:\s*(\d+)px', style)
                                    if font_size_match:
                                        px_size = float(font_size_match.group(1))
                                        # Convert px to points using actual DPI
                                        # 1 point = 1/72 inch, so px_size * (72/dpi) = points
                                        point_size = px_size * (72 / dpi)
                                
                                # Store relative dimensions and point size
                                element['relative_size'] = {
                                    'width_mm': rel_width_mm,
                                    'height_mm': rel_height_mm,
                                    'width_pt': rel_width_mm / points_to_mm,
                                    'height_pt': rel_height_mm / points_to_mm,
                                    'point_size': point_size,  # Store the extracted point size
                                    'dpi': dpi  # Store the calculated DPI for reference
                                }
                    
                    # Store the analysis result in document_data
                    self.document_data['page_structures'][str(self.current_page)] = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'structure': result
                    }
                    
                    # Show the analysis text
                    self.show_page_structure(result)
                    
                    # Enable and show bounding boxes
                    self.pdf_display.show_bounding_boxes = True
                    self.show_boxes_checkbox.setChecked(True)
                    self.pdf_display.update()  # Force repaint
                    self.translation_progress.mark_page_structure(self.current_page)
                    self.translation_progress.update()
                    self.update_page_display()
                    
                    # Auto-save session after analysis
                    self.save_session()
                    
                    # Log the state after analysis
                    logger.debug(f"Analysis complete for page {self.current_page}")
                    logger.debug(f"Page structures after analysis: {list(self.document_data['page_structures'].keys())}")
                    
                    # Show success message
                    self.status_label.showMessage(f"Page {self.current_page + 1} analysis complete", 3000)
                    # restore cursor
                    QApplication.restoreOverrideCursor()
                    return result
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                # restore cursor
                QApplication.restoreOverrideCursor()
                    
        except Exception as e:
            error_msg = f"Error during page analysis: {str(e)}"
            logger.error(error_msg)
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(error_msg, 5000)
            # restore cursor
            QApplication.restoreOverrideCursor()
            return None

    def draw_bounding_boxes(self, pixmap, analysis_result):
        """Draw bounding boxes and labels on the page image"""
        logger.debug("Drawing bounding boxes")
        try:
            if not analysis_result or 'elements' not in analysis_result:
                logger.debug("No analysis result or elements not found")
                return pixmap
            
            logger.debug(f"Analysis result: {analysis_result}")

            # Create a QPainter to draw on the pixmap
            painter = QPainter(pixmap)
            try:
                # Set up the painter
                painter.setRenderHint(QPainter.Antialiasing)
                
                # Get the pixmap dimensions for coordinate conversion
                width = pixmap.width()
                height = pixmap.height()
                
                # Set up colors for different element types
                colors = {
                    'header': QColor(255, 0, 0, 127),    # Red (semi-transparent)
                    'heading1': QColor(0, 255, 0, 127),  # Green (semi-transparent)
                    'paragraph': QColor(0, 0, 255, 127), # Blue (semi-transparent)
                    'footnote': QColor(255, 165, 0, 127),# Orange (semi-transparent)
                    'footer': QColor(128, 0, 128, 127)   # Purple (semi-transparent)
                }
                
                # Set up the font for labels
                font = QFont()
                font.setPointSize(8)
                painter.setFont(font)
                
                # Draw boxes for each element
                for element in analysis_result['elements']:
                    category = element.get('category', 'unknown')
                    coords = element.get('coordinates', [])
                    
                    if len(coords) >= 4:
                        # Convert normalized coordinates to pixel coordinates
                        x1 = int(coords[0]['x'] * width)
                        y1 = int(coords[0]['y'] * height)
                        x2 = int(coords[2]['x'] * width)
                        y2 = int(coords[2]['y'] * height)
                        
                        # Get color for this element type
                        color = colors.get(category, QColor(128, 128, 128, 127))  # Gray for unknown types
                        
                        # Draw the rectangle
                        painter.setPen(QPen(color, 2))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                        
                        # Draw the label
                        label_rect = QRectF(x1, y1 - 15, 100, 15)  # Label above the box
                        painter.fillRect(label_rect, QColor(255, 255, 255, 200))  # White background
                        painter.setPen(QPen(color.darker()))
                        painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, category)
                        
            finally:
                painter.end()
                
            return pixmap
                
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            return pixmap

    def show_page_structure(self, result):
        """Display the page structure analysis results in a formatted way"""
        try:
            #logger.debug("Starting show_page_structure with result")
            
            display_text = "=== PAGE STRUCTURE ANALYSIS ===\n\n"
            
            if not result:
                #logger.debug("Result is None or empty")
                display_text += "No analysis results available (empty result)."
                self.structure_text.setText(display_text)
                return
                
            # Handle both direct structure and nested structure
            structure_data = result
            if isinstance(result, dict) and 'structure' in result:
                structure_data = result['structure']
                
            # Add API version and model info
            display_text += f"API Version: {structure_data.get('api', 'N/A')}\n"
            display_text += f"Model: {structure_data.get('model', 'N/A')}\n"
            display_text += f"OCR Enabled: {structure_data.get('ocr', False)}\n\n"
            
            # Process elements
            if 'elements' in structure_data:
                display_text += "Document Structure:\n"
                display_text += "-" * 40 + "\n\n"
                
                for element in structure_data['elements']:
                    # Add element category
                    category = element.get('category', 'unknown')
                    display_text += f"Type: {category.upper()}\n"
                    
                    # Add content
                    if 'content' in element and 'html' in element['content']:
                        # Extract text from HTML content
                        html_content = element['content']['html']
                        # Remove HTML tags for display
                        text_content = html_content.replace('<br>', '\n')
                        for tag in ['<header.*?>', '<h1.*?>', '<p.*?>', '<footer.*?>', '</header>', '</h1>', '</p>', '</footer>']:
                            text_content = re.sub(tag, '', text_content)
                        display_text += f"Content: {text_content.strip()}\n"
                    
                    # Add position information
                    if 'coordinates' in element:
                        coords = element['coordinates']
                        if len(coords) >= 4:  # Ensure we have all corners
                            display_text += f"Position: Top-Left ({coords[0]['x']:.3f}, {coords[0]['y']:.3f}), "
                            display_text += f"Bottom-Right ({coords[2]['x']:.3f}, {coords[2]['y']:.3f})\n"
                    
                    display_text += "-" * 40 + "\n"
            
            # Add usage information
            if 'usage' in structure_data:
                display_text += f"\nPages processed: {structure_data['usage'].get('pages', 1)}\n"
            
            #logger.debug("Final display text:")
            #logger.debug(display_text)
            
            # Update the structure text display
            self.structure_text.setText(display_text)
            
        except Exception as e:
            error_msg = f"Error displaying page structure: {str(e)}"
            logger.error(error_msg)
            logger.error("Full error details:", exc_info=True)
            self.structure_text.setText(f"Error displaying analysis results:\n{str(e)}")

    def translate_text(self, force_new_translation=True):
        """Translate the extracted text"""
        try:
            # Check if we have a document
            if not self.doc:
                logger.warning("No document loaded")
                return
        
            # Get the current page structure
            if str(self.current_page) not in self.document_data['page_structures']:
                logger.warning(f"No structure found for page {self.current_page}")
                return
                
            structure = self.document_data['page_structures'][str(self.current_page)]
            if not isinstance(structure, dict) or 'structure' not in structure:
                logger.warning(f"Invalid structure format for page {self.current_page}")
                return
                
            structure_data = structure['structure']
            if 'elements' not in structure_data:
                logger.warning(f"No elements found in structure for page {self.current_page}")
                return
            
            # Get language settings from preferences
            input_lang_code = self.get_input_language()
            output_lang_code = self.get_output_language()
            output_lang_name = self.get_output_language_name()
            
            # Create cache key
            cache_key = (self.current_page, output_lang_name)
            
            # Check if we already have a translation
            if not force_new_translation and cache_key in self.document_data['translations']:
                logger.info(f"Using cached translation for page {self.current_page}")
                self.display_translation(self.document_data['translations'][cache_key])
                return
                
            # Check API key
            if not os.getenv('OPENAI_API_KEY'):
                if not self.request_api_key():
                    return
                
            # Get the selected model from preferences
            model = self.get_model()
            
            # Prepare structured text for translation
            structured_text = []
            for element in structure_data['elements']:
                category = element.get('category', 'unknown')
                content = element.get('content', {})
                html = content.get('html', '')
                if html:
                    # Remove HTML tags and convert to plain text
                    text = re.sub(r'<[^>]+>', '', html)
                    if text.strip():  # Only add if there's actual text
                        structured_text.append(f"[{category}]\n{text}")
                    
            if not structured_text:
                logger.warning("No text found in structure elements")
                return
                
            text_to_translate = "\n\n".join(structured_text)
            logger.debug(f"Structured text to translate: {text_to_translate}")
            
            # Set wait cursor
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            
            try:
                # Update UI
                self.translate_btn.setEnabled(False)
                self.status_label.showMessage(f"Translating page {self.current_page + 1} to {output_lang_name}...")
                QApplication.processEvents()  # Update UI immediately
                
                # Prepare messages for translation
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Translate the following structured text to {output_lang_name}:\n\n{text_to_translate}"
                    }
                ]
                
                # Call OpenAI API directly
                if is_new_version:
                    response = openai.chat.completions.create(
                            model=model,
                        messages=messages,
                            temperature=0.2,
                        max_tokens=2000
                    )
                    translated = response.choices[0].message.content.strip()
                else:
                    response = openai.ChatCompletion.create(
                            model=model,
                        messages=messages,
                            temperature=0.2,
                        max_tokens=2000
                    )
                    translated = response.choices[0].message['content'].strip()
                    logger.debug(f"Translated text: {translated}")
                    
                    # Process the translation
                    self.on_translation_complete(translated, self.current_page, output_lang_name)
                
            except Exception as e:
                logger.error(f"Error in translation API call: {str(e)}")
                logger.error("Full error details:", exc_info=True)
                self.status_label.showMessage(f"Translation error: {str(e)}", 3000)
                self.translate_btn.setEnabled(True)
                
            finally:
                # Restore cursor
                QApplication.restoreOverrideCursor()
                
        except Exception as e:
            logger.error(f"Error in translate_text: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Translation error: {str(e)}", 3000)
            self.translate_btn.setEnabled(True)
            # Ensure cursor is restored in case of error
            QApplication.restoreOverrideCursor()

    def parse_structured_translation(self, translated_text, original_structure):
        """Parse the structured translation into a format that preserves the original structure"""
        try:
            # Initialize the structured translation
            structured_translation = {
                'elements': [],
                'raw_text': translated_text
            }
            
            # Split the translated text into elements
            element_pattern = r'\[([A-Z_]+)\](.*?)(?=\n\[[A-Z_]+\]|\Z)'
            matches = re.finditer(element_pattern, translated_text, re.DOTALL)
            
            # Create a mapping of original elements to their translations
            for match in matches:
                element_type = match.group(1).lower()
                translated_content = match.group(2).strip()
                
                # Find the corresponding original element
                for original_element in original_structure['elements']:
                    if original_element.get('category', '').lower() == element_type:
                        # Create a new element with the translated content
                        translated_element = original_element.copy()
                        translated_element['translated_content'] = translated_content
                        structured_translation['elements'].append(translated_element)
                        break
            
            return structured_translation
            
        except Exception as e:
            logger.error(f"Error parsing structured translation: {str(e)}")
            # Return the raw text if parsing fails
            return translated_text

    def display_translation(self, translation):
        """Display the translation in the text edit"""
        try:
            if isinstance(translation, dict) and 'structure' in translation:
                # Format structured translation as a string
                formatted_text = []
                for element in translation['structure']['elements']:
                    category = element.get('category', 'unknown')
                    content = element.get('content', {})
                    text = content.get('text', '')
                    if text:
                        #formatted_text.append(f"[{category.upper()}]")
                        formatted_text.append(text)
                        formatted_text.append("")  # Add empty line between categories
                self.translated_text.setText("\n".join(formatted_text))
            else:
                # Regular translation
                self.translated_text.setText(translation)
                
        except Exception as e:
            logger.error(f"Error displaying translation: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.translated_text.setText(f"Error displaying translation: {str(e)}")

    def auto_save_session(self):
        """Automatically save the session periodically"""
        if hasattr(self, 'current_pdf_path') and self.current_pdf_path and self.doc:
            self.save_session()

    def processCurrentPage(self):
        logger.debug("=== Starting Single Page Processing ===")
        if not self.upstage_client:
            # Create or update client if API key changed
            api_key = self.api_key_input.text()
            if not api_key:
                QMessageBox.warning(self, 'Error', 'Please enter your API key.')
                return
            self.upstage_client = UpstageClient(api_key)

        # Create temporary file for PDF page if needed
        temp_path = None
        try:
            logger.debug("Processing page %d", self.current_page)
            
            if self.pdf_document:
                logger.debug("Saving current page as temporary image")
                # Save current page as temporary image
                page = self.pdf_document[self.current_page - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                temp_path = f"temp_page_{self.current_page}.png"
                pix.save(temp_path)
                file_path = temp_path
                logger.debug("Temporary file created: %s", temp_path)
            else:
                file_path = self.file_path.text()
                if not file_path or not os.path.exists(file_path):
                    QMessageBox.warning(self, 'Error', 'Please select a valid file.')
                    return
                logger.debug("Using original file: %s", file_path)

            # Process the page
            logger.debug("Starting page processing (sync=%s)", self.sync_radio.isChecked())
            if self.sync_radio.isChecked():
                self.processSynchronous(file_path, self.current_page)
            else:
                self.processAsynchronous(file_path, self.current_page)
                
        except Exception as e:
            logger.error("Error in processCurrentPage: %s", str(e))
            logger.error("Full error details:", exc_info=True)
            QMessageBox.critical(self, 'Error', f'An error occurred while processing the page: {str(e)}')
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    logger.debug("Cleaning up temporary file: %s", temp_path)
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning("Failed to remove temporary file: %s", str(e))

    def processSynchronous(self, file_path, page_number=None):
        logger.debug("=== Starting Synchronous Processing ===")
        url = "https://api.upstage.ai/v1/document-ai/document-parse"
        files = {"document": open(file_path, "rb")}
        logger.debug("Sending request for page %s", page_number)

        try:
            self.output_text.setText("Processing document...")
            result = self.upstage_client.post(url, files=files)
            logger.debug("Received synchronous response")
            self.handleResult(result, page_number)
        except Exception as e:
            logger.error("Error in processSynchronous: %s", str(e))
            logger.error("Full error details:", exc_info=True)
            raise
        finally:
            files["document"].close()


    def initialize_upstage_client(self):
        """Initialize the Upstage client with API key"""
        try:
            # Try to get API key from environment variable
            api_key = os.environ.get('UPSTAGE_API_KEY')
            
            # If not in environment, try to get from keyring
            if not api_key:
                api_key = keyring.get_password("upstage", "api_key")
            
            # If still no API key, prompt user
            if not api_key:
                api_key, ok = QInputDialog.getText(
                    self,
                    "Upstage API Key Required",
                    "Please enter your Upstage API key:",
                    QLineEdit.Password
                )
                if ok and api_key:
                    # Save to keyring for future use
                    keyring.set_password("upstage", "api_key", api_key)
                else:
                    raise ValueError("API key is required for page analysis")
            
            self.upstage_client = UpstageClient(api_key)
            logger.info("Upstage client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Upstage client: {str(e)}")
            self.status_label.showMessage("Failed to initialize Upstage client", 5000)
            raise

    def resizeEvent(self, event):
        """Handle window resize events to update PDF display"""
        super().resizeEvent(event)
        if hasattr(self, 'doc') and self.doc:
            # Update the PDF display when window is resized
            self.update_page_display()
            
            # Ensure translation is still displayed
            #current_language = self.output_language_combo.currentText()
            current_language = self.get_output_language_name()
            translation = self.get_translation(self.current_page, current_language)
            if translation:
                self.display_translation(translation)

    def cleanup_threads(self):
        """Properly terminate all running worker threads"""
        try:
            # Stop and clean up look-ahead worker if it exists
            if hasattr(self, 'look_ahead_worker') and self.look_ahead_worker and self.look_ahead_worker.isRunning():
                logger.info("=== Cleaning up look-ahead worker ===")
                logger.debug(f"Look-ahead worker thread ID: {int(self.look_ahead_worker.currentThreadId())}")
                self.look_ahead_worker.stop()
                if not self.look_ahead_worker.wait(1000):  # Wait up to 1 second
                    logger.warning("Look-ahead worker did not finish in time, terminating...")
                    self.look_ahead_worker.terminate()
                    self.look_ahead_worker.wait()
                    logger.info("Look-ahead worker terminated forcefully")
                else:
                    logger.info("Look-ahead worker stopped gracefully")
                self.look_ahead_worker = None
            
            # Stop and clean up all active workers
            if hasattr(self, 'active_workers'):
                active_count = len([w for w in self.active_workers if w.isRunning()])
                logger.info(f"=== Cleaning up {active_count} active translation workers ===")
                
                for worker in self.active_workers[:]:  # Create a copy of the list to avoid modification during iteration
                    if worker.isRunning():
                        logger.info(f"Stopping translation worker for page {worker.page_num}")
                        logger.debug(f"Worker thread ID: {int(worker.currentThreadId())}")
                        worker.stop()
                        if not worker.wait(1000):  # Wait up to 1 second
                            logger.warning(f"Translation worker for page {worker.page_num} did not finish in time")
                            worker.terminate()
                            worker.wait()
                            logger.info(f"Translation worker for page {worker.page_num} terminated forcefully")
                        else:
                            logger.info(f"Translation worker for page {worker.page_num} stopped gracefully")
                
                self.active_workers.clear()
                logger.info("All translation workers cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during thread cleanup: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            raise  # Re-raise the exception after logging

    def closeEvent(self, event):
        """Handle application close event to clean up threads and save session"""
        logger.info("=== Application Closing ===")
        
        # Save current session if a document is open
        if hasattr(self, 'current_pdf_path') and self.current_pdf_path and self.doc:
            self.save_session()
        
        # Clean up threads before closing
        self.cleanup_threads()
        
        # Ensure all threads are properly terminated
        QThread.msleep(100)  # Give threads a moment to finish
        
        self.ensure_normal_cursor()
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
        #token_estimate = self.estimate_document_tokens()
        #cost_estimate = self.estimate_translation_cost(token_estimate)
        
        # Show confirmation dialog with cost estimate
        #dialog = TranslateAllDialog(
        #    self, 
        #    page_count=len(self.doc), 
        #    token_estimate=token_estimate, 
        #    cost_estimate=cost_estimate
        #)
        
        #if dialog.exec_() != QDialog.Accepted:
        #    return  # User canceled
        
        # Start translating all pages
        self.translate_all_pages()

    def translate_all_pages(self):
        """Process and translate all pages in the document"""
        if not self.doc:
            return
        
        # Get input and output languages from preferences
        input_lang_code = self.get_input_language()
        input_lang_name = self.get_input_language_name()
        
        # If auto-detect, we'll need to detect for each page, but use English as fallback
        if input_lang_code == "auto":
            input_lang_code = "en"
        
        output_lang_name = self.get_output_language_name()
        output_lang_code = self.get_output_language()
        model = self.get_model()
        
        # Translate the current page if not already in cache
        current_cache_key = f"{self.current_page}_{output_lang_name}"
        if current_cache_key not in self.document_data['translations']:
            self.translate_text(False)  # Translate current page without forcing
        
        # Start background translation for all other pages
        pending_pages = []
        
        # Translate pages in reverse order except current page
        for i in range(len(self.doc)):
            #if i == self.current_page:
            #    continue  # Skip current page as it's already done
                
            cache_key = f"{i}_{output_lang_name}"
            if cache_key not in self.document_data['translations']:
                pending_pages.append(i)
        
        if pending_pages:
            # Process first batch of pages
            # Process current batch
            for page_num in pending_pages:
                # show the page
                self.go_to_page(page_num+1)
                self.update()
                # analyze page
                # if not analyzed, analyze page
                if not self.document_data['page_structures'].get(str(page_num)):
                    self.analyze_page(page_num+1)
                # translate page
                self.translate_text(True)

    def start_page_translation(self, page_num, input_lang_code, output_lang_code, output_lang_name, model):
        """Start translation for a specific page"""
        try:
            # Create and start translation worker
            self.worker = TranslationWorker(
                    self.extracted_text,
            page_num,
            input_lang_code,
            output_lang_code,
            output_lang_name,
            model
        )
        
        # Connect signals
            self.worker.translationComplete.connect(self.on_translation_complete)
            
                # Add to active workers list
            self.active_workers.append(self.worker)
            
            # Start the worker
            self.worker.start()
                
            logger.debug(f"Started translation worker for page {page_num}")
            
        except Exception as e:
            logger.error(f"Error starting page translation: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Error starting translation: {str(e)}", 3000)

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
            if cache_key not in self.document_data['translations'] and page_num not in in_progress_pages:
                remaining_pages.append(page_num)
        
        # Count all translated pages for the current language (not just the ones from this batch)
        total_translated_pages = 0
        for i in range(len(self.doc)):
            if (i, output_lang_name) in self.document_data['translations']:
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

    def on_output_language_changed(self):
        """Handle output language selection change"""
        if not self.doc:
            return
        
        # Get the current output language from settings
        language_name = self.get_output_language_name()
        
        # Refresh the progress bar based on the new language
        self.refresh_progress_bar()
        
        # Check if current page is already translated in the new language
        cache_key = (self.current_page, language_name)
        
        if cache_key in self.document_data['translations']:
            # Display existing translation
            self.translated_text.setText(self.document_data['translations'][cache_key])
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
        current_language = self.get_output_language_name()
        for i in range(len(self.doc)):
            if (i, current_language) in self.document_data['translations']:
                self.translation_progress.mark_page_translated(i)

    def export_translation(self):
        """Export the translated document as a text or PDF file"""
        # Check if we have a document
        if not self.doc:
            return
        
        # Check if we have any translations in the cache
        #current_language = self.output_language_combo.currentText()
        current_language = self.get_output_language_name()
        has_translations = False
        for key in self.document_data['translations']:
            if current_language in key:
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

        pdf_filename = os.path.basename(self.current_file)
        base_name = os.path.splitext(pdf_filename)[0]
        current_language = self.get_output_language_name()
        sanitized_language = current_language.replace(" ", "_").lower()
        
        # Show file save dialog for the text file
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Translation as Text", 
            f"{base_name}_{sanitized_language}.txt", 
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
            #current_language = self.output_language_combo.currentText()
            current_language = self.get_output_language_name()
            
            # Calculate how many pages are translated and get their numbers
            translated_pages = 0
            translated_page_numbers = []
            for i in range(len(self.doc)):
                key = f"{i}_{current_language}"
                if key in self.document_data['translations']:
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
                f.write(f"Translated using: {self.get_model_name()}\n")
                f.write("="*40 + "\n\n")
                
                # Write included page numbers
                f.write(f"Included pages: {', '.join(map(str, translated_page_numbers))}\n\n")
                f.write("="*40 + "\n\n")
                
                # Write only translated pages
                for i in range(len(self.doc)):
                    cache_key = f"{i}_{current_language}"
                    
                    # Skip untranslated pages
                    if cache_key not in self.document_data['translations']:
                        continue
                    
                    # Write translated page with header

                    # Add translated content based on structure
                    translation = self.document_data['translations'][cache_key]
                    if isinstance(translation, dict) and 'structure' in translation:
                        # Process structured translation
                        for element in translation['structure']['elements']:
                            category = element.get('category', 'unknown')
                            content = element.get('content', {})
                            text = content.get('text', '')
                            
                            if text:
                                # Add category heading if it's meaningful
                                #if category.lower() not in ['unknown', 'other']:
                                #    y_pos = add_text(f"{category.upper()}", y_pos, fontsize=12, is_heading=True)
                                
                                # Add the text content
                                logger.debug(f"Adding text: {text} to page {i + 1}")
                                f.write(text)
                    f.write("\n\n")

                    f.write(f"--- PAGE {i+1} ---\n\n\n\n")
                    #f.write("\n\n")
                    #f.write("-"*40 + "\n")
            
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
        """Export the translated document as a PDF file"""
        try:
            if not self.doc:
                return
                
            # Create output PDF file path
            pdf_filename = os.path.basename(self.current_file)
            base_name = os.path.splitext(pdf_filename)[0]
            current_language = self.get_output_language_name()
            sanitized_language = current_language.replace(" ", "_").lower()
            output_path, _ = QFileDialog.getSaveFileName(
            self, 
                "Save Translated PDF", 
                f"{base_name}_{sanitized_language}.pdf", 
            "PDF Files (*.pdf)"
        )
        
            if not output_path:
                return
        
            # Show progress message
            self.status_label.showMessage(f"Exporting translations to PDF...")
            QApplication.processEvents()  # Update UI immediately
            
            # Use PyMuPDF (fitz) for PDF creation which has better Unicode support
            doc = fitz.open()  # Create a new empty PDF
            
            # Add a cover page with document information
            page = doc.new_page()  # Add first page (A4 format is the default)
            
            # Calculate how many pages are translated
            translated_pages = 0
            translated_page_numbers = []
            for i in range(len(self.doc)):
                cache_key = f"{i}_{current_language}"
                if cache_key in self.document_data['translations']:
                    translated_pages += 1
                    translated_page_numbers.append(i + 1)  # Store 1-based page numbers
            
            # Create cover page content
            cover_text = f"""Translated Document

            Language: {current_language}
            Total Document Pages: {len(self.doc)}
            Translated Pages: {translated_pages}
            Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
            Translated using: {self.get_model_name()}

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
            
            # Create a font for text
            font_size = 10
            
            # Process each page
            for orig_page_num in range(len(self.doc)):
                cache_key = f"{orig_page_num}_{current_language}"
                
                # Check if we have a translation for this page
                if cache_key not in self.document_data['translations']:
                    continue
                
                # Get the translated content
                translation = self.document_data['translations'][cache_key]
                
                # Create a new page in the output PDF
                output_page = doc.new_page()
                
                # Set page header with original page number
                header_text = f"Original Page: {orig_page_num + 1} - Language: {current_language}"
                output_page.insert_text((72, 36), header_text, fontsize=8)
                
                # Draw a line under the header
                output_page.draw_line((72, 48), (output_page.rect.width - 72, 48))
                
                # Start position for content
                y_pos = 72
                
                # Helper function to add formatted text
                def add_text(text, y, fontsize=10, is_heading=False):
                    # Calculate text width and wrap if needed
                    rect = fitz.Rect(72, y, output_page.rect.width - 72, y + 1000)
                    text_options = {"fontsize": fontsize}
                    #if is_heading:
                    #    text_options["fontweight"] = "bold"
                    
                    # Insert text with wrapping
                    inserted_text = output_page.insert_textbox(rect, text, **text_options)
                    
                    # Return the new y position after the text
                    return y + inserted_text + 10  # Add a small spacing after each paragraph
                
                # Add translated content based on structure
                if isinstance(translation, dict) and 'structure' in translation:
                    # Process structured translation
                    for element in translation['structure']['elements']:
                        category = element.get('category', 'unknown')
                        content = element.get('content', {})
                        text = content.get('text', '')
                        
                        if text:
                            # Add category heading if it's meaningful
                            #if category.lower() not in ['unknown', 'other']:
                            #    y_pos = add_text(f"{category.upper()}", y_pos, fontsize=12, is_heading=True)
                            
                            # Add the text content
                            logger.debug(f"Adding text: {text} to page {orig_page_num + 1}")
                            y_pos = add_text(text, y_pos)
                else:
                    # Process plain text translation
                    y_pos = add_text(str(translation), y_pos)
            
            # Save the document
            doc.save(output_path)
            doc.close()
            
            self.status_label.showMessage(f"Translation exported to {output_path}", 5000)
            
            # Ask if user wants to open the exported file
            reply = QMessageBox.question(self, 'Open File', 
                'Export completed. Do you want to open the file now?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.Yes:
                self.open_exported_file(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting PDF: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Export error: {str(e)}", 3000)
            QMessageBox.warning(self, "Export Error", f"Failed to export PDF: {str(e)}")

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

    def go_to_page(self, page_num):
        """Handle page navigation when user enters a page number"""
        try:
            if not self.doc:
                return
                
            # Get the entered page number (1-based)
            try:
                new_page = int(page_num) - 1
            except ValueError:
                return
                
            # Validate page number
            if new_page < 0 or new_page >= len(self.doc):
                self.status_label.showMessage(f"Invalid page number. Please enter a number between 1 and {len(self.doc)}", 5000)
                self.page_input.setText(str(self.current_page + 1))
                return
                
            # Update current page
            self.current_page = new_page
            
            # Update the display
            self.update_page_display()
            self.update_buttons()
            self.translation_progress.set_current_page(self.current_page)
            self.translation_progress.update()
            
            # Only translate if auto-translate is enabled
            #if self.auto_translate_checkbox.isChecked():
            if self.is_auto_translate_enabled():
                self.translate_text()
                
            # Only do look-ahead if both auto-translate and look-ahead are enabled
            #if self.look_ahead_checkbox.isChecked():
            if self.is_look_ahead_enabled():
                self.start_look_ahead_translation()
                
        except Exception as e:
            logger.error(f"Error navigating to page: {str(e)}")
            self.status_label.showMessage(f"Error navigating to page: {str(e)}", 5000)

    def on_debug_level_changed(self):
        """Handle debug level selection change"""
        selected_level = self.debug_level_combo.currentData()
        logger.setLevel(selected_level)
        
        # Show a brief status message
        level_name = self.debug_level_combo.currentText()
        self.status_label.showMessage(f"Logging level set to: {level_name}", 3000)
        
        # Log the level change
        logger.info(f"Logging level changed to: {level_name}")

    def open_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        
        if file_path:
            # Store the current PDF path
            self.current_pdf_path = file_path
            self.current_file = os.path.abspath(file_path)
            
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
            
            # Store page dimensions in document data
            self.document_data['page_dimensions'] = {}
            for i in range(len(self.doc)):
                page = self.doc[i]
                rect = page.rect
                
                # Convert points to millimeters (1 point = 0.352778 mm)
                points_to_mm = 0.352778
                
                self.document_data['page_dimensions'][str(i)] = {
                    # Points (original PDF units)
                    'points': {
                        'width': rect.width,
                        'height': rect.height,
                        'x0': rect.x0,
                        'y0': rect.y0,
                        'x1': rect.x1,
                        'y1': rect.y1
                    },
                    # Millimeters (physical units)
                    'mm': {
                        'width': rect.width * points_to_mm,
                        'height': rect.height * points_to_mm,
                        'x0': rect.x0 * points_to_mm,
                        'y0': rect.y0 * points_to_mm,
                        'x1': rect.x1 * points_to_mm,
                        'y1': rect.y1 * points_to_mm
                    }
                }
            
            # Update page total label
            self.page_total.setText(f"/ {len(self.doc)}")
            self.translation_progress.set_total_pages(len(self.doc))
            self.translation_progress.set_current_page(self.current_page)
            self.translation_progress.update()

            
            # Clear translation cache when opening a new document
            self.document_data['translations'] = {}
            
            # Reset detected language when opening a new PDF
            self.detected_language = None
            
            # Update UI
            self.update_page_display()  # This will update the PDF display and text areas
            self.update_buttons()  # This will properly set the previous button state
            
            # Enable buttons
            self.translate_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            self.analyze_all_btn.setEnabled(True)
            self.translate_all_btn.setEnabled(True)  # Enable the Translate All button
            self.export_btn.setEnabled(True)  # Enable the Export button
            self.save_session_btn.setEnabled(True)  # Enable the Save Session button
            
            # Check for existing session and ask if user wants to load it
            pdf_filename = os.path.basename(file_path)
            # get directory of file
            self.pdf_directory = os.path.dirname(file_path)

            session_file = os.path.join(self.pdf_directory, f'{os.path.splitext(pdf_filename)[0]}.json')
            
            if os.path.exists(session_file):
                reply = QMessageBox.Yes
                #reply = QMessageBox.question(self, 'Load Session', 
                #    f'Found existing session for {pdf_filename}. Do you want to load it?',
                #    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                
                if reply == QMessageBox.Yes:
                    self.load_session(session_file)
                else:
                    # Initialize text areas with current page content
                    self.extract_text()  # This will update the original text area
                    current_language = self.get_output_language_name()
                    self.translated_text.setText("Select 'Translate' to translate to " + current_language)
                    self.structure_text.setText("Select 'Analyze' to analyze page structure")
            else:
                # Initialize text areas with current page content
                self.save_session()
                self.extract_text()  # This will update the original text area
                current_language = self.get_output_language_name()
                self.translated_text.setText("Select 'Translate' to translate to " + current_language)
                self.structure_text.setText("Select 'Analyze' to analyze page structure")

    def on_view_tab_changed(self, index):
        """Handle tab change between PDF view and translated view"""
        try:
            if index == 1:  # Translated view
                # Update the translated page display
                if self.current_page in self.document_data['page_structures']:
                    structure = self.document_data['page_structures'][self.current_page]
                    current_language = self.get_output_language_name()
                    cache_key = (self.current_page, current_language)
                    
                    if cache_key in self.document_data['translations']:
                        translation = self.document_data['translations'][cache_key]
                        self.translated_display.set_page(self.current_page, structure, translation)
                    else:
                        self.status_label.showMessage("No translation available for this page", 3000)
            
        except Exception as e:
            logger.error(f"Error in on_view_tab_changed: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Error changing view: {str(e)}", 3000)

    def on_text_scale_changed(self, value):
        """Handle text scale factor change"""
        self.settings.setValue("text_scale", value)
        self.settings.sync()
        # Update the translated page display
        if hasattr(self, 'translated_display'):
            self.translated_display.update()

    def on_line_spacing_changed(self, value):
        """Handle line spacing change"""
        self.settings.setValue("line_spacing", value)
        self.settings.sync()
        # Update the translated page display
        if hasattr(self, 'translated_display'):
            self.translated_display.update()

class PDFDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.current_page = None
        self.page_structure = None
        self.show_bounding_boxes = True

        # ---- NEW FIELDS FOR ZOOM AND PAN ----
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.is_panning = False
        self.last_mouse_pos = QPointF(0, 0)
        # -------------------------------------

    def set_page(self, page_num, pixmap):
        """Set the current page to display"""
        self.current_page = page_num
        self.pixmap = pixmap

        # Get the main window instance to access document data
        main_window = self.window()
        if hasattr(main_window, 'document_data'):
            # Get the structure for the new page
            structure = main_window.document_data['page_structures'].get(page_num)
            logger.debug(f"Setting page {page_num} structure: {structure}")
            if structure:
                self.page_structure = structure
            else:
                self.page_structure = None
                logger.debug(f"No structure found for page {page_num}")

        self.update()

    def set_page_structure(self, page_num, structure):
        """Set the structure data for the current page"""
        logger.debug(f"Setting page structure for page {page_num}: {structure}")
        if page_num == self.current_page:
            self.page_structure = structure
            self.update()

    def toggle_bounding_boxes(self, state):
        self.show_bounding_boxes = state
        self.update()

    def paintEvent(self, event):
        #logger.debug("Painting event")
        if self.pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate scaling to fit the widget
        widget_size = self.size()
        pixmap_size = self.pixmap.size()

        # The original scale to make the pixmap fit in the widget
        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()
        base_scale = min(scale_x, scale_y)

        # ---- MODIFIED PART: Multiply by zoom_factor ----
        final_scale = base_scale * self.zoom_factor
        scaled_width = pixmap_size.width() * final_scale
        scaled_height = pixmap_size.height() * final_scale

        # Calculate the position to center the scaled pixmap,
        # then apply the pan_offset
        x = (widget_size.width() - scaled_width) / 2 + self.pan_offset.x()
        y = (widget_size.height() - scaled_height) / 2 + self.pan_offset.y()

        # Draw the scaled pixmap
        painter.drawPixmap(
            QRectF(x, y, scaled_width, scaled_height),
            self.pixmap,
            QRectF(0, 0, pixmap_size.width(), pixmap_size.height())
        )

        # Draw bounding boxes if enabled and structure exists
        #logger.debug(f"Drawing bounding boxes: {self.show_bounding_boxes} and {self.page_structure}")
        if self.show_bounding_boxes and self.page_structure:
            self.draw_bounding_boxes(painter, x, y, final_scale)

        painter.end()

    def draw_bounding_boxes(self, painter, offset_x, offset_y, scale):
        """Draw bounding boxes for page elements"""
        #logger.debug("Drawing bounding boxes")
        if not self.page_structure or 'structure' not in self.page_structure:
            #logger.debug("No structure or structure data found")
                return

        structure_data = self.page_structure['structure']
        #logger.debug(f"Structure data: {structure_data}")
        if 'elements' not in structure_data:
            #logger.debug("No elements found in structure")
            return

        #logger.debug(f"Drawing bounding boxes for {len(structure_data['elements'])} elements")

        for element in structure_data['elements']:
            if 'coordinates' not in element:
                #logger.debug("Element has no coordinates")
                continue

            coords = element['coordinates']
            category = element.get('category', 'unknown').lower()
            #logger.debug(f"Drawing box for {category} with coordinates: {coords}")

            # Set color based on category
            alpha = 32
            if category == 'header':
                color = QColor(255, 0, 0, alpha)  # Red
            elif category == 'heading1':
                color = QColor(0, 255, 0, alpha)  # Green
            elif category == 'paragraph':
                color = QColor(0, 0, 255, alpha)  # Blue
            elif category == 'footnote':
                color = QColor(255, 255, 0, alpha)  # Yellow
            elif category == 'footer':
                color = QColor(255, 0, 255, alpha)  # Magenta
            else:
                color = QColor(128, 128, 128, alpha)  # Gray

            # Create polygon from coordinates
            polygon = QPolygonF()
            for point in coords:
                # Scale and offset the coordinates
                x = offset_x + (point['x'] * scale * self.pixmap.width())
                y = offset_y + (point['y'] * scale * self.pixmap.height())
                #logger.debug(f"Point coordinates: x={x}, y={y}")
                polygon.append(QPointF(x, y))

            # Draw the polygon
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawPolygon(polygon)

            # Draw category label
            painter.setPen(QPen(QColor(0, 0, 0)))
            if polygon.size() > 0:
                first_point = polygon.at(0)
                # Convert float coordinates to integers for drawText
                label_x = int(first_point.x())
                label_y = int(first_point.y() - 5)
                painter.drawText(label_x, label_y, category)

    # ----------- NEW METHODS FOR ZOOM AND PAN -----------
    def wheelEvent(self, event):
        """
        Zoom in/out centered on the mouse cursor when the user
        scrolls the mouse wheel.
        """
        # Positive delta => zoom in; negative => zoom out
        wheel_delta = event.angleDelta().y()
        if wheel_delta == 0:
                    return

        # Choose a zoom step factor. Adjust to taste.
        zoom_step = 1.2 if wheel_delta > 0 else 1 / 1.2
        
        # The position of the cursor in widget coordinates
        cursor_pos = event.position()  # In PyQt6; for PyQt5 use event.posF()
        # Convert to a QPointF
        cursor_point = QPointF(cursor_pos.x(), cursor_pos.y())

        # Compute the old scene coordinates (before zoom) of that cursor
        # relative to the scaled+offset image. We'll solve so that
        # after we change self.zoom_factor, the same scene point remains
        # under the cursor in widget coords.
        
        # First get the base scale (fit-to-widget) again to be consistent
        widget_size = self.size()
        pixmap_size = self.pixmap.size()
        if pixmap_size.isNull():
            return  # No pixmap to zoom

        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()
        base_scale = min(scale_x, scale_y)

        # Our final scale prior to this event
        old_final_scale = base_scale * self.zoom_factor
        
        # The current top-left after pan offset:
        current_width = pixmap_size.width() * old_final_scale
        current_height = pixmap_size.height() * old_final_scale
        current_x = (widget_size.width() - current_width) / 2 + self.pan_offset.x()
        current_y = (widget_size.height() - current_height) / 2 + self.pan_offset.y()

        # Convert cursor position to "scene" coords
        # where (0,0) is top-left of scaled pixmap
        scene_x = (cursor_point.x() - current_x) / old_final_scale
        scene_y = (cursor_point.y() - current_y) / old_final_scale

        # Adjust the zoom factor
        self.zoom_factor *= zoom_step
        # Prevent zoom_factor from going too small or large
        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))

        # After changing zoom_factor, compute new final scale:
        new_final_scale = base_scale * self.zoom_factor
        new_width = pixmap_size.width() * new_final_scale
        new_height = pixmap_size.height() * new_final_scale
        new_x = (widget_size.width() - new_width) / 2
        new_y = (widget_size.height() - new_height) / 2

        # Recompute what top-left would have to be so that
        # (scene_x, scene_y) is still under cursor_point.
        # We want: 
        #   cursor_point.x() = new_top_left_x + scene_x * new_final_scale
        #   cursor_point.y() = new_top_left_y + scene_y * new_final_scale
        #
        # so new_top_left_x = cursor_point.x() - scene_x * new_final_scale
        #    new_top_left_y = cursor_point.y() - scene_y * new_final_scale
        #
        # We'll store that difference in self.pan_offset (relative to the
        # default centering new_x, new_y).
        desired_top_left_x = cursor_point.x() - scene_x * new_final_scale
        desired_top_left_y = cursor_point.y() - scene_y * new_final_scale

        # Our "natural" centering is (new_x, new_y). The difference to that
        # is how much we shift with the pan_offset
        dx = desired_top_left_x - new_x
        dy = desired_top_left_y - new_y

        # Update pan_offset
        self.pan_offset = QPointF(dx, dy)

        self.update()

    def mousePressEvent(self, event):
        """
        Start panning when the user presses the left mouse button.
        """
        if event.button() == Qt.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()  # PyQt6; use event.pos() in PyQt5
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        If panning is active, update the pan_offset.
        """
        if self.is_panning:
            new_mouse_pos = event.pos()
            delta = new_mouse_pos - self.last_mouse_pos
            self.last_mouse_pos = new_mouse_pos

            # Just shift pan_offset by the mouse delta
            self.pan_offset += delta
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Stop panning on left button release.
        """
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)
    # ----------------------------------------------------


class TranslatedPageDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_page = 0
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.background_color = QColor(255, 255, 255)  # White background
        self.text_color = QColor(0, 0, 0)  # Black text
        self.pixmap = None
        self.page_dims = None
        
        # Add zoom and pan fields
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.is_panning = False
        self.last_mouse_pos = QPointF(0, 0)

    def set_page(self, page_num, structure, translation):
        """Set current page number, structure, and translation"""
        self.current_page = page_num
        # Store in parent's document_data
        self.structure = structure
        logger.debug(f"Setting structure for page {self.current_page}: {self.structure}")
        self.translation = translation
        logger.debug(f"Setting translation for page {self.current_page}: {self.translation}")
        
        # Get the page dimensions from document data
        main_window = self.window()
        if hasattr(main_window, 'document_data'):
            if 'page_dimensions' in main_window.document_data:
                self.page_dims = main_window.document_data['page_dimensions'].get(str(page_num))
                logger.debug(f"Page dimensions for page {page_num}: {self.page_dims}")

            if not self.page_dims:
                logger.warning(f"No page dimensions found for page {page_num}")
                # Try to get dimensions from the PDF document
                if hasattr(main_window, 'doc') and main_window.doc:
                    try:
                        page = main_window.doc[page_num]
                        rect = page.rect
                        points_to_mm = 0.352778  # 1 point = 0.352778 mm
                        self.page_dims = {
                            'points': {
                                'width': rect.width,
                                'height': rect.height,
                                'x0': rect.x0,
                                'y0': rect.y0,
                                'x1': rect.x1,
                                'y1': rect.y1
                            },
                            'mm': {
                                'width': rect.width * points_to_mm,
                                'height': rect.height * points_to_mm,
                                'x0': rect.x0 * points_to_mm,
                                'y0': rect.y0 * points_to_mm,
                                'x1': rect.x1 * points_to_mm,
                                'y1': rect.y1 * points_to_mm
                            }
                        }
                        logger.debug(f"Retrieved page dimensions from PDF: {self.page_dims}")
                    except Exception as e:
                        logger.error(f"Error getting page dimensions from PDF: {str(e)}")
        
        self.update()

    def paintEvent(self, event):
        """Paint the translated page with proper structure and text"""
        logger.debug(f"Painting translated page {self.current_page}")
        painter = None
        try:
            if not hasattr(self, 'structure') or not hasattr(self, 'translation'):
                logger.debug(f"No structure or translation found for page {self.current_page}")
                return
            if self.structure is None or self.translation is None:
                logger.debug(f"No structure or translation found for page {self.current_page}")
                return

            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Clear the background
            painter.fillRect(self.rect(), Qt.white)
            
            if not self.structure or not self.translation:
                logger.debug("No structure or translation to paint")
                return
                
            elements = self.structure['structure']['elements']
            translation = self.translation['structure']['elements']
            logger.debug(f"translated page view Elements: {elements}")
            logger.debug(f"translated page view Translation: {translation}")
            
            if not self.page_dims:
                logger.warning("No page dimensions found")
                return
                
            # Calculate scaling factors
            page_width = self.page_dims['points']['width']
            page_height = self.page_dims['points']['height']
            
            # Calculate base scaling to fit page in widget while maintaining aspect ratio
            scale_x = self.width() / page_width
            scale_y = self.height() / page_height
            base_scale = min(scale_x, scale_y)
            
            # Apply zoom factor
            final_scale = base_scale * self.zoom_factor
            
            # Calculate offset to center the page, then apply pan offset
            scaled_width = page_width * final_scale
            scaled_height = page_height * final_scale
            offset_x = (self.width() - scaled_width) / 2 + self.pan_offset.x()
            offset_y = (self.height() - scaled_height) / 2 + self.pan_offset.y()
            
            # Get text scale factor and line spacing from settings
            settings = QSettings("PDFTranslator", "Settings")
            text_scale = settings.value("text_scale", 0.8, type=float)
            line_spacing = settings.value("line_spacing", 1.0, type=float)
            
            # Draw each element with proper scaling
            for orig_elem, trans_elem in zip(elements, translation):
                logger.debug(f"Drawing element: {orig_elem} {trans_elem}")
                if 'coordinates' in orig_elem and 'content' in trans_elem:
                    # Get original bounding box coordinates
                    coords = orig_elem['coordinates']
                    # Get min and max x,y from the points
                    x0 = min(point['x'] for point in coords) * page_width
                    y0 = min(point['y'] for point in coords) * page_height
                    x1 = max(point['x'] for point in coords) * page_width
                    y1 = max(point['y'] for point in coords) * page_height
                    translated_text = trans_elem['content']['text']
                    
                    # Scale coordinates with zoom and pan
                    scaled_x0 = offset_x + (x0 * final_scale)
                    scaled_y0 = offset_y + (y0 * final_scale)
                    scaled_x1 = offset_x + (x1 * final_scale)
                    scaled_y1 = offset_y + (y1 * final_scale)
                    
                    # Get original point size and calculate scaled size
                    point_size = orig_elem.get('relative_size', {}).get('point_size', 12)
                    scaled_font_size = point_size * final_scale * text_scale
                    
                    # Create font with scaled size
                    font = QFont()
                    font.setPointSizeF(scaled_font_size)
                    
                    # Create QTextDocument for better text rendering
                    doc = QTextDocument()
                    doc.setDefaultFont(font)
                    doc.setPlainText(translated_text)
                    
                    # Set line spacing using QTextBlockFormat
                    cursor = QTextCursor(doc)
                    cursor.select(QTextCursor.Document)
                    format = QTextBlockFormat()
                    # Convert line spacing multiplier to percentage (e.g., 1.0 -> 100%, 1.5 -> 150%)
                    line_spacing_percent = int(line_spacing * 100)
                    format.setLineHeight(line_spacing_percent, QTextBlockFormat.ProportionalHeight)
                    cursor.mergeBlockFormat(format)
                    
                    # Set text width to enable word wrapping
                    text_width = scaled_x1 - scaled_x0
                    doc.setTextWidth(text_width)
                    
                    # Draw the document
                    painter.save()
                    painter.translate(scaled_x0, scaled_y0)
                    doc.drawContents(painter, QRectF(0, 0, text_width, scaled_y1 - scaled_y0))
                    painter.restore()
                    
        except Exception as e:
            logger.error(f"Error painting translated page: {str(e)}")
            if painter:
                painter.end()
        finally:
            if painter:
                painter.end()

    def wheelEvent(self, event):
        """Handle zoom in/out with mouse wheel"""
        wheel_delta = event.angleDelta().y()
        if wheel_delta == 0:
            return

        # Choose a zoom step factor
        zoom_step = 1.2 if wheel_delta > 0 else 1 / 1.2
        
        # The position of the cursor in widget coordinates
        cursor_pos = event.position()
        cursor_point = QPointF(cursor_pos.x(), cursor_pos.y())

        # Compute the old scene coordinates (before zoom)
        widget_size = self.size()
        page_width = self.page_dims['points']['width']
        page_height = self.page_dims['points']['height']
        
        scale_x = widget_size.width() / page_width
        scale_y = widget_size.height() / page_height
        base_scale = min(scale_x, scale_y)
        
        old_final_scale = base_scale * self.zoom_factor
        
        # The current top-left after pan offset
        current_width = page_width * old_final_scale
        current_height = page_height * old_final_scale
        current_x = (widget_size.width() - current_width) / 2 + self.pan_offset.x()
        current_y = (widget_size.height() - current_height) / 2 + self.pan_offset.y()

        # Convert cursor position to "scene" coords
        scene_x = (cursor_point.x() - current_x) / old_final_scale
        scene_y = (cursor_point.y() - current_y) / old_final_scale

        # Adjust the zoom factor
        self.zoom_factor *= zoom_step
        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))

        # After changing zoom_factor, compute new final scale
        new_final_scale = base_scale * self.zoom_factor
        new_width = page_width * new_final_scale
        new_height = page_height * new_final_scale
        new_x = (widget_size.width() - new_width) / 2
        new_y = (widget_size.height() - new_height) / 2

        # Recompute what top-left would have to be so that
        # (scene_x, scene_y) is still under cursor_point
        desired_top_left_x = cursor_point.x() - scene_x * new_final_scale
        desired_top_left_y = cursor_point.y() - scene_y * new_final_scale

        # Update pan_offset
        dx = desired_top_left_x - new_x
        dy = desired_top_left_y - new_y
        self.pan_offset = QPointF(dx, dy)

        self.update()

    def mousePressEvent(self, event):
        """Start panning when the user presses the left mouse button"""
        if event.button() == Qt.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """If panning is active, update the pan_offset"""
        if self.is_panning:
            new_mouse_pos = event.pos()
            delta = new_mouse_pos - self.last_mouse_pos
            self.last_mouse_pos = new_mouse_pos
            self.pan_offset += delta
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Stop panning on left button release"""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

class PreferencesDialog(QDialog):
    """Dialog for application preferences"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(500)
        self.settings = QSettings("PDFTranslator", "Settings")
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tabs for different settings categories
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Translation settings tab
        translation_tab = QWidget()
        translation_layout = QVBoxLayout(translation_tab)
        
        # Group: Language settings
        language_group = QGroupBox("Language Settings")
        language_layout = QFormLayout(language_group)
        
        # Input language selection
        self.input_language_combo = QComboBox()
        self.input_language_combo.addItem("English", "en")
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
        language_layout.addRow("Input Language:", self.input_language_combo)
        
        # Output language selection
        self.output_language_combo = QComboBox()
        self.output_language_combo.addItem("Korean", "ko")
        self.output_language_combo.addItem("Japanese", "ja")
        self.output_language_combo.addItem("Chinese", "zh")
        self.output_language_combo.addItem("Spanish", "es")
        self.output_language_combo.addItem("French", "fr")
        language_layout.addRow("Output Language:", self.output_language_combo)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItem("GPT-3.5 Turbo", "gpt-3.5-turbo")
        self.model_combo.addItem("GPT-4 Turbo", "gpt-4-turbo")
        self.model_combo.addItem("GPT-4o mini", "gpt-4o-mini")
        language_layout.addRow("Translation Model:", self.model_combo)
        
        translation_layout.addWidget(language_group)
        
        # Group: Translation options
        options_group = QGroupBox("Translation Options")
        options_layout = QVBoxLayout(options_group)
        
        # Auto-translate checkbox
        self.auto_translate_checkbox = QCheckBox("Auto-translate")
        self.auto_translate_checkbox.setToolTip("Automatically translate when changing pages")
        options_layout.addWidget(self.auto_translate_checkbox)
        
        # Look-ahead translation checkbox
        self.look_ahead_checkbox = QCheckBox("Look-ahead translation")
        self.look_ahead_checkbox.setToolTip("Pre-translate next page in background")
        options_layout.addWidget(self.look_ahead_checkbox)
        
        # Text scale factor
        self.text_scale_spin = QDoubleSpinBox()
        self.text_scale_spin.setRange(0.1, 2.0)
        self.text_scale_spin.setSingleStep(0.1)
        self.text_scale_spin.setValue(0.8)  # Default to 80%
        self.text_scale_spin.setToolTip("Scale factor for translated text size (0.1 to 2.0)")
        options_layout.addWidget(QLabel("Text Scale Factor:"))
        options_layout.addWidget(self.text_scale_spin)
        
        translation_layout.addWidget(options_group)
        
        # Add the translation tab
        tabs.addTab(translation_tab, "Translation")
        
        # Debug settings tab
        debug_tab = QWidget()
        debug_layout = QVBoxLayout(debug_tab)
        
        # Debug level selection
        debug_group = QGroupBox("Debug Settings")
        debug_form = QFormLayout(debug_group)
        
        self.debug_level_combo = QComboBox()
        self.debug_level_combo.addItem("DEBUG", logging.DEBUG)
        self.debug_level_combo.addItem("INFO", logging.INFO)
        self.debug_level_combo.addItem("WARNING", logging.WARNING)
        self.debug_level_combo.addItem("ERROR", logging.ERROR)
        debug_form.addRow("Debug Level:", self.debug_level_combo)
        
        debug_layout.addWidget(debug_group)
        tabs.addTab(debug_tab, "Debug")
        
        # Add buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
    def load_settings(self):
        """Load current settings from QSettings"""
        self.input_language_combo.setCurrentText(self.settings.value("input_language_name", "English"))
        self.output_language_combo.setCurrentText(self.settings.value("output_language_name", "Korean"))
        self.model_combo.setCurrentText(self.settings.value("model_name", "GPT-3.5 Turbo"))
        self.auto_translate_checkbox.setChecked(self.settings.value("auto_translate", False, bool))
        self.look_ahead_checkbox.setChecked(self.settings.value("look_ahead", False, bool))
        self.text_scale_spin.setValue(self.settings.value("text_scale", 0.8, float))
        
        # Debug level
        debug_level = self.settings.value("debug_level", logging.INFO, int)
        for i in range(self.debug_level_combo.count()):
            if self.debug_level_combo.itemData(i) == debug_level:
                self.debug_level_combo.setCurrentIndex(i)
                break
    
    def save_settings(self):
        """Save settings to QSettings"""
        # Language settings
        self.settings.setValue("input_language", self.input_language_combo.currentData())
        self.settings.setValue("output_language", self.output_language_combo.currentData())
        self.settings.setValue("input_language_name", self.input_language_combo.currentText())
        self.settings.setValue("output_language_name", self.output_language_combo.currentText())
        
        # Model setting
        self.settings.setValue("model", self.model_combo.currentData())
        self.settings.setValue("model_name", self.model_combo.currentText())
        
        # Translation options
        self.settings.setValue("auto_translate", self.auto_translate_checkbox.isChecked())
        self.settings.setValue("look_ahead", self.look_ahead_checkbox.isChecked())
        self.settings.setValue("text_scale", self.text_scale_spin.value())
        
        # Debug level
        self.settings.setValue("debug_level", self.debug_level_combo.currentData())
        
        # Sync settings to disk
        self.settings.sync()
        
        # Close the dialog
        self.accept()

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
pyinstaller --name "PDFTranslator_v0.0.3.exe" --onefile --noconsole PDFTranslator.py
'''