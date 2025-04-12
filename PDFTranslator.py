import sys
import fitz  # PyMuPDF
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                            QScrollArea, QSizePolicy, QSplitter, QTextEdit, QComboBox,
                            QDialog, QLineEdit, QDialogButtonBox, QMessageBox,
                            QStatusBar, QInputDialog, QGroupBox, QCheckBox,
                            QTabWidget)
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPainter, QPen, QColor, QFont, QBrush
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
import os
import openai  # Revert to standard import
from dotenv import load_dotenv
import time
import json
import keyring  # For secure storage of API key
from fpdf import FPDF  # Replace with fpdf2 for better Unicode support
import logging
from datetime import datetime
import pickle
import base64
import os.path
import httpx
import re
import requests

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
        
        self.current_file = None
        self.current_page = 0
        self.doc = None
        
        # Centralized document data storage
        self.document_data = {
            'page_structures': {},  # {page_num: {'timestamp': str, 'structure': dict}}
            'translations': {},     # {page_num: {language: translation}}
            'metadata': {
                'title': '',
                'author': '',
                'subject': '',
                'keywords': ''
            }
        }
        
        # Thread management
        self.active_workers = []  # Keep track of all active worker threads
        self.look_ahead_worker = None  # Keep reference to look-ahead worker thread
        
        # Check for API key on startup
        self.check_api_key()
        
        self.init_ui()
        
    def set_page_structure(self, page_num, structure):
        """Set the page structure for a specific page"""
        logger.debug(f"Setting structure for page {page_num}")
        self.document_data['page_structures'][page_num] = {
            'timestamp': datetime.now().isoformat(),
            'structure': structure
        }
        logger.debug(f"Stored structure: {self.document_data['page_structures'][page_num]}")
        self.update_page_display()

    def get_page_structure(self, page_num):
        """Get the page structure for a specific page"""
        return self.document_data['page_structures'].get(page_num)

    def set_translation(self, page_num, language, translation):
        """Set the translation for a specific page and language"""
        if page_num not in self.document_data['translations']:
            self.document_data['translations'][page_num] = {}
        self.document_data['translations'][page_num][language] = translation
        self.update_page_display()

    def get_translation(self, page_num, language):
        """Get the translation for a specific page and language"""
        return self.document_data['translations'].get(page_num, {}).get(language)

    def save_session(self):
        """Save current session data to a file"""
        try:
            if not hasattr(self, 'current_file') or not self.current_file:
                logger.debug("No current file to save session for")
                return
                
            # Create sessions directory if it doesn't exist
            if not os.path.exists('sessions'):
                os.makedirs('sessions')
                
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            session_file = os.path.join('sessions', f'{base_name}_session.json')
            
            # Convert tuple keys to strings in translations
            translations_for_json = {}
            for (page_num, language), translation in self.document_data['translations'].items():
                key = f"{page_num}_{language}"
                translations_for_json[key] = translation
            
            # Prepare session data
            session_data = {
                'filename': self.current_file,
                'current_page': self.current_page,
                'document_data': {
                    'page_structures': self.document_data['page_structures'],
                    'translations': translations_for_json,
                    'metadata': self.document_data['metadata']
                },
                'auto_translate': self.auto_translate_checkbox.isChecked(),
                'look_ahead': self.look_ahead_checkbox.isChecked(),
                'timestamp': datetime.now().isoformat(),
                'total_pages': len(self.doc) if self.doc else 0,
                'session_info': {
                    'analyzed_pages': len(self.document_data['page_structures']),
                    'translated_pages': len(self.document_data['translations']),
                    'app_version': '1.0.0'
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
        """Load session data from a file"""
        try:
            if session_file is None:
                if not hasattr(self, 'current_file') or not self.current_file:
                    return
                    
                # Get base filename without extension
                base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                session_file = os.path.join('sessions', f'{base_name}_session.json')
                old_session_file = os.path.join('sessions', f'{base_name}_session.pkl')
                
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # Load document data
                self.document_data = session_data.get('document_data', {
                    'page_structures': {},
                    'translations': {},
                    'metadata': {}
                })
                
                # Convert string page numbers to integers in page_structures
                if 'page_structures' in self.document_data:
                    self.document_data['page_structures'] = {
                        int(k): v for k, v in self.document_data['page_structures'].items()
                    }
                
                # Convert string keys back to tuples in translations
                if 'translations' in self.document_data:
                    translations = {}
                    for key, value in self.document_data['translations'].items():
                        try:
                            page_num, language = key.split('_')
                            translations[(int(page_num), language)] = value
                        except ValueError:
                            logger.warning(f"Invalid translation key format: {key}")
                    self.document_data['translations'] = translations
                
                # Load translation options
                self.auto_translate_checkbox.setChecked(session_data.get('auto_translate', False))
                self.look_ahead_checkbox.setChecked(session_data.get('look_ahead', False))
                
                # Set current page
                if 'current_page' in session_data:
                    self.current_page = session_data['current_page']
                    
                # Update display
                self.update_page_display()
                self.refresh_progress_bar()
                
                # Force update of PDF display to show bounding boxes
                logger.info(f"Loading session for page: {self.current_page}")
                logger.info(f"Page structures: {self.document_data['page_structures']}")
                logger.info(f"Translations: {self.document_data['translations']}")

                if self.current_page in self.document_data['page_structures']:
                    logger.info(f"Current page: {self.current_page}")
                    structure = self.document_data['page_structures'][self.current_page]
                    logger.info(f"Structure: {structure}")
                    # Ensure the structure is properly formatted
                    if isinstance(structure, dict) and 'structure' in structure:
                        self.pdf_display.set_page_structure(self.current_page, structure)
                        self.pdf_display.update()  # Force repaint
                        # Show the page structure in the structure text area
                        self.show_page_structure(structure['structure'])
                    else:
                        logger.warning(f"Invalid structure format for page {self.current_page}: {structure}")
                else:
                    logger.info(f"No page structure found for page {self.current_page}")
                
                logger.info(f"Session loaded successfully")
                logger.info(f"Loaded {len(self.document_data['page_structures'])} page structures")
                logger.info(f"Loaded {len(self.document_data['translations'])} translations")
                self.status_label.showMessage("Session loaded", 3000)
                
            else:
                logger.info(f"No session file found")
                
        except Exception as e:
            logger.error(f"Error loading session: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Error loading session: {str(e)}", 5000)
        
    def init_ui(self):
        self.setWindowTitle('PDF Translator')
        self.setGeometry(100, 100, 1200, 800)  # Wider window for two panes
        
        # Create central widget and main layout
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
        self.page_input.setFixedWidth(50)
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
        
        # Add output language selection combo box
        self.output_language_combo = QComboBox()
        self.output_language_combo.addItem("Korean", "ko")
        self.output_language_combo.addItem("Japanese", "ja")
        self.output_language_combo.addItem("Chinese", "zh")
        self.output_language_combo.addItem("Spanish", "es")
        self.output_language_combo.addItem("French", "fr")
        
        # Connect language change signal
        self.output_language_combo.currentTextChanged.connect(self.on_output_language_changed)
        
        # Add model selection combo box
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

        # Add analyze button
        self.analyze_btn = QPushButton('Analyze')
        self.analyze_btn.clicked.connect(lambda: self.analyze_page(True))
        self.analyze_btn.setEnabled(False)

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
        
        # Add session buttons to a new layout
        session_layout = QHBoxLayout()
        session_layout.addWidget(self.save_session_btn)
        session_layout.addWidget(self.load_session_btn)
        
        # Create translation options group
        translation_options_layout = QHBoxLayout()
        
        # Create checkboxes
        self.auto_translate_checkbox = QCheckBox("Auto-translate")
        self.auto_translate_checkbox.setToolTip("Automatically translate when changing pages")
        self.auto_translate_checkbox.setChecked(False)
        
        self.look_ahead_checkbox = QCheckBox("Look-ahead translation")
        self.look_ahead_checkbox.setToolTip("Pre-translate next page in background")
        self.look_ahead_checkbox.setChecked(False)
        
        # Add checkboxes to the translation options layout
        translation_options_layout.addWidget(self.auto_translate_checkbox)
        translation_options_layout.addWidget(self.look_ahead_checkbox)
        
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addLayout(page_layout)
        btn_layout.addWidget(QLabel("From:"))
        btn_layout.addWidget(self.input_language_combo)
        btn_layout.addWidget(QLabel("To:"))
        btn_layout.addWidget(self.output_language_combo)
        btn_layout.addWidget(QLabel("Model:"))
        btn_layout.addWidget(self.model_combo)
        btn_layout.addWidget(QLabel("Debug:"))
        btn_layout.addWidget(self.debug_level_combo)
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addWidget(self.translate_btn)
        btn_layout.addWidget(self.translate_all_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addLayout(session_layout)
        btn_layout.addLayout(translation_options_layout)
        
        # Create a widget for the status and progress bar at the bottom
        status_layout = QHBoxLayout()
        
        # Create a status bar widget
        self.status_label = QStatusBar()
        
        # Create the translation progress widget
        self.translation_progress = TranslationProgressBar()
        
        # Add widgets to the status layout
        status_layout.addWidget(self.status_label, 8)
        status_layout.addWidget(self.translation_progress, 2)
        
        # Create splitter for the two panes
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left pane - PDF View
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        
        # Create tab widget for original and translated views
        self.view_tabs = QTabWidget()
        self.view_tabs.setTabPosition(QTabWidget.North)
        
        # Create scroll area for PDF display
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Create label to display PDF
        self.pdf_display = PDFDisplayWidget(self)
        self.scroll_area.setWidget(self.pdf_display)
        
        # Create scroll area for translated page display
        self.translated_scroll_area = QScrollArea()
        self.translated_scroll_area.setWidgetResizable(True)
        
        # Create widget to display translated page
        self.translated_display = TranslatedPageDisplayWidget(self)
        self.translated_scroll_area.setWidget(self.translated_display)
        
        # Add tabs to the tab widget
        self.view_tabs.addTab(self.scroll_area, "Original")
        self.view_tabs.addTab(self.translated_scroll_area, "Translated")
        
        # Connect tab change signal
        self.view_tabs.currentChanged.connect(self.on_view_tab_changed)
        
        left_layout.addWidget(self.view_tabs)
        
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
        
        # Text display for page structure
        self.structure_text = QTextEdit()
        self.structure_text.setReadOnly(True)
        self.structure_text.setPlaceholderText("Page structure analysis will appear here")
        
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
        
        structure_container = QWidget()
        structure_layout = QVBoxLayout(structure_container)
        structure_layout.addWidget(QLabel("Page Structure:"))
        structure_layout.addWidget(self.structure_text)
        
        # Add containers to the splitter
        text_splitter.addWidget(original_container)
        text_splitter.addWidget(translated_container)
        text_splitter.addWidget(structure_container)
        
        # Set the initial sizes - 1:2:1 ratio (25% original, 50% translated, 25% structure)
        text_splitter.setSizes([200, 400, 200])
        
        # Add the splitter to the right layout
        right_layout.addWidget(text_splitter)
        
        # Add widgets to main splitter
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        self.splitter.setSizes([600, 600])  # Equal initial sizes
        
        # Add everything to main layout
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(session_layout)
        main_layout.addWidget(self.splitter)
        main_layout.addLayout(status_layout)
        
        # Set up auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_session)
        self.auto_save_timer.start(300000)  # Auto-save every 5 minutes (300,000 ms)
        
        # Add show/hide bounding boxes checkbox
        self.show_boxes_checkbox = QCheckBox("Show Bounding Boxes")
        self.show_boxes_checkbox.setChecked(True)
        self.show_boxes_checkbox.stateChanged.connect(self.pdf_display.toggle_bounding_boxes)
        
        # Add the checkbox to the button layout
        btn_layout.addWidget(self.show_boxes_checkbox)

    def on_view_tab_changed(self, index):
        """Handle tab change between original and translated views"""
        if index == 1:  # Translated view
            # Update the translated page display
            if self.current_page in self.pdf_display.page_structures:
                structure = self.pdf_display.page_structures[self.current_page]
                current_language = self.output_language_combo.currentText()
                cache_key = (self.current_page, current_language)
                
                if cache_key in self.document_data['translations']:
                    translation = self.document_data['translations'][cache_key]
                    self.translated_display.set_page(self.current_page, structure, translation)
                else:
                    self.status_label.showMessage("No translation available for this page", 3000)
        else:  # Original view
            # Update the original page display
            self.update_page_display()
        
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
            
            # Update page total label
            self.page_total.setText(f"/ {len(self.doc)}")
            
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
            self.translate_all_btn.setEnabled(True)  # Enable the Translate All button
            self.export_btn.setEnabled(True)  # Enable the Export button
            self.save_session_btn.setEnabled(True)  # Enable the Save Session button
            
            # Check for existing session and ask if user wants to load it
            pdf_filename = os.path.basename(file_path)
            session_file = os.path.join('sessions', f'{os.path.splitext(pdf_filename)[0]}_session.json')
            
            if os.path.exists(session_file):
                reply = QMessageBox.question(self, 'Load Session', 
                    f'Found existing session for {pdf_filename}. Do you want to load it?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                
                if reply == QMessageBox.Yes:
                    self.load_session(session_file)
                else:
                    # Initialize text areas with current page content
                    self.extract_text()  # This will update the original text area
                    current_language = self.output_language_combo.currentText()
                    self.translated_text.setText("Select 'Translate' to translate to " + current_language)
                    self.structure_text.setText("Select 'Analyze' to analyze page structure")
            else:
                # Initialize text areas with current page content
                self.extract_text()  # This will update the original text area
                current_language = self.output_language_combo.currentText()
                self.translated_text.setText("Select 'Translate' to translate to " + current_language)
                self.structure_text.setText("Select 'Analyze' to analyze page structure")
    
    def update_page_display(self):
        """Update the PDF display and text areas with the current page content"""
        try:
            if self.doc is None:
                return
            
            # Update PDF display
            page = self.doc.load_page(self.current_page)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            image = QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride, QImage.Format_RGB888)
            self.pdf_display.set_page(self.current_page, QPixmap.fromImage(image))
            
            # Update page number display
            self.page_input.setText(str(self.current_page + 1))
            self.page_total.setText(f"/ {len(self.doc)}")
            
            # Extract and display text
            self.extract_text()
            
            # Check for cached translation
            current_language = self.output_language_combo.currentText()
            translation = self.get_translation(self.current_page, current_language)
            
            if translation:
                if isinstance(translation, dict):
                    # If translation is a dictionary (structured translation), extract the text
                    translation_text = translation.get('text', '')
                else:
                    translation_text = translation
                self.translated_text.setText(translation_text)
            else:
                self.translated_text.clear()
            
            # Update structure text if available
            structure = self.get_page_structure(self.current_page)
            if structure:
                if isinstance(structure, dict):
                    # Format the structure data for display
                    structure_text = []
                    if 'structure' in structure:
                        structure_data = structure['structure']
                        if 'elements' in structure_data:
                            for element in structure_data['elements']:
                                category = element.get('category', 'unknown')
                                content = element.get('content', {})
                                html = content.get('html', '')
                                # Extract text from HTML if available
                                if html:
                                    # Remove HTML tags and convert to plain text
                                    import re
                                    text = re.sub(r'<[^>]+>', '', html)
                                    structure_text.append(f"{category.upper()}: {text}")
                                else:
                                    structure_text.append(f"{category.upper()}: No content")
                    self.structure_text.setText('\n\n'.join(structure_text))
                else:
                    self.structure_text.setText(str(structure))
            else:
                self.structure_text.clear()
            
            # Update buttons state
            self.update_buttons()
            
            # Update progress bar
            self.refresh_progress_bar()
            
            # Force update of PDF display to show bounding boxes
            if self.current_page in self.document_data['page_structures']:
                structure = self.document_data['page_structures'][self.current_page]
                logger.debug(f"Setting page structure for page {self.current_page}: {structure}")
                # Ensure the structure is properly formatted
                if isinstance(structure, dict) and 'structure' in structure:
                    self.pdf_display.set_page_structure(self.current_page, structure)
                    self.pdf_display.update()  # Force repaint
                    # Show the page structure in the structure text area
                    self.show_page_structure(structure['structure'])
                else:
                    logger.warning(f"Invalid structure format for page {self.current_page}: {structure}")
            else:
                logger.info(f"No page structure found for page {self.current_page}")
            
        except Exception as e:
            logger.error(f"Error updating page display: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Error updating page: {str(e)}", 3000)
    
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
        """Go to next page"""
        if self.doc and self.current_page < len(self.doc) - 1:
            self.current_page += 1
            self.update_page_display()
            self.update_buttons()
            
            # Log state after page change
            logger.debug(f"Moved to next page: {self.current_page}")
            logger.debug(f"Available structures: {list(self.document_data['page_structures'].keys())}")
            
            # Only translate if auto-translate is enabled
            if self.auto_translate_checkbox.isChecked():
                self.translate_text()
                
            # Only do look-ahead if both auto-translate and look-ahead are enabled
            if self.look_ahead_checkbox.isChecked():
                self.start_look_ahead_translation()
    
    def prev_page(self):
        """Go to previous page"""
        if self.doc and self.current_page > 0:
            self.current_page -= 1
            self.update_page_display()
            self.update_buttons()

            # Log state after page change
            logger.debug(f"Moved to previous page: {self.current_page}")
            logger.debug(f"Available structures: {list(self.document_data['page_structures'].keys())}")

            # Only translate if auto-translate is enabled
            if self.auto_translate_checkbox.isChecked():
                self.translate_text()
                
            # Only do look-ahead if both auto-translate and look-ahead are enabled
            if self.look_ahead_checkbox.isChecked():
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
        """Check if a translation for the current page and language exists in cache"""
        output_lang_name = self.output_language_combo.currentText()
        cache_key = (self.current_page, output_lang_name)
        
        if cache_key in self.document_data['translations']:
            # Translation exists in cache, display it
            self.translated_text.setText(self.document_data['translations'][cache_key])
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
        if cache_key in self.document_data['translations']:
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
            self.document_data['translations'][cache_key] = translated_text
            
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
                
            # Save session to ensure translation is persisted
            logger.info(f"Saving session after translation of page {page_num+1}")
            self.save_session()
            logger.info(f"Session saved successfully with translation for page {page_num+1}")
            
        except Exception as e:
            # Log error
            logger.error(f"Error handling translation result: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            self.status_label.showMessage(f"Error saving translation: {str(e)}", 5000)
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
    
    def analyze_page(self, force_new_analysis=True):
        """Analyze current page using Upstage API"""
        logger.info(f"Analyzing page {self.current_page + 1}")
        # wait cursor
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            # Check if analysis already exists and force_new_analysis is False
            if not force_new_analysis and self.current_page in self.document_data['page_structures']:
                logger.info(f"Using cached analysis for page {self.current_page + 1}")
                result = self.document_data['page_structures'][self.current_page]['structure']
                self.show_page_structure(result)
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
                    
                    # Store the analysis result in document_data
                    self.document_data['page_structures'][self.current_page] = {
                        'timestamp': datetime.now().isoformat(),
                        'structure': result
                    }
                    
                    # Show the analysis text
                    self.show_page_structure(result)
                    
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
        try:
            if not analysis_result or 'elements' not in analysis_result:
                return pixmap

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
            logger.debug("Starting show_page_structure with result")
            
            display_text = "=== PAGE STRUCTURE ANALYSIS ===\n\n"
            
            if not result:
                logger.debug("Result is None or empty")
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
            
            logger.debug("Final display text:")
            logger.debug(display_text)
            
            # Update the structure text display
            self.structure_text.setText(display_text)
            
        except Exception as e:
            error_msg = f"Error displaying page structure: {str(e)}"
            logger.error(error_msg)
            logger.error("Full error details:", exc_info=True)
            self.structure_text.setText(f"Error displaying analysis results:\n{str(e)}")

    def translate_text(self, force_new_translation=True):
        """Translate the extracted text using OpenAI's GPT API with structured translation"""
        # Check if there is any text to translate
        if not self.extracted_text or self.extracted_text.strip() == "":
            self.translated_text.setText("No text to translate.")
            return
        
        # Log translation request
        logger.info(f"Manual translation requested - Page {self.current_page+1} - Force new: {force_new_translation}")
        
        language_name = self.output_language_combo.currentText()
        cache_key = (self.current_page, language_name)
        
        # Check if translation is already in cache and use it if not forcing new translation
        if not force_new_translation and cache_key in self.document_data['translations']:
            self.display_translation(self.document_data['translations'][cache_key])
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
        input_lang_name = self.input_language_combo.currentText()
        
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
            
            # Get the page structure if available
            page_structure = None
            if self.current_page in self.document_data['page_structures']:
                page_structure = self.document_data['page_structures'][self.current_page]
                if isinstance(page_structure, dict) and 'structure' in page_structure:
                    page_structure = page_structure['structure']
            
            # If we have page structure, use structured translation
            if page_structure and 'elements' in page_structure:
                self.status_label.showMessage("Translating with structure...")
                QApplication.processEvents()
                
                # Prepare the messages for GPT
                messages = [
                    {"role": "system", "content": f"You are a professional translator. Translate the provided text from {input_lang_name} to {output_lang_name} accurately while maintaining the original meaning and context. Preserve the structure of the text, including headers, paragraphs, and other elements. For each element, provide the translation in this format: [ELEMENT_TYPE] translated_text"}
                ]
                
                # Add the page structure information
                structure_info = "The text is structured as follows:\n"
                for element in page_structure['elements']:
                    category = element.get('category', 'unknown')
                    if 'content' in element and 'html' in element['content']:
                        content = element['content']['html']
                        # Clean HTML tags for the prompt
                        content = re.sub('<[^<]+?>', '', content)
                        structure_info += f"\n[{category.upper()}]\n{content}\n"
                
                messages.append({"role": "user", "content": f"Translate the following structured text to {output_lang_name}:\n\n{structure_info}"})
                
                # Get the selected model
                model = self.model_combo.currentData()
                
                # Mark the current page as translating in the progress bar
                self.translation_progress.mark_page_translating(self.current_page)
                
                # Call OpenAI API with version-appropriate syntax
                if is_new_version:
                    # For openai >= 1.0.0
                    response = openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=2000
                    )
                    translated = response.choices[0].message.content.strip()
                else:
                    # For openai < 1.0.0 (older versions)
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=2000
                    )
                    translated = response.choices[0].message['content'].strip()
                
                # Parse the structured translation
                structured_translation = self.parse_structured_translation(translated, page_structure)
                
                # Store translation in cache
                self.document_data['translations'][cache_key] = structured_translation
                
                # Display the translation
                self.display_translation(structured_translation)
                
            else:
                # Fall back to regular translation if no structure available
                self.status_label.showMessage("Translating without structure...")
                QApplication.processEvents()
            
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
                        model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                translated = response.choices[0].message.content.strip()
            else:
                # For openai < 1.0.0 (older versions)
                response = openai.ChatCompletion.create(
                        model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                translated = response.choices[0].message['content'].strip()
            
            # Store translation in cache
                self.document_data['translations'][cache_key] = translated
            
            # Display the translation
            self.translated_text.setText(translated)
            
            # Start look-ahead translation for next page after completing this one
            if self.look_ahead_checkbox.isChecked():
                self.start_look_ahead_translation()
                
        except Exception as e:
            self.translated_text.setText(f"Translation error: {str(e)}\n\nTip: Try running 'pip install openai==0.28' to downgrade to a compatible version.")
            
        finally:
            # Always restore the cursor if we set it
            if not cursor_already_waiting:
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
        """Display the translation in the text area, handling both structured and unstructured translations"""
        try:
            if isinstance(translation, dict) and 'elements' in translation:
                # Display structured translation
                display_text = ""
                for element in translation['elements']:
                    category = element.get('category', 'unknown').upper()
                    content = element.get('translated_content', '')
                    display_text += f"[{category}]\n{content}\n\n"
                self.translated_text.setText(display_text.strip())
            else:
                # Display unstructured translation
                self.translated_text.setText(translation)
                
        except Exception as e:
            logger.error(f"Error displaying translation: {str(e)}")
            self.translated_text.setText(str(translation))

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

    def cleanup_threads(self):
        """Properly terminate all running worker threads"""
        try:
            # Stop and clean up look-ahead worker if it exists
            if self.look_ahead_worker and self.look_ahead_worker.isRunning():
                logger.debug("Stopping look-ahead worker")
                self.look_ahead_worker.stop()
                self.look_ahead_worker.wait()
                self.look_ahead_worker = None
            
            # Stop and clean up all active workers
            if hasattr(self, 'active_workers'):
                for worker in self.active_workers:
                    if worker.isRunning():
                        logger.debug(f"Stopping worker thread: {worker}")
                        worker.stop()
                worker.wait()
                self.active_workers.clear()
            
            logger.debug("All worker threads cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up threads: {str(e)}")
            logger.error("Full error details:", exc_info=True)

    def closeEvent(self, event):
        """Handle application close event to clean up threads and save session"""
        logger.info("=== Application Closing ===")
        
        # Save current session if a document is open
        if hasattr(self, 'current_pdf_path') and self.current_pdf_path and self.doc:
            self.save_session()
        
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
        if current_cache_key not in self.document_data['translations']:
            self.translate_text(False)  # Translate current page without forcing
        
        # Start background translation for all other pages
        pending_pages = []
        for page_num in range(len(self.doc)):
            # Skip current page as it's already being translated
            if page_num == self.current_page:
                continue
            
            # Skip pages that are already translated
            cache_key = (page_num, output_lang_name)
            if cache_key in self.document_data['translations']:
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
        try:
            # Create and start translation worker
            worker = TranslationWorker(
                self.extracted_text,
            page_num,
            input_lang_code,
            output_lang_code,
            output_lang_name,
            model
            )
        
            # Connect signals
            worker.translationComplete.connect(self.on_translation_complete)
            
            # Add to active workers list
            self.active_workers.append(worker)
        
            # Start the worker
            worker.start()  
            
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

    def on_output_language_changed(self, language_name):
        """Handle output language selection change"""
        if not self.doc:
            return
        
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
        current_language = self.output_language_combo.currentText()
        for i in range(len(self.doc)):
            if (i, current_language) in self.document_data['translations']:
                self.translation_progress.mark_page_translated(i)

    def export_translation(self):
        """Export the translated document as a text or PDF file"""
        # Check if we have a document
        if not self.doc:
            return
        
        # Check if we have any translations in the cache
        current_language = self.output_language_combo.currentText()
        has_translations = False
        for key in self.document_data['translations']:
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
                if (i, current_language) in self.document_data['translations']:
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
                    if cache_key not in self.document_data['translations']:
                        continue
                    
                    # Write translated page with header
                    f.write(f"\n\n--- PAGE {i+1} ---\n\n")
                    f.write(self.document_data['translations'][cache_key])
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
                if (i, current_language) in self.document_data['translations']:
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
                if cache_key not in self.document_data['translations']:
                    continue
                
                # Get the translated content
                content = self.document_data['translations'][cache_key]
                
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
        """Handle page navigation when user enters a page number"""
        try:
            if not self.doc:
                return
                
            # Get the entered page number (1-based)
            try:
                new_page = int(self.page_input.text()) - 1
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
            
            # Only translate if auto-translate is enabled
            if self.auto_translate_checkbox.isChecked():
                self.translate_text()
                
            # Only do look-ahead if both auto-translate and look-ahead are enabled
            if self.look_ahead_checkbox.isChecked():
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

class PDFDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pixmap = None
        self.current_page = None
        self.show_bounding_boxes = False
        self.setMinimumSize(400, 600)
        
    def set_page(self, page_num, pixmap):
        """Set the current page to display"""
        self.current_page = page_num
        self.current_pixmap = pixmap
        self.update()
        
    def set_page_structure(self, page_num, structure):
        """Set the structure data for the current page"""
        logger.debug(f"Setting page structure for page {page_num}: {structure}")
        # Get the main window instance
        main_window = self.window()
        if hasattr(main_window, 'document_data'):
            main_window.document_data['page_structures'][page_num] = structure
            logger.debug(f"Updated document_data['page_structures']: {main_window.document_data['page_structures']}")
        self.update()
        
    def toggle_bounding_boxes(self, state):
        """Toggle the display of bounding boxes"""
        self.show_bounding_boxes = state
        self.update()
        
    def paintEvent(self, event):
        """Paint the PDF page with bounding boxes"""
        painter = None
        try:
            # Get the main window instance
            main_window = self.window()
            if not main_window or not hasattr(main_window, 'document_data'):
                logger.debug("No main window or document_data found")
                return

            # Get the current page structure
            if self.current_page not in main_window.document_data['page_structures']:
                logger.debug(f"No structure found for page {self.current_page}")
                return

            structure = main_window.document_data['page_structures'][self.current_page]
            logger.debug(f"Painting page {self.current_page} with structure: {structure}")

            # Draw the base image
            if self.current_pixmap:
                painter = QPainter(self)
                painter.drawPixmap(0, 0, self.current_pixmap)
                logger.debug(f"Base image drawn at size: {self.current_pixmap.size()}")

            # Draw bounding boxes if enabled
            if self.show_bounding_boxes and structure and 'structure' in structure:
                structure_data = structure['structure']
                logger.debug(f"Drawing bounding boxes for structure: {structure_data}")

                if 'elements' in structure_data:
                    elements = structure_data['elements']
                    logger.debug(f"Found {len(elements)} elements to draw")

                    for element in elements:
                        category = element.get('category', 'unknown')
                        coords = element.get('coordinates', [])
                        logger.debug(f"Processing element: category={category}, coords={coords}")

                        if len(coords) >= 4:
                            # Get the actual image dimensions
                            if self.current_pixmap:
                                img_width = self.current_pixmap.width()
                                img_height = self.current_pixmap.height()
                            else:
                                img_width = self.width()
                                img_height = self.height()
                            
                            logger.debug(f"Image dimensions: {img_width}x{img_height}")

                            # Convert normalized coordinates to pixel coordinates
                            x1 = int(coords[0]['x'] * img_width)
                            y1 = int(coords[0]['y'] * img_height)
                            x2 = int(coords[2]['x'] * img_width)
                            y2 = int(coords[2]['y'] * img_height)
                            
                            logger.debug(f"Drawing box for {category} at ({x1}, {y1}, {x2}, {y2})")

                            # Draw the bounding box
                            painter.setPen(QPen(QColor(255, 0, 0), 2))
                            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                            
                            # Draw category label
                            painter.setPen(QPen(QColor(0, 0, 0)))
                            painter.drawText(x1, y1 - 5, category)
                        else:
                            logger.debug(f"Invalid coordinates for {category}: {coords}")
                else:
                    logger.debug("No elements found in structure")
            else:
                logger.debug("Bounding boxes disabled or no structure available")

        except Exception as e:
            logger.error(f"Error in paintEvent: {str(e)}")
            logger.error("Full error details:", exc_info=True)
        finally:
            if painter is not None:
                painter.end()

class TranslatedPageDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_page = 0
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.background_color = QColor(255, 255, 255)  # White background
        self.text_color = QColor(0, 0, 0)  # Black text

    def set_page(self, page_num, structure, translation):
        """Set current page number, structure, and translation"""
        self.current_page = page_num
        # Store in parent's document_data
        if hasattr(self.parent(), 'document_data'):
            if structure:
                self.parent().document_data['page_structures'][page_num] = structure
            if translation:
                self.parent().document_data['translations'][page_num] = translation
        self.update()

    def paintEvent(self, event):
        """Paint the translated page with proper structure and text"""
        painter = None
        try:
            if not hasattr(self.parent(), 'document_data'):
                return

            document_data = self.parent().document_data
            if self.current_page not in document_data['page_structures'] or self.current_page not in document_data['translations']:
                return

            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Fill background
            painter.fillRect(self.rect(), self.background_color)
            
            # Get the page structure and translation
            structure = document_data['page_structures'][self.current_page]
            translation = document_data['translations'][self.current_page]
            
            if not isinstance(structure, dict) or 'structure' not in structure:
                return
                
            analysis_result = structure['structure']
            
            # Set up the font for text
            font = QFont()
            font.setPointSize(11)  # Default font size
            
            # Draw each element with its translated text
            if 'elements' in analysis_result:
                for element in analysis_result['elements']:
                    category = element.get('category', 'unknown')
                    coords = element.get('coordinates', [])
                    
                    if len(coords) >= 4:
                        # Convert normalized coordinates to pixel coordinates
                        x1 = int(coords[0]['x'] * self.width())
                        y1 = int(coords[0]['y'] * self.height())
                        x2 = int(coords[2]['x'] * self.width())
                        y2 = int(coords[2]['y'] * self.height())
                        
                    # Get the translated text for this element
                    translated_text = translation.get(category, '')
                    
                    # Set up the text rectangle
                    text_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                    
                    # Draw the text
                    painter.setPen(self.text_color)
                    painter.setFont(font)
                    painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, translated_text)
        
        except Exception as e:
            logger.error(f"Error in paintEvent: {str(e)}")
            logger.error("Full error details:", exc_info=True)
        finally:
            if painter is not None:
                painter.end()

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