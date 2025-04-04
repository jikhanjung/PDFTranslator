# PDF Translator

A desktop application for translating PDF documents using OpenAI's language models.

## Features

- **PDF Viewing**: Open and navigate through PDF files with previous/next page controls
- **Text Extraction**: Automatically extracts text from PDF pages
- **AI-Powered Translation**: Translate extracted text using OpenAI's models (GPT-3.5 Turbo or GPT-4 Turbo)
- **Multiple Languages**: Support for translating between various languages including:
  - English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, and more
  - Auto-detection of input language
- **Translation Cache**: Stores translations to avoid redundant API calls
- **Look-ahead Translation**: Preemptively translates upcoming pages for smoother reading experience
- **Bulk Translation**: Option to translate the entire document at once
- **Export Options**: Export translated content as text (.txt) or PDF (.pdf) files
- **Session Management**: Save and load translation sessions to continue work later
- **Progress Tracking**: Visual progress bar showing translated pages
- **Customizable Logging**: Adjustable debug levels for troubleshooting

## Requirements

- Python 3.6+
- PyQt5
- PyMuPDF (fitz)
- OpenAI API key
- Other dependencies listed in the source code

## Usage

1. Launch the application
2. Enter your OpenAI API key when prompted (stored securely using system keyring)
3. Open a PDF file using the "Open PDF" button
4. Navigate through pages using Previous/Next buttons or direct page entry
5. Select source and target languages
6. Click "Translate" to translate the current page or "Translate All" for the entire document
7. Export translations as needed with the "Export" button

## Note

This application requires an OpenAI API key and will use API credits based on the selected model.
