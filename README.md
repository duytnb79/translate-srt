# SRT Subtitle Translator

This Python script translates SRT subtitle files from one language to Vietnamese using the OpenAI API. It processes subtitles in batches, displays a progress bar, saves translated results incrementally after each batch, and provides an estimated cost for the API usage.

## Features

*   **Batch Translation**: Translates SRT files in configurable batches to handle large files efficiently.
*   **Incremental Saving**: Saves the translated output to a file after each batch is processed. This ensures that if the script is interrupted, progress up to the last completed batch is saved.
*   **Cost Estimation**: Calculates and displays the estimated cost of OpenAI API calls based on token usage for the specified model.
*   **Progress Bar**: Uses `tqdm` to display a progress bar during the translation process.
*   **Environment Variables**: Securely manages the OpenAI API key using a `.env` file.
*   **Automatic Output Naming**: Generates output file names automatically based on input file names, placing them in an `output` directory.
*   **JSON Mode**: Leverages OpenAI's JSON mode for more reliable and structured responses.

## Prerequisites

*   Python 3.7+
*   An OpenAI API key.

## Setup

1.  **Clone the repository (if applicable) or download the script.**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenAI API Key:**
    Create a file named `.env` in the root directory of the project and add your OpenAI API key:
    ```env
    OPENAI_API_KEY='your_openai_api_key_here'
    ```

5.  **Prepare your input SRT file:**
    Place your SRT file (e.g., `input.srt`) into the `data` directory. If the `data` directory doesn't exist, create it.

## Usage

Run the script from the root directory of the project:

```bash
python main.py
```

### Configuration (in `main.py`)

You can modify the following parameters directly in the `main.py` script at the bottom:

*   `input_file_path`: Path to the input SRT file (e.g., `"data/your_subtitle_file.srt"`).
*   `output_directory`: Directory where translated files will be saved (defaults to `"output/"`).
*   `MODEL_NAME`: The OpenAI model to use for translation (e.g., `"gpt-4.1-mini"`, `"gpt-3.5-turbo"`).
*   `BATCH_SIZE`: The number of subtitle blocks to process in each API call.

### Output

*   Translated SRT files will be saved in the `output` directory (e.g., `output/your_subtitle_file_translated_batch.srt`).
*   The script will print the total prompt tokens, total completion tokens, and the estimated cost to the console after completion.
*   Progress will be displayed in the console during translation.

## Cost Calculation

The script includes a dictionary `MODEL_PRICING` to estimate costs. Ensure the prices for input (prompt) and output (completion) tokens per 1K tokens are up-to-date for the models you use. You can find the latest pricing on the OpenAI website.

```python
MODEL_PRICING = {
    "gpt-4.1-mini": {"prompt": 0.00015, "completion": 0.00060}, # Example: $0.15/1M prompt, $0.60/1M completion
    "gpt-3.5-turbo": {"prompt": 0.0000005, "completion": 0.0000015}, # Example: $0.50/1M prompt, $1.50/1M completion
    # Add other models and their pricing here
}
```

## Project Structure

```
.
├── .env                # Stores OpenAI API key (you need to create this)
├── data/
│   └── your_input.srt  # Place your input SRT files here
├── output/
│   └──                 # Translated files will be saved here
├── main.py             # The main script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
