# Medical Emergency AI Assistant

This project implements an AI-powered medical emergency assistant using Google's Gemini model and a Qdrant vector database for medical knowledge retrieval. It features asynchronous and threaded processing for improved performance, as well as LLM function calling for enhanced interaction capabilities.

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/hamees-sayed/assignment.git
   cd assignment
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the Google API key:
   - Obtain a Google API key from the [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Add your API key to the `.env` file:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

5. Run the AI assistant:
   ```
   python app.py
   ```

## Files Description

- `requirements.txt`: Lists all the Python dependencies
- `prompt.txt`: Contains the system prompt for the AI assistant
- `receptionist.py`: Main script that runs the AI assistant
- `db.ipynb`: Jupyter notebook for initializing and populating the Qdrant database (already initialized, run this if you have your own dataset)

## Usage

Once the assistant is running, you can interact with it by typing messages in the Terminal. The assistant will determine if your message is a medical emergency or a general message for Dr. Adrin and respond accordingly.