import os
import sys
import time
import queue
import logging
import warnings
import threading
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
import google.generativeai as genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google.generativeai.types import HarmCategory, HarmBlockThreshold

warnings.filterwarnings("ignore")

class MedicalAssistant:
    def __init__(self):
        load_dotenv()
        logging.basicConfig(filename='medical_assistant.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.client, self.encoder = self.initialize_db()
        self.chat = self.initialize_chat()
        self.console = Console()
        self.task_queue = queue.Queue()
        self.background_thread = threading.Thread(target=self.background_task_manager)
        self.background_thread.daemon = True
        self.background_thread.start()
        self.input_ready_event = threading.Event()

    def load_prompt(self):
        with open("prompt.txt", "r") as file:
            return file.read()

    def initialize_db(self):
        print("Initializing the database...")
        client = QdrantClient(path="medicalqna.db")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return client, encoder

    def initialize_chat(self):
        print("Initializing the chat...")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name='gemini-1.5-pro',  # 'gemini-1.5-pro-latest', 'gemini-1.5-flash', 'gemini-1.5-flash-latest'
            system_instruction=self.load_prompt(),
            tools=[self.lookup_user_emergency, self.send_email_to_doctor],
            safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return model.start_chat()

    def lookup_user_emergency(self, emergency: str):
        """
        Handle emergency situations by retrieving relevant instructions from the vector database.
        """
        logging.info(f"lookup_user_emergency called with emergency after 15 seconds: {emergency}")
        vectors = self.client.search(
            collection_name="medicalqna",
            query_vector=self.encoder.encode(emergency).tolist(),
            limit=3
        )
        context = "".join(vector.payload['answer'] for vector in vectors)
        emergency_prompt = f"""
        The user has reported an emergency: "{emergency}". Please provide immediate instructions. You already have the user's location so don't ask it again. Use the context provided below to guide the user through the emergency.
        Instructions: DON'T TELL THE USER TO CALL MEDICAL SERVICE, since Dr. Adrin is the medical service. Instead, provide instructions based on the context of emergency and steps given by doctor below.
        Doctor's Instruction context: {context}
        """
        try:
            response = self.chat.send_message(emergency_prompt)
            if response.parts and hasattr(response.parts[0], 'text'):
                self.console.print(Markdown(response.parts[0].text))
            else:
                logging.error("Chat model response was empty or in an unexpected format.")
        except Exception as e:
            logging.exception("Exception in lookup_user_emergency")

    def send_email_to_doctor(self, message: str):
        """
        Send the user's message to Dr. Adrin via email and acknowledge receipt.
        """
        logging.info(f"send_email_to_doctor called with message: {message}")
        message_prompt = f"""
        The user has left a message for Dr. Adrin: "{message}". Let the user know that the message has been forwarded to Dr. Adrin via email. They will receive a confirmation shortly.
        """
        try:
            response = self.chat.send_message(message_prompt)
            if response.parts and hasattr(response.parts[0], 'text'):
                self.console.print(Markdown(response.parts[0].text))
            else:
                logging.error("Chat model response was empty or in an unexpected format.")
        except Exception as e:
            logging.exception("Exception in send_email_to_doctor")

    def background_task_manager(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            # Wait until the input is provided
            self.input_ready_event.wait()
            task()
            self.task_queue.task_done()
            self.input_ready_event.clear()

    def handle_user_input(self, user_input):
        """
        Handle the user input by determining the appropriate action and executing it in the background if necessary.
        """
        logging.info(f"handle_user_input called with input: {user_input}")
        response = self.chat.send_message(user_input)
        
        if response.parts and response.parts[0].function_call:
            function_name = response.parts[0].function_call.name
            if function_name == "lookup_user_emergency":
                self.handle_emergency(user_input)
            elif function_name == "send_email_to_doctor":
                self.handle_message(user_input)
            else:
                logging.warning(f"Unknown function call: {function_name}")
                self.console.print(Markdown("I'm sorry, I couldn't process that request. Can you please rephrase?"))
        else:
            self.console.print(Markdown(response.parts[0].text if response.parts else "I'm sorry, I couldn't generate a response."))

    def handle_emergency(self, user_input):
        logging.info("lookup_user_emergency function added to queue")
        self.task_queue.put(lambda: self.lookup_user_emergency(user_input))
        emergency_prompt = f"User has reported an emergency: {user_input}. Let the user know that the instructions are being retrieved and ask questions to keep them engaged while you gather more information."
        response = self.chat.send_message(emergency_prompt).parts[0].text
        self.console.print(Markdown(response))
        user_location = self.console.input("> ")
        sys_prompt = f"User has reported an emergency and given this info: {user_location}, based on it ask them if anyone is with them and tell them that the doctor will be arriving by {time.strftime('%-I:%M %p', time.localtime(time.time() + 30*60))}. Use the history of the conversation for your context."
        self.console.print(Markdown(self.chat.send_message(sys_prompt).parts[0].text))

        # Signal that user input is provided, resume background task
        self.input_ready_event.set()
        self.task_queue.join()

    def handle_message(self, user_input):
        logging.info("send_email_to_doctor function added to queue")
        self.task_queue.put(lambda: self.send_email_to_doctor(user_input))

        # Signal that user input is provided, resume background task
        self.input_ready_event.set()
        self.task_queue.join()

    def start_chat(self):
        self.console.print(Markdown("Hello! I am the AI receptionist for Dr. Adrin. How can I help you today? Are you experiencing a medical emergency, or would you like to leave a message for the doctor? Type 'quit' to exit.\n"))
        while True:
            user_input = self.console.input("\n> ")
            if user_input.lower().strip() == "quit":
                self.console.print("Take care! Goodbye.")
                sys.exit(0)
            self.handle_user_input(user_input)

if __name__ == "__main__":
    assistant = MedicalAssistant()
    assistant.start_chat()
