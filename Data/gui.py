import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tkinter as tk  # For GUI components
from tkinter import scrolledtext, ttk, messagebox  # For advanced GUI components like scrolled text and message boxes
import random  # To select random responses
import json  # To read intents data from a JSON file
import pickle  # For loading serialized data (words and classes)
import numpy as np  # For numerical operations
import nltk  # Natural Language Toolkit for NLP processing
from nltk.stem import WordNetLemmatizer  # For word lemmatization (reducing words to their base form)
from keras.models import load_model  # To load the pre-trained chatbot model
import webbrowser  # To open web searches
import datetime  # To get the current time for the chatbot's responses

# Main class for the Chatbot GUI
class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.setup_gui()  # Setup the GUI layout and components
        self.load_chatbot_data()  # Load the chatbot model, words, classes, and intents
        self.conversation_history = []  # Store conversation history for saving or clearing

    # Setting up the GUI for the chatbot
    def setup_gui(self):
        self.master.title("Enhanced NLP Chatbot")  # Title of the application window
        self.master.geometry("500x600")  # Size of the application window
        self.master.configure(bg="#f0f0f0")  # Background color of the window

        # Style configuration for better GUI appearance
        style = ttk.Style()
        style.theme_use("clam")

        # Main frame for all components
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Display area for chat history
        self.chat_history = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=60, height=25, font=("Arial", 10)
        )
        self.chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_history.config(state=tk.DISABLED)  # Disable editing of chat history

        # Frame for user input and send button
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)

        # User input box
        self.user_input = ttk.Entry(input_frame, width=50, font=("Arial", 10))
        self.user_input.pack(side=tk.LEFT, padx=(0, 5), expand=True, fill=tk.X)
        self.user_input.bind("<Return>", lambda event: self.send_message())  # Enter key to send message

        # Send button to submit user input
        self.send_button = ttk.Button(
            input_frame, text="Send", command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        # Frame for additional buttons (Clear Chat, Save Chat, Help)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        # Clear Chat button to clear the chat history
        self.clear_button = ttk.Button(
            button_frame, text="Clear Chat", command=self.clear_chat
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))

        # Save Chat button to save the chat history to a file
        self.save_button = ttk.Button(
            button_frame, text="Save Chat", command=self.save_chat
        )
        self.save_button.pack(side=tk.LEFT)

        # Help button to show usage instructions
        self.help_button = ttk.Button(button_frame, text="Help", command=self.show_help)
        self.help_button.pack(side=tk.RIGHT)

    # Loading the chatbot model, words, classes, and intents
    def load_chatbot_data(self):
        self.lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
        self.intents = json.loads(
            open(
                r"F:\shivani\VSCode\Chatbot\Data\intents.json"
            ).read()
        )  # Load intents data from JSON file
        self.words = pickle.load(open("words.pkl", "rb"))  # Load words for bag of words model
        self.classes = pickle.load(open("classes.pkl", "rb"))  # Load classes for intent prediction
        self.model = load_model("chatbot_model.h5")  # Load the trained chatbot model

    # Handling user message input
    def send_message(self):
        user_message = self.user_input.get()  # Get user's input message
        self.user_input.delete(0, tk.END)  # Clear input field
        if user_message:
            self.update_chat_history(f"You: {user_message}")  # Display user message
            bot_response = self.get_bot_response(user_message)  # Get bot's response
            self.update_chat_history(f"Bot: {bot_response}")  # Display bot response
            self.conversation_history.append((user_message, bot_response))  # Store conversation history

    # Updating the chat history display
    def update_chat_history(self, message):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + "\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    # Getting bot's response based on user message
    def get_bot_response(self, user_message):
        # Special commands for exit, search, and time
        if user_message.lower() in ["exit", "quit", "bye"]:
            return "Goodbye! Have a great day!"
        elif user_message.lower().startswith("search "):
            query = user_message[7:]
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"I've opened a web search for '{query}'."
        elif user_message.lower() == "time":
            return (
                f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
            )
        else:
            ints = self.predict_class(user_message)  # Predict the class of the user message
            return self.get_response(ints)  # Get the appropriate response

    # Preprocess the sentence by tokenizing and lemmatizing
    def clean_up_sentence(self, sentence):
        return [
            self.lemmatizer.lemmatize(word.lower())
            for word in nltk.word_tokenize(sentence)
        ]

    # Creating a bag of words for the user input
    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)

    # Predicting the intent of the user message
    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [
            {"intent": self.classes[r[0]], "probability": str(r[1])} for r in results
        ]

    # Getting a response based on the predicted intent
    def get_response(self, intents_list):
        if not intents_list:
            return "I'm not sure how to respond to that. Can you please rephrase your question?"
        tag = intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm sorry, I don't have a specific response for that. Can you try asking something else?"

    # Clear chat history and reset conversation
    def clear_chat(self):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state=tk.DISABLED)
        self.conversation_history.clear()

    # Save chat history to a text file
    def save_chat(self):
        filename = (
            f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(filename, "w") as f:
            for user_msg, bot_msg in self.conversation_history:
                f.write(f"You: {user_msg}\n")
                f.write(f"Bot: {bot_msg}\n\n")
        messagebox.showinfo("Chat Saved", f"Chat history has been saved to {filename}")

    # Display help information
    def show_help(self):
        help_text = """
        Welcome to the Enhanced NLP Chatbot!
        """
        messagebox.showinfo("Chatbot Help", help_text)

if __name__ == "__main__":
    root = tk.Tk()
    ChatbotGUI(root)
    root.mainloop()  # Start the main event loop

