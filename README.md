Medical Demo Chatbot

This is a medical chatbot built using Python, LangChain, Pinecone, OpenAI, and Flask.

How to Run the Project

Step 1: Clone the repository
Open terminal and run:
git clone https://github.com/your-username/medical-demo-chatbot.git

cd medical-demo-chatbot

Step 2: Create a conda environment
conda create -n medibot python=3.10 -y
conda activate medibot

Step 3: Install required packages
pip install -r requirements.txt

Step 4: Create a .env file
Create a file named .env in the main folder and add your API keys:

PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key

Step 5: Store data in Pinecone
Run this command to upload embeddings:
python store_index.py

Step 6: Run the application
python app.py

Step 7: Open in browser
Go to this link:
http://127.0.0.1:8080

Tech Stack

Python
LangChain
Flask
OpenAI
Pinecone

Important Notes
Always activate your environment before running the project
Make sure .env file has correct API keys
Make sure templates/chat.html file exists
