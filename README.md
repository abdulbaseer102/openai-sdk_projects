# **README: OpenAI SDK Projects**

## **📌 Overview**

This repository contains four AI-powered projects built using the **OpenAI SDK**. Each project leverages **LLMs (Large Language Models)** to solve real-world problems efficiently. The projects included are:

1️⃣ **Personal Study Assistant** – An AI-powered tutor for personalized learning.\
2️⃣ **AI-Powered Travel Planner** – A smart travel assistant that helps plan trips.\
3️⃣ **Customer Support Automation System** – An AI chatbot for automated customer support.\
4️⃣ **Amazon-Walmart Product Scraper** – An AI-powered price comparison and profit analysis tool.

---

## **🛠️ Technologies Used**

- **OpenAI SDK** (GPT models for NLP & AI processing)
- **Python** (Backend logic)
- **FastAPI / Flask** (For API-based interaction)
- **BeautifulSoup/Scrapy** (For web scraping in the Product Scraper project)
- **Vector Databases** (Pinecone, ChromaDB for memory in chatbots)
- **Deployment Platforms** (AWS, GCP, Vercel)

---

## **📌 Project 1: Personal Study Assistant**

### **📖 Overview**

This AI-based study assistant helps students with explanations, summarization, quiz generation, and personalized learning plans.

### **🚀 Features**

✔ AI-powered **question answering**\
✔ **Summarizes textbooks, PDFs, and articles**\
✔ Generates **quizzes and flashcards**\
✔ Adaptive **learning plan based on user progress**

### **⚙️ Setup Instructions**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/personal-study-assistant.git
   ```

 cd personal-study-assistant

````
2. Install dependencies:  
```bash
pip install openai flask chromadb
````

3. Set up API keys in `.env`:
   ```
   OPENAI_API_KEY
   ```

OPENAI\_API\_KEY=your\_api\_key\_here

````
4. Run the application:  
```bash
python main.py
````

---

## **📌 Project 2: AI-Powered Travel Planner**

### **🌍 Overview**

This AI agent helps users **plan trips** by suggesting itineraries, accommodations, and budgeting based on preferences.

### **🚀 Features**

✔ **Generates travel itineraries** based on user input\
✔ **Recommends hotels, flights, and restaurants**\
✔ **Provides budget estimates**\
✔ **Real-time weather updates & location info**

### **⚙️ Setup Instructions**

1. Clone the repository:
   ```bash


   git clone https://github.com/yourusername/ai-travel-planner.git
   ```

git clone [https://github.com/yourusername/ai-travel-planner.git](https://github.com/yourusername/ai-travel-planner.git)
cd ai-travel-planner

````
2. Install dependencies:  
```bash
pip install openai fastapi requests
````

3. Set up API keys:
   ```
   ```

OPENAI\_API\_KEY=your\_api\_key\_here

````
4. Run the server:  
```bash
uvicorn main:app --reload
````

---

## **📌 Project 3: Customer Support Automation System**

### **💬 Overview**

A chatbot that automates customer support by answering FAQs and resolving user queries using AI.

### **🚀 Features**

✔ **AI-powered customer query handling**\
✔ **Retrieves information from knowledge base**\
✔ **Multi-language support**\
✔ **Integration with CRM tools**

### **⚙️ Setup Instructions**

1. Clone the repository:
   ```bash
   ```

git clone [https://github.com/yourusername/customer-support-ai.git](https://github.com/yourusername/customer-support-ai.git)
cd customer-support-ai

````
2. Install dependencies:  
```bash
pip install openai flask langchain chromadb
````

3. Set up API keys:
   ```
   ```

OPENAI\_API\_KEY=your\_api\_key\_here

````
4. Run the chatbot:  
```bash
python chatbot.py
````

---

## **📌 Project 4: Amazon-Walmart Product Scraper**

### **🛒 Overview**

This AI agent **scrapes product prices from Amazon**, checks availability on **Walmart**, and calculates **profit margins** for dropshipping.

### **🚀 Features**

✔ **Scrapes product details from Amazon**\
✔ **Checks stock & price on Walmart**\
✔ **Calculates potential profit margins**\
✔ **Exports results to CSV/JSON**

### **⚙️ Setup Instructions**

1. Clone the repository:
   ```bash
   ```

git clone [https://github.com/yourusername/amazon-walmart-scraper.git](https://github.com/yourusername/amazon-walmart-scraper.git)
cd amazon-walmart-scraper

````
2. Install dependencies:  
```bash
pip install openai beautifulsoup4 requests pandas
````

3. Set up API keys:
   ```
   ```

OPENAI\_API\_KEY=your\_api\_key\_here

````
4. Run the scraper:  
```bash
python scraper.py
````

---

## **🚀 Contributing**

If you want to contribute to these projects, feel free to fork the repository and submit a pull request!

---

## **📩 Contact**

For any queries, feel free to reach out:\
📧 **Email:** [baseek8@gmail.com](mailto\:baseek8@gmail.com)\
🔗 **GitHub:** [Your GitHub Profile](https://github.com/yourusername)

Let’s build intelligent AI-powered solutions together! 🚀

