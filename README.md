# 🏥 Rule-Based Chatbot for Hospital Management

A simple AI chatbot built using **Python**, **NLTK**, and **Keras** for managing basic queries in a hospital setting. It uses predefined **intents** and a trained deep learning model to classify user queries and respond accordingly.

---

## ✅ Features

* 📁 Rule-based **intents classification**
* 🧠 Trained using a **deep learning model** (Sequential/Keras)
* 💬 Handles common text queries (e.g., greetings, time/date, basic tasks)
* 🧾 Modular design for easy extension of hospital-specific commands

---

## 🛠 Technologies Used

* Python
* NLTK
* TensorFlow / Keras
* NumPy
* JSON
---

## 📁 Project Structure

```
📦 AI-Virtual-Assistent
├── bot.py               # Main logic for chatbot
├── train.py             # Model training script
├── intents.json         # Intents (patterns + responses)
├── words.pkl            # Vocabulary (generated after training)
├── classes.pkl          # Intent classes (generated after training)
├── chatbot.h5           # Trained model
├── requirements.txt     # List of dependencies
├── README.md            # Project documentation
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This will generate:

* `words.pkl`
* `classes.pkl`
* `chatbot.h5`

### 3. Run the chatbot

```bash
python bot.py
```

Type your questions into the terminal, and the bot will respond based on trained intents.

---

## 🧠 Example Intents (from `intents.json`)

```json
{
  "tags": "greetings",
  "patterns": ["hello", "hi", "hey"],
  "responses": ["Hi there!", "Hello!", "How can I help you?"]
}
```

You can customize this file to add hospital-related intents like:

* Appointment booking
* Visiting hours
* Doctor availability
* Billing help

---

## 🧪 Use Case

This chatbot can be embedded into a hospital’s internal system to:

* Answer FAQs
* Reduce front desk workload
* Provide quick information access to patients and staff

---

## 👨‍🎓 Project Info

* **Project Title:** AI Virtual Assistant for Hospital Management
* **Developed by:** *Prince Joshi*
* **Course:** B.Tech (CSE) Final Year Project
* **Year:** 2025
