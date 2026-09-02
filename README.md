# 🧠 NextGen AI — LSTM Next Word Predictor

An AI-powered text generation application that uses a **Long Short-Term Memory (LSTM)** neural network to predict and generate the next words based on the text provided by the user.

The project combines **Deep Learning, Natural Language Processing (NLP), TensorFlow/Keras, and Streamlit** to create an interactive and modern AI writing experience.

## ✨ Features

* 🧠 LSTM-based language model
* ✍️ Intelligent text continuation
* 🔢 Generate multiple words
* ⚡ Fast text generation
* 🎨 Premium dark-themed Streamlit interface
* 📱 Responsive user interface
* 🔤 Tokenization and sequence processing
* 🤖 Deep Learning-based prediction
* 🎯 Adjustable number of generated words

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Streamlit
* Pickle
* NLP
* LSTM Neural Network

## 📂 Project Structure

```text
next-word-predictor-lstm/
│
├── app.py
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ How It Works

The application follows this process:

```text
User Input
    ↓
Text Tokenization
    ↓
Sequence Preparation
    ↓
Padding
    ↓
LSTM Model
    ↓
Next Word Prediction
    ↓
Generated Word
    ↓
Updated Context
    ↓
Next Word Prediction
```

The process is repeated according to the number of words selected by the user.

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/next-word-predictor-lstm.git
```

### 2. Open the project

```bash
cd next-word-predictor-lstm
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Model Files

The application uses three trained resources:

| File            | Purpose                                 |
| --------------- | --------------------------------------- |
| `lstm_model.h5` | Trained LSTM neural network             |
| `tokenizer.pkl` | Converts words into numerical sequences |
| `max_len.pkl`   | Stores the maximum sequence length      |

## 🧪 Example

### Input

```text
Artificial intelligence is changing the
```

### Generated Output

```text
Artificial intelligence is changing the way we work and communicate with technology
```

The exact output depends on the training data and trained model.

## 🎯 Project Objective

The objective of this project is to demonstrate how deep learning can be used for **Natural Language Processing and text generation**.

The LSTM model learns patterns and relationships between words from training data and uses the learned patterns to predict possible future words.

## 🧠 Why LSTM?

LSTM networks are a type of Recurrent Neural Network (RNN) designed to handle sequential data.

For text prediction, LSTM can learn relationships between words across sequences and maintain useful information from previous inputs.

## 🔮 Future Improvements

Possible improvements include:

* Transformer-based text generation
* GPT-style architecture
* Beam search
* Temperature-based sampling
* Top-K sampling
* Top-P sampling
* Larger training datasets
* Multi-language text prediction
* Better contextual understanding
* Cloud deployment
* REST API integration

## 👨‍💻 Developer

**Ali Muhammad**

Designed & Developed as a Deep Learning and NLP project.

## 📄 License

This project is available for educational and personal use.
