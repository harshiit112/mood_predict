# Emotion Engine 🧠✨ — Read the Room

Emotion Engine is a full-stack, deep learning-powered Natural Language Processing (NLP) web application that detects and analyzes human emotions from text in real time. The project combines an advanced **Bidirectional GRU (BiGRU)** neural network built with TensorFlow/Keras with a high-performance **FastAPI** backend and a premium, mood-adaptive frontend interface.

---

## 🚀 Key Features

* **Advanced Deep Learning Backend:** Powered by a Bidirectional GRU model trained on the Hugging Face `dair-ai/emotion` dataset, achieving **~88.3% test accuracy** across six primary emotions: *sadness* 😢, *joy* 😄, *love* ❤️, *anger* 😠, *fear* 😨, and *surprise* 😲.
* **High-Performance Web API:** Built using FastAPI, featuring asynchronous server lifecycle management (model/tokenizer pre-loading) and automatic request data validation via Pydantic.
* **Mood-Adaptive UI:** A premium, dark-themed Single Page Application (SPA) with a faint instrument-panel grain and glassmorphic panels. The radial background gradients and accent colors dynamically shift in real-time depending on the predicted mood of the text.
* **Interactive Probability Dashboard:** Visualizes prediction confidence levels for all six emotions using animated progress bars.
* **Robust Preprocessing Pipeline:** Integrated text cleaning, regex filtering, tokenization, and post-padding to ensure accurate model inference.

---

## 🛠️ Tech Stack

* **Deep Learning & ML:** TensorFlow, Keras, NumPy, Pandas, Scikit-Learn, Pickle
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend UI:** Vanilla HTML5, Vanilla CSS3 (Custom Properties, Keyframes, Glassmorphism), Modern JavaScript (Fetch API, DOM)
* **Dataset:** Hugging Face `dair-ai/emotion`

---

## 📁 Repository Structure

```text
├── Artifacts/                  # Serialized model & preprocessing assets
│   ├── BiGRU_Model.keras       # Trained Bidirectional GRU model
│   └── tokenizer.pkl           # Saved Keras Tokenizer
├── static/                     # Frontend Single Page Application
│   └── index.html              # Sleek, mood-adaptive interface
├── main.py                     # FastAPI web server and API endpoints
├── nlp_model_new.ipynb         # Jupyter Notebook detailing model training
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📊 Model Training & Evaluation

The training process is fully detailed in [`nlp_model_new.ipynb`](./nlp_model_new.ipynb). During development, multiple recurrent neural network architectures were evaluated on the `emotion` dataset (16,000 training, 2,000 validation, and 2,000 test examples):

### Model Comparison & Results

| Model Architecture | Loss | Test Accuracy |
| :--- | :---: | :---: |
| Simple Recurrent Neural Network (RNN) | 1.8512 | 10.40% |
| Long Short-Term Memory (LSTM) | 1.8126 | 5.15% |
| Gated Recurrent Unit (GRU) | 1.7891 | 4.35% |
| **Advanced Bidirectional GRU (BiGRU)** | **0.3570** | **88.35%** |

*Note: The baseline RNN, LSTM, and GRU models were simple configurations; adding bidirectional tracking, larger hidden states, and dropout regularization in the BiGRU model drastically increased accuracy and generalization.*

### BiGRU Architecture Specs

```python
BiGRU = Sequential([
    Embedding(input_dim=max_words, output_dim=300, input_length=50),
    Bidirectional(GRU(units=128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(GRU(units=64)),
    Dropout(0.5),
    Dense(units=num_classes, activation='softmax')
])
```

---

## ⚙️ Installation & Setup

Follow these steps to run the FastAPI server and UI locally:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Create and Activate a Virtual Environment
**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```
**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Server
Launch the development server using Uvicorn:
```bash
uvicorn main:app --reload
```
The server will start running at `http://127.0.0.1:8000/`.

---

## 🔌 API Endpoints

Once the server is running, you can explore the API using the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### 1. Root / UI Endpoint
* **Route:** `GET /`
* **Description:** Serves the frontend single-page application (`static/index.html`).

### 2. Health Check
* **Route:** `GET /health`
* **Response Example:**
  ```json
  {
    "status": "Server is running",
    "model_loaded": true
  }
  ```

### 3. Prediction Endpoint
* **Route:** `POST /predict`
* **Request Schema:**
  ```json
  {
    "text": "I feel so happy and excited today!"
  }
  ```
* **Response Schema:**
  ```json
  {
    "text": "I feel so happy and excited today!",
    "predicted_emotion": "joy",
    "confidence": 0.9842,
    "all_probabilites": {
      "sadness": 0.0015,
      "joy": 0.9842,
      "love": 0.0083,
      "anger": 0.0021,
      "fear": 0.0011,
      "surprise": 0.0028
    }
  }
  ```

---

## 🎨 Frontend UI Aesthetics

The frontend interface features a state-of-the-art layout:
* **Glassmorphic Console:** Input text area is encased in a frosted-glass panel that blurs background elements.
* **Ambient Lighting:** Glow effects and radial gradients that automatically sync to the dominant emotion:
  * 😢 **Sadness:** Cool blue shades
  * 😄 **Joy:** Sunny yellow accents
  * ❤️ **Love:** Warm pink/rose hues
  * 😠 **Anger:** Deep crimson gradients
  * 😨 **Fear:** Dark purple tones
  * 😲 **Surprise:** Soft orange/amber glow
* **Micro-Animations:** Fluid loading indicators and smooth bar-chart expansions built with CSS transitions.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
