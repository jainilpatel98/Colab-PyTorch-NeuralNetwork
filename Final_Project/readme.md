# 🏈 Scout Report Generator

This project generates AI-powered scouting reports for NFL players by combining structured player data, web-scraped strengths/weaknesses, and community insights from Reddit. The system uses a hybrid pipeline of deep learning models (CNN + Transformer), vector databases and retrieval-augmented generation (RAG) to create detailed and personalized scouting reports.

---

## 📦 Project Structure & Pipeline Overview

### 1. **Download Official NFL Data**
- 📥 **Source**: [NFL Big Data Bowl on Kaggle](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025)
- Download data from all available years and store them in the designated raw data folder.

---

### 2. **Scrape Scouting Reports from Web**
- 📂 Navigate to `Scout_report_crawler/`
- 🚀 Run `crawler.ipynb` to scrape player strengths and weaknesses using **Selenium WebDriver**
- 🔧 Ensure `chromedriver` or the correct WebDriver is installed

---

### 3. **Fetch and Embed Reddit Insights**
- 📂 Go to `reddit_data_fetcher/`
- 🧠 Run `reddit_finetunemodel.ipynb`:
  - Authenticates with Reddit API
  - Fetches relevant posts
  - Generates sentence embeddings
  - Stores vectors in **Qdrant** (a fast vector DB)

✅ Make sure to set:
- `REDDIT_API_KEY`
- `QDRANT_API_KEY`

---

### 4. **Store Web Crawled Data in Qdrant**
- 📂 In `vector_database/`, run:
  - `send_strengths_weakness.ipynb`
- This adds scraped scouting data into Qdrant for retrieval

---

### 5. **Process Big Data Bowl Files**
- 📂 Run `1_big_data_bowl_zip_extractor.ipynb`
  - Extracts and organizes raw ZIP files

---

### 6. **Preprocess NFL Data**
- 📂 Execute `3_NFL_preprocess.ipynb`
  - Converts raw plays into machine-readable features
  - Generates:
    - `train_data`, `train_labels`
    - `test_data`, `test_labels`
  - Adds custom metrics like **play success/failure**

---

### 7. **Train CNN + Transformer Models**
- 📂 Run `4_NFL_cnn_transformer.ipynb`
  - Uses CNN for spatial features and Transformer for sequence modeling
  - Supports:
    - Binary and multi-class targets
    - 8+ variations of models with dynamic headers
- 💻 Use `model_generation/run_and_save_results.py` for distributed training
  - Models can be trained on separate machines and results saved automatically

---

### 8. **Evaluate Model Performance**
- 📂 Run `NFL_visualizer_and_evaluat Visualize performance using:
    - Confusion Matrix
    - Accuracy, Precision, Recall, F1, etc.

---

### 9. **Live Vector Database for RAG**
- Transform model outputs into sentences
- Continuously update the **Qdrant vector store**
- Enables real-time query-based retrieval for scouting insights

---

### 10. **Generate Scouting Reports**
- 📂 In `scouting_report_generator/`, run:
  - `6_generate_template_questions.ipynb`
- 🔁 Uses multiprocessing to:
  - Generate prompts from predefined templates
  - Query Gemini (or any LLM API) to create player-specific PDFs

---

### 11. **Launch AI-Powered Scouting Agent**
- 🎯 Run `ai_scout.py` using **Streamlit**:
  ```bash
  streamlit run ai_scout.py
