# LLM Evaluation Project 🚀

## 🌟 Overview
In the ever-evolving landscape of AI, evaluating Large Language Models (LLMs) has become a critical task to ensure their effectiveness and reliability. This project explores a novel approach to LLM evaluation using the cutting-edge **Arize Phoenix Platform**, combined with Python and Groq technologies. The goal of this project is to assess LLM outputs for key factors like hallucinations, exact matching, and semantic similarity, providing actionable insights into model performance.

---

## 🎥 Demo Walkthrough
👉 **[Watch the video walkthrough](https://youtu.be/J7Tv_zZ_bpQ)**

👉 **[Try the app](https://dhruv-mojila-llm-evaluation.streamlit.app/)**

---

## 🛠️ Technologies Used
- **Arize Phoenix Platform**
- **Python**
- **Groq**
- **Streamlit**
- **Streamlit Cloud**

---

## 📖 Project Outline

![outline](https://github.com/user-attachments/assets/67970fe5-d53f-4ede-b265-ab15de4416a1)

### 1. Data Generation and Annotation
- Utilized **multiple models** to create traces (data).
- Leveraged an **LLM as a judge** to annotate the generated data, creating a **Golden Dataset** or **Ground Truth** (alternatively, human annotators can also be used).

### 2. Dataset Integration
- Selected model outputs and added them to datasets hosted on the **Phoenix Platform**.

### 3. Evaluation and Comparison
- Compared outputs from various models against the **Ground Truth** to assess:
  - **Hallucinations**
  - **Exact Matching**
  - **Semantic Similarity** (e.g., cosine similarity).

### 4. Experimentation
- Conducted experiments to compute the **cosine similarity** between the **Ground Truth** and the content generated by different models.

---

## 🤔 Challenges Faced
- **Lack of Resources**: Limited guidance and documentation on evaluating LLMs.
- **Streamlit State Management**: Managing state after multiple interactions was challenging.

Despite these obstacles, extensive research and experimentation led to meaningful outcomes.

---

## 💡 Lessons Learned
- The **Arize Phoenix Platform** stands out as a robust tool for LLM Evaluation.
- Effective state management is critical in interactive web applications.

---

## 🌟 Features
- **Model Traces Creation**: Generate and annotate data using LLMs.
- **Golden Dataset Generation**: Create a reliable Ground Truth for evaluations.
- **Comparative Analysis**: Evaluate multiple models for hallucinations, exact matches, and semantic similarity.
- **Cosine Similarity Experiments**: Quantitative assessment of model outputs.

---

## 🚀 Quick Start

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/llm-evaluation
   ```

2. Navigate to the project directory:
   ```bash
   cd llm-evaluation
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app locally:
   ```bash
   streamlit run app.py
   ```
---

## 📈 Future Scope
- Explore alternative platforms for better state management.
- Expand evaluation metrics to include:
  - Sentiment Analysis
  - Response Coherence
- Incorporate user feedback mechanisms for continuous improvement.

---

## 📖 Resources
- https://docs.arize.com/phoenix
- https://app.phoenix.arize.com/
- https://arize.com/blog-course/llm-evaluation-the-definitive-guide/

---

## 📬 Contact
Feel free to reach out with feedback or collaboration ideas:
- **Email**: [dhruvmojila098@gmail.com](mailto:dhruvmojila098@gmail.com)
- **LinkedIn**: [LinkedIn](https://www.linkedin.com/in/dhruv-mojila/)

---

## 🎉 Final Thoughts
This project underscores the importance of reliable LLM evaluation in the AI ecosystem. While there were challenges, the learning experience was invaluable. Looking forward to feedback and collaboration opportunities!

