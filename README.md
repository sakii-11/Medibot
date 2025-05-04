# Medibot
It is an AI powered chatbot that provides quick, informative and conversational responses to common medical queries.Built using Streamlit and Mistral models, it aims to offer accessible healthcare related information.

---

## Features

- Conversational AI powered by open weight LLM
- Integrates Mistral language model for response generation 
- Fast and leightweight Streamlit UI
---

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python, Mistral

---

## Getting Started 

### 1. Clone the repository

```bash
git clone https://github.com/sakii-11/Medibot.git
cd Medibot
```

### 2. Install dependencies 
Python version - 3.9+
```bash
pip install -r requirements.txt
```

### 3. Environment Variables 
Create a `.env` file in the root directory and add the following:

```env
MODEL_API_URL=<your_llm_api_endpoints>
MODEL_NAME=<model_name>
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---
## RAG -> Retrieval Augumented Generation
![alt text](image.png)

### Definition

Retrieval-Augmented Generation (RAG) is an AI technique that combines information retrieval with text generation to enhance the accuracy and relevance of responses. Instead of relying solely on a model’s internal knowledge, RAG first retrieves relevant documents from an external knowledge source (e.g., a vector database) and then generates a response based on both the retrieved content and the model’s reasoning ability.


### Steps to build RAG:-
1. Query Translation (Translate the ques into a form that is better suited for query retrieval)
2. Routing (Logical / Semantic)
3. Query Construction
(main steps :)
4. Indexing
5. Retrieval
6. Generation (Active Retrieval) 

---


## ScreenShot


---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request

---

## Author

- [@sakii.codes](https://github.com/sakii-11)

---

## License

This project is licensed under the [MIT License](LICENSE).