FROM python:3.12

WORKDIR /app

COPY rag_agent_api.py .

COPY speech_generator.py .

COPY llm_logger.py . 

COPY requirements.txt .

COPY data ./data/

COPY voices ./voices/

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "rag_agent_api.py", "--server.address=0.0.0.0", "--server.port=8501"]
