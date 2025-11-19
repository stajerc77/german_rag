import logging
from datetime import datetime


# configure logging
logging.basicConfig(
    filename="chat_logger.log",
    level=logging.INFO,
    format="%(message)s",             # formatting is handled manually
    encoding="utf-8"
)


def log_llm(question: str, answer: str):
    """Log the Q&A output with timestamp

    Args:
        question (str): user query
        answer (str): LLM output
    """
    timestamp = datetime.now().strftime("%d/%m/%Y[%H:%M:%S]")
    log_entry = f"{timestamp} | user query: {question} | output: {answer}"
    logging.info(log_entry)
