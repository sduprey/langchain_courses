import os
import sys

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

from langfuse import get_client
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.llms.ollama import Ollama


from model import query, first_model, query_json   




response = query_json(f"""Donne moi la vie de lionel messi dans la forme json suivante :
                      {{
                        "Vie" : <vie>,
                        "matchs significatifs" : <matchs>
                      }}
                      Voila le json a remplir,commence :
                      """, llm=first_model,
                workflow_run_id="1",tag="playground")
response["Vie"]
print(response)
