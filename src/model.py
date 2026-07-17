"""Configuration des modèles LLM et helpers d'appel.

Ce module est la couche d'accès aux modèles. Il charge l'environnement,
configure les clients Ollama utilisés par le pipeline, active le tracing
Langfuse/OpenInference, puis expose deux fonctions :

- `query(...)` pour obtenir une réponse texte brute ;
- `query_json(...)` pour obtenir un dictionnaire Python à partir d'une réponse
  JSON produite par le modèle.
"""

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from langfuse import get_client,propagate_attributes
from dotenv import load_dotenv
from datetime import datetime
from typing import Any
import os
import json

# Charge les variables `.env` avant d'instancier les clients Ollama.
load_dotenv()

# Instrumente LlamaIndex pour que les appels LLM soient visibles dans Langfuse.
LlamaIndexInstrumentor().instrument()
langfuse = get_client()


# Modèle principal : utilisé pour les étapes structurantes du pipeline
# (questions normalisées, choix de fichiers, plan, résumés, associations).
first_model = Ollama(
    model=str(os.getenv("OLLAMA_MODEL")),
    base_url=str(os.getenv("OLLAMA_BASE_URL")),
    temperature=0.0,
    context_window=64000,
    json_mode=True,
    request_timeout=60.0,
    headers={
        "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
    }
    
    #additional_kwargs={
    #    "stop" : ["```"]
    #}
)

# Modèle secondaire
fast_model=Ollama(
    model=str(os.getenv("REVIEW_MODEL")),
    base_url="https://ollama.com",
    temperature=0.1,
    context_window=64000,
    json_mode=True,
    request_timeout=60.0,
    headers={
        "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
    }
)


def clean_json_response(content: str) -> str:
    """Nettoie une réponse LLM avant `json.loads`.

    Les modèles renvoient parfois du JSON entouré de fences Markdown
    (```json, ```mermaid, ou simple ```). Cette fonction retire ces wrappers et
    isole le premier objet JSON trouvé entre la première `{` et la dernière `}`.
    """
    content = content.strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()
    elif content.startswith("```"):
        # Certains modèles utilisent une fence générique même quand on demande
        # explicitement un JSON pur.
        content = content.removeprefix("```").strip()
    elif content.startswith("```mermaid"):
        content = content.removeprefix("```mermaid").strip()
    if content.endswith("```"):
        content = content.removesuffix("```").strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and start < end:
        # Si le modèle ajoute du texte avant/après le JSON, on garde uniquement
        # l'objet JSON principal.
        content = content[start:end + 1].strip()
    return content


def query(msg : str,llm : Ollama,workflow_run_id : str,tag : str,json : bool = False)-> str :
    """Exécute un appel chat et le trace dans Langfuse.

    `workflow_run_id` sert de session logique pour regrouper les étapes d'une
    génération documentaire. `tag` nomme l'étape courante : planning, writing,
    resume, associating, etc.
    """
    metadata={
        "workflow_run_id" : workflow_run_id,
        "name/tag" : tag,
    }
    
    with langfuse.start_as_current_observation(
        name=tag,
        as_type="span",
        input=msg,
        metadata=metadata
    ) as observation :
        
        # `propagate_attributes` attache les métadonnées au trace LlamaIndex
        # déclenché par `llm.chat(...)`.
        with propagate_attributes(
            session_id=workflow_run_id,
            trace_name= f"{datetime.now().hour}:{datetime.now().minute}-{tag}-{workflow_run_id[:4]}",
            metadata=metadata
        ) :
            if json :
                response = llm.chat([ChatMessage(content = msg)],format="json")
            else :
                response = llm.chat([ChatMessage(content = msg)])
        
        # On stocke la réponse complète dans Langfuse pour pouvoir diagnostiquer
        # les erreurs de parsing ou les hallucinations de format.
        observation.update(output=str(response))
    langfuse.flush()
    return str(response.message.content)


def query_json(msg : str,llm,workflow_run_id,tag)-> dict[Any,Any]:
    """Appelle un modèle jusqu'à obtenir une réponse JSON valide.

    Cette fonction est utilisée pour toutes les étapes où le reste du pipeline
    attend une structure exploitable. En cas de JSON invalide, elle relance le
    même prompt en changeant le tag de trace pour signaler un nouvel essai.
    """
    edit_tag = tag
    while True :
        try :
            response = query(msg=msg,llm=llm,workflow_run_id=workflow_run_id,tag=edit_tag,json=True)
            # Le nettoyage est volontairement séparé du parsing pour garder
            # visible la frontière entre sortie LLM et JSON Python.
            response = clean_json_response(response)
            response = json.loads(response)
            break
        except json.JSONDecodeError as e :
            print("\n\nLogging Error :" + e.msg + "\n\n")
            # Le prompt n'est pas modifié ici : on compte sur le json_mode du
            # modèle et sur le retry pour corriger les sorties accidentelles.
            # Pas de limite de retry pour l'instant : le pipeline privilégie
            # l'obtention d'un JSON valide, même si cela peut bloquer en cas de
            # modèle durablement non conforme.
            edit_tag="2nd:"+ tag
            continue

    return response
