"""L1 - Models, prompts and output parsers, recoded from L1-Model_prompt_parser.ipynb.

Same outline as the original:
- A direct API call (raw `ollama` client here, raw `openai` there)
- The same thing through LangChain: prompt template -> model
- Output parsers

`ResponseSchema` + `StructuredOutputParser` (the original's output-parsing
tool) were removed from LangChain along with the rest of the pre-1.0 parser
zoo. The modern replacement, used everywhere in tools_and_agent too, is
`model.with_structured_output(PydanticModel)`.
"""

import os
import sys

from common import get_model, traced
from langchain_core.prompts import ChatPromptTemplate
from ollama import Client
from pydantic import BaseModel, Field
from tracing import langfuse

sys.stdout.reconfigure(encoding="utf-8")

# --- Direct API call (no LangChain) ---

client = Client(
    host=os.environ["OLLAMA_BASE_URL"],
    headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
)
MODEL = os.environ["OLLAMA_MODEL"]


def get_completion(prompt: str, name: str) -> str:
    with langfuse.start_as_current_observation(
        name=name, as_type="generation", model=MODEL, input=prompt
    ) as generation:
        response = client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        generation.update(
            output=response.message.content,
            usage_details={"input": response.prompt_eval_count, "output": response.eval_count},
        )
        return response.message.content


print("--- direct API call ---")
print(get_completion("What is 1+1?", "L1: direct API call - 1+1"))

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks
into a style that is {style}.
text: ```{customer_email}```
"""

print("\n--- direct API call: pirate -> calm American English ---")
print(get_completion(prompt, "L1: direct API call - style transfer"))


# --- The same thing through LangChain ---

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
prompt_template = ChatPromptTemplate.from_template(template_string)
print("\ninput_variables:", prompt_template.input_variables)

model = get_model()

customer_style = "American English in a calm and respectful tone"
customer_messages = prompt_template.format_messages(style=customer_style, text=customer_email)
print("\nformatted message:", customer_messages[0])

customer_response = model.invoke(customer_messages, config=traced("L1: LangChain style transfer (customer)"))
print("\n--- LangChain: pirate -> calm American English ---")
print(customer_response.content)

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""
service_style_pirate = "a polite tone that speaks in English Pirate"

service_messages = prompt_template.format_messages(style=service_style_pirate, text=service_reply)
service_response = model.invoke(service_messages, config=traced("L1: LangChain style transfer (service)"))
print("\n--- LangChain: blunt -> polite pirate ---")
print(service_response.content)


# --- Output parsers ---

customer_review = """\
This leaf blower is pretty amazing. It has four settings: \
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price, \
and output them as a list of strings.

text: {text}
"""

plain_prompt = ChatPromptTemplate.from_template(review_template)
plain_messages = plain_prompt.format_messages(text=customer_review)
plain_response = model.invoke(plain_messages, config=traced("L1: unstructured extraction"))

print("\n--- unparsed response (plain string, .get() would fail) ---")
print(plain_response.content)
print("type:", type(plain_response.content))


class ReviewInfo(BaseModel):
    """Information extracted from a product review."""

    gift: bool = Field(description="Was the item purchased as a gift for someone else?")
    delivery_days: int = Field(description="How many days it took to arrive. -1 if not found.")
    price_value: list[str] = Field(description="Sentences about the value or price of the item.")


structured_model = model.with_structured_output(ReviewInfo, method="function_calling")
structured_chain = plain_prompt | structured_model

result = structured_chain.invoke({"text": customer_review}, config=traced("L1: structured extraction"))
print("\n--- structured extraction ---")
print(result)
print("type:", type(result))
print("delivery_days:", result.delivery_days)
