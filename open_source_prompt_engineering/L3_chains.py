"""L3 - Chains, recoded from L3-chains.ipynb.

`LLMChain`, `SimpleSequentialChain`, `SequentialChain`, `MultiPromptChain`,
`LLMRouterChain` and `RouterOutputParser` were all removed - `langchain.chains`
doesn't exist anymore in this LangChain version. LCEL (`|`) is now the only
way to build chains:

- `LLMChain` -> `prompt | model | StrOutputParser()`.
- `SimpleSequentialChain` (single in, single out) -> pipe one chain's string
  output straight into the next chain's input dict.
- `SequentialChain` (multiple named inputs/outputs) -> a stack of
  `RunnablePassthrough.assign(key=chain)` calls, each adding one more key to
  a running dict - the same pattern tools_and_agent/L6 uses for
  `agent_scratchpad`.
- The router chain -> `model.with_structured_output(RouterDecision)` picks a
  destination name (LLM-as-classifier instead of a hand-parsed markdown JSON
  blob), then a plain Python dict lookup dispatches to the chosen chain -
  the same "read AIMessage / route" idea as tools_and_agent/L5.

`Data.csv` from the original isn't included in this repo (per the top-level
README, some notebooks' local data files didn't make the mirror) - the
reviews used here live in data/reviews.csv, a small synthetic stand-in with
the same shape (Product, Review columns, one review in French to exercise
the translate/detect-language chains).
"""

import pandas as pd
from common import get_model, traced
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

df = pd.read_csv("data/reviews.csv")
print(df.head())

model = get_model(temperature=0.9)


# --- LLMChain equivalent ---

prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
chain = prompt | model | StrOutputParser()

product = "Queen Size Sheet Set"
print("\n--- LLMChain equivalent ---")
print(chain.invoke({"product": product}, config=traced("L3: LLMChain equivalent")))


# --- SimpleSequentialChain equivalent: single input, single output, piped ---

first_prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
chain_one = first_prompt | model | StrOutputParser()

second_prompt = ChatPromptTemplate.from_template("Write a 20 word description for the following company: {company_name}")
chain_two = second_prompt | model | StrOutputParser()

overall_simple_chain = chain_one | (lambda company_name: {"company_name": company_name}) | chain_two

print("\n--- SimpleSequentialChain equivalent ---")
print(overall_simple_chain.invoke({"product": product}, config=traced("L3: SimpleSequentialChain equivalent")))


# --- SequentialChain equivalent: multiple named inputs/outputs ---

translate_prompt = ChatPromptTemplate.from_template("Translate the following review to English:\n\n{review}")
chain_translate = translate_prompt | model | StrOutputParser()

summarize_prompt = ChatPromptTemplate.from_template("Can you summarize the following review in 1 sentence:\n\n{English_Review}")
chain_summarize = summarize_prompt | model | StrOutputParser()

language_prompt = ChatPromptTemplate.from_template("What language is the following review:\n\n{review}")
chain_language = language_prompt | model | StrOutputParser()

followup_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
chain_followup = followup_prompt | model | StrOutputParser()

overall_chain = (
    RunnablePassthrough.assign(English_Review=chain_translate)
    | RunnablePassthrough.assign(summary=chain_summarize)
    | RunnablePassthrough.assign(language=chain_language)
    | RunnablePassthrough.assign(followup_message=chain_followup)
)

review = df.Review[5]
print("\n--- SequentialChain equivalent (French review) ---")
result = overall_chain.invoke({"review": review}, config=traced("L3: SequentialChain equivalent"))
for key in ("English_Review", "summary", "language", "followup_message"):
    print(f"\n{key}:\n{result[key]}")


# --- Router chain equivalent ---

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise \
and easy to understand manner. When you don't know the answer to a \
question you admit that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. You are so good because \
you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the \
broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people, events \
and contexts from a range of historical periods. You have the ability \
to think, reflect, debate, discuss and evaluate the past. You have a \
respect for historical evidence and the ability to make use of it to \
support your explanations and judgements.

Here is a question:
{input}"""

computerscience_template = """You are a successful computer scientist. \
You have a passion for creativity, collaboration, forward-thinking, \
confidence, strong problem-solving capabilities, understanding of \
theories and algorithms, and excellent communication skills. You are \
great at answering coding questions.

Here is a question:
{input}"""

prompt_infos = [
    {"name": "physics", "description": "Good for answering questions about physics", "template": physics_template},
    {"name": "math", "description": "Good for answering math questions", "template": math_template},
    {"name": "History", "description": "Good for answering history questions", "template": history_template},
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "template": computerscience_template,
    },
]

destination_chains = {
    info["name"]: ChatPromptTemplate.from_template(info["template"]) | model | StrOutputParser()
    for info in prompt_infos
}
default_chain = ChatPromptTemplate.from_template("{input}") | model | StrOutputParser()

destinations_str = "\n".join(f"{info['name']}: {info['description']}" for info in prompt_infos)


class RouterDecision(BaseModel):
    """Pick which specialist prompt should answer the question."""

    destination: str = Field(
        description=f"One of the candidate prompt names, or DEFAULT if none fit well:\n{destinations_str}"
    )
    next_input: str = Field(description="The question to send to that destination, possibly reworded for clarity")


router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a raw text input to a language model, select the model prompt best suited "
            "for the input. You may also revise the original input if you think that revising it "
            "will ultimately lead to a better response from the language model.\n\n"
            f"Candidate prompts:\n{destinations_str}",
        ),
        ("user", "{input}"),
    ]
)
router_chain = router_prompt | model.with_structured_output(RouterDecision, method="function_calling")


def route(decision: RouterDecision, config=None) -> str:
    chain = destination_chains.get(decision.destination, default_chain)
    return chain.invoke({"input": decision.next_input}, config=config)


print("\n--- router chain ---")
for question in ["What is black body radiation?", "what is 2 + 2", "Why does every cell in our body contain DNA?"]:
    decision = router_chain.invoke({"input": question}, config=traced(f"L3: router decision - {question[:30]}"))
    print(f"\n> {question}")
    print(f"routed to: {decision.destination!r}")
    print(route(decision, config=traced(f"L3: router answer - {question[:30]}")))
