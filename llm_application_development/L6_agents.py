"""L6 - Agents, recoded from L6-Agents.ipynb.

`load_tools`, `initialize_agent`, `AgentType`, `create_python_agent` and
`PythonREPLTool` were all removed - `langchain.agents` in this version only
exposes `create_agent` (the LangGraph-based agent already used in
tools_and_agent/L6). Same substitution here:

- The built-in `"llm-math"` tool (an `LLMMathChain` that asks the model to
  write a Python expression, then evaluates it with `numexpr`) is replaced
  by `calculator`, a small `ast`-based safe arithmetic evaluator - no nested
  LLM call needed, since a tool-calling model can already turn "25% of 300"
  into the expression `300 * 0.25` itself when it fills in the tool's
  arguments.
- The built-in `"wikipedia"` tool is the same hand-rolled `search_wikipedia`
  used in tools_and_agent (User-Agent fix included - see `configure_wikipedia`).
- `PythonREPLTool` is reimplemented directly: `exec()` with a persistent
  globals dict, stdout captured and returned. This has **no sandboxing**,
  exactly like the original tool didn't - fine for this local, personal
  exercise; never wire this up to untrusted input.
- The custom `time` tool is unchanged from the original.
"""

import ast
import contextlib
import io
import operator
from datetime import date

import requests
import wikipedia
from common import configure_wikipedia, get_model, traced
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

configure_wikipedia()

model = get_model()


# --- Built-in LangChain tools, reimplemented ---

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. "300 * 0.25". Numbers and + - * / ** % only, no variables."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"Error evaluating {expression!r}: {e}"


@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    try:
        page_titles = wikipedia.search(query)
    except (requests.exceptions.RequestException, ValueError):
        return "No good Wikipedia Search Result was found"

    summaries = []
    for page_title in page_titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
            requests.exceptions.RequestException,
            ValueError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


tools = [calculator, search_wikipedia]
agent = create_agent(model=model, tools=tools, system_prompt="You are a helpful assistant.")


def ask(question: str, run_name: str, agent_to_use=None) -> str:
    result = (agent_to_use or agent).invoke(
        {"messages": [HumanMessage(content=question)]}, config=traced(run_name)
    )
    return result["messages"][-1].content


print("--- built-in tools: math ---")
print(ask("What is the 25% of 300?", "L6: math tool"))

print("\n--- built-in tools: wikipedia ---")
question = (
    "Tom M. Mitchell is an American computer scientist and the Founders University "
    "Professor at Carnegie Mellon University (CMU) - what book did he write?"
)
print(ask(question, "L6: wikipedia tool"))


# --- Python agent ---

_repl_globals: dict = {}


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return what it prints to stdout. No sandboxing - trusted input only."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, _repl_globals)
    except Exception as e:
        return f"Error: {e}"
    return buffer.getvalue() or "(no output - the code needs to print() its result)"


python_agent = create_agent(
    model=model,
    tools=[python_repl],
    system_prompt="You are an agent that writes and executes Python code to answer questions. Always print() the final result.",
)

customer_list = [
    ["Harrison", "Chase"],
    ["Lang", "Chain"],
    ["Dolly", "Too"],
    ["Elle", "Elem"],
    ["Geoff", "Fusion"],
    ["Trance", "Former"],
    ["Jen", "Ayai"],
]

# Note (same caveat as the original notebook): agents sometimes reason about
# the task incorrectly - e.g. writing `key=lambda x: (x[0], x[1])` while
# believing index 0 is the last name when `customer_list` actually stores
# [first, last]. That's a model mistake, not a tool bug: the executed code
# ran exactly as written. Langfuse's trace for this call shows the generated
# code if you want to see what went wrong.
print("\n--- python agent: sort customers by last name, then first name ---")
print(
    ask(
        f"Sort these customers by last name and then first name, and print the output: {customer_list}",
        "L6: python agent",
        agent_to_use=python_agent,
    )
)


# --- Define your own tool ---


@tool
def time(text: str) -> str:
    """Returns todays date, use this for any questions related to knowing todays date.
    The input should always be an empty string, and this function will always return
    todays date - any date mathmatics should occur outside this function."""
    return str(date.today())


agent_with_time = create_agent(model=model, tools=tools + [time], system_prompt="You are a helpful assistant.")

print("\n--- custom tool: what's the date today? ---")
try:
    print(ask("whats the date today?", "L6: custom time tool", agent_to_use=agent_with_time))
except Exception:
    print("exception on external access")
