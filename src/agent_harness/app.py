from .cases import CaseAtlas
from .context import ContextManager
from .external import WikipediaAdapter
from .llm import ResponsesClient
from .runtime import AgentRuntime
from .store import SQLiteStore
from .tooling import ToolRegistry
from .tools.analogy_board import AnalogyBoardTool
from .tools.calculator import CalculatorTool
from .tools.read_case import ReadCaseTool
from .tools.search import SearchTool
from .tracing import TraceLogger


def build_runtime(db_path: str = "agent.db", trace_path: str = "trace.jsonl") -> AgentRuntime:
    llm = ResponsesClient()  # validate API config before creating durable files
    store = SQLiteStore(db_path)
    atlas = CaseAtlas()
    wiki = WikipediaAdapter()
    registry = ToolRegistry()
    for tool in (CalculatorTool(), SearchTool(atlas, wiki), ReadCaseTool(atlas, wiki), AnalogyBoardTool(store)):
        registry.register(tool)
    return AgentRuntime(llm=llm, registry=registry, store=store, context=ContextManager(), tracer=TraceLogger(trace_path))
