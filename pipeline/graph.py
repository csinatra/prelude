from langgraph.graph import StateGraph, END
from pipeline.state import PipelineState
from pipeline.nodes import parse_problem, surface_signals, flag_assumptions, advise_approach

def build_graph() -> StateGraph:
    graph = StateGraph(state_schema=PipelineState)

    graph.add_node(node="parse_problem", action=parse_problem)
    graph.add_node(node="surface_signals", action=surface_signals)
    graph.add_node(node="flag_assumptions", action=flag_assumptions)
    graph.add_node(node="advise_approach", action=advise_approach)

    graph.set_entry_point(key="parse_problem")
    graph.add_edge(start_key="parse_problem", end_key="surface_signals")
    graph.add_edge(start_key="surface_signals", end_key="flag_assumptions")
    graph.add_edge(start_key="flag_assumptions", end_key="advise_approach")
    graph.add_edge(start_key="advise_approach", end_key=END)

    return graph.compile()
