def register_tool(definition_func, handler_factory, service_instance=None, agent=None):
    if agent is None:
        raise ValueError(
            "register_tool() requires an agent. Global ToolRegistry has been removed."
        )
    agent.register_tool(definition_func, handler_factory, service_instance)
