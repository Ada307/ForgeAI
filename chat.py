"""CLI Chat Runner for ForgeAI Generated Agents.

Usage:
    python chat.py --definition path/to/agent_definition.json --docs path/to/documents/
"""

import argparse
import sys
from forge.runner import AgentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgeAI Generic Agent CLI Chat Runner")
    parser.add_argument(
        "--definition",
        required=True,
        help="Path to agent_definition.json file",
    )
    parser.add_argument(
        "--docs",
        required=True,
        help="Path to documents folder for retrieval and tools",
    )
    args = parser.parse_args()

    print("Initializing ForgeAI Agent Runner...")
    try:
        runner = AgentRunner(
            agent_definition=args.definition,
            document_folder=args.docs,
        )
    except Exception as exc:
        print(f"Error loading agent runner: {exc}", file=sys.stderr)
        sys.exit(1)

    agent_name = runner.definition.get("agent_name", "ForgeAgent")
    description = runner.definition.get("description", "")
    scope = runner.definition.get("scope", "")

    print("=" * 60)
    print(f"Agent: {agent_name}")
    print(f"Description: {description}")
    print(f"Permitted Scope: {scope}")
    print(f"Authorized Tools: {runner.allowed_tools or 'None (Zero-tool assistant)'}")
    print("=" * 60)
    print("Type 'exit' or 'quit' to end session.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Session ended.")
            break

        result = runner.answer(query)

        status_tag = f"[{result.status.upper()}]"
        print(f"\n{status_tag} {result.answer}\n")

        if result.citations:
            print("Evidence Citations:")
            for cit in result.citations:
                print(f"  * [{cit.get('source')} | {cit.get('chunk_id')}] \"{cit.get('excerpt')}\"")
            print()

        if result.tool_executions:
            print("Tool Executions:")
            for t in result.tool_executions:
                tool_name = t.get("tool")
                outcome = t.get("outcome", {})
                success = outcome.get("success", False)
                print(f"  * Tool '{tool_name}' (Success: {success})")
            print()


if __name__ == "__main__":
    main()
