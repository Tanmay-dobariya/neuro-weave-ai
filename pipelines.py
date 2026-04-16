from agents import (
    create_research_agent,
    create_reader_agent,
    writer_chain,
    critic_chain,
)


def run_research_pipeline(topic: str) -> dict:
    state = {}

    print("\n" + "=" * 80)
    print(f"Research agent is running...")
    print("=" * 80 + "\n")

    research_agent = create_research_agent()
    search_response = research_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Find the most recent, reliable, relevant and detailed information about {topic}. "
                    f"Use the research_tool to search the web. "
                    f"In your final answer, ALWAYS preserve the full source URLs (starting with http:// or https://) "
                    f"for each finding so they can be followed up later. "
                    f"Format each finding as: '- <summary> (URL: <full_url>)'.",
                )
            ]
        }
    )

    state["search_result"] = search_response.get("messages")[-1].content
    print(state["search_result"])

    print("\n" + "=" * 80)
    print(f"Reader agent is running...")
    print("=" * 80 + "\n")

    reader_agent = create_reader_agent()
    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Below are search results about {topic}. "
                    f"Pick ONE concrete URL (must start with http:// or https://) from the results "
                    f"and call the reader_tool with that URL as the 'url' argument to scrape its content. "
                    f"If no URL is present, reply with 'No URL found' and do not call the tool.\n\n"
                    f"Search results:\n{state.get('search_result')}",
                )
            ]
        }
    )

    state["reader_result"] = reader_result.get("messages")[-1].content
    print(state["reader_result"])

    combined_research = (
        f'Search result:\n{state.get("search_result")}\n\n'
        f'Reader result:\n{state.get("reader_result")}'
    )

    state["report"] = writer_chain.invoke(
        {"topic": topic, "research": combined_research}
    )
    print(state["report"])

    print("\n" + "=" * 80)
    print(f"Critic agent is running...")
    print("=" * 80 + "\n")

    state["critique"] = critic_chain.invoke({"report": state.get("report")})
    print(state["critique"])

    return state


if __name__ == "__main__":
    topic = input("Enter the topic you want to research: ")
    run_research_pipeline(topic)
