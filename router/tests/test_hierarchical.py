from hierarchical import route_hierarchical


class FakeClient:
    def __init__(self):
        self.calls = []

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        if "category" in questions:
            return {
                "answers": {
                    "category": {
                        "choice": "files",
                        "confidence": 0.7,
                        "probabilities": {"files": 0.7, "shell": 0.3},
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        return {
            "answers": {
                "tool": {
                    "choice": "files_read_document",
                    "confidence": 0.95,
                    "probabilities": {
                        "files_read_document": 0.95,
                        "files_search_content": 0.05,
                    },
                }
            },
            "usage": {"input_tokens": 20, "output_tokens": 3},
        }


def test_hierarchical_router_uses_two_choice_calls_and_combines_metrics():
    client = FakeClient()
    result = route_hierarchical(
        "review a contract",
        "read docs/contract.md",
        catalog={
            "files_read_document": "read a document",
            "files_search_content": "search a document",
            "shell_run_command": "run a command",
        },
        client=client,
    )

    assert result.tool == "files_read_document"
    assert result.category == "files"
    assert result.category_confidence == 0.7
    assert result.confidence == 0.7
    assert result.calls == 2
    assert result.usage == {"input_tokens": 30, "output_tokens": 5}
    assert len(client.calls) == 2
    assert len(client.calls[0][1]["category"]["criteria"]) == 2
    assert len(client.calls[1][1]["tool"]["criteria"]) == 2
    assert client.calls[1][0]["selected_category"] == "files"
