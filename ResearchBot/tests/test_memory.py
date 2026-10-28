from langchain_core.messages import AIMessage, HumanMessage

from researchbot.memory import recent_pairs, split_pairs


def convo(n_pairs: int, pending: bool = False):
    msgs = []
    for i in range(n_pairs):
        msgs += [HumanMessage(f"q{i}"), AIMessage(f"a{i}")]
    if pending:
        msgs.append(HumanMessage("pending"))
    return msgs


def test_split_pairs_keeps_pending_question_separate():
    pairs, pending = split_pairs(convo(2, pending=True))
    assert len(pairs) == 2
    assert [m.content for m in pending] == ["pending"]


def test_split_pairs_no_pending():
    pairs, pending = split_pairs(convo(3))
    assert len(pairs) == 3
    assert pending == []


def test_recent_pairs_returns_last_three_pairs():
    msgs = recent_pairs(convo(5, pending=True), n=3)
    assert [m.content for m in msgs] == ["q2", "a2", "q3", "a3", "q4", "a4"]


def test_recent_pairs_empty():
    assert recent_pairs([HumanMessage("first question")]) == []
