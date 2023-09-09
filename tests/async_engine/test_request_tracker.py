import pytest

from aphrodite.engine.async_aphrodite import RequestTracker
from aphrodite.common.outputs import RequestOutput

def test_request_tracker():
    tracker = RequestTracker()
    stream_1 = tracker.add_request("1")
    new, finished = tracker.get_new_and_finished_requests()
    assert len(new) == 1
    assert new[0]["request_id"] == "1"
    assert not finished
    assert not stream_1.finished

    stream_2 = tracker.add_request("2")
    stream_3 = tracker.add_request("3")
    new, finished = tracker.get_new_and_finished_requests()
    assert len(new) == 2
    assert new[0]["request_id"] == "2"
    assert new[1]["request_id"] == "3"
    assert not finished
    assert not stream_2.finished
    assert not stream_3.finished

    with pytest.raises(KeyError):
        tracker.add_request("1")

    tracker.abort_request("1")
    new, finished = tracker.get_new_and_finished_requests()
    assert len(finished) == 1
    assert "1" in finished

    stream_1 = tracker.add_request("4")
    tracker.abort_request("4")
    new, finished = tracker.get_new_and_finished_requests()
    assert len(new) == 1
    assert "4" in finished
    assert not new
    assert not stream_4.finished

    stream_5 = tracker.add_request("5")
    tracker.process_request_output(
        RequestOutput('2', 'output', [], [], finished=True))
    new, finished = tracker.get_new_and_finished_requests()
    assert len(finished) == 1
    assert "2" in finished
    assert len(new) == 1
    assert new[0]["request_id"] == "5"
    assert stream_2.finished
    assert not stream_5.finished