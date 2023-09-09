import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pytest
import requests

def _query_server(prompt: str) -> dict:
    response = requests.post("http://localhost:8000/generate",
                             json={
                                 "prompt": prompt,
                                 "max_tokens": 100,
                                 "temperature": 0,
                                 "ignore_eos": True
                             })
    response.raise_for_status()
    return response.json()

@pytest.fixture
def api_server():
    script_path = Path(__file__).parent.joinpath(
        "api_server_async_engine.py").absolute()
    uvicorn_process = subprocess.Popen([
        sys.executable, "-u",
        str(script_path), "--model", "PygmalionAI/pygmalion-6b"
    ])
    yield
    uvicorn_process.terminate()

def test_api_server(api_server):
    with Pool(32) as pool:
        prompts = ["Hello world"] * 1
        result = None
        while not result:
            try:
                for result in pool.map(_query_server, prompts):
                    break
            except:
                time.sleep(1)

        for result in pool.map(_query_server, prompts):
            assert result
        
        num_aborted_requests = requests.get(
            "http://localhost:8000/stats").json()["num_aborted_requests"]
        assert num_aborted_requests == 0

        prompts = ["Hello world"] * 100
        for result in pool.map(_query_server, prompts):
            assert result

        pool.map_async(_query_server, prompts)
        time.sleep(0.01)
        pool.terminate()
        pool.join()

        num_aborted_requests = requests.get(
            "http://localhost:8000/stats").json()["num_aborted_requests"]
        assert num_aborted_requests > 0

    with Pool(32) as pool:
        prompts = ["Hello world"] * 100
        for result in pool.map(_query_server, prompts):
            assert result
