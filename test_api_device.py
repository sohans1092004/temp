"""api.py's UDK_DEVICE switch, and the Whisper CUDA -> CPU fallback: a GPU/cuDNN
failure must fall back to CPU Whisper, never to MockSTT (= no English
transcription at all). Stubbed, no real models: python test_api_device.py"""
import os
import sys
import warnings

import api

calls = []
FAIL_CUDA = False


class FakeWhisper:
    def __init__(self, model_size, device, compute_type):
        calls.append((model_size, device, compute_type))
        self.device = device

    def transcribe(self, pcm):
        if self.device == "cuda" and FAIL_CUDA:
            raise RuntimeError("Could not load library libcudnn_ops.so.9")
        return None


sys.modules["stt"].FasterWhisperSTT = FakeWhisper  # the loader imports it from stt at call time
os.environ["UDK_ENABLE_STT"] = "1"

for env, want in (("cpu", "cpu"), ("cuda", "cuda")):
    api._DEVICE = None
    os.environ["UDK_DEVICE"] = env
    assert api._device() == want, (env, api._device())

api._DEVICE, FAIL_CUDA, calls[:] = "cuda", False, []
stt = api._load_stt_backend()
assert calls == [("small.en", "cuda", "float16")] and stt.device == "cuda", calls

api._DEVICE, FAIL_CUDA, calls[:] = "cuda", True, []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    stt = api._load_stt_backend()
assert isinstance(stt, FakeWhisper) and stt.device == "cpu", "must fall back to CPU Whisper, never MockSTT"
assert calls == [("small.en", "cuda", "float16"), ("small.en", "cpu", "int8")], calls
assert any("falling back to CPU" in str(x.message) for x in w)

api._DEVICE, calls[:] = "cpu", []
api._load_stt_backend()
assert calls == [("small.en", "cpu", "int8")], calls
print("api device switch: all checks passed")
