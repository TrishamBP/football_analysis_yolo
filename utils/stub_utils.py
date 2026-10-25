import hashlib
import os
import pickle


def get_video_signature(frames):
    # Frame count + hash of the first frame identifies the video a stub was built from
    return len(frames), hashlib.md5(frames[0].tobytes()).hexdigest()


def read_stub(stub_path, frames):
    if stub_path is None or not os.path.exists(stub_path):
        return None

    with open(stub_path, 'rb') as f:
        stub = pickle.load(f)

    # Ignore stubs that were generated from a different video
    if not isinstance(stub, dict) or stub.get('video_signature') != get_video_signature(frames):
        print(f"Stub {stub_path} does not match the input video, recomputing")
        return None

    return stub['data']


def save_stub(stub_path, frames, data):
    if stub_path is None:
        return

    os.makedirs(os.path.dirname(stub_path) or '.', exist_ok=True)
    with open(stub_path, 'wb') as f:
        pickle.dump({'video_signature': get_video_signature(frames), 'data': data}, f)
