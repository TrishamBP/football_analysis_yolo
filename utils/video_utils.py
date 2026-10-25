import cv2

class VideoFrames:
    # List-like view of a video that decodes frames on demand instead of keeping
    # every frame in memory (750 frames of 1080p video is ~4.7 GB)
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.next_index = 0
        self.cached_index, self.cached_frame = None, None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.length))]
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        if index == self.cached_index:
            return self.cached_frame

        # Sequential reads are fast; only seek when jumping around
        if index != self.next_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError(index)
        self.next_index = index + 1
        self.cached_index, self.cached_frame = index, frame
        return frame

    def __iter__(self):
        for i in range(self.length):
            yield self[i]

def read_video(video_path):
    return VideoFrames(video_path)

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 24

def save_video(ouput_video_frames,output_video_path,fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    # Frames may be a generator, so open the writer once the first frame's size is known
    for frame in ouput_video_frames:
        if out is None:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        out.write(frame)
    if out is not None:
        out.release()