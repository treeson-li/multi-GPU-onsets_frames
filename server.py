# Echo server program
import socket
import pyaudio
import wave
import time
import numpy as np
import pickle
from onsets_frames_record import Transcription
from onsets_frames_record import parse_onset_frame
import real_time_constant as const

CHUNK = 512
CHUNK_NUM = 4 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "server_output.wav"
WIDTH = 2
frames = []
HOST = '192.168.1.119'        # Symbolic name meaning all available interfaces
PORT = 50007              # Arbitrary non-privileged port
PLOT_LENGTH = 400

def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    See Also
    --------
    buf_to_float

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in `x`

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the dd of the data
    dd = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Redd and format the data buffer
    return dd * np.frombuffer(x, fmt).astype(dtype)

transcription =  Transcription()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))#绑定IP 和 端口
s.listen(1)
conn, addr = s.accept()
print ('Connected by'+str(addr))
# data = conn.recv(12288)
# print(len(data))
# decode = buf_to_float(data, dtype=np.float32)
# print(len(decode))
# onset, frame, velocity, spec = transcription.transcrib(decode)
# onset_as_bytes = pickle.dumps(onset)
# print(len(onset_as_bytes))
# conn.sendall(onset_as_bytes)

onset_frame = [np.zeros([PLOT_LENGTH, 88]), np.zeros([PLOT_LENGTH, 88]), np.zeros([PLOT_LENGTH, 88]),
                np.zeros([PLOT_LENGTH, 229]), np.zeros([PLOT_LENGTH, 88])]
while True:
    # stream.write(data)
    data = conn.recv(12*512*2) 
    print('>>>>>>>>>>>>>>>>>>>>>>data len', len(data))
    decode = buf_to_float(data, dtype=np.float32)
    print('>>>>>>>>>>>>>>>>>>>>>>decode.shape', decode.shape, type(decode))
    onset, frame, velocity, spec = transcription.transcrib(decode)
    piano_queue = [onset, frame, velocity,  spec, None]

    for i in range(4):
        onset_frame[i] = np.vstack((onset_frame[i], piano_queue[i]))
        onset_frame[i] = onset_frame[i][-PLOT_LENGTH:, :]

    onset_frame[const.MERGE] = parse_onset_frame(onset_frame)   # numpy.float32
    print('type shape of merge', type(onset_frame[const.MERGE][0, 0]), onset_frame[const.MERGE].shape)
    send_bytes = onset_frame[const.MERGE].tobytes()
    print('len send bytes', len(send_bytes))
    print(send_bytes[:11])
    conn.sendall(send_bytes)

# conn.close()