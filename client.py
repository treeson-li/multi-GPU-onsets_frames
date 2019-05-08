import socket
import pyaudio
import wave
import numpy as np
import pickle

#record
CHUNK = 6144
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 30

# HOST = '192.168.1.119'    # The remote host
# HOST = 'localhost'
HOST = '192.168.1.118'
PORT = 50007              # The same port as used by the server

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("============================ recording")

frames = []

while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    print('len of send data', len(data))
    s.sendall(data)
    result_data = s.recv(141504)
    print('len of recieve', len(result_data))
    print(result_data[:11])
    y = np.fromstring(result_data, np.float32)
    print('>>>>>>>>>>>>>>>>>len decode', len(y))
    y = y.reshape(402, 88)
    print(y)
    frames.append(data)
 

print("*done recording")

stream.stop_stream()
stream.close()
p.terminate()
s.close()

print("*closed")