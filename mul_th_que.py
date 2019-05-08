import threading  
import time  
import queue
import numpy as np
from animation_plot import plot
# from onsets_frames_client_record import Record, Transcription
from onsets_frames_record import Record, Transcription, Record_analog
import real_time_constant as constant

PLOT_TYPE = constant.MERGE #PLUS SPEC MERGE
lock = threading.Lock()
class Record_thread(threading.Thread): 
    def __init__(self, queue):  
        threading.Thread.__init__(self, name='record')  
        self.queue=queue
        self.record= Record_analog()
        self.start()
    
    def run(self):
        while True:
            next_record = next(self.record.recording())
            self.queue.put(next_record, block=True)
            print('---------------------------- put record queue ==>> size', self.queue.qsize())

class Transcription_thread(threading.Thread):
    def __init__(self, record_queue, piano_queue):
        threading.Thread.__init__(self, name='record')
        self.record_queue = record_queue
        self.piano_queue = piano_queue
        self.transcription = Transcription()
        self.start()

    def run(self):
        while True:
            priority, samples = self.record_queue.get(block=True)
            self.record_queue.task_done()
            if priority<20: continue
            print('---------------------------- get record queue <<== size', self.record_queue.qsize())
            
            onset, frame, velocity, spec = self.transcription.transcrib(samples)
            self.piano_queue.put((priority, [onset, frame, velocity, spec, None]), block=True)
            print('---------------------------- put piano queue ==>> size', self.piano_queue.qsize())

class Plot_thread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self, name='plot')
        self.queue = queue
        self.plot = plot
        self.start()
    def run(self):
        self.plot(self.queue, PLOT_TYPE)
    
if __name__ == "__main__":  
    q_record =queue.PriorityQueue()
    q_piano = queue.PriorityQueue()

    threads = []
    threads.append(Plot_thread(q_piano))
    threads.append(Record_thread(q_record))
    threads.append(Transcription_thread(q_record, q_piano))
    # threads.append(Transcription_thread(q_record, q_piano))
    # threads.append(Transcription_thread(q_record, q_piano))
    # threads.append(Transcription_thread(q_record, q_piano))
    # threads.append(Transcription_thread(q_record, q_piano))
    for th in threads:
        if th.isAlive():
            th.join()
