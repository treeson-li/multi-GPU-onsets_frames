import numpy as np 
import pickle 

class Transcription_analog:
    def __init__(self):
        import pickle
        self.pickle_dir = '/tmp/pickle_file'
        self.sample_size = None
        self.piano_roll = pickle.load(open(self.pickle_dir, 'rb'))
        self.step = 4
        self.length = 400
        self.i = 0

    def transcrib(self, samples):
        assert(len(samples), self.sample_size)
        print(self.i)
        if self.i >= self.piano_roll.shape[0]: self.i = 0 
        self.i += self.step
        return self.piano_roll[self.i:self.i+self.length, :]
        
if __name__=='__main__':
    t  = Transcription_analog()
    for i in range(111):
        r = t.transcrib([None])
        print(np.sum(r))