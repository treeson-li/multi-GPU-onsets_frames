import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter, Locator, NullLocator
import os
from matplotlib.animation import FuncAnimation
import queue

from onsets_frames_record import parse_onset_frame
import real_time_constant as const

PLOT_LENGTH = 400   # around 12 sceconds
class myLocator(Locator):
    def __init__(self, scale=1):
        super(myLocator, self).__init__()
        self.scale=scale
    def __call__(self):
        return self.tick_values(None,None)
    def tick_values(self, vmin, vmax):
        return np.arange(0.5, 88, self.scale)
def funcx(x, pos):
    return int(x/50)
def funcy(x, pos):
    # print(x, pos)
    return int(x+20.5)

def plot(q, k):
    if k==const.SPEC or k==const.PLUS: vmax=60
    else: vmax=1
    print('============================ animating')
    onset_frame = [np.zeros([PLOT_LENGTH, 88]), np.zeros([PLOT_LENGTH, 88]), np.zeros([PLOT_LENGTH, 88]),
                    np.zeros([PLOT_LENGTH, 229]), None, None, None]

    _, piano_queue = q.get(block=True)
    q.task_done()

    for i in range(4):
        onset_frame[i] = np.vstack((onset_frame[i], piano_queue[i]))
        onset_frame[i] = onset_frame[i][-PLOT_LENGTH:, :]

    onset_frame[const.MERGE] = parse_onset_frame(onset_frame)
    onset_frame[const.PLUS] = np.concatenate((50*onset_frame[const.MERGE][-PLOT_LENGTH:, :], 10*np.ones([PLOT_LENGTH, 2]), onset_frame[const.SPEC]), axis=1)
    onset_frame[const.MULTI] = np.concatenate((onset_frame[const.ONSET], np.ones([PLOT_LENGTH, 1]), 
                                            onset_frame[const.FRAME], np.ones([PLOT_LENGTH, 1]), onset_frame[const.MERGE][-PLOT_LENGTH:, :]), axis=1)
    fig = plt.figure(figsize=(30, 20), dpi=100)
    pcolor = plt.pcolormesh(onset_frame[k].T, cmap='jet', vmin=0, vmax=vmax)

    # fig settings
    xLocator = MultipleLocator(30)
    yLocator = myLocator(6)
    yminLocator = myLocator(1)

    # plt.gca().xaxis.set_major_locator(xLocator)
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(funcx))

    # plt.gca().yaxis.set_major_formatter(FuncFormatter(funcy))
    # plt.gca().yaxis.set_major_locator(yLocator)
    # plt.gca().yaxis.set_minor_locator(yminLocator)

    # plt.grid(color='g', linestyle='--', linewidth=0.5)
    # plt.colorbar()

    # animate func
    plot_time = time.time()
    def animate(*args):
        nonlocal plot_time, onset_frame
        try:
            priority, piano_queue=q.get(block=True, timeout=10)   #接收消息
            q.task_done()

            for i in range(4):
                onset_frame[i] = np.vstack((onset_frame[i], piano_queue[i]))
                onset_frame[i] = onset_frame[i][-PLOT_LENGTH:, :]

            onset_frame[const.MERGE] = parse_onset_frame(onset_frame)
            onset_frame[const.PLUS] = np.concatenate((50*onset_frame[const.MERGE][-PLOT_LENGTH:, :], 10*np.ones([PLOT_LENGTH, 2]), onset_frame[const.SPEC]), axis=1)
            onset_frame[const.MULTI] = np.concatenate((onset_frame[const.ONSET], np.ones([PLOT_LENGTH, 1]), 
                                                    onset_frame[const.FRAME], np.ones([PLOT_LENGTH, 1]), onset_frame[const.MERGE][-PLOT_LENGTH:, :]), axis=1)
            print('---------------------------- priority', priority)
            pcolor.set_array(onset_frame[k].T.ravel())
        except queue.Empty:
            print("plot queue is empty！")
            
        print('|| plot time', time.time() - plot_time)
        plot_time = time.time()
        return [pcolor]

    # onclick call back
    def onClick(event):
        if anim.running:
            anim.event_source.stop()
        else:
            anim.event_source.start()
        anim.running ^= True
    # onpress call back
    def on_press(event):
        if event.key.isspace():
            if anim.running:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            anim.running ^= True
        elif event.key == 'left':
            time.sleep(1)
        elif event.key == 'right':
            animate('')

    fig.canvas.mpl_connect('button_press_event', onClick)
    fig.canvas.mpl_connect('key_press_event', on_press)
    anim = FuncAnimation(fig, animate, interval=10, blit=True)
    anim.running = True 
    plt.show()

if __name__=='__main__':
    q = queue.Queue()
    for i in range(10):
        q.put(np.random.rand(627, 88).T)
    plot(q)
