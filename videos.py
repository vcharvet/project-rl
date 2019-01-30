import numpy as np
import matplotlib.pyplot as plt

from skimage.viewer import CollectionViewer
import time


def choose_sample(segment1, segment2):
    plt.ion()


    # fig, (ax0, ax1) = plt.subplots(1, 2)
    # for i in range(segment1.shape[0]):
    #     ax0.imshow(segment1[i, :, :, :])
    #     ax0.set_title('Firt Segment - {}th frame'.format(i))
    #     ax1.imshow(segment2[i, :, :, :])
    #     ax1.set_title('Second Segment - {}th frame'.format(i))
    #     fig.show()
    axes = AxesSequence()
    for i in range(segment1.shape[0]):
        ax1, ax2 = axes.new()
        ax1.imshow(segment1[i, :, :, :])
        ax1.set_title('First Segment - {}th frame'.format(i))
        # ax[1].imshow(segment2[i, :, :, :])
        # ax[1].set_title('Second Segment - {}th frame'.format(i))
    # for i, ax in zip(range(segment2.shape[0]), axes):
        ax2.imshow(segment2[i, :, :, :])
        ax2.set_title('Second Segment - {}th frame'.format(i))
    axes.show()

    user_preference = input("Choose either segment 1 or segment 2:")
    plt.close('all')
    if user_preference == 1:
        res = 1, 0
    elif user_preference == 2:
        res = (0, 1)
    else:
        res = 0.5, 0.5

    return res

#https://stackoverflow.com/questions/13443474/matplotlib-sequence-of-figures-in-the-same-window
class AxesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 16))
        self.axes = []
        self._i = 0 # Currently displayed axes index
        self._n = 0 # Last created axes index
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        # ax = self.fig.add_axes([0.15, 0.1, 0.8, 0.8],
        #                        visible=False, label=self._n)
        ax1 = self.fig.add_subplot(211, visible=False, label=self._n)
        ax2 = self.fig.add_subplot(212, visible=False, label=self._n)
        self._n += 1
        self.axes.append((ax1, ax2))
        return ax1, ax2

    def on_keypress(self, event):
        if event.key == 'right':
            self.next_plot()
        elif event.key == 'left':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self._i < len(self.axes):
            self.axes[self._i][0].set_visible(False)
            self.axes[self._i+1][0].set_visible(True)

            self.axes[self._i][1].set_visible(False)
            self.axes[self._i+1][1].set_visible(True)
            self._i += 1

    def prev_plot(self):
        if self._i > 0:
            self.axes[self._i][0].set_visible(False)
            self.axes[self._i-1][0].set_visible(True)

            self.axes[self._i][1].set_visible(False)
            self.axes[self._i-1][1].set_visible(True)

            self._i -= 1

    def show(self):
        self.axes[0][0].set_visible(True)
        self.axes[0][1].set_visible(True)
        plt.show()