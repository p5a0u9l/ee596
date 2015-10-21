import matplotlib.pyplot as plt
import subprocess
import re
import numpy as np
# import ipdb; ipdb.set_trace()


class ScreenImage():

    def __init__(self):
        xr = subprocess.Popen('xrandr', stdout=subprocess.PIPE)
        output = subprocess.check_output(('grep', '*'), stdin=xr.stdout)
        screen_res = [int(x) for x in re.findall('\d{4}', output)]
        self.max = np.array(screen_res)
        print self.max
        self.cur = np.array([0, 0])

    def show(self, im, name):
        a = np.array(im[0].shape[:2])
        fig = plt.subplot(131)
        plt.imshow(im[0])
        self.im_config(name[0])

        fig = plt.subplot(132)
        plt.imshow(im[1], cmap="gray")
        self.im_config(name[1])

        fig = plt.subplot(133)
        plt.imshow(im[2])
        self.im_config(name[2])

        plt.show()
        self.cur[0] += a[1]
        if self.cur[0] > self.max[0]:
            self.cur[1] += a[0]
            self.cur[0] = 0

    def im_config(self, name):
        plt.title(name)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
