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
        self.cur = np.array([0, 0])

    def show(self, im, name):
        a = np.array(im[0].shape[:2])
        f = plt.figure(figsize=(12, 10))
        # print "## %s" % (title,)
        ax = f.add_subplot(221)
        ax.imshow(im[0])
        self.im_config(name[0])

        ax = f.add_subplot(222)
        ax.imshow(im[1][:, :, 0], cmap="gray")
        self.im_config(name[1])

        ax = f.add_subplot(223)
        ax.imshow(im[2])
        self.im_config(name[2])

        ax = f.add_subplot(224)
        ax.imshow(im[3])
        self.im_config(name[3])

        # ax = f.add_subplot(235)
        # ax.imshow(im[4], cmap="gray")
        # self.im_config(name[4])

        # ax = f.add_subplot(236)
        # ax.imshow(im[5], cmap="gray")
        # self.im_config(name[5])

        plt.show()

    def im_config(self, name):
        plt.title(name)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
