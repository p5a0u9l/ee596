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

    def show(self, im, name, sqr_shape):
        # print "## %s" % (title,)
        f = plt.figure(figsize=(14, 14))
        if sqr_shape == 2:
            a = 2; b = 1
        elif sqr_shape == 4:
            a = b = 2

        ax = f.add_subplot(a, b, 1)
        ax.imshow(im[0])
        self.im_config(name[0].replace("_", " ").replace("/", " "))

        ax = f.add_subplot(a, b, 2)
        ax.imshow(im[1])
        self.im_config(name[1].replace("_", " ").replace("/", " "))

        if sqr_shape == 4:
            ax = f.add_subplot(a, b, 3)
            ax.imshow(im[2])
            self.im_config(name[2].replace("_", " ").replace("/", " "))

            ax = f.add_subplot(a, b, 4)
            ax.imshow(im[3])
            self.im_config(name[3].replace("_", " ").replace("/", " "))

        plt.tight_layout()
        plt.show()
        # ax = f.add_subplot(235)
        # ax.imshow(im[4], cmap="gray")
        # self.im_config(name[4])

        # ax = f.add_subplot(236)
        # ax.imshow(im[5], cmap="gray")
        # self.im_config(name[5])

    def im_config(self, name):
        plt.title(name, {'fontsize': 20, 'fontweight': 'bold'})
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
