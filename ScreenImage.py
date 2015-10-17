import subprocess
import re
import numpy as np
import cv2


class ScreenImage():

    def __init__(self):
        xr = subprocess.Popen('xrandr', stdout=subprocess.PIPE)
        output = subprocess.check_output(('grep', '*'), stdin=xr.stdout)
        screen_res = [int(x) for x in re.findall('\d{4}', output)]
        self.max = np.array(screen_res)
        print self.max
        self.cur = np.array([0, 0])

    def show(self, im, name):
        a = np.array(im.shape[:2])
        cv2.imshow(name, im)
        cv2.moveWindow(name, self.cur[0], self.cur[1])
        self.cur[0] += a[1]
        if self.cur[0] > self.max[0]:
            self.cur[1] += a[0]
            self.cur[0] = 0

        # Closing
        cv2.waitKey(0)
        cv2.destroyAllWindows()
