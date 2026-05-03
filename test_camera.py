import sys
import cv2

sys.path.append('/home/cat/cat_linux_project')
from Drivers.camera import Camera

WIN = 'Camera Test'

with Camera(device=9, width=640, height=480, fps=60) as cam:
    print('Press q or ESC to quit')
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    while True:
        frame = cam.read()
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()
