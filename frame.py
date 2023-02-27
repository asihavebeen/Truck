import cv2
import numpy as np
import copy
import another


#parameter
bounding_top = 170        #检测区域的上边界
bounding_bottom = 380     #检测区域的下边界
distance_thresh = 60.0    #在这个距离以内的中心点被认为是同一辆车
valid_thresh = 10         #持续15帧以上的才计入车


#launch
way = []    #记录方向，0为下行，1为上行
color_all = []     #记录颜色，1-7
color_bound = []      #框的颜色，画图使用
rect_all = []     #矩形类，记录检测出的矩形框
begin_place = []     #记录起始位置，用于判断方向和大小型车
valid = []    #记录识别是否有效，大于15帧才算有效
going = []    #记录汽车是否已脱离区域
pic_out = []      #记录是否已经输出图像，输出过的不再输出
num = 0      #记录识别出的总车数，但不全都有效
car_cal = 0;      #统计实际有效车数


cap = cv2.VideoCapture(0)
ret, img = cap.read()
# img = cv2.resize(img, (640, 360))
# img = img[100:, :]
previmg = img    #第一帧
previmg = cv2.cvtColor(previmg, cv2.COLOR_BGR2GRAY)
previmg = cv2.GaussianBlur(previmg, (3, 3), 0)
height, width = img.shape[:2]

'''保存视频'''
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps = cap.get(cv2.CAP_PROP_FPS)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out_result = cv2.VideoWriter('output/result01.avi', fourcc, fps, size)

while cap.isOpened():
    ret, img = cap.read()
    # img = cv2.resize(img, (640, 360))
    # img = img[100:, :]
    nextimg = img
    nextimg = cv2.cvtColor(nextimg, cv2.COLOR_BGR2GRAY)
    nextimg = cv2.GaussianBlur(nextimg, (3, 3), 0)

    if ret:
        '''帧差法'''
        diff = cv2.absdiff(previmg, nextimg)
        diff[:150, :] = 0
        # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        '''连通域去噪'''
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary, dtype=np.uint8)
        for (i, label) in enumerate(np.unique(labels)):
            # 如果是背景，忽略
            if label == 0:
                continue
            if stats[i][-1] > 200:
                filtered[labels == i] = 1
        # cv2.imshow('noise', filtered)

        '''膨胀连通边缘'''
        kernel = np.ones((35, 35), dtype=np.uint8)
        # process = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        process = cv2.dilate(filtered, kernel, iterations=1)

        '''洪泛'''
        copy_fl = copy.deepcopy(process)
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(copy_fl, mask, (0, 0), 255)
        copy_inv = cv2.bitwise_not(copy_fl)
        flood = copy_inv | process
        result = cv2.erode(flood, kernel, iterations=1)

        '''矩形框'''
        another.record_cars(dst, frame, going, num, bounding_top)
        see = another.find_rec(frame, num, valid, valid_thresh, going, rect_all, color_bound)
        another.cout_csv(num, valid, valid_thresh, way)

        # out_result.write(img)

        cv2.imshow('video', diff)
        cv2.imshow('origin', img)
        cv2.imshow('process', process)
        cv2.imshow('result', result)
        previmg = nextimg   #帧差法 背景变化
        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
