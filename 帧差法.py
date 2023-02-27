import cv2
import numpy as np
import copy


cap = cv2.VideoCapture('video/test01.mp4')
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
        contours, _ = cv2.findContours(flood, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            # 外接矩形
            x, y, w, h = cv2.boundingRect(cont)
            # 在原图上画出预测的矩形
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
