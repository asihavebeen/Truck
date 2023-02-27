import cv2
import math


def cal_grad(srcGray, thresh_th):
    gradX = cv2.Sobel(srcGray, cv2.CV_16S, 1, 0, 3)
    gradY = cv2.Sobel(srcGray, cv2.CV_16S, 0, 1, 3)

    gradX = cv2.convertScaleAbs(gradX)
    gradY = cv2.convertScaleAbs(gradY)

    dst = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)

    _, dst = cv2.threshold(dst, thresh_th, 255, cv2.THRESH_BINARY)

    return dst


def is_inside(rect1, rect2):
    return rect1 == (rect1 & rect2)


def getCenterPoint(rect):
    cpt = ()
    cpt_x = rect.x + round(rect.width/2.0)
    cpt_y = rect.y + round(rect.height/2.0)
    cpt = (cpt_x, cpt_y)
    return cpt


def getDistance(pointO, pointA):
    distance = math.sqrt(math.pow((pointO[0] - pointA[0]), 2) + math.pow((pointO[1] - pointA[1]), 2))
    return distance


def find_rec(frame, num, valid, valid_thresh, going, rect_all, color_bound):
    drawing = frame.copy()
    for i in range(num):
        if(valid[i] > valid_thresh and going[i] > 0):
            cv2.rectangle(drawing, rect_all[i].tl(), rect_all[i].br(), color_bound[i], 2, 8, 0)
    return drawing


def judgeColor(frame):
    global car_cal
    clone_one = None
    for i in range(num):
        destiny = "../saved/"
        count = str(car_cal)
        destiny_back = ".jpg"

        destiny += count
        destiny += destiny_back

        if valid[i] == valid_thresh and pic_out[i] < 1:
            clone_one = frame[rect_all[i][1]:rect_all[i][1]+rect_all[i][3], rect_all[i][0]:rect_all[i][0]+rect_all[i][2]]
            cv2.imwrite(destiny, clone_one)

            car_cal += 1
            pic_out[i] = 1
    
    return 0

def judgeArea(i):
    if way[i] < 1:
        # 下行看结束大小
        if rect_all[i].area() > 30000:
            return 2 # 大型车
        elif rect_all[i].area() > 7000:
            return 1 # 中型车
        else:
            return 0 # 小型车
    else:
        # 上行看初始大小
        if begin_place[i].area() > 30000:
            return 2 # 大型车
        elif begin_place[i].area() > 7000:
            return 1 # 中型车
        else:
            return 0 # 小型车

def read_color(place):
    global color_five
    destiny = "../saved/"
    count = str(place)
    destiny_back = ".jpg"
    destiny += count
    destiny += destiny_back

    read_one = cv2.imread(destiny)
    HSVMat = cv2.cvtColor(read_one, cv2.COLOR_BGR2HSV)
    planes = cv2.split(HSVMat)

    color_five = [0, 0, 0, 0, 0] # 按顺序分别是黑、白、黄、红、蓝

    for i in range(read_one.shape[0]):
        for j in range(read_one.shape[1]):
            if planes[2][i][j] < 30:
                color_five[0] += 1 # 黑
            else:
                if planes[1][i][j] < 43:
                    if planes[2][i][j] < 200:
                        continue
                    else:
                        color_five[1] += 1 # 白
                else:
                    if planes[0][i][j] > 100 and planes[0][i][j] < 124:
                        color_five[4] += 1 # 蓝
                    else:
                        if planes[0][i][j] > 3 and planes[0][i][j] < 50:
                            color_five[2] += 1 # 黄
                        if planes[0][i][j] > 160 or planes[0][i][j] < 3:
                            color_five[3] += 1 # 红

    max_record = 0
    max_out = 0
    for i in range(5):
        if color_five[i] > max_record:
            max_record = color_five[i]
            max_out = i

    return max_out


def cout_csv():
    csv_cal = 0

    # 打开输出文件
    p = open("../saved/output.csv", "w")

    # 写入表头
    p.write("序号,车型,方向,颜色\n")

    # 遍历车辆并输出
    for i in range(num):
        if valid[i] < valid_thresh:
            continue

        count = str(csv_cal)
        p.write(count + ",")

        if judgeArea(i) == 0:
            p.write("小型车,")
        elif judgeArea(i) == 1:
            p.write("中型车,")
        else:
            p.write("大型车,")

        if way[i] < 1:
            p.write("下行,")
        else:
            p.write("上行,")

        switch_dict = {
            0: "黑色",
            1: "白色",
            2: "黄色",
            3: "红色",
            4: "蓝色"
        }
        color = read_color(csv_cal)
        p.write(switch_dict.get(color, "彩色") + "\n")

        csv_cal += 1

    # 关闭输出文件
    p.close()


def change_car(car, frame, num, going, rect_all, distance_thresh, begin_place, way,
               color_all, valid, color_bound, pic_out):     # 修改记录数据的部分，颜色、矩形框等的数据
    in_or_out = False

    for i in range(num):
        going[i] = False
        if getDistance(getCenterPoint(car), getCenterPoint(rect_all[i])) < distance_thresh:    # 距离在范围内就更新，也就是同一辆车
            rect_all[i] = car
            if getCenterPoint(car)[1] - getCenterPoint(begin_place[i])[1] > 0:
                way[i] = 0
            else:
                way[i] = 1
            color_all[i] = judgeColor(frame)
            valid[i] += 1
            going[i] = True

            in_or_out = True
            break

    if not in_or_out:    # 距离不在范围内就新加入一个
        rect_all[num] = car
        begin_place[num] = car
        way[num] = 0
        color_all[num] = 0
        valid[num] = 1
        color_bound[num] = (rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255))
        going[num] = True
        pic_out[num] = 0

        num += 1

    if num == 0:    # 第一辆车加入一下
        rect_all[0] = car
        begin_place[0] = car
        way[0] = 0
        color_all[0] = 0
        valid[0] = 1
        color_bound[0] = (rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255))
        going[0] = True
        pic_out[0] = 0

        num += 1


def record_cars(dst, frame):
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundRect = []
    for contour in contours:
        boundRect.append(cv2.boundingRect(contour))

    judge = 0
    for i, rect in enumerate(boundRect):
        if getCenterPoint(rect).y > bounding_top and rect.area() > 3000:
            change_car(rect, frame)
            judge = 1
    if judge < 1:
        for i in range(num):
            going[i] = 0


def main():
    main_capture = cv2.VideoCapture()
    frame_gray = backgroud_truth = dst = see = None
    dsize = (960, 544)

    backgroud_truth = build_backgroud().clone()
    print("BackGround Done!")

    if not main_capture.open("../video/1.mp4"):
        print("can not open ...")
        return -1

    _, frame = main_capture.read()
    frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)

    # rate = frame.shape[1] / frame.shape[0]
    # rows_real = 540
    # cols_real = int(540.0 * rate)

    # dsize = (cols_real, rows_real)
    # frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dst_mid = 100 * np.ones((frame.shape[0] // 2, frame.shape[1] // 2), dtype=np.uint8)
    record = frame_gray.copy()

    cv2.namedWindow("output_origin", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output_origin", 960, 544)
    cv2.namedWindow("output_gray", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output_gray", 960, 544)
    cv2.namedWindow("output_see", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output_see", 960, 544)
    cv2.namedWindow("output_dst", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output_dst", 960, 544)

    M = 2
    cols_real = frame.shape[1] // M
    rows_real = frame.shape[0] // M

    thresh_all = 0
    record_i = 0
    record_j = 0

    diff_record = np.zeros((rows_real, cols_real), dtype=np.int)

    for i in range(rows_real):
        for j in range(cols_real):
            diff_record[i, j] = 0
    
    while True:
    ret, frame = main_capture.read()
    
    if not ret:
        break
        
    frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)
    
    cv2.imshow("output_origin", frame)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("output_gray", frame_gray)
    
    dark = is_dark(frame_gray)
    
    thresh_all = 0
    diff_record = np.zeros((rows_real, cols_real), dtype=np.int32)
    
    for i in range(rows_real):
        for j in range(cols_real):
            diff_record[i][j] = 0
            
    for i in range(rows_real):
        for j in range(cols_real):
            for ii in range(M):
                for jj in range(M):
                    diff_record[i][j] += abs(record[N*i+ii, N*j+jj] - frame_gray[N*i+ii, N*j+jj])
            
            thresh_all += diff_record[i][j]
            
    thresh_all = thresh_all // (cols_real * rows_real)
    
    dst = np.zeros((N*rows_real, N*cols_real), dtype=np.uint8)
    for i in range(rows_real):
        for j in range(cols_real):
            record_i = M*i
            record_j = M*j
            
            if diff_record[i][j] >= thresh_all:
                for ii in range(M):
                    for jj in range(M):
                        dst[record_i+ii, record_j+jj] = 250
            else:
                for ii in range(M):
                    for jj in range(M):
                        dst[record_i+ii, record_j+jj] = 5
    
    dst = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY)[1]
    dst_mid = np.zeros((2*rows_real, 2*cols_real), dtype=np.uint8)
    for i in range(2*rows_real):
        for j in range(2*cols_real):
            dst_mid[i, j] = dst[i, j]
            
    dst = cv2.resize(dst_mid, dsize, interpolation=cv2.INTER_AREA)
    dst = cv2.dilate(dst, kernel_1515, iterations=1)
    dst = cv2.dilate(dst, kernel_1515, iterations=1)
    dst = cv2.dilate(dst, kernel_1515, iterations=1)
    dst = cv2.erode(dst, kernel_77, iterations=1)
    dst = cv2.erode(dst, kernel_77, iterations=1)
    dst = cv2.erode(dst, kernel_77, iterations=1)
    dst = cv2.erode(dst, kernel_1515, iterations=1)
    
    if dark < 1:
        for i in range(dst.shape[0]):
            for j in range(5):
                dst[i, 430+j] = 0
                
    if dark < 1:
        record_cars(dst, frame)
    else:
        record_cars_dark(frame, frame_gray, dst)
        circle_to_rect(frame)
        
    cv2.imshow("output_dst", dst)
    
    if dark < 1:
        see = find_rec(frame).copy()
    else:
        see = find_rec_dark(frame).copy()
        
    cv2.imshow("output_see", see)
    
    record = frame_gray.copy()

    if mean < dark:
        cout_csv_dark()
    else:
        cout_csv()

    cap.release()