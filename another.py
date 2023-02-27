import cv2
import math
import random
import csv


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
    cpt_x = rect[0] + round(rect[2]/2.0)
    cpt_y = rect[1] + round(rect[3]/2.0)
    cpt = (cpt_x, cpt_y)
    return cpt


def getarea(rect):
    return rect[2] * rect[3]


def getDistance(pointO, pointA):
    distance = math.sqrt(math.pow((pointO[0] - pointA[0]), 2) + math.pow((pointO[1] - pointA[1]), 2))
    return distance


def find_rec(frame, num, valid, valid_thresh, going, rect_all, color_bound):
    drawing = frame.copy()
    for i in range(num[0]):
        if(valid[i] > valid_thresh and going[i] > 0):
            cv2.rectangle(drawing, (rect_all[i][0], rect_all[i][1]), (rect_all[i][0] + rect_all[i][2], 
                                    rect_all[i][1] + rect_all[i][3]), color_bound[i], 2, 8, 0)
    return drawing


def saveColor(frame, valid, pic_out, rect_all, valid_thresh, num, car_cal):
    for i in range(num[0]):
        destiny = "./saved/"
        count = str(car_cal[0])
        destiny_back = ".jpg"

        destiny += count
        destiny += destiny_back

        if valid[i] >= valid_thresh and pic_out[i] < 1 and getarea(rect_all[i]) > 2000:
            clone_one = frame[rect_all[i][1]:rect_all[i][1]+rect_all[i][3], rect_all[i][0]:rect_all[i][0]+rect_all[i][2]]
            cv2.imwrite(destiny, clone_one)

            car_cal[0] += 1
            pic_out[i] = 1
    
    return 0


def judgeArea(i, way, rect_all, begin_place):
    if way[i] < 1:
        # 下行看结束大小
        if getarea(rect_all[i]) > 30000:
            return 2 # 大型车
        elif getarea(rect_all[i]) > 7000:
            return 1 # 中型车
        else:
            return 0 # 小型车
    else:
        # 上行看初始大小
        if getarea(begin_place[i]) > 30000:
            return 2 # 大型车
        elif getarea(begin_place[i]) > 7000:
            return 1 # 中型车
        else:
            return 0 # 小型车


def read_color(place):
    color_five = [0, 0, 0, 0, 0]

    destiny = "./saved/"
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


def cout_csv(num, valid, valid_thresh, way, rect_all, begin_place):
    csv_cal = 0

    # 打开输出文件
    with open("./saved/output.csv", "w", newline='') as csvfile:
        p = csv.writer(csvfile)

        # 写入表头
        p.writerow(["序号","车型","方向","颜色"])

        # 遍历车辆并输出
        for i in range(num[0]):
            if valid[i] < valid_thresh:
                continue

            count = str(csv_cal)

            if judgeArea(i, way, rect_all, begin_place) == 0:
                big = "小型车"
            elif judgeArea(i, way, rect_all, begin_place) == 1:
                big = "中型车"
            else:
                big = "大型车"

            if way[i] < 1:
                go = "下行"
            else:
                go = "上行"

            switch_dict = {
                0: "黑色",
                1: "白色",
                2: "黄色",
                3: "红色",
                4: "蓝色"
            }
            color = read_color(csv_cal)
            p.writerow([count, big, go, switch_dict[color]])

            csv_cal += 1


def change_car(car, frame, num, going, rect_all, distance_thresh, begin_place, way,
               color_all, valid, color_bound, pic_out, valid_thresh, car_cal):     # 修改记录数据的部分，颜色、矩形框等的数据
    in_or_out = False

    for i in range(num[0]):
        going[i] = False
        if getDistance(getCenterPoint(car), getCenterPoint(rect_all[i])) < distance_thresh:    # 距离在范围内就更新，也就是同一辆车
            rect_all[i] = car
            if getCenterPoint(car)[1] - getCenterPoint(begin_place[i])[1] > 0:
                way[i] = 0
            else:
                way[i] = 1
            color_all[i] = saveColor(frame, valid, pic_out, rect_all, valid_thresh, num, car_cal)
            valid[i] += 1
            going[i] = True

            in_or_out = True
            break

    if not in_or_out:    # 距离不在范围内就新加入一个
        rect_all.append(car)
        begin_place.append(car)
        way.append(0)
        color_all.append(0)
        valid.append(1)
        color_bound.append((random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255)))
        going.append(True)
        pic_out.append(0)

        num[0] += 1

    if num[0] == 0:    # 第一辆车加入一下
        rect_all.append(car)
        begin_place.append(car)
        way.append(0)
        color_all.append(0)
        valid.append(1)
        color_bound.append((random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255)))
        going.append(True)
        pic_out.append(0)

        num[0] += 1


def record_cars(dst, frame, bounding_top, num, going, rect_all, distance_thresh, begin_place, way,
               color_all, valid, color_bound, pic_out, valid_thresh, car_cal):
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundRect = []
    for contour in contours:
        boundRect.append(cv2.boundingRect(contour))

    judge = 0
    for i, rect in enumerate(boundRect):
        if getCenterPoint(rect)[1] > bounding_top and getarea(rect) > 3000:
            change_car(rect, frame, num, going, rect_all, distance_thresh, begin_place, 
                        way, color_all, valid, color_bound, pic_out, valid_thresh, car_cal)
            judge = 1

    if judge < 1:
        for i in range(num[0]):
            going[i] = 0
