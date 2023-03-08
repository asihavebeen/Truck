#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <queue>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;


#define N 8   // 背景块的划分大小
#define M 4   // 帧差块的划分大小
#define region 2  // 定义邻域的大小
#define back_refine 4*255  // 背景修正的阈值，超过认为不是背景
#define thresh_th 100  // 梯度的阈值
#define high_th 20  // 与背景差的高阈值
// #define low_th 20   // 与背景差的低阈值
#define bounding_top 170   // 检测区域的上边界
#define bounding_bottom 380   // 检测区域的下边界
#define distance_thresh 60.0    // 在这个距离以内的中心点被认为是同一辆车
#define distance_thresh_light 60.0    // 在这个距离以内的中心点被认为是同一个圆
#define valid_thresh 10    // 持续15帧以上的才计入车
#define valid_thresh_dark 20    // 持续5帧以上的才计入车,黑夜
#define dark_thresh 50     // 平均像素灰度达到此值认为是晚上
#define line_thresh 3     // 小于这个值认为在同一水平线


RNG rng(12345);    // 随机数生成器，全局变量，产生随机颜色

int way[40];    // 记录方向，0为下行，1为上行
int color_all[40];     // 记录颜色，1-7
Scalar color_bound[40];      // 框的颜色，画图使用
Rect rect_all[40];     // 矩形类，记录检测出的矩形框
Point point_all[100];     // 点类，记录检测出的圆
float radius_all[100];     // 记录检测出的半径
float begin_radius[100];     // 记录起始半径，用于判断方向和大小型车
Rect begin_place[40];     // 记录起始位置，用于判断方向和大小型车
Point begin_point[100];      // 记录车灯的起始位置，用于判断方向和大小型车
int valid[100];    // 记录识别是否有效，大于15帧才算有效
int valid_dark_car[40];     // // 记录识别是否有效，大于5帧才算有效
int going[40];    // 记录汽车是否已脱离区域
int going_dark[100];    // 记录车灯是否已脱离区域
int pic_out[40];      // 记录是否已经输出图像，输出过的不再输出
int num = 0;      // 记录识别出的总车数，但不全都有效
int num_dark = 0;      // 记录识别出的黑暗总车数，全都有效
int car_cal = 0;      // 统计实际有效车数
int dark = 0;     // 记录是否天黑
Rect circle_rect[40];     // 由有效车灯形成的矩形
Rect begin_circle_rect[40];

Mat kernel_33 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));      // 定义结构元
Mat kernel_55 = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
Mat kernel_77 = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
Mat kernel_99 = getStructuringElement(MORPH_RECT, Size(9, 9), Point(-1, -1));
Mat kernel_1111 = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
Mat kernel_1515 = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));


Mat cal_grad(Mat srcGray)    // 计算梯度，返回二值化的二值图
{
    Mat gradX, gradY;
    Sobel(srcGray, gradX, CV_16S, 1, 0, 3);
    Sobel(srcGray, gradY, CV_16S, 0, 1, 3);
    // Scharr(srcGray, gradX, CV_16S, 1, 0);
    // Scharr(srcGray, gradY, CV_16S, 0, 1);
    convertScaleAbs(gradX, gradX);  // 计算绝对值
    convertScaleAbs(gradY, gradY);

    Mat dst;
    addWeighted(gradX, 0.5, gradY, 0.5, 0, dst);

    cv::threshold(dst, dst, thresh_th, 255, cv::THRESH_BINARY);
    
    return dst;
}

bool isInside(Rect rect1, Rect rect2)
{
    return (rect1 == (rect1&rect2));
}

Mat build_backgroud()     // 建立背景的函数
{
    int count = 0;
    VideoCapture back_capture;
    Mat back_frame, back_frame_gray, backgroud;
    back_frame = back_capture.open("../video/1.mp4");
    Size dsize = Size(960, 544);

    if(!back_capture.isOpened())
    {
        printf("can not open ...\n");
        return back_frame;
    }

    back_capture.read(back_frame);
    resize(back_frame, back_frame, dsize, 0, 0, INTER_AREA);
    // GaussianBlur(back_frame, back_frame, Size(5,5), 3, 3);
    cv::cvtColor(back_frame, back_frame_gray, cv::COLOR_BGR2GRAY);  // 预处理
    backgroud = back_frame_gray.clone();

    while(back_capture.read(back_frame))
    {
        resize(back_frame, back_frame, dsize, 0, 0, INTER_AREA);

        // GaussianBlur(back_frame, back_frame, Size(5,5), 3, 3);

        cv::cvtColor(back_frame, back_frame_gray, cv::COLOR_BGR2GRAY);  // 预处理

        // backgroud = (backgroud * (count + 1) + back_frame_gray) / (count + 2);
        addWeighted(backgroud, float(count + 1) / float(count + 2), back_frame_gray, 1.0 / float(count + 2), 0, backgroud);

        count = count + 1;
    }

    back_capture.release();

    cv::imwrite("../Background.jpg", backgroud);

    return backgroud;
}

Mat valid_backgroud()      // 背景局部更新的函数
{
    VideoCapture valid_capture1, valid_capture2;
    Mat valid_frame1, valid_frame2, valid_frame_gray, valid_backgroud, temp;
    valid_frame1 = valid_capture1.open("../video/1.mp4");
    valid_frame2 = valid_capture2.open("../video/2.mp4");
    Size dsize = Size(960, 544); 

    if(!valid_capture1.isOpened())
    {
        printf("can not open ...\n");
        return valid_frame1;
    }

    valid_capture1.read(valid_frame1);
    resize(valid_frame1, valid_frame1, dsize, 0, 0, INTER_AREA);
    GaussianBlur(valid_frame1, valid_frame1, Size(5,5), 3, 3);
    cv::cvtColor(valid_frame1, valid_frame_gray, cv::COLOR_BGR2GRAY);  // 预处理
    valid_capture2.read(valid_frame2);
    resize(valid_frame2, valid_frame2, dsize, 0, 0, INTER_AREA);
    GaussianBlur(valid_frame2, valid_frame2, Size(5,5), 3, 3);
    valid_backgroud = build_backgroud().clone();

    int cols_real = valid_frame1.cols/N;
    int rows_real = valid_frame1.rows/N;

    int back_record[rows_real][cols_real];
    int frame_record[rows_real][cols_real];
    int count_record[rows_real][cols_real];

    for(int i=0; i<rows_real; i++)
    {
        for(int j=0; j<cols_real; j++)
        {
            frame_record[i][j] = 64 * 255;
        }
    }

    temp = cal_grad(valid_backgroud).clone();

    for(int i=0; i<rows_real; i++)
    {
        for(int j=0; j<cols_real; j++)
        {
            for(int ii=0; ii<N; ii++)
            {
                for(int jj=0; jj<N; jj++)
                {
                    back_record[i][j] += temp.at<uchar>(N*i + ii, N*j + jj);
                }
            }
        }
    }

    int mid = 0;
    int count = 1;
    int record_i, record_j;

    while(valid_capture1.read(valid_frame1))
    {
        resize(valid_frame1, valid_frame1, dsize, 0, 0, INTER_AREA);

        GaussianBlur(valid_frame1, valid_frame1, Size(5,5), 3, 3);

        cv::cvtColor(valid_frame1, valid_frame_gray, cv::COLOR_BGR2GRAY);  // 预处理

        for(int i=0; i<rows_real; i++)
        {
            for(int j=0; j<cols_real; j++)
            {
                mid = 0;
                record_i = N*i;
                record_j = N*j;

                temp = cal_grad(valid_frame_gray).clone();

                for(int ii=0; ii<N; ii++)
                {
                    for(int jj=0; jj<N; jj++)
                    {
                        mid += temp.at<uchar>(record_i + ii, record_j + jj);
                    }
                }

                mid = abs(mid - back_record[i][j]);

                if(mid < back_refine && mid < frame_record[i][j])
                {
                    frame_record[i][j] = mid;
                    count_record[i][j] = count;
                }
            }
        }

        count = count + 1;

        if(count > 70)
        {
            break;
        }
    }

    count = 1;

    while(valid_capture2.read(valid_frame2))
    {
        resize(valid_frame2, valid_frame2, dsize, 0, 0, INTER_AREA);

        GaussianBlur(valid_frame2, valid_frame2, Size(5,5), 3, 3);

        cv::cvtColor(valid_frame2, valid_frame_gray, cv::COLOR_BGR2GRAY);  // 预处理

        for(int i=0; i<rows_real; i++)
        {
            for(int j=0; j<cols_real; j++)
            {
                if(count_record[i][j] == count)
                {
                    for(int ii=0; ii<N; ii++)
                    {
                        for(int jj=0; jj<N; jj++)
                        {
                            valid_backgroud.at<uchar>(N*i + ii, N*j + jj) = 
                            valid_frame_gray.at<uchar>(N*i + ii, N*j + jj);
                        }
                    }
                }
            }
        }

        count = count + 1;
    }

    valid_capture1.release();
    valid_capture2.release();

    cv::imwrite("../115.jpg", valid_backgroud);

    return valid_backgroud;
}

Point getCenterPoint(Rect rect)     // 获取中心点
{
    Point cpt;

    cpt.x = rect.x + cvRound(rect.width/2.0);
    cpt.y = rect.y + cvRound(rect.height/2.0);

    return cpt;
}

float getDistance(Point pointO, Point pointA)     // 获取距离
{
    float distance;
    distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
    distance = sqrtf(distance);
    return distance;
}

Mat find_rec(Mat frame)
{
    Mat drawing = frame.clone();
 
	/// 画包围的矩形框
	for (int i = 0; i< num; i++)
	{
		// drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
        if((valid[i] > valid_thresh) && (going[i] > 0))
        {
            rectangle(drawing, rect_all[i].tl(), rect_all[i].br(), color_bound[i], 2, 8, 0);
        }
	}

    return drawing;
}

Mat find_rec_dark(Mat frame)
{
    Mat drawing = frame.clone();
 
	/// 画包围的矩形框
	for (int i = 0; i< num_dark; i++)
	{
		// drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
        if(valid_dark_car[i] > valid_thresh_dark && going[i] > 0)
        {
            rectangle(drawing, circle_rect[i].tl(), circle_rect[i].br(), color_bound[i], 2, 8, 0);
	    }
    }

    return drawing;
}

int judgeColor(Mat frame)    // 每次读取图像都调用，因为要截取车辆的图片
{
    Mat clone_one;
    for (int i = 0; i < num; i++)
    {
        string destiny = "../saved/";
        string count = to_string(car_cal);
        string destiny_back = ".jpg";

        destiny += count;
        destiny += destiny_back;

        if(valid[i] == valid_thresh && pic_out[i] < 1)
        {
            clone_one = frame(rect_all[i]);
            cv::imwrite(destiny, clone_one);

            car_cal += 1;
            pic_out[i] = 1;
        }
    }
    
    return 0;
}

int judgeArea(int i)     // 整个识别完再判断
{
    if(way[i] < 1)
    {   // 下行看结束大小
        if(rect_all[i].area() > 30000)    // 大型车
        {
            return 2;
        }
        else
        {
            if(rect_all[i].area() > 7000)    // 中型车
            {
                return 1;
            }
            else
            {
                return 0;     // 小型车
            }
        }
    }
    else
    {   // 上行看初始大小
        if(begin_place[i].area() > 30000)    // 大型车
        {
            return 2;
        }
        else
        {
            if(begin_place[i].area() > 7000)    // 中型车
            {
                return 1;
            }
            else
            {
                return 0;     // 小型车
            }
        }
    }
}

int judgeArea_dark(int i)     // 整个识别完再判断
{
    if(way[i] < 1)
    {   // 下行看结束大小
        if(circle_rect[i].area() > 30000)    // 大型车
        {
            return 2;
        }
        else
        {
            if(circle_rect[i].area() > 7000)    // 中型车
            {
                return 1;
            }
            else
            {
                return 0;     // 小型车
            }
        }
    }
    else
    {   // 上行看初始大小
        if(begin_circle_rect[i].area() > 30000)    // 大型车
        {
            return 2;
        }
        else
        {
            if(begin_circle_rect[i].area() > 7000)    // 中型车
            {
                return 1;
            }
            else
            {
                return 0;     // 小型车
            }
        }
    }
}

int read_color(int place)     // 也是整个识别完，再读入存的图片做判断
{
    Mat read_one, HSVMat;

    string destiny = "../saved/";
    string count = to_string(place);
    string destiny_back = ".jpg";

    destiny += count;
    destiny += destiny_back;

    read_one = cv::imread(destiny);
    cv::cvtColor(read_one, HSVMat, COLOR_BGR2HSV);
    vector<Mat> planes;

    split(HSVMat, planes);

    int color_five[5];    // 按顺序分别是黑、白、黄、红、蓝
    for (int i = 0; i < 5; i++)
    {
        color_five[i] = 0;
    }

    for (int i = 0; i < read_one.rows; i++)
    {
        for (int j = 0; j < read_one.cols; j++)
        {
            if(planes[2].at<uchar>(i,j) < 30)
            {
                color_five[0] += 1;    // 黑
            }
            else
            {
                if(planes[1].at<uchar>(i,j) < 43)
                {
                    if(planes[2].at<uchar>(i,j) < 200)
                    {
                        continue;
                    }
                    else
                    {
                        color_five[1] += 1;    // 白
                    }
                }
                else
                {
                    if(planes[0].at<uchar>(i,j) > 100 && planes[0].at<uchar>(i,j) < 124)
                    {
                        color_five[4] += 1;    // 蓝
                    }
                    else
                    {
                        if(planes[0].at<uchar>(i,j) > 3 && planes[0].at<uchar>(i,j) < 50)
                        {
                            color_five[2] += 1;    // 黄
                        }
                        if(planes[0].at<uchar>(i,j) > 160 || planes[0].at<uchar>(i,j) < 3)
                        {
                            color_five[3] += 1;    // 红
                        }
                    }
                }
            }
        }
    }

    int max_record = 0, max_out;
    for (int i = 0; i < 5; i++)
    {
        if(color_five[i] > max_record)
        {
            max_record = color_five[i];
            max_out = i;
        }
    }
    
    return max_out;
}

void cout_csv()      // 输出表格的函数
{
    int csv_cal = 0;
    ofstream p;

    p.open("../saved/output.csv",ios::out|ios::trunc);    //打开文件路径
    p.trunc;
    p<<"序号"<<","<<"车型"<<","<<"方向"<<","<<"颜色"<<endl;    //输入内容，注意要用逗号，隔开

    for (int i = 0; i < num; i++)
    {
        if(valid[i] < valid_thresh)
        {
            continue;
        }

        string count = to_string(csv_cal);
        p<<count<<",";

        switch(judgeArea(i)){
            case 0 :
                p<<"小型车"<<",";
                break;
            case 1 :
                p<<"中型车"<<",";
                break;

            default :
                p<<"大型车"<<",";
        }

        if(way[i] < 1)
        {
            p<<"下行"<<",";
        }
        else
        {
            p<<"上行"<<",";
        }

        switch(read_color(csv_cal)){
            case 0 :
                p<<"黑色"<<endl;
                break;
            case 1 :
                p<<"白色"<<endl;
                break;
            case 2 :
                p<<"黄色"<<endl;
                break;
            case 3 :
                p<<"红色"<<endl;
                break;
            case 4 :
                p<<"蓝色"<<endl;
                break;

            default :
                p<<"彩色"<<endl;
        }

        csv_cal = csv_cal + 1;
    }

    p.close();
}

void cout_csv_dark()      // 输出表格的函数
{
    int csv_cal = 0;
    ofstream p;

    p.open("../saved/output_dark.csv",ios::out|ios::trunc);    //打开文件路径
    p.trunc;
    p<<"序号"<<","<<"车型"<<","<<"方向"<<","<<"颜色"<<endl;    //输入内容，注意要用逗号，隔开

    for (int i = 0; i < num_dark; i++)
    {
        string count = to_string(csv_cal);
        p<<count<<",";

        switch(judgeArea_dark(i)){
            case 0 :
                p<<"小型车"<<",";
                break;
            case 1 :
                p<<"中型车"<<",";
                break;

            default :
                p<<"大型车"<<",";
        }

        if(way[i] < 1)
        {
            p<<"下行"<<",";
        }
        else
        {
            p<<"上行"<<",";
        }

        p<<"不统计"<<endl;

        csv_cal = csv_cal + 1;
    }

    p.close();
}

void change_car(Rect car,Mat frame)      // 修改记录数据的部分，颜色、矩形框等的数据
{
    bool in_or_out = 0;

    for(int i = 0; i < num; i++)
    {
        going[i] = 0;
        if(getDistance(getCenterPoint(car), getCenterPoint(rect_all[i])) < distance_thresh)    // 距离在范围内就更新，也就是同一辆车
        {
            rect_all[i] = car;
            if((getCenterPoint(car).y - getCenterPoint(begin_place[i]).y) > 0)
            {
                way[i] = 0;
            }
            else
            {
                way[i] = 1;
            }
            color_all[i] = judgeColor(frame);
            valid[i] = valid[i] + 1;
            going[i] = 1;

            in_or_out = 1;
            break;
        }
    }

    if(!in_or_out)    // 距离不在范围内就新加入一个
    {
        rect_all[num] = car;
        begin_place[num] = car;
        way[num] = 0;
        color_all[num] = 0;
        valid[num] = 1;
        color_bound[num] = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        going[num] = 1;
        pic_out[num] = 0;

        num += 1;
    }

    if(num == 0)    // 第一辆车加入一下
    {
        rect_all[0] = car;
        begin_place[0] = car;
        way[0] = 0;
        color_all[0] = 0;
        valid[0] = 1;
        color_bound[0] = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        going[0] = 1;
        pic_out[0] = 0;

        num += 1;
    }
}

void change_dark(Rect car)      // 修改记录数据的部分，颜色、矩形框等的数据
{
    bool in_or_out = 0;

    for(int i = 0; i < num_dark; i++)
    {
        going[i] = 0;
        if(getDistance(getCenterPoint(car), getCenterPoint(circle_rect[i])) < distance_thresh)    // 距离在范围内就更新，也就是同一辆车
        {
            circle_rect[i] = car;
            if((getCenterPoint(car).y - getCenterPoint(begin_circle_rect[i]).y) > 0)
            {
                way[i] = 0;
            }
            else
            {
                way[i] = 1;
            }
            going[i] = 1;
            valid_dark_car[i] += 1;

            in_or_out = 1;
            break;
        }
    }

    if(!in_or_out)    // 距离不在范围内就新加入一个
    {
        circle_rect[num_dark] = car;
        begin_circle_rect[num_dark] = car;
        way[num_dark] = 0;
        color_all[num_dark] = 0;
        valid_dark_car[num_dark] = 1;
        color_bound[num_dark] = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        going[num_dark] = 1;
        pic_out[num_dark] = 0;

        num_dark += 1;
    }

    if(num_dark == 0)    // 第一辆车加入一下
    {
        circle_rect[0] = car;
        begin_circle_rect[0] = car;
        way[0] = 0;
        color_all[0] = 0;
        valid_dark_car[0] = 1;
        color_bound[0] = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        going[0] = 1;
        pic_out[0] = 0;

        num_dark += 1;
    }
}

void change_car_dark(Point car,float radius_num)      // 修改记录数据的部分，颜色、矩形框等的数据
{
    bool in_or_out = 0;

    for(int i = 0; i < num; i++)
    {
        going_dark[i] = 0;
        if(getDistance(car, point_all[i]) < distance_thresh_light)    // 距离在范围内就更新，也就是同一辆车
        {
            point_all[i] = car;
            radius_all[i] = radius_num;
            valid[i] = valid[i] + 1;
            going_dark[i] = 1;

            in_or_out = 1;
            break;
        }
    }

    if(!in_or_out)    // 距离不在范围内就新加入一个
    {
        point_all[num] = car;
        radius_all[num] = radius_num;
        begin_point[num] = car;
        valid[num] = 1;
        going_dark[num] = 1;

        num += 1;
    }

    if(num == 0)    // 第一辆车加入一下
    {
        point_all[0] = car;
        radius_all[0] = radius_num;
        begin_point[0] = car;
        valid[0] = 1;
        going_dark[0] = 1;

        num += 1;
    }
}

void record_cars(Mat dst, Mat frame)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
 
	/// 找到轮廓
	findContours(dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    /// 获取矩形边界框
	vector<Rect> boundRect(contours.size());
 
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(Mat(contours[i]));
	}

    int judge = 0;
    for (int i = 0; i< contours.size(); i++)
	{
        if(getCenterPoint(boundRect[i]).y > bounding_top && boundRect[i].area() > 3000)    // 在范围内就进行判断
        {
            change_car(boundRect[i], frame);     // 判断
            judge = 1;
        }
	}
    if(judge < 1)
    {
        for(int i = 0; i < num; i++)
        {
            going[i] = 0;
        }
    }
}

void record_cars_dark(Mat frame, Mat frame_gray, Mat thresh_dst)
{
    Mat dst_dark = frame_gray.clone();

    int to_average = 0;

    for(int i = 0; i < frame_gray.rows; i++)
    {
        for(int j = 0; j < frame_gray.cols; j++)
        {
            to_average += frame_gray.at<uchar>(i, j);
        }
    }
    to_average = to_average / (frame_gray.cols * frame_gray.rows);

    for(int i=0; i < frame_gray.rows; i++)     // 统计变化较大的像素
    {
        for(int j=0; j < frame_gray.cols; j++)
        {
            if(frame_gray.at<uchar>(i, j)>= 140)
            {    
                dst_dark.at<uchar>(i, j) = 250;
            }
            else
            {
                dst_dark.at<uchar>(i, j) = 5;
            }
        }
    }
    cv::threshold(dst_dark, dst_dark, 128, 255, cv::THRESH_BINARY);

    // cv::dilate(dst_dark, dst_dark, kernel_77, Point(-1,-1), 1);
    // cv::dilate(dst_dark, dst_dark, kernel_99, Point(-1,-1), 1);
    // cv::dilate(dst_dark, dst_dark, kernel_77, Point(-1,-1), 1);
    // cv::erode(dst_dark, dst_dark, kernel_77, Point(-1,-1), 1);
    // cv::erode(dst_dark, dst_dark, kernel_77, Point(-1,-1), 1);

    cv::dilate(dst_dark, dst_dark, kernel_33, Point(-1,-1), 1);
    cv::dilate(dst_dark, dst_dark, kernel_33, Point(-1,-1), 1);
    cv::dilate(dst_dark, dst_dark, kernel_33, Point(-1,-1), 1);
    cv::dilate(dst_dark, dst_dark, kernel_33, Point(-1,-1), 1);
    cv::erode(dst_dark, dst_dark, kernel_55, Point(-1,-1), 1);

    // cv::bitwise_and(dst_dark, thresh_dst, dst_dark);

    vector<vector<Point> > contours1;
	vector<Vec4i> hierarchy1;
    /// 找到轮廓
	findContours(thresh_dst, contours1, hierarchy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    /// 获取矩形边界框
	vector<Rect> boundRect(contours1.size());
    Mat dst_dark_record = Mat::zeros(dst_dark.rows, dst_dark.cols, CV_8UC1);
 
	for (int i = 0; i < contours1.size(); i++)
	{
		boundRect[i] = boundingRect(Mat(contours1[i]));
	}

    for (int i = 0; i< contours1.size(); i++)
	{
        for (int x = 0; x < boundRect[i].height; x++)
        {
            for (int y = 0; y < boundRect[i].width; y++)
            {
                dst_dark_record.at<uchar>(int(boundRect[i].y + x), int(boundRect[i].x + y)) = 255;
            }
        }
    }
    cv::bitwise_and(dst_dark_record, dst_dark, dst_dark);

    cv::imwrite("../dst.jpg", dst_dark);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
 
	/// 找到轮廓
	findContours(dst_dark, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    /// 多边形逼近轮廓 + 获取矩形边界框
	vector<Point2f> center(contours.size());
	vector<float> radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		minEnclosingCircle(contours[i], center[i], radius[i]);
	}

	int judge = 0;
    for (int i = 0; i< center.size(); i++)
	{
        if(center[i].y > bounding_top && center[i].y < bounding_bottom + 1000 && radius[i] < 30)    // 在范围内就进行判断
        {
            change_car_dark(center[i], radius[i]);     // 判断
            judge = 1;
        }
	}
    if(judge < 1)
    {
        for(int i = 0; i < num; i++)
        {
            going_dark[i] = 0;
        }
    }
}

int is_dark(Mat frame_gray)
{
    int many = 0;
    for(int i = 0; i < frame_gray.rows; i++)
    {
        for(int j = 0; j < frame_gray.cols; j++)
        {
            many += frame_gray.at<uchar>(i, j);
        }
    }
    many = many / (frame_gray.rows * frame_gray.cols);

    if(many < dark_thresh)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void circle_to_rect(Mat frame)
{
    vector<Point> point_record;
    vector<float> radius_record;
    vector<Point> point_last;
    vector<float> radius_last;

    for (int i = 0; i < num; i++)
    {
        if(valid[i] >= valid_thresh)
        {
            point_record.push_back(point_all[i]);
            radius_record.push_back(radius_all[i]);
        }
    }
    
    Rect rect_build;
    vector<int> element_record;
    vector<vector<int>> element_in;
    vector<int> temp;
    int judge_continue = 0;
    float max_record;
    int place_record;

    for (int i = 0; i < point_record.size(); i++)
	{
        judge_continue = 0;
        max_record = 1000;

        for (int l = 0; l < element_record.size(); l++)
        {
            if(element_record[l] == i)
            {
                judge_continue = 1;
            }
        }
        if(judge_continue > 0)
        {
            continue;
        }
        
        for (int j = 0; j < point_record.size(); j++)
        {
            if(i == j)
            {
                continue;
            }
            if(abs(point_record[i].y - point_record[j].y) < max_record)
            {
                max_record = abs(point_record[i].y - point_record[j].y);
                place_record = j;
            }
        }
        if(abs(point_record[i].y - point_record[place_record].y) < line_thresh 
        && getDistance(point_record[i], point_record[place_record]) > 45
        && getDistance(point_record[i], point_record[place_record]) < 200)
        {
            point_last.push_back(point_record[i]);
            radius_last.push_back(radius_record[i]);
            point_last.push_back(point_record[place_record]);
            radius_last.push_back(radius_record[place_record]);

            element_record.push_back(place_record);
        }
	}

    Mat drawing = frame.clone();
	for (int i = 0; i< point_last.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		circle(drawing, point_last[i], (int)radius_last[i], color, 2, 8, 0);
	}
    cv::imwrite("../circle.jpg", drawing);

    std::cout<<point_last.size()<<endl;

    int judge = 0;
    for (int i = 0; i < point_last.size(); i = i + 2)
    {
        if(radius_last[i] > radius_last[i + 1])
        {
            if(point_last[i].x < point_last[i + 1].x)
            {
                rect_build = Rect(int(point_last[i].x - radius_last[i]), int(point_last[i].y - radius_last[i]),
                    int(abs(point_last[i + 1].x - point_last[i].x) + 2 * radius_last[i]), int(2 * radius_last[i]));
            }
            else
            {
                rect_build = Rect(int(point_last[i + 1].x - radius_last[i]), int(point_last[i + 1].y - radius_last[i]),
                    int(abs(point_last[i].x - point_last[i + 1].x) + 2 * radius_last[i]), int(2 * radius_last[i]));
            }
        }
        else
        {
            if(point_last[i].x < point_last[i + 1].x)
            {
                rect_build = Rect(int(point_last[i].x - radius_last[i + 1]), int(point_last[i].y - radius_last[i + 1]),
                    int(abs(point_last[i + 1].x - point_last[i].x) + 2 * radius_last[i + 1]), int(2 * radius_last[i + 1]));
            }
            else
            {
                rect_build = Rect(int(point_last[i + 1].x - radius_last[i + 1]), int(point_last[i + 1].y - radius_last[i + 1]),
                    int(abs(point_last[i].x - point_last[i + 1].x) + 2 * radius_last[i + 1]), int(2 * radius_last[i + 1]));
            }
        }
        change_dark(rect_build);
        judge = 1;
    }
}

int main()
{
    VideoCapture main_capture;
    Mat frame, frame_gray, backgroud_truth, dst, see;
    Size dsize = Size(960, 544);

    backgroud_truth = build_backgroud().clone();
    std::cout << "BackGround Done!" << endl;

    frame = main_capture.open("../video/01.mp4");
    if(!main_capture.isOpened())
    {
        printf("can not open ...\n");
        return -1;
    }

    main_capture.read(frame);
    resize(frame, frame, dsize, 0, 0, INTER_AREA);

    // float rate = float(frame.cols) / float(frame.rows);
    // int rows_real = 540;
    // int cols_real = int(540.0 * rate);

    // Size dsize = Size(cols_real, rows_real); 
    // resize(frame, frame, dsize, 0, 0, INTER_AREA);

    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);         // 灰度图
    cv::cvtColor(frame, dst, cv::COLOR_BGR2GRAY);         // 灰度图
    Mat dst_mid(frame.rows / 2, frame.cols / 2, CV_8U, 100);             // 目标检测结果
    Mat record = frame_gray.clone();                         // 记录上一帧

    cv::namedWindow("output_origin", CV_WINDOW_NORMAL);
    cv::resizeWindow("output_origin", 960, 544);
    cv::namedWindow("output_gray", CV_WINDOW_NORMAL);
    cv::resizeWindow("output_gray", 960, 544);
    cv::namedWindow("output_see", CV_WINDOW_NORMAL);
    cv::resizeWindow("output_see", 960, 544);
    cv::namedWindow("output_dst", CV_WINDOW_NORMAL);
    cv::resizeWindow("output_dst", 960, 544);

    int cols_real = frame.cols/M;
    int rows_real = frame.rows/M;

    int thresh_all = 0;
    int record_i = 0;
    int record_j = 0;
    // int min = 1000;

    int diff_record[rows_real][cols_real];
    for (int i = 0; i < rows_real; i++)
    {
        for (int j = 0; j < cols_real; j++)
        {
            diff_record[i][j] = 0;
        }
        
    }   

    while (main_capture.read(frame))
    {
        resize(frame, frame, dsize, 0, 0, INTER_AREA);

        // GaussianBlur(frame, frame, Size(5,5), 3, 3);
        cv::imshow("output_origin", frame);

        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);  // 预处理
        cv::imshow("output_gray", frame_gray);

        dark = is_dark(frame_gray);

        thresh_all = 0;
        for (int i = 0; i < rows_real; i++)    // 记录差值的
        {
            for (int j = 0; j < cols_real; j++)
            {
                diff_record[i][j] = 0;
            }
        }

        for(int i = 0; i < rows_real; i++)
        {
            for(int j = 0; j < cols_real; j++)
            {
                for(int ii = 0; ii < M; ii++)
                {
                    for(int jj = 0; jj < M; jj++)
                    {
                        // cout<<abs(record.at<uchar>(N*i + ii, N*j + jj) - frame_gray.at<uchar>(N*i + ii, N*j + jj))<<endl;
                        diff_record[i][j] += abs(record.at<uchar>(N*i + ii, N*j + jj) - frame_gray.at<uchar>(N*i + ii, N*j + jj));
                    }
                }
                thresh_all += diff_record[i][j];
            }
        }
        thresh_all = thresh_all / (cols_real * rows_real);

        for(int i=0; i<rows_real; i++)     // 统计变化较大的像素
        {
            for(int j=0; j<cols_real; j++)
            {
                record_i = M*i;
                record_j = M*j;

                if(diff_record[i][j] >= thresh_all)
                {    
                    for(int ii = 0; ii < M; ii++)
                    {
                        for(int jj = 0; jj < M; jj++)
                        {
                            dst.at<uchar>(record_i + ii, record_j + jj) = 250;
                        }
                    } 
                    // for(int ii = 0; ii < M; ii++)     // 与背景的区别
                    // {
                    //     for(int jj = 0; jj < M; jj++)
                    //     {
                    //         // min = 1000;
                    //         if(abs(backgroud_truth.at<uchar>(record_i + ii, record_j + jj) - frame_gray.at<uchar>(record_i + ii, record_j + jj)) > high_th)
                    //         {
                    //             dst.at<uchar>(record_i + ii, record_j + jj) = 254;
                    //         }
                    //         else
                    //         {
                    //             dst.at<uchar>(record_i + ii, record_j + jj) = 1;
                    //         }  
                    //     }
                    // }    
                }
                else
                {
                    for(int ii = 0; ii < M; ii++)
                    {
                        for(int jj = 0; jj < M; jj++)
                        {
                            dst.at<uchar>(record_i + ii, record_j + jj) = 5;
                        }
                    } 
                }
            }
        }

        cv::threshold(dst, dst, 128, 255, cv::THRESH_BINARY);
        for(int i = 0; i < 2 * rows_real; i++)
        {
            for(int j = 0; j < 2 * cols_real; j++)
            {
                dst_mid.at<uchar>(i, j) = dst.at<uchar>(i, j);
            }
        }
        resize(dst_mid, dst, dsize, 0, 0, INTER_AREA);

        cv::dilate(dst, dst, kernel_1515, Point(-1,-1), 1);
        cv::dilate(dst, dst, kernel_1515, Point(-1,-1), 1);
        cv::dilate(dst, dst, kernel_1515, Point(-1,-1), 1);
        cv::erode(dst, dst, kernel_77, Point(-1,-1), 1);
        cv::erode(dst, dst, kernel_77, Point(-1,-1), 1);
        cv::erode(dst, dst, kernel_77, Point(-1,-1), 1);
        cv::erode(dst, dst, kernel_1515, Point(-1,-1), 1);

        if(dark < 1)
        {
            for (int i = 0; i < dst.rows; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    dst.at<uchar>(i, 430 + j) = 0;
                }
            }
        }
        
        if(dark < 1)
        {
            record_cars(dst, frame);
        }
        else
        {
            record_cars_dark(frame, frame_gray, dst);
            circle_to_rect(frame);
        }

	    cv::imshow("output_dst", dst);

        if(dark < 1)
        {
            see = find_rec(frame).clone();
        }
        else
        {
            see = find_rec_dark(frame).clone();
        }
        
        cv::imshow("output_see", see);

        record = frame_gray.clone();
        thresh_all = 0;

        cv::waitKey(100);
    }

    main_capture.release();

    if(dark < 1)
    {
        cout_csv();
    }
    else
    {
        cout_csv_dark();
    }

    return 0;
}


// Mat find_rec(Mat frame, Mat dst)
// {
//     Mat threshold_output;
// 	vector<vector<Point> > contours;
// 	vector<Vec4i> hierarchy;
 
// 	/// 使用Threshold检测边缘
// 	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
// 	/// 找到轮廓
// 	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
 
// 	/// 多边形逼近轮廓 + 获取矩形和圆形边界框
// 	vector<vector<Point> > contours_poly(contours.size());
// 	vector<Rect> boundRect(contours.size());
// 	vector<Point2f>center(contours.size());
// 	vector<float>radius(contours.size());
 
// 	//for (int i = 0; i < contours.size(); i++)
// 	//{
// 	//	approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
// 	//	boundRect[i] = boundingRect(Mat(contours_poly[i]));
// 	//	minEnclosingCircle(contours_poly[i], center[i], radius[i]);
// 	//}
// 	for (int i = 0; i < contours.size(); i++)
// 	{
// 		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
// 		boundRect[i] = boundingRect(Mat(contours[i]));
// 		minEnclosingCircle(contours[i], center[i], radius[i]);
// 	}
 
// 	/// 画多边形轮廓 + 包围的矩形框 + 圆形框
// 	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
// 	for (int i = 0; i< contours.size(); i++)
// 	{
// 		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
// 		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
// 		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
// 		circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
// 	}
 
// 	/// 显示在一个窗口
// 	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
// 	imshow("Contours", drawing);
// }

