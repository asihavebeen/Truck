#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

float gauss(float mid, float diff, float goal)  //算概率的函数
{
    return std::exp(-(goal - mid)*(goal - mid)/(2*diff*diff)) / (diff * 2.5066);  //高斯分布计算公式
}

class mixture{
    public:
    int k = 4; //混合模型数
    int h, w;
    float gate = 0.9;  //背景的阈值
    float learn = 0.01;

    mixture(int high, int width);
    int change(int i, int j, cv::Mat mat, float w[4], int mid[4], int diff[4]);
};

mixture::mixture(int high, int width)
{
    h = high;
    w = width;
}

int mixture::change(int i, int j, cv::Mat mat, float w[4], int mid[4], int diff[4])
{
    float possibility;
    int point = 254;

    int decide = 0;  //判定是否背景
    for(int l=0; l<k; l++)
    {
        if(mat.at<uchar>(i,j) - mid[l] < 3*diff[l])  //更新w和均值方差
        {
            w[l] = (1 - learn) * w[l] + learn;
            //possibility =  w[l] * gauss(mid[l], diff[l], mat.at<int>(i,j));
            possibility =  learn / w[l];

            mid[l] = mid[l] * (1 - possibility) + possibility * mat.at<uchar>(i,j);
            diff[l] = std::sqrt(diff[l] * diff[l] * (1 - possibility) + possibility * (mat.at<uchar>(i,j) - mid[l]) * (mat.at<uchar>(i,j) - mid[l]));

            decide = 1;
        }
        else
        {
            w[l] = (1 - learn) * w[l];
        }
    }

    if(decide < 1)  //无匹配分布添加新分布
    {
        float min = 1;
        int rankord = 0;
        for(int l=0; l<k; l++)
        {
            if(w[l] < min)
            {
                min = w[l];
                rankord = l;
            }
        }
        w[rankord] = learn;
        mid[rankord] = mat.at<uchar>(i,j);
        diff[rankord] = 15;
    }

    for(int x=0; x<k; x++)  //交换次序
    {
        float max = 0;
        int rankord = 0;
        for(int y=x; y<k; y++)
        {
            float xx = w[y] / (diff[y] * diff[y]);
            if(xx > max)
            {
                max = xx;
                rankord = y;
            }
        }
        float middle_w = w[x];
        int middle_mid = mid[x];
        int middle_diff = diff[x];

        w[x] = w[rankord];
        mid[x] = mid[rankord];
        diff[x] = diff[rankord];

        w[rankord] = middle_w;
        mid[rankord] = middle_mid;
        diff[rankord] = middle_diff;
    }

    float all = w[0] + w[1] + w[2] + w[3];
    for(int l=0; l<k; l++)  //归一化
    {
        w[l] = w[l] / all;
    }

    int dicide_last = 0;  //给定背景排序
    float sumit = 0;
    for(int l=0; l<k; l++)
    {
        sumit = sumit + w[l];
        if(sumit > gate)
        {
            dicide_last = l;
            break;
        }
    }
    for(int l=0; l<dicide_last + 1; l++)
    {
        if(mat.at<uchar>(i,j) - mid[l] < 3*diff[l])
        {
            point = 1;
        }
    }

    return point;
}

int main()
{
    cv::VideoCapture capture;
    cv::Mat frame, frame_gray, frame_gray_mid, frame_blur;
    Mat aChannel[3];
    frame = capture.open("../video/1.mp4");
    if(!capture.isOpened())
    {
        printf("can not open ...\n");
        return -1;
    }

    capture.read(frame);

    float rate = float(frame.cols) / float(frame.rows);
    int rows_real = 540;
    int cols_real = int(540.0 * rate);

    Size dsize = Size(cols_real, rows_real); 
    resize(frame, frame, dsize, 0, 0, INTER_AREA);
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    mixture back(rows_real, cols_real);

    cv::namedWindow("output_origin", CV_WINDOW_NORMAL);
    cv::resizeWindow("output_origin", cols_real, rows_real);
    //cv::namedWindow("output_gray", CV_WINDOW_NORMAL);
    //cv::resizeWindow("output_gray", cols_real, rows_real);
    cv::namedWindow("output_diffrence", CV_WINDOW_NORMAL);
    cv::resizeWindow("output_diffrence", cols_real, rows_real);
    cv::namedWindow("output_gauss", CV_WINDOW_NORMAL);
    cv::resizeWindow("output_gauss", cols_real, rows_real);
    cv::namedWindow("output_edge", CV_WINDOW_NORMAL);
    cv::resizeWindow("output_edge", cols_real, rows_real);
    cv::namedWindow("output_diff", CV_WINDOW_NORMAL);
    cv::resizeWindow("output_diff", cols_real, rows_real);

    float w1[rows_real][cols_real][4];
    int mid1[rows_real][cols_real][4];
    int diff1[rows_real][cols_real][4];
    float w2[rows_real][cols_real][4];
    int mid2[rows_real][cols_real][4];
    int diff2[rows_real][cols_real][4];
    float w3[rows_real][cols_real][4];
    int mid3[rows_real][cols_real][4];
    int diff3[rows_real][cols_real][4];

    for(int i=0; i<rows_real; i++)
    {
        for(int j=0; j<cols_real; j++)
        {
            for(int l=0; l<4; l++)
            {
                if(l==0)
                {
                    w1[i][j][0] = 1;
                    mid1[i][j][0] = frame.at<Vec3b>(i, j)[0];
                    diff1[i][j][0] = 15;
                    w2[i][j][0] = 1;
                    mid2[i][j][0] = frame.at<Vec3b>(i, j)[1];
                    diff2[i][j][0] = 15;
                    w3[i][j][0] = 1;
                    mid3[i][j][0] = frame.at<Vec3b>(i, j)[2];
                    diff3[i][j][0] = 15;
                }
                else
                {
                    w1[i][j][l] = 0;
                    mid1[i][j][l] = 0;
                    diff1[i][j][l] = 0;
                    w2[i][j][l] = 0;
                    mid2[i][j][l] = 0;
                    diff2[i][j][l] = 0;
                    w3[i][j][l] = 0;
                    mid3[i][j][l] = 0;
                    diff3[i][j][l] = 0;
                }
            }
        }
    }

    cv::Mat image, img_binary, edges, img_dst;
    cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame, img_dst, cv::COLOR_BGR2GRAY);

    int right1, right2, right3;

    cv::Mat kernel_33 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));  //定义结构元
    cv::Mat kernel_55 = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    cv::Mat kernel_77 = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
    cv::Mat kernel_99 = getStructuringElement(MORPH_RECT, Size(9, 9), Point(-1, -1));
    cv::Mat kernel_1515 = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));

    while (capture.read(frame))
    {
        resize(frame, frame, dsize, 0, 0, INTER_AREA);
        GaussianBlur(frame, frame_blur, Size(5,5), 3, 3);
        
	split(frame_blur, aChannel);
        
        imshow("output_origin", frame);

        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);  //预处理
        medianBlur(frame_gray, frame_gray_mid, 9);
        //imshow("output_gray", frame_gray);

        Canny(frame_gray_mid, edges, 160, 190, 3, false);  //边缘提取
        //dilate(edges, edges, kernel_1515, Point(-1,-1), 1);
        //dilate(edges, edges, kernel_77, Point(-1,-1), 1);
        //erode(edges, edges, kernel_3030, Point(-1,-1), 1);
        //dilate(edges, edges, kernel_2020, Point(-1,-1), 1);
        //erode(edges, edges, kernel_3030, Point(-1,-1), 1);
        imshow("output_edge", edges);

        for(int i=0; i<rows_real; i++)
        {
            for(int j=0; j<cols_real; j++)
            {
                right1 = back.change(i, j, aChannel[0], w1[i][j], mid1[i][j], diff1[i][j]);  //高斯背景建模
                right2 = back.change(i, j, aChannel[1], w2[i][j], mid2[i][j], diff2[i][j]);  //高斯背景建模
                right3 = back.change(i, j, aChannel[2], w3[i][j], mid3[i][j], diff3[i][j]);  //高斯背景建模
                if(right1 > 127 && right2 > 127 && right3 > 127)
                {
                    image.at<uchar>(i,j) = 254;
                }
                else
                {
                    image.at<uchar>(i,j) = 1;
                }
            }
        }
        threshold(image, img_binary, 127, 255, cv::THRESH_BINARY);
        dilate(img_binary, img_binary, kernel_77, Point(-1,-1), 1);
        //erode(dst, dst, kernel_33, Point(-1,-1), 1);
        imshow("output_gauss", img_binary);

        //blur(img_binary, dst, Size(3,3), Point(-1,-1));
    	//morphologyEx(img_binary, dst, MORPH_CLOSE, kernel_3030);
        //morphologyEx(img_binary, dst, MORPH_OPEN, kernel_1515);
        //morphologyEx(img_binary, dst, MORPH_OPEN, kernel_3030);

        for(int i=0; i<rows_real; i++)
        {
            for(int j=0; j<cols_real; j++)
            {
                if(edges.at<uchar>(i,j) == 0 || img_binary.at<uchar>(i,j) == 0)
                {
                    img_dst.at<uchar>(i,j) = 1;
                }
                else
                {
                    img_dst.at<uchar>(i,j) = 254;
                }
                if(i < 80)
                {
                    img_dst.at<uchar>(i,j) = 1;
                }
            }
        }
        threshold(img_dst, img_dst, 127, 255, cv::THRESH_BINARY);
        dilate(img_dst, img_dst, kernel_1515, Point(-1,-1), 1);
        dilate(img_dst, img_dst, kernel_1515, Point(-1,-1), 1);
        dilate(img_dst, img_dst, kernel_1515, Point(-1,-1), 1);
        dilate(img_dst, img_dst, kernel_1515, Point(-1,-1), 1);
        //erode(img_dst, img_dst, kernel_1515, Point(-1,-1), 1);
        //erode(img_dst, img_dst, kernel_1515, Point(-1,-1), 1);
        //erode(img_dst, img_dst, kernel_1515, Point(-1,-1), 1);
        //erode(img_dst, img_dst, kernel_55, Point(-1,-1), 1);
        //erode(dst, dst, kernel_33, Point(-1,-1), 1);
        imshow("output_diffrence", img_dst);

        waitKey(10);
    }

    capture.release();
    return 0;
}
