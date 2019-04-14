#include	<iostream>
#include	<assert.h>
#include	<fstream>
#include	<opencv2/core.hpp>
#include	<opencv2/core/utility.hpp>
#include	<opencv2/videoio.hpp>
#include	<opencv2/video/video.hpp>
#include	<opencv2/highgui.hpp>
#include	<opencv2/imgproc.hpp>
#include	<opencv2/ml/ml.hpp>
#include	<opencv2/opencv.hpp>
//#include	<opencv/ml.h>////opencv version==310

using namespace cv;
using namespace cv::ml;
using namespace std;
const int DISSAPPEAR_TIME_LIMIT = 200;
//Size	WinSize = Size(50, 50),
//BlockSize = Size(20, 20),
//BlockStride = Size(10, 10),
//CellSize = Size(10, 10);
//int		Nbins = 9;

 
Mat SHOW_Label(Mat choice_label, string s1,int length = 560);
void recall_for_ready(int event, int x, int y, int flags, void* userdata);
Rect readtxt(string file, bool& thresh_flag);
void MovingAverage(Mat &A, int N);
vector<int> find_wavetop(Mat A);
Mat Thresh_Process_Func(const Mat  ); 
int HOG_predict(const Mat  );

bool break_ = false;
uint show_frame_flag = 0;
int dissappear_time = 0;

int main()
{
	//	clock_t start, end;
	//	start = clock();
	//	int framecnt = 0;
	bool thresh_flag;
	Rect ROI = readtxt("location.txt", thresh_flag);

	cout << ROI << endl;
	VideoCapture video;
	video.open(0);
	if (!video.isOpened())
	{
		cout << "can't open video" << endl; 
		return -1;
	}
	 


	Mat choice_label = Mat::zeros(Size(80, 40), CV_8UC3),show1_label;
	rectangle(choice_label, Rect(1, 1, 79, 39), Scalar(0, 255, 255),2);
	putText(choice_label, "QUIT", Point(5, 30), FONT_HERSHEY_COMPLEX, 0.95, Scalar(0, 255, 255), 2);
	
	 

	Mat frame;


	namedWindow("frame");
	setMouseCallback("frame", recall_for_ready);
	 
	while (true)
	{
		video >> frame;
		assert(!frame.empty());
		Mat img = frame(ROI).clone();
		if (thresh_flag) {
			Mat img_thresh= Thresh_Process_Func(  img );

			cv::resize(img_thresh, img_thresh, Size(32, 32));
			imshow("img_thresh", img_thresh);
			
			show1_label = SHOW_Label(choice_label, "test predict: " + to_string(HOG_predict(img_thresh)), frame.cols - 80);
			cout << "test:" << HOG_predict(img_thresh) << endl;
		}
		else {
			cvtColor(img, img, COLOR_BGR2GRAY); 
			cv::resize(img, img, Size(32, 32)); 
			show1_label = SHOW_Label(choice_label, "test predict: " + to_string(HOG_predict(img)), frame.cols - 80);
			cout << "test:" << HOG_predict(img) << endl;
		}
		if (show_frame_flag == 0)
		{
			rectangle(frame, ROI, Scalar(0, 255, 0), 1);
			vconcat(show1_label, frame, show1_label);
		}
		 if (show_frame_flag == 1 ||dissappear_time==DISSAPPEAR_TIME_LIMIT-1)
		{
			show_frame_flag = 2; 
			dissappear_time = DISSAPPEAR_TIME_LIMIT;
		}
		 if (waitKey(300) == uint8_t('s'))
		 {
			 show_frame_flag = 0;  
			 dissappear_time = 0; 
		 }
		imshow("frame", show1_label);

		 if (dissappear_time < DISSAPPEAR_TIME_LIMIT)
			 dissappear_time++;


		//		framecnt++;//计算程序消耗时间所用
		if (break_)
		{
			destroyAllWindows();
			break;
		}

	}

	//	end = clock();
	//	cout << "Time consume£º" << double(end - start) / CLOCKS_PER_SEC;
	//	cout << "Framecnt:"<<framecnt << endl;

	video.release(); 
	return 0;

}





int HOG_predict(const Mat image)
{
	int predict_number=0;
	Size	WinSize =	Size(32, 32),
		BlockSize =		Size(12, 16),
		BlockStride =	Size(4, 8),
		CellSize =		Size(6, 8);
	int		Nbins = 9;
	auto hog = HOGDescriptor(WinSize,BlockSize,BlockStride,CellSize,Nbins);
	vector<float>  result_hog;
	Ptr<ml::SVM> svm_ = ml::SVM::load("svm.xml");

	hog.compute(image, result_hog);
	predict_number = svm_->predict(Mat(result_hog, CV_32FC1).reshape(0, 1));

	return predict_number;
}
 
void recall_for_ready(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDBLCLK)
	{
		show_frame_flag = 1;
	}
	if (event == EVENT_LBUTTONUP)
	{
		if (y > 1 && y < 40 && x>1 && x < 80)
			break_ = true;
	}
}

Rect readtxt(string file, bool& thresh_flag)
{
	FileStorage  infile("location.xml", FileStorage::READ);
	int x1 = 0, y1 = 0, y2 = 0, x2 = 0;

	infile["ix1"] >> x1;
	infile["iy1"] >> y1;
	infile["iy2"] >> y2;
	infile["ix2"] >> x2;
	infile["key_choose_thresh"] >> thresh_flag;

	Rect temp(Point(x1,y1),Point(x2,y2));
	infile.release();
	return temp;

}

void MovingAverage(Mat &A, int N)
{
	assert(A.size[0] == 1 ^ A.size[1] == 1);
	int sizenum = (A.size[0] == 1) ? A.size[1] : A.size[0];
	if (A.size[0] == 1)
	{
		for (int a = 0; a < sizenum - N; a++)
			A.at<float>(0, a) = sum(A.colRange(a, a + N))[0] / N;
		for (int a = sizenum - N; a < sizenum; a++)
			A.at<float>(0, a) = 0;

		hconcat(Mat::zeros(Size(1, 1), CV_32FC1), A, A);
	}
	else
	{
		for (int a = 0; a < sizenum - N; a++)
			A.at<float>(a, 0) = sum(A.rowRange(a, a + N))[0] / N;
		for (int a = sizenum - N; a <sizenum; a++)
			A.at<float>(a, 0) = 0;

		vconcat(Mat::zeros(Size(1, 1), CV_32FC1), A, A);

	}
}

vector<int> find_wavetop(Mat A)
{
	vector<int > store_int = { 0 };
	int sizenum = (A.size[0] <A.size[1]) ? A.size[1] : A.size[0];
	int num = 0;
	for (int a = 0; a < sizenum - 1; a++)
	{
		if (A.at<float>(a) < A.at<float>(a + 1))
			num = a;
		if (A.at<float>(a) > A.at<float>(a + 1) && num != *(store_int.cend() - 1))
			store_int.push_back(num);
	}

	return store_int;
}

Mat SHOW_Label(Mat choice_label, string s1,int length)
{
	Mat temp;
	Mat SHOW_Label1 = Mat::zeros(Size(length, 40), CV_8UC3);
	putText(SHOW_Label1, s1, Point(5, 30), FONT_HERSHEY_COMPLEX, 0.95, Scalar(0, 255, 0), 2);
	hconcat(choice_label, SHOW_Label1, temp);
	return temp;
}

Mat Thresh_Process_Func(const Mat  image) {
	//////做二值化

	Mat  B, G, R, histormB, histormG, histormR;
	vector<Mat> channel_split;
	int hisnum = 255;
	const float range[] = { 0,255 };
	const float* histRange = { range };
	/////分量提取并作后续工作
	split(image.clone(), channel_split);

	B = channel_split.at(0);
	G = channel_split.at(1);
	R = channel_split.at(2);

	calcHist(&B, 1, 0, Mat(), histormB, 1, &hisnum, &histRange, true, false);
	calcHist(&G, 1, 0, Mat(), histormG, 1, &hisnum, &histRange, true, false);
	calcHist(&R, 1, 0, Mat(), histormR, 1, &hisnum, &histRange, true, false);

	MovingAverage(histormB, 50);
	MovingAverage(histormG, 50);
	MovingAverage(histormR, 50);

	vector<int> wavetopB = find_wavetop(histormB);
	vector<int> wavetopG = find_wavetop(histormG);
	vector<int> wavetopR = find_wavetop(histormR);

	int B_diff = *(wavetopB.cend() - 1) - *(wavetopB.cbegin() + 1);
	int G_diff = *(wavetopG.cend() - 1) - *(wavetopG.cbegin() + 1);
	int R_diff = *(wavetopR.cend() - 1) - *(wavetopR.cbegin() + 1);

	bool flag = false;

	if (B_diff > 150 && (G_diff > 150 || R_diff > 150))
		flag = true;
	else if (G_diff > 150 && R_diff > 150)
		flag = true;
	Mat img_thresh;

	int loc_thresh = 127;
	if (!flag)
	{
		int max_value = max(max(B_diff, G_diff), R_diff);

		if (B_diff == max_value)
		{
			img_thresh = B;
			loc_thresh = (*(wavetopB.cend() - 1) + *(wavetopB.cbegin() + 1)) / 2;
		}
		else if (G_diff == max_value)
		{
			img_thresh = G;
			loc_thresh = (*(wavetopG.cend() - 1) + *(wavetopG.cbegin() + 1)) / 2;
		}
		else
		{
			img_thresh = R;
			loc_thresh = (*(wavetopR.cend() - 1) + *(wavetopR.cbegin() + 1)) / 2;
		}
	}
	else
		cvtColor(image.clone(), img_thresh, COLOR_BGR2GRAY);
	threshold(img_thresh, img_thresh, loc_thresh, 255, THRESH_BINARY);
	return img_thresh;
}