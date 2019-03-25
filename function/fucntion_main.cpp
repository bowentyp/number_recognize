#include	<iostream>
#include	<assert.h>
#include	<fstream>
#include	<opencv2\core.hpp>
#include	<opencv2\core\utility.hpp>
#include	<opencv2\videoio.hpp>
#include	<opencv2\video\video.hpp>
#include	<opencv2\highgui.hpp>
#include	<opencv2\imgproc.hpp>
#include	<opencv2\ml\ml.hpp>
#include	<opencv2\opencv.hpp>
#include	<time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

bool break_ = false;
void recall(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONUP)
	{
		if (y > 5 && y < 58 && x>5 && x < 95)
			break_ = true;
	}
}

Rect readtxt(string file ,bool& thresh_flag)
{
	ifstream infile;
	infile.open(file.data(),ios::in);
	assert(infile.is_open()); 

	string s;
	int x1=0, y1=0, w=0, h=0;
	getline(infile, s, ',');
	x1 = atoi(s.c_str());
	getline(infile, s, ',');
	y1 = atoi(s.c_str());
	getline(infile, s, ',');
	w = atoi(s.c_str());
	getline(infile, s, ',');
	h = atoi(s.c_str());
	getline(infile, s, ',');
	thresh_flag = atoi(s.c_str())!=0;
	
	if (w < x1)
	{
		int ss = w;
			w = x1 - w;
			x1 = ss;
	}
	else
		w = w - x1;

	if (h < y1)
	{
		int ss = h;
			h = y1 - h;
			y1 = ss;
	}
	else
		h = h - y1;

	Rect temp(x1, y1, w, h); 
		
	infile.close();

	return temp;

}


Size	WinSize		=Size(50,50), 
		BlockSize	=Size(20,20), 
		BlockStride	=Size(10,10), 
		CellSize	=Size(10,10);
int		Nbins		=9;

auto hog = HOGDescriptor(WinSize   ,
						BlockSize	,
						BlockStride	,
						CellSize	,
						Nbins);	

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
	vector<int > store_int = {0};
	int sizenum = (A.size[0] <A.size[1]) ? A.size[1] : A.size[0];
	int num = 0;
	for (int a = 0; a < sizenum-1; a++)
	{
		if (A.at<float>(a) < A.at<float>(a + 1))
			num = a;
		if (A.at<float>(a) > A.at<float>(a + 1) && num != *(store_int.cend() - 1))
			store_int.push_back(num);
	}

	return store_int;
}
 
int main()
{
	clock_t start, end;
	start = clock();
	bool thresh_flag;
	Rect ROI=readtxt("C:\\Users\\typ97\\Desktop\\data\\final\\location.txt", thresh_flag);
	cout << ROI << endl;
	VideoCapture video ;
	video.open(0);
	if (!video.isOpened())
	{
		cout << "cant open video" << endl;
	
		system("pause");
		return -1;
	}
	
	namedWindow("record");
	setMouseCallback("record",recall );
	
	Mat choice_label=Mat::zeros(Size(50,25),CV_8UC3);
	rectangle(choice_label, Rect(0, 0, 49, 24), Scalar(0, 255, 255));
	putText(choice_label, "QUIT", Point(2, 18), FONT_HERSHEY_COMPLEX, 0.6, Scalar(0, 255, 255), 2);
		
	imshow("record", choice_label);
	
	Ptr<ml::SVM> svm_ = ml::SVM::load("C:\\Users\\typ97\\Desktop\\data\\final\\svm.dat");
	
	
	Mat frame,predict,B,G,R,histormB,histormG,histormR;
	vector<float>  result_hog;	 
	vector<Mat> channel;
	int hisnum = 255;
	const float range[] = { 0,255 };
	const float* histRange = { range };
	int framecnt = 0;

	while (true)
	{														
		video >> frame;										
		assert(!frame.empty());								
		Mat img = frame(ROI);
		if (thresh_flag) {
			split(img.clone(), channel);
	
			B = channel.at(0);
			G = channel.at(1);
			R = channel.at(2);
	
			calcHist(&B, 1, 0, Mat(), histormB, 1, &hisnum, &histRange, true, false);
			calcHist(&G, 1, 0, Mat(), histormG, 1, &hisnum, &histRange, true, false);
			calcHist(&R, 1, 0, Mat(), histormR, 1, &hisnum, &histRange, true, false);

			MovingAverage(histormB,50)	 ;
			MovingAverage(histormG,50)	 ;
			MovingAverage(histormR,50)	 ;

			vector<int> wavetopB=find_wavetop(histormB);
			vector<int> wavetopG=find_wavetop(histormG);
			vector<int> wavetopR=find_wavetop(histormR);
			
			int B_diff = *(wavetopB.cend() - 1) - *(wavetopB.cbegin()+1);
			int G_diff = *(wavetopG.cend() - 1) - *(wavetopG.cbegin()+1);
			int R_diff = *(wavetopR.cend() - 1) - *(wavetopR.cbegin()+1);
			 
			bool flag=false;

			if (B_diff > 150 && (G_diff > 150 || R_diff > 150))
				flag = true;
			else if (G_diff > 150 && R_diff > 150)
				flag = true;
			Mat img_thresh ;

			int loc_thresh=127;
			if (!flag)
			{
				int max_value = max(max(B_diff, G_diff), R_diff);
				cout << max_value;
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
				cvtColor(img.clone(), img_thresh, COLOR_BGR2GRAY);
			threshold(img_thresh, img_thresh, loc_thresh, 255, THRESH_BINARY);
			imshow("img_thresh", img_thresh);
			cv::resize(img_thresh, img_thresh, Size(50, 50));
			hog.compute(img_thresh, result_hog);
			cout << "test:" << svm_->predict(Mat(result_hog, CV_32FC1).reshape(0, 1)) << endl;
		}
		else {
			cvtColor(img, img, COLOR_BGR2GRAY);
			imshow("record", choice_label);
			cv::resize(img, img, Size(50, 50));
	
			hog.compute(img, result_hog); 
			cout << "test:" << svm_->predict(Mat(result_hog, CV_32FC1).reshape(0, 1)) << endl;
	
			
	
		}
		imshow("img", frame);
		if (waitKey(1)==27)  ;


		framecnt++;
		if (break_)
		{
			destroyAllWindows();
			break;
		}

	}	

	end = clock();
	cout << "Time consume£º" << double(end - start) / CLOCKS_PER_SEC;
	cout << "Framecnt:"<<framecnt << endl;

	video.release(); 
	system("pause");
	return 0;
		
	}