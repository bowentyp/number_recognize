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
//#include	<opencv\ml.h>////opencv version==310

#include	<time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

bool break_ = false;
void recall_for_quit(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONUP)
	{
		if (y > 5 && y < 58 && x>5 && x < 95)
			break_ = true;
	}
}


bool recall_for_back=false, recall_for_next = false, recall_for_esc = false, recall_for_no= false, recall_for_ok = false;
void recall_for_setup(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONUP)
	{
		if (y > 5 && y < 58)
		{
			if (x > 5 && x < 95)
				recall_for_back = true;
			else	if (x > 105 && x < 195)
				recall_for_next = true;
			else	if (x > 205 && x < 295)
				recall_for_ok = true;
			else	if (x > 305 && x < 395)
				recall_for_no = true;
			else	if (x > 405 && x < 495)
				recall_for_esc = true;
		}
	}
}

Mat img;
int ix1 = -1, iy1 = -1, ix2 = -1, iy2 = -1 ;
bool drawing_ = false;
void recall_for_draw(int event, int x, int y, int flags, void* userdata)
{
	//当按下左键时返回起始位置坐标
	if (event == EVENT_LBUTTONDOWN)
	{
		drawing_ = true;
		ix1 = x; iy1 = y;  
	}
	if (event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_LBUTTON)
	{
		if (drawing_)
		{ 
			Mat img2 = img.clone();
			rectangle(img2, Point(ix1, iy1),Point(x, y), Scalar(0, 255, 255), 1);
			imshow("img", img2);
		}
	}
	//当鼠标松开时停止绘图
	if (event == EVENT_LBUTTONUP)
	{
		drawing_ = false;
		ix2 = x; iy2 = y;
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

auto hog = HOGDescriptor(WinSize,BlockSize,BlockStride,CellSize,Nbins);	

void MovingAverage(Mat &A, int N)
{
	assert((A.size[0] == 1) ^( A.size[1] == 1));
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

Mat choice_for_setup()
{
	Mat choice_label = Mat::zeros(Size(500, 60), CV_8UC3);
	rectangle(choice_label, Rect(5, 5, 90, 53), Scalar(0, 255, 255), 2);
	rectangle(choice_label, Rect(105, 5, 90, 53), Scalar(0, 255, 255), 2);
	rectangle(choice_label, Rect(205, 5, 90, 53), Scalar(0, 255, 255), 2);
	rectangle(choice_label, Rect(305, 5, 90, 53), Scalar(0, 255, 255), 2);
	rectangle(choice_label, Rect(405, 5, 90, 53), Scalar(0, 255, 255), 2);
	putText(choice_label, "BACK", Point(8, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);
	putText(choice_label, "NEXT", Point(108, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);
	putText(choice_label, "OK", Point(230, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);
	putText(choice_label, "NO", Point(330, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);
	putText(choice_label, "ESC", Point(420, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);

	return choice_label;
}

Mat SHOW_Label(Mat choice_label, string s1,bool lines=false )
{	
	Mat temp;
	Mat SHOW_Label1 = Mat::zeros(Size(500, 40), CV_8UC3);
	if (lines)
		line(SHOW_Label1, Point(0, 5), Point(500, 5), Scalar(0, 255, 0), 3);
	putText(SHOW_Label1, s1, Point(5, 30), FONT_HERSHEY_COMPLEX, 0.95, Scalar(0, 255, 0), 2);
	vconcat(choice_label, SHOW_Label1, temp);
	return temp;
}

Mat SHOW_once(string s1, int overtime = 0, Mat A = Mat())
{
	Mat temp = Mat::zeros(Size(500, 40), CV_8UC3);
	putText(temp, s1, Point(5, 30), FONT_HERSHEY_COMPLEX, 0.95, Scalar(0, 255, 0), 2);
	if (A.size() != Size(0, 0))
		vconcat(temp, A, temp);

	if (overtime)
	{
		imshow("show", temp);
		waitKey(overtime);
		destroyWindow("show");
	}
	return temp;
}

int main()
{
	Mat choice_label = choice_for_setup();
		
	//////以下是录制视频的代码	
	
	Mat choice_label_show= SHOW_Label( SHOW_Label(choice_label, 
								"When camera ready,clik OK",true),
								"to record ; ESC to quit");
	
	namedWindow("recall_for_setup");
	setMouseCallback("recall_for_setup", recall_for_setup);
	imshow("recall_for_setup", choice_label_show);
	
	VideoCapture cap(0);
	cap.set(CAP_PROP_AUTO_EXPOSURE, 1);
	int fourcc = VideoWriter::fourcc('M', 'P', '4', '2');
	auto video_writor = VideoWriter("train.avi", fourcc, 10, 
		Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)),true);
	
	Mat frame;
	int cnt_record = 0;
	while (true)
	{
		cap >> frame;
		assert(!frame.empty());
		imshow("frame", frame);
		waitKey(500);
		if (recall_for_ok)
		{
			video_writor.write(frame);
			cout << "frame record:" << cnt_record << " pics" << endl;
			cnt_record++;
		}
		if (recall_for_esc)
		{
			if (cnt_record == 0)
			{
				SHOW_once("clik OK to record", 1000);
				continue;
			}
			destroyWindow("frame");
			//destroyWindow("recall_for_setup");
			break;
		}
	}
	video_writor.release();
	cap.release();
	cap.open("train.avi");
	choice_label_show = SHOW_Label(SHOW_Label(choice_label,
						"Draw fixed ROI Firstly", true),
						"Then adjust label(BACK/NEXT)");
		
	vector<Mat> train_img;
	vector<int> chars_label;
	int label_cnt = 1;
	namedWindow("img");
	setMouseCallback("img", recall_for_draw);
	imshow("recall_for_setup", choice_label_show);

	for (int cnt_cap = 0; cnt_cap < cap.get(CAP_PROP_FRAME_COUNT); cnt_cap++)
	{
		cap.read(frame);
		img = frame.clone();
		rectangle(img, Point(ix1, iy1), Point(ix2, iy2), Scalar(0, 255, 0), 2);
		imshow("img", img);

		while (true)
		{
			recall_for_back = false;
			recall_for_next = false;
			recall_for_esc = false;
			recall_for_ok = false;
			recall_for_no = false;
			waitKey(100);
			if (recall_for_ok)
			{
				if (ix1 == -1 || iy1 == -1 || ix2 == -1 || iy2 == -1)
				{
					SHOW_once("Draw ROI firstly" , 1000);
					continue;
				}
				train_img.push_back(frame(Rect(Point(ix1, iy1), Point(ix2, iy2))).clone());
				chars_label.push_back(label_cnt);
				cout << "comfirm the label cnt:" << label_cnt << endl;
				
				break;
			}
			if (recall_for_back)
			{
				label_cnt--;
				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_label,
								"Add to train(OK/NO),Quit(ESC)", true),
								"Change label(BACK/NEXT)"),
								"NOW label is: " + to_string(label_cnt)));
			}
			if (recall_for_next)
			{
				 label_cnt++;
				 imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_label,
								 "Add to train(OK/NO),Quit(ESC)", true),
								 "Change label(BACK/NEXT)"),
								  "NOW label is: "+to_string(label_cnt)));
			}
			if (recall_for_no)
			{
				if (ix1 == -1 || iy1 == -1 || ix2 == -1 || iy2 == -1)
				{
					SHOW_once("Draw ROI firstly", 1000);
					continue;
				}
				cout << "you have not choose this frame" << endl;
				break;
			}
			if (recall_for_esc)
			{
				if (ix1 == -1 || iy1 == -1 || ix2 == -1 || iy2 == -1)
				{
					SHOW_once("Draw ROI firstly", 1000);
					continue;
				}
				SHOW_once("You have leaved", 500);
				break;
			}
		}
		if (recall_for_esc)
			break;
	}

	destroyWindow("img");

	ofstream location_file;
	location_file.open("..\\location.txt", ios::out);
	location_file << ix1 << "," << iy1 << "," << ix2 << "," << iy2 << endl;
	 
	cap.release();

	vector<int> numdelete;
	vector<Mat> train_thresh;  

	//////做二值化
	for (int num_record = 0; num_record < train_img.size(); num_record++)
	{
		//cout << "label:" << chars_label[num_record] << endl;
		//imshow("train_img", train_img[num_record]);
		//waitKey(1);

		/////分量提取并作后续工作
		Mat  B, G, R, histormB, histormG, histormR;
		vector<Mat> channel_split;
		int hisnum = 255;
		const float range[] = { 0,255 };
		const float* histRange = { range };

		split(train_img[num_record].clone(), channel_split);

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
			cvtColor(train_img[num_record].clone(), img_thresh, COLOR_BGR2GRAY);
		threshold(img_thresh, img_thresh, loc_thresh, 255, THRESH_BINARY);/*
		imshow("img_thresh", img_thresh);
		waitKey(1);*/
		train_thresh.push_back(img_thresh);
	}
	
	//////做一些删减
	choice_label_show = SHOW_Label(SHOW_Label(SHOW_Label(choice_label,
									"Delete this image(OK)", true),
									"Not delete(NEXT)"),
									"NOW label is:" + to_string(chars_label[0]));
	imshow("recall_for_setup", choice_label_show); 
	for (int num_record1 = 0; num_record1 < train_img.size(); num_record1++)
	{
		imshow("img_thresh", train_thresh[num_record1]);
		imshow("train_img", train_img[num_record1]);
		waitKey(1); 
		while (true)
		{
			recall_for_ok = false;
			recall_for_next= false;
			recall_for_back = false;

			waitKey(50);//can't lose this sentense 
			if (recall_for_ok)
			{
				if (num_record1 < 0)
				{
					SHOW_once("NO images", 500);
					break;
				}
				SHOW_once("Delete Done:" + to_string(chars_label[num_record1]), 500);
				train_img.erase(train_img.cbegin() + num_record1);
				train_thresh.erase(train_thresh.cbegin() + num_record1);
				chars_label.erase(chars_label.cbegin() + num_record1); 
				num_record1--;
				choice_label_show = SHOW_Label(SHOW_Label(SHOW_Label(choice_label,
					"Delete this image(OK)", true),
					"Not delete(NEXT)"),
					"NOW label is:" + to_string(chars_label[num_record1 + 1]));
				imshow("recall_for_setup", choice_label_show);
				break;

			}
			if (recall_for_back)
			{
				if (num_record1==0)
				{
					SHOW_once("NO images",500);
					continue;
				}
				num_record1--;
				imshow("img_thresh", train_thresh[num_record1]);
				imshow("train_img", train_img[num_record1]);
				choice_label_show = SHOW_Label(SHOW_Label(SHOW_Label(choice_label,
									"Delete this image(OK)", true),
									"Not delete(NEXT)"),
									"NOW label is:" + to_string(chars_label[num_record1]));
				imshow("recall_for_setup", choice_label_show);
				recall_for_back = false;
				waitKey(50);
				continue;
			}
			if (recall_for_next)
			{
				choice_label_show = SHOW_Label(SHOW_Label(SHOW_Label(choice_label,
					"Delete this image(OK)", true),
					"Not delete(NEXT)"),
					"NOW label is:" + to_string(chars_label[num_record1+1]));
				imshow("recall_for_setup", choice_label_show);
				break;
			}
		}
	}
	destroyWindow("img_thresh");
	destroyWindow("train_img");


	bool key_choose_thresh = false;

	choice_label_show = SHOW_Label( SHOW_Label(choice_label,
								"Use Binarization image to ", true),
								"make model file?(OK/NO)");
	  
	imshow("recall_for_setup", choice_label_show);

	while (true)
	{
		recall_for_esc = false;
		recall_for_ok = false;
		waitKey(100);
		if (recall_for_ok)
		{
			key_choose_thresh = true;
			break;
		}
		else if (recall_for_no) 
			break;
	}
	vector<Mat> chars_img;
	if (key_choose_thresh)
	{
		chars_img = train_thresh;
		SHOW_once("You choose Thresh",500);
	}
	else
	{
		chars_img = train_img;
		SHOW_once("You choose Gray",500);
	}
	location_file << key_choose_thresh << endl;
	location_file.close();

	Mat svm_image_data  ;
	Mat svm_label_data = Mat(chars_label);
	cout << svm_label_data << ";\n"<< svm_label_data.size()<< endl;
	
	vector<float> result_hog;

	resize(chars_img[0], img, Size(50, 50));
	hog.compute(img, result_hog);
	transpose(Mat(result_hog, CV_32FC1), svm_image_data);
	
	for (int img_cnt = 1; img_cnt<chars_img.size();img_cnt++)
	{
		Mat temp;
		resize(chars_img[img_cnt], img, Size(50, 50));
		hog.compute(img, result_hog);
		transpose(Mat(result_hog, CV_32FC1), temp);
		vconcat(svm_image_data, temp.reshape(0, 1), svm_image_data);
	}
	
	cout << "img.size():" << svm_image_data.size() << endl;
	
	
	
	//////训练开始
	cout << "begin training" << endl;

	SHOW_once("prepare data welly", 1000, SHOW_once("begin training"));

	Ptr<ml::SVM> svm_ = ml::SVM::create();
	svm_->setC(100);
	svm_->setType(SVM::C_SVC);
	svm_->setKernel(SVM::RBF);
	svm_->setGamma(0.3);
	
	svm_->train(svm_image_data, ROW_SAMPLE, svm_label_data);
	svm_->save("..\\svm.dat");

	//Ptr<SVM> svm_ =  Algorithm::load<SVM>("svm.dat");	////opencv310	下load函数并没有在SVM里面
	//Ptr<ml::SVM> svm_ = ml::SVM::load("svm.dat");	////opencv401	下load函数在SVM里面


	system("pause");
	return 0;
		
	}
