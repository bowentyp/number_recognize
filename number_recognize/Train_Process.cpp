#include	"Train_Process.h"
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
#include	<numeric>

using namespace cv;
using namespace std;

int ix1 = -1, iy1 = -1, ix2 = -1, iy2 = -1;
bool drawing_ = false;
bool recall_for_back = false, recall_for_next = false, recall_for_esc = false, recall_for_no = false, recall_for_ok = false;

/**
滑动平均
*/
void MovingAverage(Mat &A, int N)
{
	assert(A.channels()==1);//make sure that the image is single channel
	assert((A.size[0] == 1) ^ (A.size[1] == 1));//make sure that the image is vector
	A=A.reshape(1, 1);
	
	for (int a = 0; a < A.size[1] - N; a++)
		A.at<float>(a) = sum(A.colRange(a, a + N))[0] / N;
	for (int a = A.size[1] - N; a < A.size[1]; a++)
		A.at<float>(a) = 0;

	hconcat(Mat::zeros(Size(1, 1), CV_32FC1), A, A);
	
}
/**
滑动平均
*/
void MovingAverage(vector<float> &V, int N)
{
	Mat A = Mat(V, CV_32FC1).reshape(1, 1);

	for (int a = 0; a < A.size[1] - N; a++)
		A.at<float>(a) = sum(A.colRange(a, a + N))[0] / N;
	for (int a = A.size[1] - N; a < A.size[1]; a++)
		A.at<float>(a) = 0;

	hconcat(Mat::zeros(Size(1, 1), CV_32FC1), A, A);

	V = static_cast<vector<float> >(A.reshape(1, 1));
	
}
/**
找波峰点
*/
vector<int> find_wavetop(Mat A)
{
	vector<int > store_int = { 0 };
	assert((A.size[0] == 1) ^ (A.size[1] == 1));
	
	vector<int> A_transformTo_another(A.reshape(1, 1));

	int num = 0;
	for (int a = 0; a < A_transformTo_another.size() - 1; a++)
	{
		if (A_transformTo_another[a] <= A_transformTo_another[a + 1])
			num = a+1;
		if (A_transformTo_another[a] > A_transformTo_another[a + 1] && num != *(store_int.cend() - 1))
			store_int.push_back(num);
	}	
	store_int.erase(store_int.begin());
	return store_int;
}
/**
找波峰点
*/
vector<int> find_wavetop(vector<float> A)
{
	vector<int > store_int = { 0 }; 
	
	int num = 0;
	for (int a = 0; a < A.size() - 1; a++)
	{
		if (static_cast<int>(A[a]) <= static_cast<int>(A[a + 1]))
			num = a + 1;
		if (static_cast<int>(A[a]) >  static_cast<int>(A[a + 1]) && num != *(store_int.cend() - 1))
			store_int.push_back(num);
	}
	store_int.erase(store_int.begin());
	return store_int;
}

/**
数字分割，寻找最佳分割点
*/
int find_separated_loc(const Mat Img)
{
	assert(Img.channels() == 1);//必须是单通道的图片，因为只有这样，才能做后续操作
	//if (Img.channels() != 1)
	//	cout << Img.channels() << endl;
	vector<float> sum_row_mat;

	for (int cnt = 0; cnt < Img.cols; cnt++)
	{
		sum_row_mat.push_back(sum(Img.col(cnt))[0] / 255);
	}

	MovingAverage(sum_row_mat, 5);
	vector<int> index_result = find_wavetop(sum_row_mat);

	auto mi1n1 = min_element(sum_row_mat.cbegin() + *index_result.cbegin(), sum_row_mat.cbegin() + *(index_result.cend() - 1));
	return  distance(sum_row_mat.cbegin(), mi1n1);
}

/**
短暂显示一片数字
@s1			显示内容
@overtime	停留时间
@A			右边拼接的图片，但必须保证图片的类型是CV_8UC3
*/
Mat SHOW_once(String s1, int overtime = 0, Mat A = Mat())
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

/**overload
面向train过程
@choice_label	上接矩阵
@s1				显示内容
@lines			是否划线
*/
Mat SHOW_Label(Mat choice_label, String s1, bool lines = false )
{
	Mat temp;
	Mat SHOW_Label1 = Mat::zeros(Size(500, 40), CV_8UC3);
	if (lines)
		line(SHOW_Label1, Point(0, 5), Point(500, 5), Scalar(0, 255, 0), 3);
	putText(SHOW_Label1, s1, Point(5, 30), FONT_HERSHEY_COMPLEX, 0.95, Scalar(0, 255, 0), 2);
	vconcat(choice_label, SHOW_Label1, temp);
	return temp;
}

/**overload
面向test过程
@choice_label	上接矩阵
@s1				显示内容
@length			显示长度
*/
Mat SHOW_Label(Mat choice_label, string s1, int length)
{
	Mat temp;
	Mat SHOW_Label1 = Mat::zeros(Size(length, 40), CV_8UC3);
	putText(SHOW_Label1, s1, Point(5, 30), FONT_HERSHEY_COMPLEX, 0.95, Scalar(0, 255, 0), 2);
	hconcat(choice_label, SHOW_Label1, temp);
	return temp;
}

/**
创造一个功能区显示图片
*/
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

/**
鼠标回调函数--确定功能区
*/
void  recall_for_setup(int event, int x, int y, int flags, void* userdata)
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

/**
面向训练过程
主要是后门，用于按键退出安装
*/
void Key_quit_waitkey(int waittime)
{
	if (waitKey(waittime) == 27)
		exit(0);
}

/**
鼠标回调函数---画圈以确定ROI
*/
void recall_for_draw(int event, int x, int y, int flags, void* userdata)
{
	static int weight;
	//当按下左键时返回起始位置坐标
	if (event == EVENT_LBUTTONDOWN)
	{
		drawing_ = true;
		ix1 = x; iy1 = y;
	}
	if (event == cv::EVENT_MOUSEMOVE)// && ( flags == EVENT_FLAG_LBUTTON))
	{
		if (drawing_)
		{
			weight = (abs(x - ix1) > abs(y - iy1)) ? abs(x - ix1) : abs(y - iy1);
			ix2 = ix1 + weight; iy2 = iy1 + weight;

			//Mat img2 = image.clone();
			//weight = (abs(x - ix1) > abs(y - iy1)) ? abs(x - ix1) : abs(y - iy1);
			//rectangle(img2, Point(ix1, iy1), Point(ix1 + weight, iy1 + weight), Scalar(0, 255, 255), 1);
			//line(img2, Point(ix1 + int(weight / 2), iy1), Point(ix1 + int(weight / 2), iy1 + weight), Scalar(0, 255, 255), 1);
			//imshow("img", img2);
		}
	}
	//当鼠标松开时停止绘图
	if (event == EVENT_LBUTTONUP)
	{
		drawing_ = false;

		ix2 = ix1 + weight; iy2 = iy1 + weight;
	}
}

Train_Process::Train_Process()
{
	//HOG_Init();
}

Train_Process::~Train_Process()
{
}

void Train_Process::HOG_Init(
							Size	WinSize ,
							Size BlockSize ,
							Size BlockStride ,
							Size CellSize,
							int	Nbins)
{
	hog=HOGDescriptor(WinSize, BlockSize, BlockStride, CellSize, Nbins);
	 
	//hog=&hog1;
}

Mat Train_Process::Thresh_Process_Func(const Mat  image)
{
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

void Train_Process::Camera_Write_To_File(String train_file)
{
	namedWindow("recall_for_setup");
	setMouseCallback("recall_for_setup",  recall_for_setup);
	imshow("recall_for_setup", SHOW_Label(SHOW_Label(choice_for_setup(),
		"When camera ready,clik OK", true),
		"to record ; ESC to quit"));
	waitKey(1);

	cap.open(0);
	//cap.set(CAP_PROP_AUTO_EXPOSURE, -3);
	int fourcc = VideoWriter::fourcc('M', 'P', '4', '2');
	auto video_writor = VideoWriter(train_file, fourcc, 10,
		Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), true);

	Mat frame;
	int cnt_record = 0;
	while (true)
	{
		cap >> frame;
		assert(!frame.empty());
		imshow("frame", frame);
		Key_quit_waitkey(500);
		if (recall_for_ok)
		{
			video_writor.write(frame);
			cnt_record++;
			imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
				"When camera ready,clik OK", true),
				"to record ; ESC to quit"),
				"record " + to_string(cnt_record) + " pics"));
			cout << "frame record:" << cnt_record << " pics" << endl;
		}
		if (recall_for_esc)
		{
			if (cnt_record == 0)
			{
				SHOW_once("clik OK to record", 1000);
				continue;
			}
			break;
		}
	}
	destroyAllWindows();
	video_writor.release();
	cap.release();

}
 
void Train_Process::DrawROI_Then_Choose_Image(String video_file)
{
	cap.open(video_file);

	int label_cnt = 1;
	namedWindow("img");
	setMouseCallback("img", recall_for_draw);
	namedWindow("recall_for_setup");
	setMouseCallback("recall_for_setup", recall_for_setup);
	imshow("recall_for_setup", SHOW_Label(SHOW_Label(choice_for_setup(),
		"Draw fixed ROI Firstly", true),
		"Then adjust label(BACK/NEXT)"));
	Mat frame;
	for (int cnt_cap = 0; cnt_cap < cap.get(CAP_PROP_FRAME_COUNT); cnt_cap++)
	{
		cap.read(frame);
		image = frame.clone();
		rectangle(image, Point(ix1, iy1), Point(ix2, iy2), Scalar(0, 255, 0), 2);
		imshow("img", image);

		while (true)
		{
			if (drawing_)
			{
				Mat img2 = image.clone();
				rectangle(img2, Point(ix1, iy1), Point(ix2, iy2), Scalar(0, 255, 255), 1);
				//	line(img2, Point(int(ix1 + weight/2), iy1), Point(ix1 + int(weight / 2), iy1 + weight),Scalar(0, 255, 255), 1);
				imshow("img", img2);
				waitKey(1);
			}
			recall_for_back = false;
			recall_for_next = false;
			recall_for_esc = false;
			recall_for_ok = false;
			recall_for_no = false;
			Key_quit_waitkey(100);
			if (recall_for_ok)
			{
				if (ix1 == -1 || iy1 == -1 || ix2 == -1 || iy2 == -1)
				{
					SHOW_once("Draw ROI firstly", 1000);
					continue;
				}
				train_img.push_back(frame(Rect(Point(ix1, iy1), Point(ix2, iy2))).clone());
				int_label.push_back(label_cnt);
				cout << "comfirm the label cnt:" << label_cnt << endl;

				break;
			}
			if (recall_for_back)
			{
				label_cnt--;
				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
					"Add to train(OK/NO),Quit(ESC)", true),
					"Change label(BACK/NEXT)"),
					"NOW label is: " + to_string(label_cnt)));
			}
			if (recall_for_next)
			{
				label_cnt++;
				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
					"Add to train(OK/NO),Quit(ESC)", true),
					"Change label(BACK/NEXT)"),
					"NOW label is: " + to_string(label_cnt)));
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

	destroyAllWindows();
	cap.release();
}

void Train_Process::Thresh_Then_Delete(String LOC_FILE_NAME)
{
	vector<int> numdelete;
	vector<Mat> train_thresh,train_gray;
	Mat gray_img;
	namedWindow("recall_for_setup");
	setMouseCallback("recall_for_setup", recall_for_setup);
	for (int num_record = 0; num_record < train_img.size(); num_record++)
	{
		train_thresh.push_back(Thresh_Process_Func(train_img[num_record]));
		cvtColor(train_img[num_record], gray_img, COLOR_BGR2GRAY);
		train_gray.push_back(gray_img.clone());
	}
	//////做一些删减 
	imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
							"Delete this image(OK)", true),
							"Not delete(NEXT)"),
							"NOW label is:" + to_string(int_label[0])));
	for (int num_record1 = 0; num_record1 < train_img.size(); num_record1++)
	{
		imshow("img_thresh", train_thresh[num_record1]);
		imshow("train_gray", train_gray[num_record1]);
		waitKey(1);
		while (true)
		{
			recall_for_ok = false;
			recall_for_next = false;
			recall_for_back = false;

			Key_quit_waitkey(50);//can't lose this sentense
			if (recall_for_ok)
			{
				if (num_record1 < 0)
				{
					SHOW_once("NO images", 500);
					break;
				}
				SHOW_once("Delete Done:" + to_string(int_label[num_record1]), 500);
				train_img.erase(train_img.cbegin() + num_record1);
				train_thresh.erase(train_thresh.cbegin() + num_record1);
				train_gray.erase(train_gray.cbegin() + num_record1);
				int_label.erase(int_label.cbegin() + num_record1);
				num_record1--;
				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
					"Delete this image(OK)", true),
					"Not delete(NEXT)"),
					"NOW label is:" + to_string(int_label[num_record1 + 1])));
				break;

			}
			if (recall_for_back)
			{
				if (num_record1 == 0)
				{
					SHOW_once("NO images", 500);
					continue;
				}
				num_record1--;
				imshow("img_thresh", train_thresh[num_record1]);
				imshow("train_gray", train_gray[num_record1]);
				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
					"Delete this image(OK)", true),
					"Not delete(NEXT)"),
					"NOW label is:" + to_string(int_label[num_record1])));
				recall_for_back = false;
				waitKey(50);
				continue;
			}
			if (recall_for_next)
			{
				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
					"Delete this image(OK)", true),
					"Not delete(NEXT)"),
					"NOW label is:" + to_string(int_label[num_record1 + 1])));
				break;
			}
		}
	}
	destroyWindow("img_thresh");
	destroyWindow("train_gray");


	bool key_choose_thresh = false;


	imshow("recall_for_setup", SHOW_Label(SHOW_Label(choice_for_setup(),
		"Use Binarization image to ", true),
		"make model file?(OK/NO)"));

	while (true)
	{
		recall_for_esc = false;
		recall_for_ok = false;
		Key_quit_waitkey(100);
		if (recall_for_ok)
		{
			key_choose_thresh = true;
			break;
		}
		else if (recall_for_no)
			break;
	}

 	if (key_choose_thresh)
	{
		train_img=train_thresh;
		SHOW_once("You choose Thresh", 500);
	}
	else
	{
		train_img=train_gray;
		SHOW_once("You choose Gray", 500);
	}
	destroyAllWindows();
	
	FileStorage  location_file(LOC_FILE_NAME, FileStorage::WRITE);
	location_file << "ix1" << ix1 << "iy1" << iy1 << "ix2" << ix2 << "iy2" << iy2;
	location_file << "key_choose_thresh" << key_choose_thresh;
	location_file.release();


}

void Train_Process::Hog_Compute()
{
	//hog compute
	Mat img_hog;
	
	vector<float> result_hog;
	resize(train_img[0], img_hog, Size(32, 32));
	hog.compute(img_hog, result_hog);
	transpose(Mat(result_hog, CV_32FC1),   hog_result_to_svm);

	for (int img_cnt = 1; img_cnt<train_img.size(); img_cnt++)
	{
		Mat temp;
		resize(train_img[img_cnt], img_hog, Size(32, 32));
		hog.compute(img_hog, result_hog);
		transpose(Mat(result_hog, CV_32FC1), temp);
		vconcat(  hog_result_to_svm,temp.reshape(0, 1), hog_result_to_svm);
	}
}

void Train_Process::SVM_train(const String SVM_MODE_NAME ,const Mat label_data, const Mat image_data)
{
	Mat svm_image_data, svm_label_data;
	if (image_data.size()==Size(0,0) )
		svm_image_data = hog_result_to_svm;
	else
		svm_image_data = image_data;

	if (label_data.size() == Size(0, 0))
		svm_label_data = Mat(int_label);
	else
		svm_label_data = label_data;

	Ptr<ml::SVM> svm_ = ml::SVM::create();
	svm_->setC(100);
	svm_->setType(ml::SVM::C_SVC);
	svm_->setKernel(ml::SVM::RBF);
	svm_->setGamma(0.3);
	//////训练开始
	cout << "begin training" << endl;
	SHOW_once("prepare data welly", 1000, SHOW_once("begin training"));
	
	svm_->train(svm_image_data, ml::ROW_SAMPLE, svm_label_data);
	
	
	cout << "train Done" << endl;
	//Ptr<SVM> svm_ =  Algorithm::load<SVM>("svm.dat");////opencv310 下load函数并没有在SVM里面
	
	svm_->save(SVM_MODE_NAME);//如果没有二分类以上，这样子会出错。
}

void Train_Process::SVM_train_for_two(const String save_file_name)
{
	vector<int> train_label_copy;

	for (auto c : int_label)
	{
		if (c > 9)
			train_label_copy.push_back(1);
		else
			train_label_copy.push_back(0);
	}
	swap(train_label_copy, int_label);
	HOG_Init(Size(32, 32), Size(16, 16), Size(8, 16), Size(8, 8), 4);
	Hog_Compute();
	SVM_train(save_file_name);
	swap(train_label_copy, int_label);
}

void Train_Process::Separate_Number_For_Train(String LOC_FILE_NAME)
{
	
	vector<Mat > another_train_img;
	vector<int > another_int_label;
	vector<int > collect_middle_loc;

	for (int cnt = 0; cnt < train_img.size(); cnt++)
	{
		if (int_label[cnt] > 9)
		{
			collect_middle_loc.push_back(  find_separated_loc(train_img[cnt]));
		}
	}
	

	int max_loc = *max_element(collect_middle_loc.cbegin(), collect_middle_loc.cend());
	int min_loc = *min_element(collect_middle_loc.cbegin(), collect_middle_loc.cend());;
	
	if(max_loc-min_loc>10)
	for (auto cnt = collect_middle_loc.cbegin(); cnt != collect_middle_loc.cend(); cnt++)
	{
		if (*cnt == max_loc)
			collect_middle_loc.erase(cnt--);
		else if (*cnt == min_loc)
			collect_middle_loc.erase(cnt--);
	}

	int middle_loc = std::accumulate(collect_middle_loc.cbegin(), collect_middle_loc.cend(), 0) / collect_middle_loc.size();
	 
	for (int cnt = 0; cnt < train_img.size(); cnt++)
	{
		if (int_label[cnt] > 9)
		{
			int label_separate_to_ten = int_label[cnt] / 10,
				label_separate_to_one = int_label[cnt] % 10;
			Mat img1 = train_img[cnt].colRange(0, middle_loc).clone(), img2 = train_img[cnt].colRange(middle_loc, train_img[cnt].cols).clone();
			copyMakeBorder(img1, img1, 0, 0, 3, train_img[cnt].cols - middle_loc - 3, BORDER_CONSTANT, Scalar(0, 0, 0));
			copyMakeBorder(img2, img2, 0, 0, middle_loc - 3, 3, BORDER_CONSTANT, Scalar(0, 0, 0));

			another_train_img.push_back(img1.clone());
			another_train_img.push_back(img2.clone());
			another_int_label.push_back(label_separate_to_ten);
			another_int_label.push_back(label_separate_to_one);
		}
		else
		{
			another_train_img.push_back(train_img[cnt].clone());
			another_int_label.push_back(int_label[cnt]);
		}
	}
	FileStorage  location_file(LOC_FILE_NAME, FileStorage::APPEND);
	location_file << "middle_loc" << middle_loc ;
	location_file.release();

	train_img = another_train_img;
	int_label = another_int_label;
	HOG_Init();
	Hog_Compute();
	SVM_train();
}

void Train_Process::Add_Data(String file_name)
{
	FileStorage file_data(file_name, FileStorage::READ);
	assert(file_data.isOpened() );
	int store_num, label_num;
	file_data["store_sum"] >> store_num;
	Mat img;
	String img_name, label_name;
	for (int cnt = 0; cnt<store_num; cnt++)
	{
		img_name = "img_" + to_string(cnt);
		label_name = "label_" + to_string(cnt);
		file_data[img_name] >> img;
		file_data[label_name] >> label_num;
		train_img.push_back(img.clone());
		int_label.push_back(label_num);
	}

	file_data.release();
}

bool Train_Process::Save_Train_Data(String save_file_name)
{
	FileStorage  save_file(save_file_name, FileStorage::WRITE);
	if (!save_file.isOpened())
		return false;

	String img_name, label_name;
	save_file << "store_sum" << (int)train_img.size();
	for (int cnt = 0; cnt < train_img.size(); cnt++)
	{
		img_name = "img_" + to_string(cnt);
		label_name = "label_" + to_string(cnt);

		save_file << img_name << train_img[cnt];
		save_file << label_name <<  int_label[cnt];
	}
	save_file.release();
	return true;
}

bool Train_Process::AutoTrain()
{
	Camera_Write_To_File();
	DrawROI_Then_Choose_Image();
	Thresh_Then_Delete();
	HOG_Init();
	Hog_Compute();
	SVM_train();

	bool flag_detect_ten = false;
	for (int cnt = 0; cnt < int_label.size(); cnt++)
	{
		if (int_label[cnt] > 9)
		{
			SVM_train_for_two();
			Separate_Number_For_Train();
			Save_Train_Data();
			break;
		}
	}
	for (int cnt = 0; cnt < int_label.size(); cnt++)
	{
		if (int_label[cnt] > 9)
		{
			imshow("img", train_img[cnt]);
			if (waitKey(0) == 27);
		}
	}
	return true;
}

void Find_Separate_line(Mat A)
{
	vector<int> sum_row_mat;

	for (int cnt = 0; cnt < A.cols; cnt++)
	{
		sum_row_mat.push_back(sum(A.col(cnt))[0]);
	}

}

void Train_Process::Test(const String SVM_MODE_NAME, const String location_file_name)
{
	cap.open(0);
	assert(cap.isOpened());
	assert(SVM_MODE_NAME.length());
	Ptr<ml::SVM> svm_1 = ml::SVM::load("Svm_for_two.xml");
	Ptr<ml::SVM> svm_2 = ml::SVM::load(SVM_MODE_NAME);

	bool  thresh_flag;
	FileStorage  infile(location_file_name, FileStorage::READ);
	int x1 = 0, y1 = 0, y2 = 0, x2 = 0, middle_loc;
	infile["ix1"] >> x1;
	infile["iy1"] >> y1;
	infile["iy2"] >> y2;
	infile["ix2"] >> x2;
	infile["key_choose_thresh"] >> thresh_flag;
	infile["middle_loc"] >> middle_loc;
	Rect ROI(Point(x1, y1), Point(x2, y2));

	Mat choice_label = Mat::zeros(Size(80, 40), CV_8UC3), show1_label;
	rectangle(choice_label, Rect(1, 1, 79, 39), Scalar(0, 255, 255), 2);
	putText(choice_label, "QUIT", Point(5, 30), FONT_HERSHEY_COMPLEX, 0.95, Scalar(0, 255, 255), 2);
	vector<float>  result_hog;
	int num_predict =0 ;
	//HOG_Init(Size(32, 32), Size(16, 16), Size(8, 16), Size(8, 8), 4);
	
	while (true)
	{
		cap >> image;
		assert(!image.empty());
		Mat img = image(ROI).clone();
		Mat separate_for_ten, separate_for_one;
		if (thresh_flag) {
			Mat img_thresh = Thresh_Process_Func(img);

			cv::resize(img_thresh, img_thresh, Size(32, 32));
			imshow("img_thresh", img_thresh);
			waitKey(1);

			HOG_Init(Size(32, 32), Size(16, 16), Size(8, 16), Size(8, 8), 4);
			hog.compute(img_thresh, result_hog);
			num_predict = svm_1->predict(Mat(result_hog, CV_32FC1).reshape(0, 1));

			if (num_predict)
			{
				HOG_Init();
				int medle_loc = find_separated_loc(img_thresh);
				Mat img1 = img_thresh.colRange(0, medle_loc), img2 = img_thresh.colRange(medle_loc, img_thresh.cols);
			}
			else
			{
				HOG_Init();
				hog.compute(img_thresh, result_hog);
				num_predict = svm_2->predict(Mat(result_hog, CV_32FC1).reshape(0, 1));

				show1_label = SHOW_Label(choice_label, "test predict: " + to_string(num_predict), image.cols - 80);
				cout << "test:" << num_predict << endl;
			}
		}
		else {
			cvtColor(img, img, COLOR_BGR2GRAY);
			Mat img2;
			cv::resize(img, img2, Size(32, 32));
			HOG_Init(Size(32, 32), Size(16, 16), Size(8, 16), Size(8, 8), 4);
			hog.compute(img2, result_hog);
			num_predict = svm_1->predict(Mat(result_hog, CV_32FC1).reshape(0, 1));
			HOG_Init();

			if (num_predict)
			{
				separate_for_ten=img.colRange(0, middle_loc).clone();
				separate_for_one=img.colRange(middle_loc, img.cols).clone();
				copyMakeBorder(separate_for_ten, separate_for_ten, 0, 0, 3, img.cols - middle_loc - 3, BORDER_CONSTANT, Scalar(0, 0, 0));
				copyMakeBorder(separate_for_one, separate_for_one, 0, 0, middle_loc - 3, 3, BORDER_CONSTANT, Scalar(0, 0, 0));
				
				//imshow("separate_for_ten", separate_for_ten);
				//imshow("separate_for_one", separate_for_one);
				//if (waitKey(0) == 27)
				//	destroyAllWindows();
				cv::resize(separate_for_one, separate_for_one, Size(32, 32));
				cv::resize(separate_for_ten, separate_for_ten, Size(32, 32));
				hog.compute(separate_for_ten, result_hog);
				num_predict=svm_2->predict(Mat(result_hog, CV_32FC1).reshape(0, 1));
				hog.compute(separate_for_one, result_hog);
				num_predict = num_predict*10+svm_2->predict(Mat(result_hog, CV_32FC1).reshape(0, 1));


			}
			else
			{
				cv::resize(img, img, Size(32, 32));
				hog.compute(img, result_hog);
				num_predict = svm_2->predict(Mat(result_hog, CV_32FC1).reshape(0, 1));
			}
			show1_label = SHOW_Label(choice_label, "test predict: " + to_string(num_predict),  image.cols - 80);
			cout << "test:" << num_predict << endl;
		}

	}



	cap.release();

}