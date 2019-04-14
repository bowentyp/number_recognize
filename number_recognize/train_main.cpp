#include "Train_Process.h"
#include <iostream>
#include <numeric>


using namespace std;

//void MovingAverage(Mat &A, int N);
//void MovingAverage(vector<float> &V, int N);
//vector<int> find_wavetop(vector<float> A);
//vector<int> find_wavetop(Mat A);
int find_separated_loc(const Mat Img);





int main(int argc, char ** argv)
{
	FileStorage  save_file("Train_Data.xml", FileStorage::READ);
	if (!save_file.isOpened())
	{
		cout << "error read file" << endl;
		return -1;
	}
	String img_name, label_name; 
	Mat train_img;
	int train_img_size,int_label;
	save_file["store_sum" ]>> train_img_size;
	for (int cnt = 0; cnt <train_img_size; cnt++)
	{
		img_name = "img_" + to_string(cnt);
		label_name = "label_" + to_string(cnt);

		save_file[img_name] >> train_img ;
		save_file[ label_name ]>> int_label ;
		imshow("img", train_img);
		cout << int_label << endl;
		if (waitKey(0) == 27);
	}

	//Train_Process number_recognizer;
	//number_recognizer.Add_Data();
	//number_recognizer.Test() ;
	//number_recognizer.AutoTrain();
	//	//cout << argc << endl << argv[1] << endl;
	//	if (argc > 1 && argv[1][0] == '1')
	//		 number_recognizer.AutoTrain();
	//	else 
	//		number_recognizer.Test("Svm_for_two.xml");


	//Mat img = imread("SYC (26).jpg",0 );
	//threshold(img, img, 127, 255, THRESH_BINARY_INV);
	//resize(img, img, Size(64, 64));
	//imshow("imsf", img);
	//
	//int  min_index = find_separated_loc(img);

	//Mat img1 = img.colRange(0, min_index),img2=img.colRange(  min_index, img.cols);
	//imshow("img1", img1);
	//imshow("img2", img2);
	//waitKey(0);



	system("pause");
	return 0;
}

//#include	<iostream>
//#include	<assert.h>
//#include	<fstream>
//#include	<opencv2/core.hpp>
//#include	<opencv2/core/utility.hpp>
//#include	<opencv2/videoio.hpp>
//#include	<opencv2/video/video.hpp>
//#include	<opencv2/highgui.hpp>
//#include	<opencv2/imgproc.hpp>
//#include	<opencv2/ml/ml.hpp>
//#include	<opencv2/opencv.hpp>
////#include	<opencv/ml.h>////opencv version==310
//
//#include	<time.h>
//
//using namespace cv;
//using namespace cv::ml;
//using namespace std;
//
//
//
//Size	WinSize = Size(32, 32),
//BlockSize = Size(12, 16),
//BlockStride = Size(4, 8),
//CellSize = Size(6, 8);
//int		Nbins = 9;
////Size	WinSize = Size(50, 50),
////BlockSize = Size(10, 30),
////BlockStride = Size(5, 10),
////CellSize = Size(5, 10);
////int		Nbins = 9;
//HOGDescriptor hog( WinSize, BlockSize, BlockStride, CellSize, Nbins);
//
//Mat img;
//int ix1 = -1, iy1 = -1, ix2 = -1, iy2 = -1;
//bool drawing_ = false;
//bool recall_for_back = false, recall_for_next = false, recall_for_esc = false, recall_for_no = false, recall_for_ok = false;
//vector<Mat> train_img;
//vector<int> chars_label;
//const String	LOC_FILE_NAME = "C:\\A_Cprj\\number_recognize\\function\\location.xml",
//				SVM_MODE_NAME = "C:\\A_Cprj\\number_recognize\\function\\svm.xml";
//
//void  Key_quit_waitkey(int waittime);
//void recall_for_setup(int event, int x, int y, int flags, void* userdata);
//void recall_for_draw(int event, int x, int y, int flags, void* userdata);
//void MovingAverage(Mat &A, int N);
//vector<int> find_wavetop(Mat A);
//Mat choice_for_setup();
//Mat SHOW_Label(Mat choice_label, string s1, bool lines = false);
//Mat SHOW_once(string s1, int overtime = 0, Mat A = Mat());
//Mat Thresh_Then_Delete();
//void addition(vector<Mat> &img_, vector<int> &label_);
//void Camera_Write_To_File();
//void DrawROI_Then_Choose_Image();
//Mat Thresh_Process_Func(const Mat );
//void SVM_train(const Mat svm_label_data,const Mat svm_image_data);
//void SVM_FIRST_Train(const Mat svm_label_data, const Mat svm_image_data);
//int main()
//{ 
//	Camera_Write_To_File();
//	DrawROI_Then_Choose_Image(); 
//	//addition(chars_img,chars_label);
//	Mat img = Thresh_Then_Delete();
//	SVM_FIRST_Train(Mat(chars_label), img);
//	SVM_train(Mat(chars_label), img);
//
//	return 0;
//
//}
//
//void SVM_FIRST_Train( const Mat  image_data, const Mat  label_data)
//{
//	Ptr<ml::SVM> svm_ = ml::SVM::create();
//	svm_->setC(100);
//	svm_->setType(SVM::C_SVC);
//	svm_->setKernel(SVM::RBF);
//	svm_->setGamma(0.3);
//	Mat label_data;
//
//
//
//
//	cout << "begin training number separation" << endl;
//	svm_->train( image_data, ROW_SAMPLE,  label_data);
//
//}
//
//void Camera_Write_To_File()
//{
//	namedWindow("recall_for_setup");
//	setMouseCallback("recall_for_setup", recall_for_setup);
//	imshow("recall_for_setup", SHOW_Label(SHOW_Label(choice_for_setup(),
//		"When camera ready,clik OK", true),
//		"to record ; ESC to quit"));
//	waitKey(1);
//
//	VideoCapture cap(0);
//	//cap.set(CAP_PROP_AUTO_EXPOSURE, 1);
//	int fourcc = VideoWriter::fourcc('M', 'P', '4', '2');
//	auto video_writor = VideoWriter("train.avi", fourcc, 10,
//		Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), true);
//
//	Mat frame;
//	int cnt_record = 0;
//	while (true)
//	{
//		cap >> frame;
//		assert(!frame.empty());
//		imshow("frame", frame);
//		Key_quit_waitkey(500);
//		if (recall_for_ok)
//		{
//			video_writor.write(frame);
//			cnt_record++;
//			choice_label_show = SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
//				"When camera ready,clik OK", true),
//				"to record ; ESC to quit"),
//				"record " + to_string(cnt_record) + " pics");
//			imshow("recall_for_setup", choice_label_show);
//			cout << "frame record:" << cnt_record << " pics" << endl;
//		}
//		if (recall_for_esc)
//		{
//			if (cnt_record == 0)
//			{
//				SHOW_once("clik OK to record", 1000);
//				continue;
//			}
//			break;
//		}
//	}
//	destroyAllWindows();
//	video_writor.release();
//	cap.release();
//}
//
//void DrawROI_Then_Choose_Image()
//{
//	VideoCapture cap("train.avi");
//
//	int label_cnt = 1;
//	namedWindow("img");
//	setMouseCallback("img", recall_for_draw);
//	namedWindow("recall_for_setup");
//	setMouseCallback("recall_for_setup", recall_for_setup);
//	imshow("recall_for_setup", SHOW_Label(SHOW_Label(choice_for_setup(),
//									"Draw fixed ROI Firstly", true),
//									"Then adjust label(BACK/NEXT)"));
//	Mat frame;
//	for (int cnt_cap = 0; cnt_cap < cap.get(CAP_PROP_FRAME_COUNT); cnt_cap++)
//	{
//		cap.read(frame);
//		img = frame.clone();
//		rectangle(img, Point(ix1, iy1), Point(ix2, iy2), Scalar(0, 255, 0), 2);
//		imshow("img", img);
//
//		while (true)
//		{
//			recall_for_back = false;
//			recall_for_next = false;
//			recall_for_esc = false;
//			recall_for_ok = false;
//			recall_for_no = false;
//			Key_quit_waitkey(100);
//			if (recall_for_ok)
//			{
//				if (ix1 == -1 || iy1 == -1 || ix2 == -1 || iy2 == -1)
//				{
//					SHOW_once("Draw ROI firstly", 1000);
//					continue;
//				}
//				train_img.push_back(frame(Rect(Point(ix1, iy1), Point(ix2, iy2))).clone());
//				chars_label.push_back(label_cnt);
//				cout << "comfirm the label cnt:" << label_cnt << endl;
//
//				break;
//			}
//			if (recall_for_back)
//			{
//				label_cnt--;
//				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
//					"Add to train(OK/NO),Quit(ESC)", true),
//					"Change label(BACK/NEXT)"),
//					"NOW label is: " + to_string(label_cnt)));
//			}
//			if (recall_for_next)
//			{
//				label_cnt++;
//				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
//					"Add to train(OK/NO),Quit(ESC)", true),
//					"Change label(BACK/NEXT)"),
//					"NOW label is: " + to_string(label_cnt)));
//			}
//			if (recall_for_no)
//			{
//				if (ix1 == -1 || iy1 == -1 || ix2 == -1 || iy2 == -1)
//				{
//					SHOW_once("Draw ROI firstly", 1000);
//					continue;
//				}
//				cout << "you have not choose this frame" << endl;
//				break;
//			}
//			if (recall_for_esc)
//			{
//				if (ix1 == -1 || iy1 == -1 || ix2 == -1 || iy2 == -1)
//				{
//					SHOW_once("Draw ROI firstly", 1000);
//					continue;
//				}
//				SHOW_once("You have leaved", 500);
//				break;
//			}
//		}
//		if (recall_for_esc)
//			break;
//	}
//
//	destroyAllWindows();
//	cap.release();
//}
//
//Mat Thresh_Then_Delete()
//{
//	vector<int> numdelete;
//	vector<Mat> train_thresh;
//	namedWindow("recall_for_setup");	
//	setMouseCallback("recall_for_setup", recall_for_setup);
//	for (int num_record = 0; num_record < train_img.size(); num_record++)
//	{
//		train_thresh.push_back(Thresh_Process_Func(train_img[num_record]));
//	}
//	//////做一些删减 
//	imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
//								"Delete this image(OK)", true),
//								"Not delete(NEXT)"),
//								"NOW label is:" + to_string(chars_label[0])));
//	for (int num_record1 = 0; num_record1 < train_img.size(); num_record1++)
//	{
//		imshow("img_thresh", train_thresh[num_record1]);
//		imshow("train_img", train_img[num_record1]);
//		waitKey(1);
//		while (true)
//		{
//			recall_for_ok = false;
//			recall_for_next = false;
//			recall_for_back = false;
//
//			Key_quit_waitkey(50);//can't lose this sentense
//			if (recall_for_ok)
//			{
//				if (num_record1 < 0)
//				{
//					SHOW_once("NO images", 500);
//					break;
//				}
//				SHOW_once("Delete Done:" + to_string(chars_label[num_record1]), 500);
//				train_img.erase(train_img.cbegin() + num_record1);
//				train_thresh.erase(train_thresh.cbegin() + num_record1);
//				chars_label.erase(chars_label.cbegin() + num_record1);
//				num_record1--;
//				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
//											"Delete this image(OK)", true),
//											"Not delete(NEXT)"),
//											"NOW label is:" + to_string(chars_label[num_record1 + 1])));
//				break;
//
//			}
//			if (recall_for_back)
//			{
//				if (num_record1 == 0)
//				{
//					SHOW_once("NO images", 500);
//					continue;
//				}
//				num_record1--;
//				imshow("img_thresh", train_thresh[num_record1]);
//				imshow("train_img", train_img[num_record1]);
//				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
//												"Delete this image(OK)", true),
//												"Not delete(NEXT)"),
//												"NOW label is:" + to_string(chars_label[num_record1])));
//				recall_for_back = false;
//				waitKey(50);
//				continue;
//			}
//			if (recall_for_next)
//			{
//				imshow("recall_for_setup", SHOW_Label(SHOW_Label(SHOW_Label(choice_for_setup(),
//												"Delete this image(OK)", true),
//												"Not delete(NEXT)"),
//												"NOW label is:" + to_string(chars_label[num_record1 + 1])));
//				break;
//			}
//		}
//	}
//	destroyWindow("img_thresh");
//	destroyWindow("train_img");
//
//
//	bool key_choose_thresh = false;
//	 
//
//	imshow("recall_for_setup", SHOW_Label(SHOW_Label(choice_for_setup(),
//								"Use Binarization image to ", true),
//								"make model file?(OK/NO)"));
//
//	while (true)
//	{
//		recall_for_esc = false;
//		recall_for_ok = false;
//		Key_quit_waitkey(100);
//		if (recall_for_ok)
//		{
//			key_choose_thresh = true;
//			break;
//		}
//		else if (recall_for_no)
//			break;
//	}
//
//	vector<Mat> chars_img;
//	if (key_choose_thresh)
//	{
//		chars_img = train_thresh;
//		SHOW_once("You choose Thresh", 500);
//	}
//	else
//	{
//		chars_img = train_img;
//		SHOW_once("You choose Gray", 500);
//	}
//
//	FileStorage  location_file(LOC_FILE_NAME, FileStorage::WRITE);
//	location_file << "ix1" << ix1 << "iy1" << iy1 << "ix2" << ix2 << "iy2" << iy2;
//	location_file << "key_choose_thresh" << key_choose_thresh;
//	location_file.release();
//	
//	Mat hog_result_transpose;
//	vector<float> result_hog;
//	resize(chars_img[0], img, Size(32, 32));
//	hog.compute(img, result_hog);
//	transpose(Mat(result_hog, CV_32FC1), hog_result_transpose);
//
//	for (int img_cnt = 1; img_cnt<chars_img.size(); img_cnt++)
//	{
//		Mat temp;
//		resize(chars_img[img_cnt], img, Size(32, 32));
//		hog.compute(img, result_hog);
//		transpose(Mat(result_hog, CV_32FC1), temp);
//		vconcat(hog_result_transpose, temp.reshape(0, 1), hog_result_transpose);
//	}
//	destroyAllWindows();
//	return hog_result_transpose;
//}
//
//void SVM_train(const Mat svm_label_data, const Mat svm_image_data)
//{
//	Ptr<ml::SVM> svm_ = ml::SVM::create();
//	svm_->setC(100);
//	svm_->setType(SVM::C_SVC);
//	svm_->setKernel(SVM::RBF);
//	svm_->setGamma(0.3);
//	//////训练开始
//	cout << "begin training" << endl;
//	SHOW_once("prepare data welly", 1000, SHOW_once("begin training"));
//	svm_->train(svm_image_data, ROW_SAMPLE, svm_label_data);
//	cout << "train Done" << endl;
//	//Ptr<SVM> svm_ =  Algorithm::load<SVM>("svm.dat");////opencv310 下load函数并没有在SVM里面
//
//	
//	svm_->save(SVM_MODE_NAME);
//}
//
//Mat Thresh_Process_Func(const Mat  image) {
//	//////做二值化
// 
//		Mat  B, G, R, histormB, histormG, histormR;
//		vector<Mat> channel_split;
//		int hisnum = 255;
//		const float range[] = { 0,255 };
//		const float* histRange = { range };
//		/////分量提取并作后续工作
//		split(image.clone(), channel_split);
//
//		B = channel_split.at(0);
//		G = channel_split.at(1);
//		R = channel_split.at(2);
//
//		calcHist(&B, 1, 0, Mat(), histormB, 1, &hisnum, &histRange, true, false);
//		calcHist(&G, 1, 0, Mat(), histormG, 1, &hisnum, &histRange, true, false);
//		calcHist(&R, 1, 0, Mat(), histormR, 1, &hisnum, &histRange, true, false);
//
//		MovingAverage(histormB, 50);
//		MovingAverage(histormG, 50);
//		MovingAverage(histormR, 50);
//
//		vector<int> wavetopB = find_wavetop(histormB);
//		vector<int> wavetopG = find_wavetop(histormG);
//		vector<int> wavetopR = find_wavetop(histormR);
//
//		int B_diff = *(wavetopB.cend() - 1) - *(wavetopB.cbegin() + 1);
//		int G_diff = *(wavetopG.cend() - 1) - *(wavetopG.cbegin() + 1);
//		int R_diff = *(wavetopR.cend() - 1) - *(wavetopR.cbegin() + 1);
//
//		bool flag = false;
//
//		if (B_diff > 150 && (G_diff > 150 || R_diff > 150))
//			flag = true;
//		else if (G_diff > 150 && R_diff > 150)
//			flag = true;
//		Mat img_thresh;
//
//		int loc_thresh = 127;
//		if (!flag)
//		{
//			int max_value = max(max(B_diff, G_diff), R_diff);
//
//			if (B_diff == max_value)
//			{
//				img_thresh = B;
//				loc_thresh = (*(wavetopB.cend() - 1) + *(wavetopB.cbegin() + 1)) / 2;
//			}
//			else if (G_diff == max_value)
//			{
//				img_thresh = G;
//				loc_thresh = (*(wavetopG.cend() - 1) + *(wavetopG.cbegin() + 1)) / 2;
//			}
//			else
//			{
//				img_thresh = R;
//				loc_thresh = (*(wavetopR.cend() - 1) + *(wavetopR.cbegin() + 1)) / 2;
//			}
//		}
//		else
//			cvtColor(image.clone(), img_thresh, COLOR_BGR2GRAY);
//		threshold(img_thresh, img_thresh, loc_thresh, 255, THRESH_BINARY);
//		return img_thresh;
//}
//
//void  Key_quit_waitkey(int waittime)
//{
//	if (waitKey(waittime) == 27)
//		exit(0);
//}
//
//void recall_for_setup(int event, int x, int y, int flags, void* userdata)
//{
//	if (event == EVENT_LBUTTONUP)
//	{
//		if (y > 5 && y < 58)
//		{
//			if (x > 5 && x < 95)
//				recall_for_back = true;
//			else	if (x > 105 && x < 195)
//				recall_for_next = true;
//			else	if (x > 205 && x < 295)
//				recall_for_ok = true;
//			else	if (x > 305 && x < 395)
//				recall_for_no = true;
//			else	if (x > 405 && x < 495)
//				recall_for_esc = true;
//		}
//	}
//}
//
//void recall_for_draw(int event, int x, int y, int flags, void* userdata)
//{
//	static int weight;
//	//当按下左键时返回起始位置坐标
//	if (event == EVENT_LBUTTONDOWN)
//	{
//		drawing_ = true;
//		ix1 = x; iy1 = y;
//	}
//	if (event == cv::EVENT_MOUSEMOVE)// && ( flags == EVENT_FLAG_LBUTTON))
//	{
//		if (drawing_)
//		{
//			Mat img2 = img.clone();
//			weight = (abs(x - ix1) > abs(y - iy1)) ? abs(x - ix1) : abs(y - iy1);
//			rectangle(img2, Point(ix1, iy1), Point(ix1+weight, iy1+weight), Scalar(0, 255, 255), 1);
//			line(img2, Point(ix1 +int( weight/2), iy1), Point(ix1 + int(weight / 2), iy1 + weight),Scalar(0, 255, 255), 1);
//			imshow("img", img2);
//		}
//	}
//	//当鼠标松开时停止绘图
//	if (event == EVENT_LBUTTONUP)
//	{
//		drawing_ = false;
//		
//			ix2 = ix1 + weight; iy2 = iy1 + weight;
//	}
//}
//
//void MovingAverage(Mat &A, int N)
//{
//	assert((A.size[0] == 1) ^ (A.size[1] == 1));
//	int sizenum = (A.size[0] == 1) ? A.size[1] : A.size[0];
//	if (A.size[0] == 1)
//	{
//		for (int a = 0; a < sizenum - N; a++)
//			A.at<float>(0, a) = sum(A.colRange(a, a + N))[0] / N;
//		for (int a = sizenum - N; a < sizenum; a++)
//			A.at<float>(0, a) = 0;
//
//		hconcat(Mat::zeros(Size(1, 1), CV_32FC1), A, A);
//	}
//	else
//	{
//		for (int a = 0; a < sizenum - N; a++)
//			A.at<float>(a, 0) = sum(A.rowRange(a, a + N))[0] / N;
//		for (int a = sizenum - N; a <sizenum; a++)
//			A.at<float>(a, 0) = 0;
//
//		vconcat(Mat::zeros(Size(1, 1), CV_32FC1), A, A);
//
//	}
//}
//
//vector<int> find_wavetop(Mat A)
//{
//	vector<int > store_int = { 0 };
//	int sizenum = (A.size[0] <A.size[1]) ? A.size[1] : A.size[0];
//	int num = 0;
//	for (int a = 0; a < sizenum - 1; a++)
//	{
//		if (A.at<float>(a) < A.at<float>(a + 1))
//			num = a;
//		if (A.at<float>(a) > A.at<float>(a + 1) && num != *(store_int.cend() - 1))
//			store_int.push_back(num);
//	}
//
//	return store_int;
//}
//
//Mat choice_for_setup()
//{
//	Mat choice_label = Mat::zeros(Size(500, 60), CV_8UC3);
//	rectangle(choice_label, Rect(5, 5, 90, 53), Scalar(0, 255, 255), 2);
//	rectangle(choice_label, Rect(105, 5, 90, 53), Scalar(0, 255, 255), 2);
//	rectangle(choice_label, Rect(205, 5, 90, 53), Scalar(0, 255, 255), 2);
//	rectangle(choice_label, Rect(305, 5, 90, 53), Scalar(0, 255, 255), 2);
//	rectangle(choice_label, Rect(405, 5, 90, 53), Scalar(0, 255, 255), 2);
//	putText(choice_label, "BACK", Point(8, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);
//	putText(choice_label, "NEXT", Point(108, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);
//	putText(choice_label, "OK", Point(230, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);
//	putText(choice_label, "NO", Point(330, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);
//	putText(choice_label, "ESC", Point(420, 40), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2);
//
//	return choice_label;
//}
//
//Mat SHOW_Label(Mat choice_label, string s1, bool lines  )
//{
//	Mat temp;
//	Mat SHOW_Label1 = Mat::zeros(Size(500, 40), CV_8UC3);
//	if (lines)
//		line(SHOW_Label1, Point(0, 5), Point(500, 5), Scalar(0, 255, 0), 3);
//	putText(SHOW_Label1, s1, Point(5, 30), FONT_HERSHEY_COMPLEX, 0.95, Scalar(0, 255, 0), 2);
//	vconcat(choice_label, SHOW_Label1, temp);
//	return temp;
//}
//
//Mat SHOW_once(string s1, int overtime  , Mat A )
//{
//	Mat temp = Mat::zeros(Size(500, 40), CV_8UC3);
//	putText(temp, s1, Point(5, 30), FONT_HERSHEY_COMPLEX, 0.95, Scalar(0, 255, 0), 2);
//	if (A.size() != Size(0, 0))
//		vconcat(temp, A, temp);
//
//	if (overtime)
//	{
//		imshow("show", temp);
//		waitKey(overtime);
//		destroyWindow("show");
//	}
//	return temp;
//}
//
//void addition(vector<Mat> &img_,vector<int> &label_)
//{
//	FileStorage file_img("img.xml", FileStorage::READ);
//	FileStorage file_label("label.xml", FileStorage::READ);
//	assert(file_img.isOpened() && file_label.isOpened());
//	int store_num, label_num;
//	file_label["store_sum"] >> store_num;
//	Mat img;
//	String img_name, label_name;
//	for (int cnt = 0; cnt<store_num; cnt++)
//	{
//		img_name = "img_" + to_string(cnt);
//		label_name = "label_" + to_string(cnt);
//		file_img[img_name] >> img;
//		file_label[label_name] >> label_num;
//		img_.push_back(img);
//		label_.push_back(label_num);
//	}
//
//	file_img.release();
//	file_label.release();
//}
