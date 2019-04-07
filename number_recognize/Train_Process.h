#ifndef __TRAIN_PROCESS_HPP__
#define __TRAIN_PROCESS_HPP__

#pragma once
#include	<opencv2/core.hpp>
#include	<opencv2/core/utility.hpp>
#include	<opencv2/videoio.hpp>
#include	<opencv2/video/video.hpp>
#include	<opencv2/highgui.hpp>
#include	<opencv2/imgproc.hpp>
#include	<opencv2/ml/ml.hpp>
#include	<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Train_Process
{
public:
	Train_Process();
	~Train_Process();

	void	Camera_Write_To_File(String train_file="train.avi");
	void	HOG_Init(Size	WinSize = Size(32, 32),	
					Size BlockSize = Size(12, 16),
					Size BlockStride = Size(4, 8),
					Size CellSize = Size(6, 8),	
					int	Nbins = 9);
	void	DrawROI_Then_Choose_Image(String video_file= "train.avi");
	void	Thresh_Then_Delete(String LOC_FILE_NAME = "location.xml");
	void	SVM_train(const Mat label_data=Mat(), const Mat image_data=Mat(),String SVM_MODE_NAME = "svm.xml");
	void	Add_Data(String file_name = "img.xml");
	void	hog_compute();
	bool	Save_Train_Data(String save_file_name ="Train_Data.xml");
	Mat		Thresh_Process_Func(const Mat  image);
	bool	AutoTrain();
private:
	VideoCapture cap;
	HOGDescriptor hog;

	Mat image;
	vector<Mat> train_img;
	vector<int> int_label;
	Mat hog_result_to_svm;
};

#endif