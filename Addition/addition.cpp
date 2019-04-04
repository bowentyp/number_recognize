#include	<opencv2\core.hpp>
#include	<opencv2\highgui.hpp>
#include	<opencv2\imgproc.hpp>
#include	<iostream> 
#include	<fstream>  
#include	<io.h>
#include	<opencv2/opencv.hpp>

using namespace cv;
using namespace std;
void take();
void store();
void concat();
int main()
{
	 
	//Size	WinSize = Size(32, 32),
	//	BlockSize = Size(12, 16),
	//	BlockStride = Size(4, 8),
	//	CellSize = Size(6, 8);
	//int		Nbins = 9;
	//HOGDescriptor hog(WinSize, BlockSize, BlockStride, CellSize, Nbins);

	//vector<float> result_hog;
	//Mat img = imread("C:\\A_Cprj\\number_recognize\\number_data\\3\\1_3.jpg",0);
	//resize(img, img, Size(32, 32));
	//hog.compute(img, result_hog);
	//cout << result_hog.size() << endl;
	//waitKey(1);
	//system("pause");
	//return 0;
}

void take()
{
	FileStorage file_img("img.xml", FileStorage::READ);
	FileStorage file_label("label.xml", FileStorage::READ);
	int store_num,label_num;
	file_label["store_sum"] >> store_num;
	cout << "store_num:" <<store_num << endl;
	Mat img;
	String img_name, label_name;
	for (int cnt =0;cnt<store_num;cnt++)
	{
		img_name	="img_"+	to_string(cnt);
		label_name = "label_"+	to_string(cnt);
		file_img[img_name] >> img;
		file_label[label_name] >> label_num;
		imshow("img", img);
		cout << "label:" << label_num << endl;
		if (waitKey(0) == 27)
			break;		
	}

	file_img.release();
	file_label.release();
 

}



void store()
{ 
	string path = "C:\\A_Cprj\\number_recognize\\number_data\\";
	FileStorage file_img("img.xml", FileStorage::WRITE);
	FileStorage file_label("label.xml", FileStorage::WRITE);
	vector<string> filename;
	//vector<Mat>		img_list;
	//vector<string>		img_label;
	int store_num = 0;
	for (int num = 0; num <= 39; num++)
	{
		cout << path + to_string(num) + "\\*" << endl;
		glob(path + to_string(num) + "\\*", filename);
		for (auto C : filename)
		{
			//img_list.push_back();
			//img_label.push_back(to_string());
			file_img << "img_" + to_string(store_num)<<imread(C, 0);
			file_label << "label_" + to_string(store_num)<<num;
			store_num++;
		}
	}
	file_label << "store_sum" << store_num;
	file_img.release();
	file_label.release();
	//ofstream file_img("img.txt", ios::out | ios::binary), file_label("label.txt", ios::out | ios::binary);
	//num = 0;
	//while (num < img_list.size() || num < img_label.size())
	//{
	//	file_img << img_list[num] << endl;
	//	file_label << img_label[num] << endl;
	//	num++;
	//}
	//file_img.close();
	//file_label.close();
}

void concat()
{
	vector<string> filename;
	vector<Mat>		img_;
	vector<vector<Mat> >	img_list;
	for (int cnt = 0; cnt < 10; cnt++)
	{
		string path = "C:\\A_Cprj\\number_recognize\\Addition\\" + to_string(cnt) + "\\*";
		glob(path, filename);
		for (auto C : filename)
		{
			Mat hh = imread(C, 0);
			threshold(hh, hh, 127, 255, THRESH_BINARY);
			img_.push_back(hh);
		}
		img_list.push_back(img_);
		img_.clear();
	}
	Mat img;
	for (int typ_r = 1; typ_r < 4; typ_r++)
	{
		for (int typ_c = 0; typ_c < 10; typ_c++)
		{
			if (_access(to_string(10 * typ_r + typ_c).c_str(), 0) == -1)
			{
				string path1 = "mkdir " + to_string(10 * typ_r + typ_c);
				system(path1.c_str());
			}
			else
				cout << "file exsited" << endl;
			for (int typ_num = 0; typ_num < 35; typ_num++)
			{
				hconcat(img_list[typ_r][typ_num], img_list[typ_c][typ_num], img);

				vector<int> num_row;
				for (int i = 0; i < img.size[1]; i++)
				{
					if (sum(img.col(i))[0] != 0)
						num_row.push_back(i);
				}


				Mat img1;
				int i = 0;
				for (; i < num_row.size() - 1; i++)
				{
					if (num_row[i + 1] - num_row[i] != 1)
					{
						hconcat(img.colRange(num_row[0], num_row[i]), Mat::zeros(Size(10, 50), img.type()), img1);
						break;
					}
				}
				hconcat(img1, img.colRange(num_row[i + 1], num_row[num_row.size() - 1]), img1);

				copyMakeBorder(img1, img1, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0, 0, 0));
				resize(img1, img1, Size(50, 50));
				string path1 = "C:\\A_Cprj\\number_recognize\\Addition\\" + to_string(10 * typ_r + typ_c) + "\\" + to_string(typ_num) + "_" + to_string(10 * typ_r + typ_c) + ".jpg";

				imwrite(path1, img1);
			}
		}
	}
}