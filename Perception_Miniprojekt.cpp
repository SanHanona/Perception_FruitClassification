#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;


vector<Mat> loadFolder(String folder, vector<Mat> &vecSize);
vector<Mat> thres(vector<Mat> vecHSV);
vector<vector<double>> conturs(vector<Mat> morphClose, vector<Mat> &vecSize, vector<Mat> vecHSV);


int main()
{
	vector<Mat> vecApple;
	vector<Mat> vecOrange;
	vector<Mat> vecBanana;
	vector<Mat> vecTest;

	///////////////    Behandling af æble billeder   ////////////////////////////
	vector<Mat> apples = loadFolder("æble", vecApple);
	vector<Mat> MORPHapples = thres(apples);
	vector<vector<double>> conApple = conturs(MORPHapples, vecApple, apples);

	///////////////    Behandling af appelsin billeder   ////////////////////////
	vector<Mat> oranges = loadFolder("appelsin", vecOrange);
	vector<Mat> MORPHoranges = thres(oranges);
	vector<vector<double>> conOrange = conturs(MORPHoranges, vecOrange, oranges);

	///////////////    Behandling af banan billeder   //////////////////////////
	vector<Mat> bananas = loadFolder("banan", vecBanana);
	vector<Mat> MORPHbananas = thres(bananas);
	vector<vector<double>> conBanana = conturs(MORPHbananas, vecBanana, bananas);

	///////////////    Behandling af test billeder   ///////////////////////////
	vector<Mat> test = loadFolder("frugtTest", vecTest);
	vector<Mat> MORPHtest = thres(test);
	vector<vector<double>> conTest = conturs(MORPHtest, vecTest, test);
	

	///////////////    Gemmer features for æbler i "Apples.txt"   //////////////
	ofstream out1("Apples.txt");
	for (int i = 0; i < 16; i++)
	{
		cout << i << ":  ";
		for (int j = 0; j < conApple.size(); j++)
		{
			out1 << conApple[j][i] << ",";
			cout << conApple[j][i] << "  ";
		}
		cout << endl;
		out1 << endl;
	}
	out1.close();

	///////////////    Gemmer features for appelsiner i "Orange.txt"   ////////
	ofstream out2("Orange.txt");
	for (int i = 0; i < 16; i++)
	{
		cout << i << ":  ";
		for (int j = 0; j < conOrange.size(); j++)
		{
			out2 << conOrange[j][i] << ",";
			cout << conOrange[j][i] << "  ";
		}
		cout << endl;
		out2 << endl;
	}
	out2.close();

	///////////////    Gemmer features for bananer i "Banana.txt"   /////////
	ofstream out3("Banana.txt");
	for (int i = 0; i < 16; i++)
	{
		cout << i << ":  ";
		for (int j = 0; j < conBanana.size(); j++)
		{
			out3 << conBanana[j][i] << ",";
			cout << conBanana[j][i] << "  ";
		}
		cout << endl;
		out3 << endl;
	}
	out3.close();

	///////////////    Gemmer features for bananer i "Banana.txt"   /////////
	ofstream out4("Test.txt");
	for (int i = 0; i < 16; i++)
	{
		cout << i << ":  ";
		for (int j = 0; j < conTest.size(); j++)
		{
			out4 << conTest[j][i] << ",";
			cout << conTest[j][i] << "  ";
		}
		cout << endl;
		out4 << endl;
	}
	out4.close();

	return 0;
}


/////////////////////////////////////////////////////////////////////////
////////   Loader mappe med billeder & konvertere fra RGB til HSV  //////
vector<Mat> loadFolder(String folder, vector<Mat> &vecSize)
{
	String path("C:\\Users\\PC\\OneDrive - Aalborg Universitet\\Billeder\\Filmrulle\\" + folder + "\\*.jpg");
	vector<String> vecFile;
	vector<Mat> vecFruit;
	vector<Mat> vecHSV;

	glob(path, vecFile, false);

	for (int i = 0; i < vecFile.size(); ++i)
	{
		Mat fruit = imread(vecFile[i]);
		vecFruit.push_back(fruit);
		vecSize.push_back(fruit);
		vecHSV.push_back(fruit);

		resize(vecFruit[i], vecSize[i], Size(), 0.25, 0.25);

		cvtColor(vecSize[i], vecHSV[i], COLOR_BGR2HSV);
	}

	return vecHSV;
}


////////////////////////////////////////////////////////////////////////
////////   Laver threshold med "inrange", og reducere støj  ///////////
vector<Mat> thres(vector<Mat> vecHSV)
{
	vector<Mat> vecThres1(vecHSV.size());
	vector<Mat> vecThres2(vecHSV.size());
	vector<Mat> mask(vecHSV.size());
	vector<Mat> morphOpen(vecHSV.size());
	vector<Mat> morphClose(vecHSV.size());

	for (int i = 0; i < vecHSV.size(); ++i)
	{
		inRange(vecHSV[i], Scalar(0, 80, 140), Scalar(50, 255, 255), vecThres1[i]);
		inRange(vecHSV[i], Scalar(150, 80, 140), Scalar(255, 255, 255), vecThres2[i]);

		mask[i] = vecThres1[i] | vecThres2[i];

		Mat elem = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

		morphologyEx(mask[i], morphOpen[i], MORPH_OPEN, elem);
		morphologyEx(morphOpen[i], morphClose[i], MORPH_CLOSE, elem);
		morphologyEx(morphClose[i], morphClose[i], MORPH_ERODE, elem);
	}

	return morphClose; 
}
 

/////////////////////////////////////////////////////////////////////////
////////   Tegner store BLOBS, finder features: ratio,   ///////////////
////////   med rotated rectangel og laver color feature  ///////////////
vector<vector<double>> conturs(vector<Mat> morphClose, vector<Mat> &vecSize, vector<Mat> vecHSV)
{
	vector<vector<Point>> vecContours;
	double area;
	RotatedRect minRect;
	double ratio;
	vector<double> featureRatio;
	vector<double> avgColor;
	vector<vector<double>> matFeature;

	for (int i = 0; i < morphClose.size(); ++i)
	{
		findContours(morphClose[i], vecContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		for (int j = 0; j < vecContours.size(); j++)
		{
			area = contourArea(vecContours[j]);

			if (area > 4000)
			{
				drawContours(vecSize[i], vecContours, j, Scalar(0, 0, 255), 2);

				minRect = minAreaRect(vecContours[j]);

				if (minRect.size.height > minRect.size.width)
				{
					ratio = minRect.size.height / minRect.size.width;
				}
				else
				{
					ratio = minRect.size.width / minRect.size.height;
				}

				featureRatio.push_back(ratio);

				for (int x = 0; x < morphClose[i].cols; x++)
				{
					for (int y = 0; y < morphClose[i].rows; y++)
					{
						if (morphClose[i].at<uchar>(Point(x, y)) < 200)
						{
							vecHSV[i].at<Vec3b>(Point(x, y)) = 0;
						}
					}
				};

				Scalar colorMean = mean(vecHSV[i]);
				avgColor.push_back(colorMean[1] / 255);
			}
		}
	}

	matFeature.push_back(featureRatio);
	matFeature.push_back(avgColor);

	return matFeature;
}