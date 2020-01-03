#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include "opencv2\highgui.hpp"
#include "opencv2\imgcodecs.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\ml\ml.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
//#include <Windows.h>

using namespace std;
using namespace cv; 

const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

char op;
int ends;
double hasil;
string awal,akhir;
string bagi = "/";
string kurang = "-";
string tambah = "+";
string kali = "x";
char oper_bagi = '/';
char oper_tambah = '+';
char oper_kurang = '-';
char oper_kali = 'x';

// #define KNN_PROC

///////////////////////////////////////////////////////////////////////////////////////////////////

class ContourWithData {

public:

    // member variables ///////////////////////////////////////////////////////////////////////////
    std::vector<cv::Point> ptContour;           // contour
    cv::Rect boundingRect;                      // bounding rect for contour
    float fltArea;                              // area of contour

    ///////////////////////////////////////////////////////////////////////////////////////////////

    bool checkIfContourIsValid() {                              // obviously in a production grade program
        if (fltArea < MIN_CONTOUR_AREA) return false;           // we would have a much more robust function for 
        return true;                                            // identifying if a contour is valid !!
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////

    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
    return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right
    }
};

int main() {

    std::vector<ContourWithData> allContoursWithData,allContoursWithData2;           // declare empty vectors,
    std::vector<ContourWithData> validContoursWithData,validContoursWithData2;         // we will fill these shortly

    // read in training classifications ///////////////////////////////////////////////////

    cv::Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector
    cv::FileStorage fsClassifications("classifications8.xml", cv::FileStorage::READ);        // open the classifications file

    if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
        std::cout << "error, unable to open training classifications file, exiting program\n\n";    // show error message
        return(0);                                                                                  // and exit program
    }

    fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
    fsClassifications.release();                                        // close the classifications file
	// read in training images ////////////////////////////////////////////////////////////

    cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector
    cv::FileStorage fsTrainingImages("images8.xml", cv::FileStorage::READ);          // open the training images file

    if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
        std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
        return(0);                                                                              // and exit program
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
    fsTrainingImages.release();                                                 // close the traning images file

    // train //////////////////////////////////////////////////////////////////////////////
    cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object

      // finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
	 // even though in reality they are multiple images / numbers

    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    // test ///////////////////////////////////////////////////////////////////////////////
	VideoCapture cap(0);

	// pilihan menu input
	// printf("\nKNN TEXT RECOGNITION PROGRAM\n");
	// printf("_________________________________\n");

	// printf("Menu: \n 1.\tTake input\n2.\tClasification");

	cv::Mat  matTestingNumbers,matTestingNumbers1,matTestingNumbers2;            // read in the test numbers image
	cout << "Tekan huruf 'x' pada keyboard untuk mengambil gambar" << endl;

	while(1)
	{
		cap >> matTestingNumbers; //get a new frame from camera
		
		// matTestingNumbers = cv::imread("data_input_5.png");
		imshow("CAMERA 1", matTestingNumbers);


		if (matTestingNumbers.empty()) {                                // if unable to open image
			std::cout << "error: image not read from file\n\n";         // show error message on command line
			return(0);                                                  // and exit program
		}

		if (waitKey(28) == 'x') {
				imwrite("data_input_image.png", matTestingNumbers);
				cout << "Input ke SATU sudah terambil" <<endl;
				break;
		}
	}


	matTestingNumbers1 = cv::imread("data_input_image.png");
	cv::Mat matGrayscale;           //
	cv::Mat matBlurred;             // declare more image variables
	cv::Mat matThresh;              //
	cv::Mat matThreshCopy;          //

	cv::cvtColor(matTestingNumbers1, matGrayscale, CV_BGR2GRAY);         // convert to grayscale
																			// blur
	cv::GaussianBlur(matGrayscale,              // input image
					 matBlurred,                // output image
					 cv::Size(5, 5),            // smoothing window width and height in pixels
					 0);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value


	// filter image from grayscale to black and whit
	cv::adaptiveThreshold(matBlurred,                           // input image
						  matThresh,                            // output image
						  255,                                  // make pixels that pass the threshold full white
						  cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
						  cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
						  11,                                   // size of a pixel neighborhood used to calculate threshold value
						  2);                                   // constant subtracted from the mean or weighted mean


	matThreshCopy = matThresh.clone();              // make a copy of the thresh image, this in necessary b/c findContours modifies the image

	std::vector<std::vector<cv::Point>> ptContours;        // declare a vector for the contours
	std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

	cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
					 ptContours,                             // output contours
					 v4iHierarchy,                           // output hierarchy
					 cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
					 cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points
	for (int i = 0; i < ptContours.size(); i++) {               // for each contour

		ContourWithData contourWithData;                                                    // instantiate a contour with data object
		contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
		contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
		contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
		allContoursWithData.push_back(contourWithData);                                     // add contour with data object to list of all contours with data

	}


	for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
		if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
			validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
		}
	}

	// sort contours from left to right

	std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);
	std::string strFinalString;         // declare final string, this will have the final number sequence by the end of the program

	for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour
																			// draw a green rect around the current char

		cv::rectangle(matTestingNumbers1,                            // draw rectangle on original image
					  validContoursWithData[i].boundingRect,        // rect to draw
					  cv::Scalar(0, 255, 0),                        // green
					  2);                                           // thickness

		cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect
		cv::Mat matROIResized;
		cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage
		cv::Mat matROIFloat;

		matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest
		cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);
		cv::Mat matCurrentChar(0, 0, CV_32F);

		kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!

		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

		strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string

	}
	
	// put text
	putText(matTestingNumbers1, "Bilangan " + strFinalString, Point(0, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0), 2);
	
	cout << "numbers read = " << strFinalString << "\n\n";       // show the full string
	imshow("matTestingNumbers", matTestingNumbers1); // show input image with green boxes drawn around found digits
	//Sleep(100);
	
	//parsing string
	for(int i=0;i<=15;i++){
		if(strFinalString.at(i) == oper_bagi){
			auto start = 0U;
			auto end = strFinalString.find(bagi);
			while (end != std::string::npos){
				cout << strFinalString.substr(start, end - start) << endl;
				awal = strFinalString.substr(start, end - start);
				start = end + bagi.length();
				end = strFinalString.find(bagi, start);
			}
			cout << strFinalString.substr(start, end) << endl;
			akhir = strFinalString.substr(start, end);

			double num2 = atof(akhir.c_str());
			double num = atof(awal.c_str());

			cout << "angka dari input pertama : " << num << endl;
			cout << "angka dari input kedua : " << num2 << endl;

			if(num2 == 0){
				cout << " Yah hasilnya tak terdefinisi";
			}
			cout << "Berapakah hasil dari " << num << " / " << num2 << " ? " ;
			cin >> hasil;
			if(hasil == num/num2){
				cout << " YeaY Jawaban kamu benar " << endl;
			}
			else{
				cout << " Yah jawabanmu salah, yang benar adalah " <<  num/num2 << endl;
			}
			break;
		}

		else if(strFinalString.at(i) == oper_tambah){
			auto start = 0U;
			auto end = strFinalString.find(tambah);
			while (end != std::string::npos){
				cout << strFinalString.substr(start, end - start) << endl;
				awal = strFinalString.substr(start, end - start);
				start = end + bagi.length();
				end = strFinalString.find(bagi, start);
			}
			cout << strFinalString.substr(start, end) << endl;
			akhir = strFinalString.substr(start, end);

			double num2 = atof(akhir.c_str());
			double num = atof(awal.c_str());

			cout << "angka dari input pertama : " << num << endl;
			cout << "angka dari input kedua : " << num2 << endl;

			cout << "Berapakah hasil dari " << num << " + " << num2 << " ? " ;
			cin >> hasil;
			if(hasil == num+num2){
				cout << " YeaY Jawaban kamu benar " << endl;
			}
			else{
				cout << " Yah jawabanmu salah, yang benar adalah " <<  num+num2 << endl;
			}
			break;
		}

		else if(strFinalString.at(i) == oper_kurang){
			auto start = 0U;
			auto end = strFinalString.find(kurang);
			while (end != std::string::npos){
				cout << strFinalString.substr(start, end - start) << endl;
				awal = strFinalString.substr(start, end - start);
				start = end + bagi.length();
				end = strFinalString.find(bagi, start);
			}
			cout << strFinalString.substr(start, end) << endl;
			akhir = strFinalString.substr(start, end);

			double num2 = atof(akhir.c_str());
			double num = atof(awal.c_str());

			cout << "angka dari input pertama : " << num << endl;
			cout << "angka dari input kedua : " << num2 << endl;

			cout << "Berapakah hasil dari " << num << " - " << num2 << " ? " ;
			cin >> hasil;
			if(hasil == num-num2){
				cout << " YeaY Jawaban kamu benar " << endl;
			}
			else{
				cout << " Yah jawabanmu salah, yang benar adalah " <<  num-num2 << endl;
			}
			break;
		}

		else if(strFinalString.at(i) == oper_kali){
			auto start = 0U;
			auto end = strFinalString.find(kali);
			while (end != std::string::npos){
				cout << strFinalString.substr(start, end - start) << endl;
				awal = strFinalString.substr(start, end - start);
				start = end + bagi.length();
				end = strFinalString.find(bagi, start);
			}
			cout << strFinalString.substr(start, end) << endl;
			akhir = strFinalString.substr(start, end);

			double num2 = atof(akhir.c_str());
			double num = atof(awal.c_str());

			cout << "angka dari input pertama : " << num << endl;
			cout << "angka dari input kedua : " << num2 << endl;

			cout << "Berapakah hasil dari " << num << " x " << num2 << " ? " ;
			cin >> hasil;
			if(hasil == num*num2){
				cout << " YeaY Jawaban kamu benar " << endl;
			}
			else{
				cout << " Yah jawabanmu salah, yang benar adalah " <<  num*num2 << endl;
			}
			break;
		}
	
	}

	waitKey(0);

	cap.release();
    return(0);
}