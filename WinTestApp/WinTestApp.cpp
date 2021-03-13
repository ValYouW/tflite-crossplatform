#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../ObjectDetection/ObjectDetector.h"

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("od_test.jpg");

	ifstream ifile("..\\models\\od_model.tflite", ifstream::binary);
	ifile.seekg(0, ifile.end);
	int length = ifile.tellg();
	ifile.seekg(0, ifile.beg);
	char* modelBuffer = new char[length];
	ifile.read(modelBuffer, length);


	ObjectDetector detector = ObjectDetector(modelBuffer, length, false);
	DetectResult* res = detector.detect(img);

	for (int i = 0; i < detector.DETECT_NUM; ++i) {
		int label = res[i].label;
		float score = res[i].score;
		float ymin = res[i].ymin;
		float xmin = res[i].xmin;
		float ymax = res[i].ymax;
		float xmax = res[i].xmax;

		rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
		putText(img, to_string(label) + "-" + to_string(score), Point(xmin, ymin), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255), 2);
	}

	imshow("img", img);
	waitKey(0);

	return 0;
}
