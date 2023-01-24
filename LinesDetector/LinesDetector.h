#pragma once
#include <opencv2/core.hpp>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"

using namespace std;
using namespace cv;

struct Line {
	Point2f start;
	Point2f end;

	Line(Point2f s, Point2f e) {
		start = s;
		end = e;
	}
};

class LinesDetector 
{
public:
	// Methods
	LinesDetector(const char* mcDetectionModel, long modelSize = 0, bool useNNAPI = false);
	~LinesDetector();
	vector<Vec4i> detect(Mat img);

private:
	// Members
	const int DETECTION_MODEL_SIZE = 320;
	const int DETECTION_MODEL_CNLS = 4;
	const int DETECTION_OUTPUT_COUNT = 200;

	bool m_hasDetectionModel = false;
	char* m_modelBytes = nullptr;
	TfLiteModel* m_model;
	TfLiteInterpreterOptions* m_interpreterOptions = nullptr;
	TfLiteInterpreter* m_interpreter;
	TfLiteDelegate* m_nnapi_delegate;
	TfLiteTensor* m_input_tensor = nullptr;
	const TfLiteTensor* m_output_centers = nullptr;
	const TfLiteTensor* m_output_scores = nullptr;
	const TfLiteTensor* m_output_center_offset = nullptr;
	void initDetectionModel(const char* mcDetectionModel, long modelSize, bool useNNAPI);
};
