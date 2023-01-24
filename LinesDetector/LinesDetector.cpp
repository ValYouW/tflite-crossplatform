#include <cmath>
#include "LinesDetector.h"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

LinesDetector::LinesDetector(const char* mcDetectionModel, long modelSize, bool useNNAPI)
{
	if (modelSize > 0) {
		initDetectionModel(mcDetectionModel, modelSize, useNNAPI);
	}
}

LinesDetector::~LinesDetector() {
	if (m_modelBytes != nullptr) {
		free(m_modelBytes);
		m_modelBytes = nullptr;
	}

	if (m_interpreter != nullptr)
		TfLiteInterpreterDelete(m_interpreter);

	if (m_nnapi_delegate != nullptr)
		TfLiteNnapiDelegateDelete(m_nnapi_delegate);

	if (m_interpreterOptions != nullptr)
		TfLiteInterpreterOptionsDelete(m_interpreterOptions);

	if (m_model != nullptr)
		TfLiteModelDelete(m_model);

	m_hasDetectionModel = false;
}

void LinesDetector::initDetectionModel(const char* mcDetectionModel, long modelSize, bool useNNAPI) {
	if (modelSize < 1) { return; }

	// Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
	m_modelBytes = (char*)malloc(modelSize);
	memcpy(m_modelBytes, mcDetectionModel, modelSize);
	m_model = TfLiteModelCreate(m_modelBytes, modelSize);

	if (m_model == nullptr) {
		printf("Failed to load model");
		return;
	}

	// Build the interpreter
	m_interpreterOptions = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(m_interpreterOptions, 1);

	if (useNNAPI) {
		TfLiteNnapiDelegateOptions xnnOpts = TfLiteNnapiDelegateOptionsDefault();
		m_nnapi_delegate = TfLiteNnapiDelegateCreate(&xnnOpts);
		TfLiteInterpreterOptionsAddDelegate(m_interpreterOptions, m_nnapi_delegate);
	}

	// Create the interpreter.
	m_interpreter = TfLiteInterpreterCreate(m_model, m_interpreterOptions);
	if (m_interpreter == nullptr) {
		printf("Failed to create interpreter");
		return;
	}

	// Allocate tensor buffers.
	if (TfLiteInterpreterAllocateTensors(m_interpreter) != kTfLiteOk) {
		printf("Failed to allocate tensors!");
		return;
	}

	// Find input tensors.
	if (TfLiteInterpreterGetInputTensorCount(m_interpreter) != 1) {
		printf("Detection model graph needs to have 1 and only 1 input!");
		return;
	}

	m_input_tensor = TfLiteInterpreterGetInputTensor(m_interpreter, 0);
	if (m_input_tensor->type != kTfLiteFloat32) {
		printf("Detection model input should be kTfLiteFloat32!");
		return;
	}

	if (m_input_tensor->dims->data[0] != 1 ||
		m_input_tensor->dims->data[1] != DETECTION_MODEL_SIZE ||
		m_input_tensor->dims->data[2] != DETECTION_MODEL_SIZE ||
		m_input_tensor->dims->data[3] != DETECTION_MODEL_CNLS) {
		printf("Detection model must have input dims of 1x%ix%ix%i", DETECTION_MODEL_SIZE,
			DETECTION_MODEL_SIZE, DETECTION_MODEL_CNLS);
		return;
	}

	// Find output tensors.
	if (TfLiteInterpreterGetOutputTensorCount(m_interpreter) != 3) {
		printf("Detection model graph needs to have 3 and only 3 outputs!");
		return;
	}

	m_output_centers = TfLiteInterpreterGetOutputTensor(m_interpreter, 0);
	if (m_output_centers->type != kTfLiteInt32 ||
		m_output_centers->dims->data[0] != 1 ||
		m_output_centers->dims->data[1] != DETECTION_OUTPUT_COUNT ||
		m_output_centers->dims->data[2] != 2) {
		printf("Output tensor of Centers should be Int32 of size [1, 200, 2]");
		return;
	}

	m_output_scores = TfLiteInterpreterGetOutputTensor(m_interpreter, 1);
	if (m_output_scores->type != kTfLiteFloat32 ||
		m_output_scores->dims->data[0] != 1 ||
		m_output_scores->dims->data[1] != DETECTION_OUTPUT_COUNT) {
		printf("Output tensor of Scores should be of size [1, 200]");
		return;
	}

	m_output_center_offset = TfLiteInterpreterGetOutputTensor(m_interpreter, 2);
	int vmap_size = (DETECTION_MODEL_SIZE / 2);
	if (m_output_center_offset->type != kTfLiteFloat32 ||
		m_output_center_offset->dims->data[0] != 1 ||
		m_output_center_offset->dims->data[1] != vmap_size ||
		m_output_center_offset->dims->data[2] != vmap_size ||
		m_output_center_offset->dims->data[3] != 4) {
		printf("Output tensor of VMAP should be of size [1, %d, %d, 4]", vmap_size, vmap_size);
		return;
	}

	m_hasDetectionModel = true;
}

vector<Vec4i> LinesDetector::detect(Mat src) {
	if (src.empty() || !m_hasDetectionModel) {
		return {};
	}

	float hScale = static_cast<float>(src.rows) / DETECTION_MODEL_SIZE;
	float wScale = static_cast<float>(src.cols) / DETECTION_MODEL_SIZE;

	Mat image;
	resize(src, image, Size(DETECTION_MODEL_SIZE, DETECTION_MODEL_SIZE), 0, 0, INTER_AREA);
	int cnls = image.type();
	if (cnls == CV_8UC1) {
		cvtColor(image, image, COLOR_GRAY2RGBA);
	}
	else if (cnls == CV_8UC3) {
		cvtColor(image, image, COLOR_BGR2RGBA);
	}

	Mat fimage;
	image.convertTo(fimage, CV_32FC4);
	image.release();
	// Copy image into input tensor
	float* dst = m_input_tensor->data.f;
	memcpy(dst, fimage.data, sizeof(float) * DETECTION_MODEL_SIZE * DETECTION_MODEL_SIZE * DETECTION_MODEL_CNLS);

	if (TfLiteInterpreterInvoke(m_interpreter) != kTfLiteOk) {
		printf("Error invoking detection model");
		return {};
	}

	const int* centers = m_output_centers->data.i32;
	const float* scores = m_output_scores->data.f;
	const float* center_offset = m_output_center_offset->data.f;

	vector<Vec4i> res;
	int offset_mtx_size = (DETECTION_MODEL_SIZE / 2);
	for (int i = 0; i < 2 * DETECTION_OUTPUT_COUNT; i += 2) {
		int centerY = centers[i];
		int centerX = centers[i + 1];
		float score = scores[i / 2];
		if (score <= 0.2)
			continue;

		// center_offset is a matrix of m*n*4 where per x/y it holds the offset
		// of the line start/end from the line center
		int idx = centerY * offset_mtx_size * 4 + centerX * 4;
		float startX = centerX + center_offset[idx];
		float startY = centerY + center_offset[idx + 1];
		float endX = centerX + center_offset[idx + 2];
		float endY = centerY + center_offset[idx + 3];
		float dist = sqrt(pow((endX - startX), 2) + pow((endY - startY), 2));
		if (dist < 20)
			continue;

		// scale back the point to the size of the input image.
		// First multiple by 2 as the result points are for image that is half the size
		// of the model input. Then we scale back from the model input to the original img size
		startX = 2 * wScale * startX;
		startY = 2 * hScale * startY;
		endX = 2 * wScale * endX;
		endY = 2 * hScale * endY;
		res.push_back(Vec4i(startX, startY, endX, endY));
	}

	fimage.release();
	return res;
}
