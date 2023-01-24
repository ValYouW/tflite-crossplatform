#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "ObjectDetector.h"

using namespace cv;

extern "C" JNIEXPORT jlong JNICALL
Java_com_vyw_tflite_ObjectDetection_initDetector(JNIEnv* env, jobject p_this, jobject assetManager) {

    char *buffer = nullptr;
    long size = 0;

    if (!(env->IsSameObject(assetManager, NULL))) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        AAsset *asset = AAssetManager_open(mgr, "mobilenetv1.tflite", AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);

        size = AAsset_getLength(asset);
        buffer = (char *) malloc(sizeof(char) * size);
        AAsset_read(asset, buffer, size);
        AAsset_close(asset);
    }

    jlong res = (jlong) new ObjectDetector(buffer, size, true, false);
    free(buffer); // ObjectDetector duplicate it
    return res;
}

extern "C" JNIEXPORT void JNICALL
Java_com_vyw_tflite_ObjectDetection_destroyDetector(JNIEnv* env, jobject p_this, jlong p_native_ptr) {
    if (p_native_ptr)
        delete (ObjectDetector *) p_native_ptr;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_vyw_tflite_ObjectDetection_detect(JNIEnv* env, jobject p_this,
                                            jlong detectorAddr, jbyteArray src, int width,
                                            int height)
{
    jbyte *_rgba = env->GetByteArrayElements(src, 0);
    Mat frame(height, width, CV_8UC4, _rgba);
    cvtColor(frame, frame, COLOR_RGBA2BGRA);

    env->ReleaseByteArrayElements(src, _rgba, 0);


    ObjectDetector *detector = (ObjectDetector *) detectorAddr;
    DetectResult *res = detector->detect(frame);

    int arrlen = 6 * detector->DETECT_NUM + 1;
    jfloat* jres = new jfloat[arrlen];
    jres[0] = detector->DETECT_NUM;

    for (int i = 0; i < detector->DETECT_NUM; ++i) {
        int pos = i * 6 + 1;
        jres[pos + 0] = res[i].score;
        jres[pos + 1] = res[i].label;
        jres[pos + 2] = res[i].xmin;
        jres[pos + 3] = res[i].ymin;
        jres[pos + 4] = res[i].xmax;
        jres[pos + 5] = res[i].ymax;
    }

    jfloatArray output = env->NewFloatArray(arrlen);
    env->SetFloatArrayRegion(output, 0, arrlen, jres);

    return output;
}
