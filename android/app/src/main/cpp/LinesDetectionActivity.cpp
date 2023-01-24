#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "LinesDetector.h"

using namespace cv;

extern "C" JNIEXPORT jlong JNICALL
Java_com_vyw_tflite_LinesDetection_initDetector(JNIEnv* env, jobject p_this, jobject assetManager) {

    char *buffer = nullptr;
    long size = 0;

    if (!(env->IsSameObject(assetManager, NULL))) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        AAsset *asset = AAssetManager_open(mgr, "M-LSD_320_tiny_fp16.tflite", AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);

        size = AAsset_getLength(asset);
        buffer = (char *) malloc(sizeof(char) * size);
        AAsset_read(asset, buffer, size);
        AAsset_close(asset);
    }

    jlong res = (jlong) new LinesDetector(buffer, size, false);
    free(buffer); // ObjectDetector duplicate it
    return res;
}

extern "C" JNIEXPORT void JNICALL
Java_com_vyw_tflite_LinesDetection_destroyDetector(JNIEnv* env, jobject p_this, jlong p_native_ptr) {
    if (p_native_ptr)
        delete (LinesDetector*) p_native_ptr;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_vyw_tflite_LinesDetection_detect(JNIEnv* env, jobject p_this,
                                            jlong detectorAddr, jbyteArray src, int width,
                                            int height)
{
    jbyte *_rgba = env->GetByteArrayElements(src, 0);
    Mat frame(height, width, CV_8UC4, _rgba);
    // frame is already RGBA and this is what we need

    env->ReleaseByteArrayElements(src, _rgba, 0);

    LinesDetector* detector = (LinesDetector*)detectorAddr;
    vector<Vec4i> res = detector->detect(frame);

    int arrlen = 4 * res.size() + 1;
    jint* jres = new jint[arrlen];
    jres[0] = res.size();

    for (int i = 0; i < res.size(); ++i) {
        int pos = i * 4 + 1;
        jres[pos + 0] = res[i][0];
        jres[pos + 1] = res[i][1];
        jres[pos + 2] = res[i][2];
        jres[pos + 3] = res[i][3];
    }

    jintArray output = env->NewIntArray(arrlen);
    env->SetIntArrayRegion(output, 0, arrlen, jres);

    return output;
}
