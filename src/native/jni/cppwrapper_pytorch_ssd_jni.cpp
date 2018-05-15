
#include <cstdlib>
#include <cstdio>
#include <string>
#include <dlfcn.h>

#include "object_detection_intf.hpp"
#include "org_cripac_isee_ObjectDetection.h"

// static void* dl = NULL;

JNIEXPORT jlong JNICALL Java_org_cripac_isee_ObjectDetection_loadLibrary
    (JNIEnv* env, jobject obj, jstring lib_name) {
    const int kLibNameLen = env->GetStringUTFLength(lib_name);
    char* c_lib_name = new char[kLibNameLen+1];

    env->GetStringUTFRegion(lib_name, 0, kLibNameLen, c_lib_name);
    c_lib_name[kLibNameLen] = '\0';

    void* dl = dlopen(c_lib_name, RTLD_LAZY|RTLD_GLOBAL);
    if (!dl) {
        fprintf(stderr, "Load dll FAILED!\n");
        fprintf(stderr, "%s\n", dlerror());
        delete[] c_lib_name;
        c_lib_name = NULL;
        exit(-1);
    }
    delete[] c_lib_name;
    c_lib_name = NULL;

    return (jlong)dl;
}

JNIEXPORT jlong JNICALL Java_org_cripac_isee_ObjectDetection_initialize
    (JNIEnv *env, jobject obj, jlong jdl, jint gpu_id, jstring model_path) {
    const int kModelPathLen = env->GetStringUTFLength(model_path);
    char* c_model_path = new char[kModelPathLen + 1];
    env->GetStringUTFRegion(model_path, 0, kModelPathLen, c_model_path);
    c_model_path[kModelPathLen] = '\0';
    
    //PyTorchSSDCppWrapper* detector = new PyTorchSSDCppWrapper;
    void* dl = (void*)jdl;
    create_t create_func = (create_t) dlsym(dl, "create");
    if (!create_func) {
        fprintf(stderr, "Load sysmbol (create) FAILED!\n");
        delete[] c_model_path;
        c_model_path = NULL;
        return -1;
    }
    ObjectDetectionIntf* detector = create_func();

    int ret = detector->init((const char*)c_model_path);

    delete[] c_model_path;
    c_model_path = NULL;

    if (ret < 0) {
        fprintf(stderr, "Error: detector initailize FAILED!\n");
        return (jlong)NULL;
    } else {
        return (jlong)detector;
    }
}

JNIEXPORT jint JNICALL Java_org_cripac_isee_ObjectDetection_detect
    (JNIEnv *env, jobject obj, jlong handle, jbyteArray jframe, 
    jint h, jint w, jint c, jfloatArray jbbs, jfloatArray jscores) {
    ObjectDetectionIntf* detector = (ObjectDetectionIntf *)handle;
    jbyte *frame = env->GetByteArrayElements(jframe, NULL);

    const int kMaxObjsNum = 128;
    float* c_bbs = new float[kMaxObjsNum*4];
    float* c_scores = new float[kMaxObjsNum];

    int ret = detector->detect((jboolean*)frame, h, w, c, c_bbs, c_scores);

    if (ret < 0) {
        fprintf(stderr, "Error: detector detect FAILED!\n");
    } else {
        env->SetFloatArrayRegion(jbbs, 0, ret*4, c_bbs);
        env->SetFloatArrayRegion(jscores, 0, ret*4, c_scores);
    }

    delete[] c_bbs;
    c_bbs = NULL;
    delete[] c_scores;
    c_scores = NULL;

    return (jint)ret;
}

JNIEXPORT jint JNICALL Java_org_cripac_isee_ObjectDetection_release
    (JNIEnv *env, jobject obj, jlong handle, jlong jdl) {
    ObjectDetectionIntf* detector = (ObjectDetectionIntf *)handle;
    int ret = -1;
    if (detector) {
        ret = detector->release();
    }
    void* dl = (void*)jdl;
    destroy_t destroy_func = (destroy_t) dlsym(dl, "destroy");
    if (!destroy_func) {
        fprintf(stderr, "Load sysmbol (destroy) FAILED!\n");
        dlclose(dl);
        exit(-1);
    }
    destroy_func(detector);
 
    return ret;
}

JNIEXPORT void JNICALL Java_org_cripac_isee_ObjectDetection_closeLibrary
    (JNIEnv *env, jobject obj, jlong jdl) {
    void* dl = (void*)jdl;
    dlclose(dl);
}
