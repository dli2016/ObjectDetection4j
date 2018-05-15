#ifndef _CPPWRAPPER_PYTORCH_SSD_H_
#define _CPPWRAPPER_PYTORCH_SSD_H_

//#include "boost/python.hpp"
#include "Python.h"
#include "object_detection_intf.hpp"

#define MAX_PEDESTRIAN_NUM 128

typedef struct lcpr_bounding_box_t {      
    int top;      
    int left;      
    int bottom;      
    int right;      
} LCPRBoundingBox;

typedef struct pedstrian_t {      
    float score;      
    LCPRBoundingBox bb;      
} Pedestrian;

typedef struct detection_output_t {      
    int num;      
    Pedestrian detected_pedestrians[MAX_PEDESTRIAN_NUM];      
} DetectionOutput;

typedef struct lcpr_input_image_t {      
    int img_w;      
    int img_h;      
    int num_channels;      
    unsigned char* img_data;  // bgr, bgr, bgr, ...    
} LCPRInputIMG;

class PyTorchSSDCppWrapper: public ObjectDetectionIntf {

  public:
    PyTorchSSDCppWrapper(void);
     ~PyTorchSSDCppWrapper(void);

    // Initialization.
    int init(const char* weight_filename);
    // Detect.
    // 1. For test.
    int detect0(const char* image_filename);
    int detect1(unsigned char* data, int h, int w, int c);
    // 2. normal interface.
    int detect2(const LCPRInputIMG& img, DetectionOutput& detected);
    int detect(unsigned char* data, int h, int w, int c, float* bbs, 
        float* scores);
    // Release.
    int release(void);

  private:
    void parseDetectRes(PyObject* res);
    int parseDetectRes_v2(PyObject* res, float* bbs, float* scores);

  private:
    //PyObject* net;
    //PyObject* transform;
    PyObject* init_res;
    PyObject* p_module;
};

#endif //_CPPWRAPPER_PYTORCH_SSD_H_

