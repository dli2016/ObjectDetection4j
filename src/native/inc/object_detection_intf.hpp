
/**
 * object_detection_intf.hpp
 **/

#ifndef _OBJECT_DETECTION_INTF_HPP_

class ObjectDetectionIntf {
  public:
    virtual ~ObjectDetectionIntf() {};
    
    // Initialization.
    virtual int init(const char* weight_filename) = 0;
    // Detection
    virtual int detect(unsigned char* data, int h, int w, int c, float* bbs,
        float* scores) = 0;
    virtual int detect0(const char* image_filename) = 0;
    // Release
    virtual int release(void) = 0;
};

#ifdef __cplusplus
extern "C" { 
#endif

typedef ObjectDetectionIntf* (*create_t)();
typedef void (*destroy_t)(ObjectDetectionIntf*);

#ifdef __cplusplus
}
#endif

#endif  // _OBJECT_DETECTION_INTF_HPP_
