
/**
 * object_detection_intf.hpp
 **/

#ifndef _OBJECT_DETECTION_INTF_HPP_

typedef enum obj_det_err_type_t_ {
    SUCCESS = 0,
    FILE_NOT_EXIST = -1,
    PY_MOD_IMPORT_ERR = -2,
    PY_CLS_IMPORT_ERR = -3,
    PY_CREATE_OBJ_ERR = -4,
    PY_METHOD_ERR = -5,
    NULL_STR = -6,
    NULL_DATA= -7,
} ObjDetErrorTypes;

class ObjectDetectionIntf {
  public:
    virtual ~ObjectDetectionIntf() {};
    
    // Initialization.
    virtual int init(const char* weight_filename) = 0;
    // Detection
    virtual int detect(unsigned char* data, int h, int w, int c, float* bbs,
        float* scores) = 0;
    virtual int detect(const char* image_filename) = 0;
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
