#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
using namespace std;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"
#include "cppwrapper_pytorch_ssd.h"

PyTorchSSDCppWrapper::PyTorchSSDCppWrapper(void) {
    py_instance = NULL;
}

PyTorchSSDCppWrapper::~PyTorchSSDCppWrapper(void) {
    release();
}

int PyTorchSSDCppWrapper::initPyEnv(const char* py_home, 
    const char* py_path) {
    if (py_home == NULL || py_path == NULL) {
        fprintf(stderr, "Python Home or python path is NULL!\n");
        return NULL_STR;
    }
    // Set python home ...
    size_t len = strlen(py_home);
    wchar_t* wc_python_home = Py_DecodeLocale(py_home, &len);
    Py_SetPythonHome(wc_python_home);
    // Set python path ...
    //setenv("PYTHONPATH","");
    //const char* c_python_path = "";
    //len = strlen(c_python_path);
    //wchar_t* wc_python_path = Py_DecodeLocale(c_python_path, &len);
    //Py_SetPath(wc_python_path);
    
    // Initialize the Python Interpreter
    printf("-99999999\n");
    Py_Initialize();
    printf("000000000\n");
    if (PyArray_API==NULL) {
        import_array();
    }
    // Set search path ...
    string python_path_self = py_path;
    //"/home/data/da.li/projects/ObjectDetection4j/src/native/lib";
    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyUnicode_FromString(python_path_self.c_str()));

    // Get python home.
    wchar_t* my_python_home = Py_GetPythonHome();
    char* mph = (char *)malloc( 1024 );
    len = wcstombs( mph, my_python_home, 1024);
    printf("%s\n", mph);
    // Get python path.
    wchar_t* python_path = Py_GetPath();
    len = wcstombs( mph, python_path, 1024);
    printf("%s\n", mph);
    free(mph);

    return 0;
}

int PyTorchSSDCppWrapper::init(const char* weight_filename) {
    // Set python environments
    const char* py_home = "/home/da.li/anaconda3/";
    const char* py_path =
        "/home/data/da.li/projects/ObjectDetection4j/src/native/lib";
    int ret = initPyEnv(py_home, py_path);
    if (ret < 0) {
        fprintf(stderr, "Set python environment FAILED!\n");
        return ret;
    }
    // Initialize
    char* input_module_name = (char*)"ssd_detector_pytorch_v2";
    //char* input_func_name = (char*)"initialize";
    // Import module ...
    printf("111111111\n");
    PyObject* module_name = PyUnicode_DecodeFSDefault(input_module_name);
    printf("222222222\n");
    PyObject* py_module = PyImport_Import(module_name);
    printf("333333333\n");
    Py_DECREF(module_name);
    if (!py_module) {
        fprintf(stderr, "In init: Import %s FAILED!\n", input_module_name);
        return PY_MOD_IMPORT_ERR;
    } else {
        fprintf(stdout, "In init: %s import successfully!\n", input_module_name);
    }
    
    // Import class.
    char* class_name = (char*)"SSDPyTorch";
    PyObject* py_dict = PyModule_GetDict(py_module);
    PyObject* py_class= PyDict_GetItemString(py_dict, class_name);
    // Create an instance of the class
    if (PyCallable_Check(py_class)) {
        PyObject* py_args = PyTuple_New(1);
        PyObject* py_val = PyBytes_FromString((char*)weight_filename);
        PyTuple_SetItem(py_args, 0, py_val);
        py_instance = PyObject_CallObject(py_class, py_args);
        if (!py_instance) {
            fprintf(stderr, "In init: create object FAILED!\n");
            Py_DECREF(py_args);
            return PY_CREATE_OBJ_ERR;
        }
        Py_DECREF(py_args);
    } else {
        fprintf(stderr, "Load class %s FAILED!\n", class_name);
        release();
        return PY_CLS_IMPORT_ERR;
    }
    fprintf(stdout, "Initialize successfully!\n");
    return SUCCESS;
}

int PyTorchSSDCppWrapper::parseDetectRes_v2(PyObject* res, float* bbs, 
    float* scores) {
    PyArrayObject* py_bbs_object = 
        (PyArrayObject*)PyTuple_GetItem(res, 0);
    PyArrayObject* py_conf_object =
        (PyArrayObject*)PyTuple_GetItem(res, 1);
    int bbs_h = PyArray_DIM(py_bbs_object, 0);
    int bbs_w = PyArray_DIM(py_bbs_object, 1);
    int conf_n = PyArray_DIM(py_conf_object, 0);
    if (py_bbs_object==NULL || py_conf_object==NULL) {
        fprintf(stderr, "Parse return object FAILED!\n");
        return -1;
    }
    float* cpp_bbs = (float*)PyArray_DATA(py_bbs_object);
    memcpy(bbs, cpp_bbs, bbs_h*bbs_w*4);
    double* cpp_scores = (double*)PyArray_DATA(py_conf_object);
    for (int n = 0; n < conf_n; ++n) {
        scores[n] = (float)cpp_scores[n];
    }
    printf("The detection results are successfully parsed ...\n");
    return conf_n;
}

int PyTorchSSDCppWrapper::detect(const char* image_filename) {
    if (!image_filename) {
        fprintf(stderr, "In detect: input filename is NULL\n");
        return NULL_STR;
    }
    if (py_instance) {
        PyObject* py_method_name = PyUnicode_FromString("run_v2");
        PyObject* py_input_val = PyBytes_FromString((char*)image_filename);
        PyObject* py_args_in = PyTuple_New(1);
        PyTuple_SetItem(py_args_in, 0, py_input_val);
        PyObject* py_return_val = PyObject_CallMethodObjArgs(py_instance, 
            py_method_name, py_args_in);
        Py_DECREF(py_args_in);
        if (py_return_val) {
            printf("Detect successfully!\n");
            // Parse results:
            parseDetectRes(py_return_val);
            Py_DECREF(py_return_val);
        } else {
            PyErr_Print();
            fprintf(stderr,"In detect: Call %s failed\n", "run_v2");
            return PY_METHOD_ERR;
        }
    } else {
        fprintf(stderr, "In detect: Object Instance is NULL\n");
        return PY_METHOD_ERR;
    }
    return SUCCESS;
}

int PyTorchSSDCppWrapper::detect(unsigned char* data, int h, int w, int c, 
    float* bbs, float* scores) {
    if (!data) {
        fprintf(stderr, "In detect: input data is NULL!\n");
        return NULL_DATA;
    }

    int obj_num = -1;
    if (py_instance) {
        PyObject* py_method_name = PyUnicode_FromString("run_v4");
        // Convert cpp data to python object.
        int nd = 3;
        npy_intp dims[] = {h, w, c};
        PyObject* py_input_val = PyArray_SimpleNewFromData(nd, dims, NPY_UINT8,
            (void*)data);
        PyObject* py_args_in = PyTuple_New(1);
        PyTuple_SetItem(py_args_in, 0, py_input_val);
        PyObject* py_return_val = PyObject_CallMethodObjArgs(py_instance,
            py_method_name, py_args_in);
        Py_DECREF(py_args_in);
        if (py_return_val) {
            printf("Detect successfully!\n");
            // Parse results:
            if (!bbs || !scores) {
                fprintf(stderr, 
                    "In detect: allocate memory to store detection results\n");
                return NULL_DATA;
            }
            obj_num = parseDetectRes_v2(py_return_val, bbs, scores);
            Py_DECREF(py_return_val);
        } else {
            PyErr_Print();
            fprintf(stderr,"In detect: Call %s failed\n", "run_v4");
            return PY_METHOD_ERR;
        }
    } else {
        fprintf(stderr, "In detect: Object Instance is NULL\n");
        return PY_METHOD_ERR;
    }
    return obj_num;
}

int PyTorchSSDCppWrapper::detect1(unsigned char* data, int h, int w, int c) {
    /*
    }*/
    return 0;
}

int PyTorchSSDCppWrapper::test(const char* input_str) {
    /*
    */
    return 0;
}

void PyTorchSSDCppWrapper::parseDetectRes(PyObject* res) {
    PyArrayObject* py_bbs_object = 
        (PyArrayObject*)PyTuple_GetItem(res, 0);
    PyArrayObject* py_conf_object =
        (PyArrayObject*)PyTuple_GetItem(res, 1);
    int bbs_h = PyArray_DIM(py_bbs_object, 0);
    int bbs_w = PyArray_DIM(py_bbs_object, 1);
    int conf_n = PyArray_DIM(py_conf_object, 0);
    printf("element_n = %d\n", conf_n);
    //int conf_h= PyArray_DIM(py_conf_object, 0);
    //int conf_w= PyArray_DIM(py_conf_object, 1);
    //printf("(h=%d, w=%d)\n", conf_h, conf_w);
    if (py_bbs_object==NULL || py_conf_object==NULL) {
        fprintf(stderr, "Parse return object FAILED!\n");
        return;
    }
    printf("bbs: %d - %d\n", PyArray_TYPE(py_bbs_object), NPY_FLOAT32);
    float* cpp_bbs = (float*)PyArray_DATA(py_bbs_object);
    // Test
    for (int h = 0; h < bbs_h; ++h) {
        for (int w = 0; w < bbs_w; ++w) {
            printf("%f ", cpp_bbs[h*bbs_w + w]);
            if ((w+1) % 4 == 0) {
                printf("\n");
            }
        }
    }
    printf("score: %d - %d\n", PyArray_TYPE(py_conf_object), NPY_DOUBLE);
    double* cpp_scores = (double*)PyArray_DATA(py_conf_object);
    for (int n = 0; n < conf_n; ++n) {
        printf("%f ", cpp_scores[n]);
    }
    printf("\n");
    // End
    printf("The detection results are successfully parsed ...\n");
    return;
}

int PyTorchSSDCppWrapper::detect2(const LCPRInputIMG& img,
    DetectionOutput& detected) {
    return -1;
}

int PyTorchSSDCppWrapper::release(void) {
    if (py_instance) {
        Py_DECREF(py_instance);
        py_instance = NULL;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

ObjectDetectionIntf* create() {
    return new PyTorchSSDCppWrapper;
}

void destroy(ObjectDetectionIntf* obj) {
    delete obj;
    obj = NULL;
}

#ifdef __cplusplus
}
#endif
