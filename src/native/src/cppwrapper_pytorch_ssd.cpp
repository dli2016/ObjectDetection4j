#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"
#include "cppwrapper_pytorch_ssd.h"

PyTorchSSDCppWrapper::PyTorchSSDCppWrapper(void) {
    init_res = NULL;
    p_module = NULL;
}

PyTorchSSDCppWrapper::~PyTorchSSDCppWrapper(void) {
    release();
}

int PyTorchSSDCppWrapper::init(const char* weight_filename) {
    // Set PYTHONPATH TO working directory
    //setenv("PYTHONPATH",
    //    "/home/data/da.li/projects/ObjectDetection4j/src/native/lib",1);
    setenv("PYTHONPATH",
        "/home/da.li/anaconda3/lib/python3.6/site-packages:/home/data/da.li/projects/ObjectDetection4j/src/native/lib",1);
    printf("%s\n", getenv("PYTHONPATH"));

    char* input_module_name = (char*)"ssd_detector_pytorch";
    char* input_func_name = (char*)"initialize";
    // Initialize the Python Interpreter
    // printf("-99999999\n");
    Py_Initialize();
    //printf("000000000\n");
    if (PyArray_API==NULL) {
        import_array();
    }
 
    //printf("111111111\n");
    PyObject* module_name = PyUnicode_DecodeFSDefault(input_module_name);
    //printf("222222222\n");
    p_module = PyImport_Import(module_name);
    //printf("333333333\n");
    Py_DECREF(module_name);
    
    // init the net
    if (p_module != NULL) {
        PyObject* p_func = PyObject_GetAttrString(p_module, input_func_name);
        if (p_func && PyCallable_Check(p_func)) {
            //PyObject* p_args = PyTuple_Pack(1,(char*)weight_filename);
            PyObject* p_args = PyTuple_New(1);
            PyObject* p_val = PyBytes_FromString((char*)weight_filename);
            PyTuple_SetItem(p_args, 0, p_val);
            init_res = PyObject_CallObject(p_func, p_args);
            Py_DECREF(p_args);
            if (init_res) {
                printf("Initialize successfully!\n");
            } else {
                Py_DECREF(p_func);
                PyErr_Print();
                fprintf(stderr,"Call %s failed\n", input_func_name);
                return -1;
            }
        }
        else {
            if (PyErr_Occurred()) {
                PyErr_Print();
            }
            fprintf(stderr, "Cannot find function \"%s\"\n", input_func_name);
            return -1;
        }
        Py_XDECREF(p_func);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", input_module_name);
        return -1;
    }
    return 0;
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

int PyTorchSSDCppWrapper::detect(unsigned char* data, int h, int w, int c, 
    float* bbs, float* scores) {
    if (!p_module) {
        fprintf(stderr, "The pointer to the module is NULL!");
        return -1;
    }
    int obj_num = 0;
    PyObject* p_func = PyObject_GetAttrString(p_module, "run_v4");
    if (p_func && PyCallable_Check(p_func)) {
        // Convert cpp data to python object.
        int nd = 3;
        npy_intp dims[] = {h, w, c};
        PyObject* py_data = PyArray_SimpleNewFromData(nd, dims, NPY_UINT8,
            (void*)data);
        PyObject* p_args = PyTuple_New(2);
        if (init_res == NULL) {
            fprintf(stderr, "The net is NULL!");
            return -1;
        }
        PyTuple_SetItem(p_args, 0, init_res);
        PyTuple_SetItem(p_args, 1, py_data);
        PyObject* detect_res = PyObject_CallObject(p_func, p_args);
        Py_DECREF(p_args);
        if (detect_res) {
            printf("Detect successfully!\n");
            // Parse results:
            obj_num = parseDetectRes_v2(detect_res, bbs, scores);
            Py_DECREF(detect_res);
        } else {
            Py_DECREF(p_func);
            PyErr_Print();
            fprintf(stderr,"Call %s failed\n", "run_v4");
            return -1;
        }
    } else {
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
        fprintf(stderr, "Cannot find function \"%s\"\n", "run_v4");
        return -1;
    }
    Py_XDECREF(p_func);

    return obj_num;
}

int PyTorchSSDCppWrapper::detect1(unsigned char* data, int h, int w, int c) {
    if (p_module != NULL) {
        PyObject* p_func = PyObject_GetAttrString(p_module, "run_v3");
        if (p_func && PyCallable_Check(p_func)) {
            // Convert cpp data to python object.
            int nd = 3;
            npy_intp dims[] = {h, w, c};
            PyObject* py_data = PyArray_SimpleNewFromData(nd, dims, NPY_UINT8,
                (void*)data);
            PyObject* py_h = PyLong_FromSize_t((size_t)h);
            PyObject* py_w = PyLong_FromSize_t((size_t)w);
            PyObject* py_c = PyLong_FromSize_t((size_t)c);
            PyObject* p_args = PyTuple_New(4);
            PyTuple_SetItem(p_args, 0, py_data);
            PyTuple_SetItem(p_args, 1, py_h);
            PyTuple_SetItem(p_args, 2, py_w);
            PyTuple_SetItem(p_args, 3, py_c);
            PyObject_CallObject(p_func, p_args);
            Py_DECREF(p_args);
        } else {
            if (PyErr_Occurred()) {
                PyErr_Print();
            }
            fprintf(stderr, "Cannot find function \"%s\"\n", "run_v3");
            return -1;
        }
        Py_XDECREF(p_func);
    } else {
        fprintf(stderr, "The pointer to the module is NULL!");
        return -1;
    }
    return 0;
}

int PyTorchSSDCppWrapper::detect0(const char* image_filename) {
    if (p_module != NULL) {
        PyObject* p_func = PyObject_GetAttrString(p_module, "run_v2");
        if (p_func && PyCallable_Check(p_func)) {
            PyObject* p_args = PyTuple_New(2);
            PyObject* p_val = PyBytes_FromString((char*)image_filename);
            if (init_res == NULL) {
                fprintf(stderr, "The net is NULL!");
                return -1;
            }
            PyTuple_SetItem(p_args, 0, init_res);
            PyTuple_SetItem(p_args, 1, p_val);
            PyObject* detect_res = PyObject_CallObject(p_func, p_args);
            Py_DECREF(p_args);
            if (detect_res) {
                printf("Detect successfully!\n");
                // Parse results:
                parseDetectRes(detect_res);
                Py_DECREF(detect_res);
            } else {
                Py_DECREF(p_func);
                PyErr_Print();
                fprintf(stderr,"Call %s failed\n", "run_v2");
                return -1;
            }
        } else {
            if (PyErr_Occurred()) {
                PyErr_Print();
            }
            fprintf(stderr, "Cannot find function \"%s\"\n", "run_v2");
            return -1;
        }
        Py_XDECREF(p_func);
    } else {
        fprintf(stderr, "The pointer to the module is NULL!");
        return -1;
    }
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
    if (p_module) {
        Py_DECREF(p_module);
        p_module = NULL;
    }
    if (init_res) {
        Py_DECREF(init_res);
        init_res = NULL;
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
