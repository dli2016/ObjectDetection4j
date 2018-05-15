# ObjectDetection4j
A simple example to achieve object detection using pytorch SSD in java environment. It is also an example to call python code (CPython) from java.

Notes:
1. The code only test under anaconda-python3.6.4 and java 1.8.0_172 in Linux (centos7).
2. To achieve the gaol, i.e., calling python code from java, the python code is first embedded into c, and then java calls c code using JNI.
3. The python c API and numpy c API are used in our c code.
4. Make sure to load the dll under the mode of "RTLD_LAZY|RTLD_GLOBAL". (Thanks the answer in https://github.com/ContinuumIO/anaconda-issues/issues/6401)
5. ssd.pytorch is used to do object detection with python (Thanks the source code in https://github.com/amdegroot/ssd.pytorch).

Please feel free to email dli1988@126.com for any prolembs.
