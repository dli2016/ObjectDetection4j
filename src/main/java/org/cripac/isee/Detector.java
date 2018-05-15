
package org.cripac.isee;

import org.cripac.isee.ObjectDetection.DetectionOutput;

public interface Detector {
    DetectionOutput process(byte[] frame, int h, int w, int c);
    void free();
}
