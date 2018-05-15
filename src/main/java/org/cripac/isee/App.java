package org.cripac.isee;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Object detetion by call pyTorch ssd.
 *
 */
public class App 
{
    static {
        Loader.load(opencv_core.class);
    }
    public static void main( String[] args ) throws Exception
    {
        if (args.length != 1) {
            System.out.println("Error: BAD number of input args!");
            System.out.println("Info: 2 args are needed.");
            return;
        }
        //System.out.println("Performing validness test...");
        //System.out.println("Native library path: " + 
        //    System.getProperty("java.library.path"));
        System.out.println("Creating Detector...");

        Detector detector = new ObjectDetection(
            -1,
            new File("src/native/test/weights/ssd300_mAP_77.43_v2.pth")
        );
        System.out.println("Create object detector SUCCESSFULLY!");

        // Load image
        String image_filename = args[0];
        opencv_core.Mat img = imread(image_filename, IMREAD_COLOR);
        byte[] img_data = new byte[img.rows() * img.cols() * img.channels()];
        img.data().get(img_data);

        // Detect
        detector.process(img_data, img.rows(), img.cols(), img.channels());
        img.release();
        detector.free();
    }
}
