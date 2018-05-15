
package org.cripac.isee;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.CharacterCodingException;
import java.nio.file.AccessDeniedException;
import java.util.Collection;

public class ObjectDetection implements Detector {

    static {
        try {
            System.out.println("Load native libraries from " + 
                System.getProperty("java.library.path"));
            System.loadLibrary("cppwrapper_pytorch_ssd_jni");
            System.out.println("Native library loaded successfully!");
        } catch (Throwable t) {
            System.out.println("Failed to load native library!");
            t.printStackTrace();
        }
    }

    private long handle;
    private long dl;

    private native long loadLibrary(String lib_name);
    private native long initialize(long dl, int gpu, String modelPath);
    private native int detect(long handle, byte[] imgBuf, int h, int w, int c, 
        float[] bbs, float[] scores);
    private native int release(long handle, long dl);
    private native void closeLibrary(long dl);

    // Constructor
    public ObjectDetection(int gpu, File modelFile) throws 
        FileNotFoundException, AccessDeniedException, CharacterCodingException {
        
        if (!modelFile.exists()) {
            throw new FileNotFoundException("Cannot find " + 
                modelFile.getPath());
        }
        if (!modelFile.canRead()) {
            throw new AccessDeniedException("Cannot read " + 
                modelFile.getPath());
        }
        
        // Load library.
        dl = loadLibrary("libpytorch_ssd.so");
        if (dl > 0) {
            System.out.println("Load library of python code Done!");
        } else {
            System.exit(-1);
        }
        System.out.println("Initializing ...");
        // Initialize.
        handle = initialize(dl, gpu, modelFile.getPath());
        if (handle > 0) {
            System.out.println("Initialization Done!");
        } else {
            System.out.println("Initialize FAILED!");
            System.exit(-1);
        }
    }

    @Override
    public void free() {
        release(handle, dl);
        closeLibrary(dl);
        //super.finalize();
        System.out.println("Resources RELEASED!");
    }

    @Override
    public DetectionOutput process(byte[] frame, int h, int w, int c) {
        int kBBCoordValsNum = 4;
        int kMaxObjectsNum = 128;
        float[] bbs = new float[kBBCoordValsNum*kMaxObjectsNum];
        float[] scores = new float[kMaxObjectsNum];
        int objsNum = detect(handle, frame, h, w, c, bbs, scores);
        System.out.println("#Objects " + objsNum);
        for (int i = 0; i < objsNum*4; ++i) {
            System.out.println("position:" + bbs[i]);
        }
        for (int i = 0; i < objsNum; ++i) {
            System.out.println("score:" + scores[i]);
        }
        return null; // Test
    }

    @Override
    protected void finalize() throws Throwable {
        //release(handle, dl);
        //closeLibrary(dl);
        super.finalize();
        //System.out.println("Resources RELEASED!");
    }

    public static class DetectionOutput {
        int numObjects = -1;
        BoundingBox[] bbs = null;
        float[] scores = null;
        public static class BoundingBox {
            public int x = 0;
            public int y = 0;
            public int width = 0;
            public int height = 0;
        }
    }    
}
