package com.iwatake.viewandroidtflite;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.util.Formatter;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    /*** Fixed values ***/
    private static final String TAG = "MyApp";
    private int REQUEST_CODE_FOR_PERMISSIONS = 1234;;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    private enum AppStatus {
        NotInitialized,
        Initialized,
        Running,
    };

    /*** Views ***/
    private PreviewView previewView;
    private ImageView imageView;
    private ImageView imageView2;
    private Button buttonCamera;
    private Button buttonVr;
    private Button buttonVrExit;
    private Button buttonCmd0;
    private Button buttonCmd1;
    private Button buttonCmd2;
    private TextView textViewFps;
    private TextView textViewImageProcessTime;

    /*** For CameraX ***/
    private Camera camera = null;
    private Preview preview = null;
    private ImageAnalysis imageAnalysis = null;
    private ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();

    private int lensFacing = CameraSelector.LENS_FACING_BACK;
    private AppStatus appStatus = AppStatus.NotInitialized;

    static {
        System.loadLibrary("native-lib");
        System.loadLibrary("opencv_java4");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getViews();
        setEventListeners();
        exitVrMode();

        if (checkPermissions()) {
            if (appStatus == AppStatus.NotInitialized) {
                if (ImageProcessorInitialize() == 0) {
                    ImageProcessorCommand(0);
                    appStatus = AppStatus.Initialized;
                }
            }
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_FOR_PERMISSIONS);
        }
    }

    private void getViews() {
        previewView = findViewById(R.id.previewView);
        imageView = findViewById(R.id.imageView);
        imageView2 = findViewById(R.id.imageView2);
        buttonCamera = findViewById(R.id.buttonCamera);
        buttonVr = findViewById(R.id.buttonVr);
        buttonVrExit = findViewById(R.id.buttonVrExit);
        buttonCmd0 = findViewById(R.id.buttonCmd0);
        buttonCmd1 = findViewById(R.id.buttonCmd1);
        buttonCmd2 = findViewById(R.id.buttonCmd2);
        textViewFps = findViewById(R.id.textViewFps);
        textViewImageProcessTime = findViewById(R.id.textViewImageProcessTime);
    }

    private void setEventListeners() {
        buttonCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                    lensFacing = CameraSelector.LENS_FACING_FRONT;
                    buttonCamera.setText("BACK");
                } else {
                    lensFacing = CameraSelector.LENS_FACING_BACK;
                    buttonCamera.setText("FRONT");
                }
                startCamera();
            }
        });

        buttonCmd0.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ImageProcessorCommand((0));
            }
        });

        buttonCmd1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ImageProcessorCommand((1));
            }
        });

        buttonCmd2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ImageProcessorCommand((2));
            }
        });

        buttonVr.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                enterVrMode();
            }
        });

        buttonVrExit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                exitVrMode();
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (appStatus == AppStatus.Initialized) {
            appStatus = AppStatus.NotInitialized;
            ImageProcessorFinalize();
        }
    }

    private void enterVrMode() {
        imageView2.setVisibility(View.VISIBLE);
        imageView2.setLayoutParams(new TableRow.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f));
        buttonCamera.setVisibility(View.INVISIBLE);
        buttonVr.setVisibility(View.INVISIBLE);
        buttonCmd0.setVisibility(View.INVISIBLE);
        buttonCmd1.setVisibility(View.INVISIBLE);
        buttonCmd2.setVisibility(View.INVISIBLE);
        textViewFps.setVisibility(View.INVISIBLE);
        textViewImageProcessTime.setVisibility(View.INVISIBLE);
        buttonVrExit.setVisibility(View.VISIBLE);
    }

    private void exitVrMode() {
        imageView2.setVisibility(View.INVISIBLE);
        imageView2.setLayoutParams(new TableRow.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 0f));
        buttonCamera.setVisibility(View.VISIBLE);
        buttonVr.setVisibility(View.VISIBLE);
        buttonCmd0.setVisibility(View.VISIBLE);
        buttonCmd1.setVisibility(View.VISIBLE);
        buttonCmd2.setVisibility(View.VISIBLE);
        textViewFps.setVisibility(View.VISIBLE);
        textViewImageProcessTime.setVisibility(View.VISIBLE);
        buttonVrExit.setVisibility(View.INVISIBLE);
    }

    private void startCamera() {
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        Context context = this;
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
//                    preview = new Preview.Builder().build();
                    imageAnalysis = new ImageAnalysis.Builder().build();
                    imageAnalysis.setAnalyzer(cameraExecutor, new MyImageAnalyzer());
                    CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(lensFacing).build();

                    cameraProvider.unbindAll();
//                    camera = cameraProvider.bindToLifecycle((LifecycleOwner)context, cameraSelector, preview, imageAnalysis);
                    camera = cameraProvider.bindToLifecycle((LifecycleOwner)context, cameraSelector,  imageAnalysis);
//                    preview.setSurfaceProvider(previewView.createSurfaceProvider(camera.getCameraInfo()));
                } catch(Exception e) {
                    Log.e(TAG, "[startCamera] Use case binding failed", e);
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private class MyImageAnalyzer implements ImageAnalysis.Analyzer {
        private long previousTime = System.nanoTime();
        private float averageFPS = 0;
        private long averageImageProcessTime = 0;
        private int frameCount = 0;

        @Override
        public void analyze(@NonNull ImageProxy image) {
            if (previewView.getDisplay() == null || appStatus != AppStatus.Initialized || image.getWidth() == 0) {
                image.close();
                return;
            }
            /* Create cv::mat(RGB888) from image(NV21) */
            Mat matOrg = getMatFromImage(image);
            /* Fix image rotation (it looks image in PreviewView is automatically fixed by CameraX???) */
            Mat mat = fixMatRotation(matOrg);
//            Log.i(TAG, "[analyze] width = " + image.getWidth() + ", height = " + image.getHeight() + "Rotation = " + previewView.getDisplay().getRotation());
//            Log.i(TAG, "[analyze] mat width = " + matOrg.cols() + ", mat height = " + matOrg.rows());

            /* Do some image processing */
            appStatus = AppStatus.Running;
            long imageProcessTimeStart = System.nanoTime();
            ImageProcessorProcess(mat.getNativeObjAddr());
            long imageProcessTimeEnd = System.nanoTime();
            appStatus = AppStatus.Initialized;
            Mat matOutput = mat;
//            Mat matOutput = new Mat(mat.rows(), mat.cols(), mat.type());
//            if (matPrevious == null) matPrevious = mat;
//            Core.absdiff(mat, matPrevious, matOutput);
//            matPrevious = mat;

            /* Calculate FPS */
            long currentTime = System.nanoTime();
            float fps = 1000000000 / (currentTime - previousTime);
            previousTime = currentTime;
            frameCount++;
            averageFPS = (averageFPS * (frameCount - 1) + fps) / frameCount;
            Formatter fmFps = new Formatter();
            fmFps.format("%4.1f (%4.1f) [FPS]", averageFPS, fps);

            long imageProcessTime = imageProcessTimeEnd - imageProcessTimeStart;
            averageImageProcessTime = (long)((averageImageProcessTime * (frameCount - 1) + imageProcessTime) / frameCount);
            Formatter fmImageProcessTime = new Formatter();
            fmImageProcessTime.format("%d (%d) [msec]", averageImageProcessTime/1000000, imageProcessTime/1000000);

            /* Convert cv::mat to bitmap for drawing */
            Bitmap bitmap = Bitmap.createBitmap(matOutput.cols(), matOutput.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matOutput, bitmap);

            /* Display the result onto ImageView */
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    imageView.setImageBitmap(bitmap);
                    if (imageView2.getVisibility() == View.VISIBLE)  imageView2.setImageBitmap(bitmap);
                    textViewFps.setText(fmFps.toString());
                    textViewImageProcessTime.setText(fmImageProcessTime.toString());
                }
            });

            /* Close the image otherwise, this function is not called next time */
            image.close();
        }

        private Mat getMatFromImage(ImageProxy image) {
            /* https://stackoverflow.com/questions/30510928/convert-android-camera2-api-yuv-420-888-to-rgb */
            ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
            ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
            ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();
            byte[] nv21 = new byte[ySize + uSize + vSize];
            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);
            Mat yuv = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
            yuv.put(0, 0, nv21);
            Mat mat = new Mat();
            Imgproc.cvtColor(yuv, mat, Imgproc.COLOR_YUV2RGB_NV21, 3);
            return mat;
        }

        private Mat fixMatRotation(Mat matOrg) {
            Mat mat;
            switch (previewView.getDisplay().getRotation()){
                default:
                case Surface.ROTATION_0:
                    mat = new Mat(matOrg.cols(), matOrg.rows(), matOrg.type());
                    Core.transpose(matOrg, mat);
                    if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                        Core.flip(mat, mat, 1);
                    } else {
                        Core.flip(mat, mat, 0);
                    }
                    break;
                case Surface.ROTATION_90:
                    mat = matOrg;
                    break;
                case Surface.ROTATION_270:
                    mat = matOrg;
                    Core.flip(mat, mat, -1);
                    break;
            }

            return mat;
        }
    }

    private boolean checkPermissions(){
        for(String permission : REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == REQUEST_CODE_FOR_PERMISSIONS){
            if(checkPermissions()){
                ImageProcessorInitialize();
                ImageProcessorCommand(0);
                appStatus = AppStatus.Initialized;
                startCamera();
            } else{
                Log.i(TAG, "[onRequestPermissionsResult] Failed to get permissions");
                this.finish();
            }
        }
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native int ImageProcessorInitialize();
    public native int ImageProcessorProcess(long objMat);
    public native int ImageProcessorFinalize();
    public native int ImageProcessorCommand(int cmd);
}
