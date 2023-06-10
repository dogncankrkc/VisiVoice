/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

// Yüksek seviye kütüphaneler

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;

// UI kütüphaneleri
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.bottomsheet.BottomSheetBehavior;

// I/O ve util kütüphaneleri
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Objects;

// TensorFlow ve uygulama özel kütüphaneler
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;

public abstract class CameraActivity extends AppCompatActivity
        implements OnImageAvailableListener,
        Camera.PreviewCallback,
        View.OnClickListener {
    private static final Logger LOGGER = new Logger();

    // İzinler için sabit değerler
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;

    // Diğer değişkenlerin tanımlanması
    private static final String ASSET_PATH = "";
    protected int previewWidth = 0;
    protected int previewHeight = 0;
    protected Handler handler;
    private HandlerThread handlerThread;
    private boolean useCamera2API;
    private boolean isProcessingFrame = false;
    private final byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    protected int defaultModelIndex = 0;
    protected int defaultDeviceIndex = 0;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    protected ArrayList<String> modelStrings = new ArrayList<>();

    // UI elementlerinin tanımlanması
    private LinearLayout gestureLayout;
    private BottomSheetBehavior<LinearLayout> sheetBehavior;

    // Metin görüntüleyicilerinin tanımlanması
    protected TextView frameValueTextView;
    protected TextView cropValueTextView;
    private ImageView plusImageView, minusImageView;
    protected TextView threadsTextView;

    /**
     * Current indices of device and model.
     */
    int currentDevice = -1;
    int currentModel = -1;
    int currentNumThreads = -1;

    ArrayList<String> deviceStrings = new ArrayList<>();

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        // Logger'ı kullanarak debug mesajını yazdır
        LOGGER.d("onCreate " + this);
        // Üst sınıfın onCreate metodunu çağır
        super.onCreate(null);
        // Ekranın sürekli açık kalmasını sağla
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // Layout'u belirle
        setContentView(R.layout.tfe_od_activity_camera);
        // Toolbar'u bul ve ayarla
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        Objects.requireNonNull(getSupportActionBar()).setDisplayShowTitleEnabled(false);

        // Kamera izni var mı kontrol et
        if (hasPermission()) {
            // Fragment'i ayarla
            setFragment();
        } else {
            // Kamera izni yoksa izin iste
            requestPermission();
        }

        // Varsayılan model ve cihazı belirle
        currentNumThreads = 2;
        deviceStrings.add("CPU");
        currentDevice = defaultDeviceIndex;

        // AssetManager ve path kullanarak model listesini al
        modelStrings = getModelStrings(getAssets(), ASSET_PATH);

        // Model seçeneklerini içeren ArrayAdapter'ı oluştur
        currentModel = defaultModelIndex;
    }

    // getRgbBytes metodu, RGB baytlarını döndürür
    protected int[] getRgbBytes() {
        imageConverter.run();
        return rgbBytes;
    }

    // getModelStrings metodu, AssetManager ve path kullanarak model listesini döndürür
    protected ArrayList<String> getModelStrings(AssetManager mgr, String path) {
        ArrayList<String> res = new ArrayList<String>();
        try {
            String[] files = mgr.list(path);
            for (String file : files) {
                String[] splits = file.split("\\.");
                if (splits[splits.length - 1].equals("tflite")) {
                    res.add(file);
                }
            }

        } catch (IOException e) {
            System.err.println("getModelStrings: " + e.getMessage());
        }
        return res;
    }

    // onPreviewFrame metodu, kamera önizlemesi için çağırılır
    @Override
    public void onPreviewFrame(final byte[] bytes, final Camera camera) {
        // Eğer bir çerçeve işleniyorsa, çerçeve düşürülür
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!");
            return;
        }

        try {
            // Çözünürlük bilindiğinde depolama bitmap'lerini bir kez başlatın
            if (rgbBytes == null) {
                Camera.Size previewSize = camera.getParameters().getPreviewSize();
                previewHeight = previewSize.height;
                previewWidth = previewSize.width;
                rgbBytes = new int[previewWidth * previewHeight];
                onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
            }
        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
            return;
        }
        // Çerçeve işleniyor olarak işaretlenir
        isProcessingFrame = true;
        yuvBytes[0] = bytes;
        yRowStride = previewWidth;

        // Görüntüyü dönüştürmek için imageConverter'ı çalıştır
        imageConverter =
                () -> ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);

        // İşlem sonrası callback'i belirle
        postInferenceCallback =
                () -> {
                    camera.addCallbackBuffer(bytes);
                    isProcessingFrame = false;
                };
        // Görüntüyü işle
        processImage();
    }

    // onImageAvailable metodu, kullanılabilir bir görüntü olduğunda çağırılır
    @Override
    public void onImageAvailable(final ImageReader reader) {
        // onPreviewSizeChosen'dan bazı boyutlara sahip olana kadar beklememiz gerekiyor
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            Trace.beginSection("imageAvailable");
            final Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            // Görüntüyü dönüştürmek için imageConverter'ı çalıştır
            imageConverter =
                    () -> ImageUtils.convertYUV420ToARGB8888(
                            yuvBytes[0],
                            yuvBytes[1],
                            yuvBytes[2],
                            previewWidth,
                            previewHeight,
                            yRowStride,
                            uvRowStride,
                            uvPixelStride,
                            rgbBytes);

            // İşlem sonrası callback'i belirle
            postInferenceCallback =
                    () -> {
                        image.close();
                        isProcessingFrame = false;
                    };

            // Görüntüyü işle
            processImage();
        } catch (final Exception e) {
            LOGGER.e(e, "Exception!");
            Trace.endSection();
            return;
        }
        Trace.endSection();
    }

    // onStart metodu, Activity başladığında çağırılır
    @Override
    public synchronized void onStart() {
        LOGGER.d("onStart " + this);
        super.onStart();
    }

    // onResume metodu, Activity devam ettiğinde çağırılır
    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();

        // Thread'i başlat
        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    // onPause metodu, Activity durakladığında çağırılır
    @Override
    public synchronized void onPause() {
        LOGGER.d("onPause " + this);

        // Thread'i sonlandır
        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }

        super.onPause();
    }

    // onStop metodu, Activity durduğunda çağırılır
    @Override
    public synchronized void onStop() {
        LOGGER.d("onStop " + this);
        super.onStop();
    }

    // onDestroy metodu, Activity yok edildiğinde çağırılır
    @Override
    public synchronized void onDestroy() {
        LOGGER.d("onDestroy " + this);
        super.onDestroy();
    }

    // runInBackground metodu, arka planda bir işlemi çalıştırır
    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    // onRequestPermissionsResult metodu, izin talebine cevap verir
    @Override
    public void onRequestPermissionsResult(
            final int requestCode, @NonNull final String[] permissions, @NonNull final int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                setFragment();
            } else {
                requestPermission();
            }
        }
    }

    // allPermissionsGranted metodu, tüm izinlerin verilip verilmediğini kontrol eder
    private static boolean allPermissionsGranted(final int[] grantResults) {
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    // hasPermission metodu, kamera izni olup olmadığını kontrol eder
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    // requestPermission metodu, kamera izni istemek için çağırılır
    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                                CameraActivity.this,
                                "Camera permission is required for this demo",
                                Toast.LENGTH_LONG)
                        .show();
            }
            requestPermissions(new String[]{PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
        }
    }

    // isHardwareLevelSupported metodu, gereken donanım seviyesinin desteklenip desteklenmediğini kontrol eder
    private boolean isHardwareLevelSupported(CameraCharacteristics characteristics) {
        int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            return false;
        }
        return android.hardware.camera2.CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL <= deviceLevel;
    }

    // chooseCamera metodu, kullanılacak kamerayı seçer
    @RequiresApi(api = Build.VERSION_CODES.M)
    private String chooseCamera() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (final String cameraId : manager.getCameraIdList()) {
                final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                // Ön kamera kullanmayın
                final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                // StreamConfigurationMap al
                final StreamConfigurationMap map =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (map == null) {
                    continue;
                }

                // Kamera2 API kullanılacak mı kontrol et
                useCamera2API =
                        (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                                || isHardwareLevelSupported(
                                characteristics);
                LOGGER.i("Camera API lv2?: %s", useCamera2API);
                return cameraId;
            }
        } catch (CameraAccessException e) {
            LOGGER.e(e, "Not allowed to access camera");
        }

        return null;
    }

    // setFragment metodu, fragmenti ayarlar
    protected void setFragment() {
        String cameraId = null;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
            cameraId = chooseCamera();
        }

        Fragment fragment;
        if (useCamera2API) {
            CameraConnectionFragment camera2Fragment =
                    CameraConnectionFragment.newInstance(
                            (size, rotation) -> {
                                previewHeight = size.getHeight();
                                previewWidth = size.getWidth();
                                CameraActivity.this.onPreviewSizeChosen(size, rotation);
                            },
                            this,
                            getLayoutId(),
                            getDesiredPreviewFrameSize());

            camera2Fragment.setCamera(cameraId);
            fragment = camera2Fragment;
        } else {
            fragment =
                    new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
        }

        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }

    // fillBytes metodu, byte dizilerini doldurur
    protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    // isDebug metodu, debug modunda olup olmadığını döndürür
    public boolean isDebug() {
        return false;
    }

    // readyForNextImage metodu, bir sonraki görüntüye hazır olduğunu belirtir
    protected void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }

    // getScreenOrientation metodu, ekranın dönme durumunu döndürür
    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    // showFrameInfo metodu, çerçeve bilgilerini gösterir
    protected void showFrameInfo() {
    }

    // showCropInfo metodu, kırpma bilgisini gösterir
    protected void showCropInfo(String cropInfo) {
    }

    // showInference metodu, sonucu gösterir
    protected void showInference() {
    }

    // processImage metodu, görüntüyü işler
    protected abstract void processImage();

    // onPreviewSizeChosen metodu, önizleme boyutu seçildiğinde çağırılır
    protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

    // getLayoutId metodu, layout ID'sini döndürür
    protected abstract int getLayoutId();

    // getDesiredPreviewFrameSize metodu, istenen önizleme boyutunu döndürür
    protected abstract Size getDesiredPreviewFrameSize();

}
