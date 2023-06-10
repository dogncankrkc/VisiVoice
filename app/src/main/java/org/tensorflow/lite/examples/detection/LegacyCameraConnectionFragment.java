package org.tensorflow.lite.examples.detection;

/*
 * Bu sınıf, TensorFlow Lite nesne tespiti uygulamasında kullanılan Legacy Camera API'sini
 * temsil eder. Bu API, Android Kamera sınıfını kullanarak kamera önizlemesi yapar.
 */

import android.annotation.SuppressLint;
import android.app.Fragment;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;

import java.io.IOException;
import java.util.List;

import org.tensorflow.lite.examples.detection.customview.AutoFitTextureView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;

@SuppressLint("ValidFragment")
public class LegacyCameraConnectionFragment extends Fragment {
    private static final Logger LOGGER = new Logger();
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();

    // Döndürme değerlerini atanmış bir SparseIntArray oluşturur.
    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    private Camera camera;
    private final Camera.PreviewCallback imageListener;
    private final Size desiredSize;
    private final int layout;
    private AutoFitTextureView textureView;

    // SurfaceTextureListener için bir nesne oluşturur.
    private final TextureView.SurfaceTextureListener surfaceTextureListener =
            new TextureView.SurfaceTextureListener() {
                @Override
                public void onSurfaceTextureAvailable(
                        final SurfaceTexture texture, final int width, final int height) {

                    // Kamera nesnesini açar
                    int index = getCameraId();
                    camera = Camera.open(index);

                    try {
                        // Kamera parametrelerini alır
                        Camera.Parameters parameters = camera.getParameters();
                        List<String> focusModes = parameters.getSupportedFocusModes();

                        // Sürekli odaklama modunu destekliyorsa, odaklama modunu ayarlar
                        if (focusModes != null
                                && focusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE)) {
                            parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
                        }

                        // Desteklenen önizleme boyutlarını alır
                        List<Camera.Size> cameraSizes = parameters.getSupportedPreviewSizes();
                        Size[] sizes = new Size[cameraSizes.size()];
                        int i = 0;
                        for (Camera.Size size : cameraSizes) {
                            sizes[i++] = new Size(size.width, size.height);
                        }

                        // En uygun önizleme boyutunu seçer
                        Size previewSize =
                                CameraConnectionFragment.chooseOptimalSize(
                                        sizes, desiredSize.getWidth(), desiredSize.getHeight());
                        parameters.setPreviewSize(previewSize.getWidth(), previewSize.getHeight());

                        // Kamera önizlemesini döndürür
                        camera.setDisplayOrientation(90);
                        camera.setParameters(parameters);
                        camera.setPreviewTexture(texture);
                    } catch (IOException exception) {
                        // Hata durumunda kamera serbest bırakılır
                        camera.release();
                    }

                    // Kamera önizlemesi için geri arama belirler
                    camera.setPreviewCallbackWithBuffer(imageListener);

                    // Kamera önizleme boyutunu alır ve görüntü tamponunu ekler
                    Camera.Size s = camera.getParameters().getPreviewSize();
                    camera.addCallbackBuffer(new byte[ImageUtils.getYUVByteSize(s.height, s.width)]);

                    // TextureView'nin en-boy oranını ayarlar
                    textureView.setAspectRatio(s.height, s.width);

                    // Kamera önizlemesini başlatır
                    camera.startPreview();
                }

                @Override
                public void onSurfaceTextureSizeChanged(
                        final SurfaceTexture texture, final int width, final int height) {
                    // Yüzeyin boyutu değiştiğinde yapılacak işlemler burada tanımlanır
                }

                @Override
                public boolean onSurfaceTextureDestroyed(final SurfaceTexture texture) {
                    // Yüzey yok edildiğinde yapılacak işlemler burada tanımlanır
                    return true;
                }

                @Override
                public void onSurfaceTextureUpdated(final SurfaceTexture texture) {
                    // Yüzey güncellendiğinde yapılacak işlemler burada tanımlanır
                }
            };
    private HandlerThread backgroundThread;

    public LegacyCameraConnectionFragment(
            final Camera.PreviewCallback imageListener, final int layout, final Size desiredSize) {
        this.imageListener = imageListener;
        this.layout = layout;
        this.desiredSize = desiredSize;
    }

    @Override
    public View onCreateView(
            final LayoutInflater inflater, final ViewGroup container, final Bundle savedInstanceState) {
        // Fragment'in görünümünü oluşturur
        return inflater.inflate(layout, container, false);
    }

    @Override
    public void onViewCreated(final View view, final Bundle savedInstanceState) {
        // Görünüm oluşturulduğunda yapılacak işlemler burada tanımlanır
        textureView = view.findViewById(R.id.texture);
    }

    @Override
    public void onActivityCreated(final Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
    }

    @Override
    public void onResume() {
        super.onResume();

        // Arka plan iş parçacığını başlatır
        startBackgroundThread();

        if (textureView.isAvailable()) {
            // TextureView kullanılabilir durumdaysa kamera önizlemesini başlatır
            camera.startPreview();
        } else {
            // TextureView kullanılabilir durumda değilse SurfaceTextureListener'ı ayarlar
            textureView.setSurfaceTextureListener(surfaceTextureListener);
        }
    }

    @Override
    public void onPause() {
        // Kamerayı durdurur ve arka plan iş parçacığını durdurur
        stopCamera();
        stopBackgroundThread();
        super.onPause();
    }

    /**
     * Arka plan iş parçacığını ve ilgili {@link Handler}'ı başlatır.
     */
    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("CameraBackground");
        backgroundThread.start();
    }

    /**
     * Arka plan iş parçacığını ve ilgili {@link Handler}'ı durdurur.
     */
    private void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }
    }

    /**
     * Kamerayı durdurur.
     */
    protected void stopCamera() {
        if (camera != null) {
            camera.stopPreview();
            camera.setPreviewCallback(null);
            camera.release();
            camera = null;
        }
    }

    /**
     * Kamera ID'sini alır.
     */
    private int getCameraId() {
        CameraInfo ci = new CameraInfo();
        for (int i = 0; i < Camera.getNumberOfCameras(); i++) {
            Camera.getCameraInfo(i, ci);
            if (ci.facing == CameraInfo.CAMERA_FACING_BACK) return i;
        }
        return -1; // Kamera bulunamadı
    }
}
