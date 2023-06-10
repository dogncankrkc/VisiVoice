package org.tensorflow.lite.examples.detection;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Handler;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;
import org.tensorflow.lite.examples.detection.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import android.speech.tts.TextToSpeech;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import java.util.Locale;
import java.util.Queue;

/**
 * Bu sınıf, nesne algılama ve metin tanıma yeteneklerine sahip bir TensorFlow Lite modelini kullanarak kameradan görüntüleri işler.
 * Nesneler algılandığında ve metin tanındığında konuşma çıktısı üretir.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private TextToSpeech tts; // Metni konuşma çıktısı üretmek için kullanılan TextToSpeech sınıfı
    private static final Logger LOGGER = new Logger();
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.8f; // Nesne algılaması için minimum güven düzeyi
    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(416, 416); // İstenen önizleme boyutu
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10; // Metin boyutu
    OverlayView trackingOverlay;
    private YoloV5Classifier detector; // Nesne algılama sınıflandırıcısı
    private Bitmap rgbFrameBitmap = null; // Kameradan alınan görüntünün RGB formatındaki kopyası
    private Bitmap croppedBitmap = null; // Kesilmiş görüntü
    private Bitmap cropCopyBitmap = null; // Kesilmiş görüntünün bir kopyası
    private boolean computingDetection = false; // Nesne algılama işlemi yapılıyor mu?
    private boolean computingOCR = false; // Metin tanıma işlemi yapılıyor mu?
    private long timestamp = 0; // Görüntü zaman damgası
    private Matrix frameToCropTransform; // Çerçeve kesimi dönüşüm matrisi
    private Matrix cropToFrameTransform; // Kesimden çerçeve dönüşüm matrisi
    private MultiBoxTracker tracker; // Nesne takipçisi
    private long lastOCRUpdateTime = 0; // Son metin tanıma güncelleme zamanı
    private boolean isObjectDetected = false; // Nesne algılandı mı?
    private final Queue<String> objectSpeakQueue = new LinkedList<>(); // Algılanan nesneler için konuşma kuyruğu
    private final Queue<String> textSpeakQueue = new LinkedList<>(); // Tanınan metinler için konuşma kuyruğu
    private int utteranceId = 0; // Konuşma kimliği

    /**
     * Aktivite oluşturulduğunda çağrılır. Gerekli ayarlamaları yapar ve TTS'yi başlatır.
     */
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.tfe_od_activity_camera);

        TextView bottomSheetText = findViewById(R.id.bottom_sheet_text);
        Handler handler = new Handler();
        String[] texts = new String[]{"Bu uygulama görme engelli bireylere yardımcı olması için yapılmıştır.", "", "Bu uygulama video kaydı yapmamaktadır."};
        Runnable runnable = new Runnable() {
            int i = 0;

            @Override
            public void run() {
                // Animasyonları oluştur
                ObjectAnimator disappear = ObjectAnimator.ofFloat(bottomSheetText, "alpha", 1f, 0f);
                disappear.setDuration(2000); // 1 saniye
                ObjectAnimator appear = ObjectAnimator.ofFloat(bottomSheetText, "alpha", 0f, 1f);
                appear.setDuration(2000); // 1 saniye

                // Animasyonları sırayla çalıştır
                AnimatorSet animatorSet = new AnimatorSet();
                animatorSet.playSequentially(disappear, appear);
                animatorSet.addListener(new AnimatorListenerAdapter() {
                    @Override
                    public void onAnimationEnd(Animator animation) {
                        super.onAnimationEnd(animation);
                        // Animasyonlar bittiğinde metni değiştir
                        bottomSheetText.setText(texts[i]);
                        i++;
                        if (i == texts.length) {
                            i = 0;
                        }
                    }
                });

                animatorSet.start();

                handler.postDelayed(this, 4000); // Zamanı animasyon sürelerinin iki katına ayarlayın (disappear+appear)
            }
        };

        handler.post(runnable);

        tts = new TextToSpeech(getApplicationContext(), status -> {
            if (status != TextToSpeech.ERROR) {
                tts.setLanguage(new Locale("tr", "TR")); // Türkçe dilini ayarlayın
                tts.setSpeechRate(1.0f);
                tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                    @Override
                    public void onStart(String utteranceId) {
                        // Hiçbir şey yapma
                    }

                    @Override
                    public void onDone(String utteranceId) {
                        speak();
                    }

                    @Override
                    public void onError(String utteranceId) {
                        // Hiçbir şey yapma
                    }
                });
                tts.speak("Uygulama kullanıma hazır", TextToSpeech.QUEUE_FLUSH, null, "StartID"); // Konuş
            }
        });
    }

    /**
     * Konuşma kuyruğundaki metni konuşur.
     */
    private void speak() {
        String text = null;
        String idPrefix = null;
        if (!objectSpeakQueue.isEmpty()) {
            text = objectSpeakQueue.poll();
            idPrefix = "ObjectSpeak";
        } else if (!textSpeakQueue.isEmpty()) {
            text = textSpeakQueue.poll();
            idPrefix = "TextSpeak";
        }
        if (text != null) {
            String id = idPrefix + utteranceId++;
            tts.speak(text, TextToSpeech.QUEUE_ADD, null, id);
        }
    }

    /**
     * Nesne algılama sonuçlarını günceller ve konuşma kuyruğunu güncellemek için çağırır.
     */
    /**
     * Nesne algılama sonuçlarını günceller ve konuşma kuyruğunu güncellemek için çağırır.
     */
    private void updateSpeak(final List<Classifier.Recognition> results) {
        objectSpeakQueue.clear();  // Mevcut kuyruğu temizle
        if (results != null && !results.isEmpty()) {
            for (Classifier.Recognition result : results) {
                if (result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                    objectSpeakQueue.add(result.getTitle() + " Algılandı.");
                }
            }
        }
    }

    /**
     * Önizleme boyutu seçildiğinde çağrılır ve gerekli ayarlamaları yapar.
     */
    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        BorderedText borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        final int modelIndex = 0;
        final String modelString = modelStrings.get(modelIndex);

        try {
            detector = DetectorFactory.getDetector(getAssets(), modelString);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        int cropSize = detector.getInputSize();

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        int sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> {
                    tracker.draw(canvas);
                    if (isDebug()) {
                        tracker.drawDebug(canvas);
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    /**
     * Görüntü işleme için çağrılır. Nesne algılama ve metin tanıma işlemlerini yürütür.
     */
    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // Kilitleme gerekmediği için mutex gerekmez.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // Gerçek TF girişini incelemek için.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                () -> {
                    LOGGER.i("Running detection on image " + currTimestamp);
                    final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);


                    Log.e("CHECK", "run: " + results.size());

                    if (results.isEmpty()) {
                        if (isObjectDetected && !computingOCR) {
                            runTextRecognition(croppedBitmap);
                        } else {
                            runOnUiThread(this::clearSpeakQueue);
                        }
                    } else {
                        isObjectDetected = true;
                        updateSpeak(results);
                    }

                    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                    final Canvas canvas1 = new Canvas(cropCopyBitmap);
                    final Paint paint = new Paint();
                    paint.setColor(Color.RED);
                    paint.setStyle(Style.STROKE);
                    paint.setStrokeWidth(2.0f);

                    runTextRecognition(croppedBitmap);

                    final List<Classifier.Recognition> mappedRecognitions =
                            new LinkedList<>();

                    for (final Classifier.Recognition result : results) {
                        final RectF location = result.getLocation();
                        if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                            canvas1.drawRect(location, paint);

                            cropToFrameTransform.mapRect(location);

                            result.setLocation(location);
                            mappedRecognitions.add(result);

                        }
                    }

                    tracker.trackResults(mappedRecognitions, currTimestamp);
                    trackingOverlay.postInvalidate();

                    computingDetection = false;

                    runOnUiThread(
                            () -> {
                                showFrameInfo();
                                showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                showInference();
                            });
                });
    }

    /**
     * Verilen bir bitmap üzerinde metin tanıma işlemini yürütür.
     */
    private void runTextRecognition(Bitmap bitmap) {
        long currentTime = System.currentTimeMillis();
        final List<String> ignoreList = Arrays.asList("DUR", "STOP", "D", "P", "20", "30", "40", "50", "70", "90", "120");

        if (currentTime - lastOCRUpdateTime < 1500) {
            return;  // Son metin tanıma üzerinden 1.5 saniye geçmediyse, sadece dön
        }
        lastOCRUpdateTime = currentTime;  // Son metin tanıma zamanını güncelle

        InputImage image = InputImage.fromBitmap(bitmap, 0);
        TextRecognizer recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        recognizer.process(image)
                .addOnSuccessListener(
                        texts -> {
                            computingOCR = false;
                            StringBuilder completeText = new StringBuilder();
                            boolean isTextDetected = false;
                            for (Text.TextBlock block : texts.getTextBlocks()) {
                                for (Text.Line line : block.getLines()) {
                                    for (Text.Element element : line.getElements()) {
                                        String elementText = element.getText().toUpperCase();
                                        if (!ignoreList.contains(elementText)) {
                                            completeText.append(elementText).append(" ");
                                            isTextDetected = true;
                                        }
                                    }
                                }
                            }
                            boolean finalIsTextDetected = isTextDetected;
                            runOnUiThread(() -> {
                                if (finalIsTextDetected) {
                                    textSpeakQueue.add(completeText.toString());
                                    speak();
                                } else {
                                    clearSpeakQueue();
                                }
                            });
                        }
                )
                .addOnFailureListener(
                        e -> {
                            computingOCR = false;
                            isObjectDetected = false;
                            clearSpeakQueue();
                            Log.d("DetectorActivity", "Fail to recognize text from image");
                        }
                );
    }

    /**
     * Layout kimliğini döndürür.
     */
    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    /**
     * İstenen önizleme çerçeve boyutunu döndürür.
     */
    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    /**
     * Konuşma kuyruğunu temizler.
     */
    private void clearSpeakQueue() {
        if (tts != null) {
            tts.stop();
        }
        objectSpeakQueue.clear();
        textSpeakQueue.clear();
    }

    /**
     * Bir görünüm tıklaması işlendiğinde çağrılır.
     */
    @Override
    public void onClick(View view) {
    }
}
