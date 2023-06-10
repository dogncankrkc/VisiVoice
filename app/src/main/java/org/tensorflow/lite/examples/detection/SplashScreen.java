package org.tensorflow.lite.examples.detection;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.speech.tts.TextToSpeech;

import java.util.Locale;

public class SplashScreen extends Activity {
    private TextToSpeech tts;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        // Zaman aşımı süresi
        int SPLASH_TIME_OUT = 5000;

        // Metni seslendirmek için TextToSpeech örneği oluşturulur
        tts = new TextToSpeech(getApplicationContext(), status -> {
            if (status != TextToSpeech.ERROR) {
                tts.setLanguage(new Locale("tr", "TR")); // Türkçe dilini ayarlar
                tts.speak("Uygulama başlatılıyor", TextToSpeech.QUEUE_FLUSH, null); // Mesajı Türkçe olarak seslendirir
            }
        });

        // Belirtilen süre sonunda yeni bir aktiviteye geçiş yapılır
        new Handler().postDelayed(() -> {
            Intent intent = new Intent(SplashScreen.this, DetectorActivity.class);
            startActivity(intent);
            finish();
        }, SPLASH_TIME_OUT);
    }

    @Override
    protected void onDestroy() {
        // TextToSpeech örneği kapatılır
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        super.onDestroy();
    }
}
