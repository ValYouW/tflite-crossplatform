package com.vyw.tflite

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Example of a call to a native method

    }

    fun btnObjectDetection_click(view: View) {
        val intent = Intent(this, ObjectDetection::class.java)
        startActivity(intent)
    }

    fun btnLinesDetection_click(view: View) {
        val intent = Intent(this, LinesDetection::class.java)
        startActivity(intent)
    }
}
