package com.vyw.tflite

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.vyw.tflite.databinding.ActivityLinesDetectionBinding
import kotlinx.android.synthetic.main.activity_lines_detection.*
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class LinesDetection : AppCompatActivity(), ImageAnalysis.Analyzer {

	companion object {
		private const val TAG = "LinesDetection"
		private const val REQUEST_CODE_PERMISSIONS = 10
		private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

		init {
			System.loadLibrary("native-lib")
		}
	}

	private lateinit var cameraExecutor: ExecutorService
	private var imageAnalyzer: ImageAnalysis? = null
	private var detectorAddr = 0L
	private lateinit var rgbaFrame: ByteArray
	private val labelsMap = arrayListOf<String>()
	private val _paint = Paint()
	private lateinit var binding: ActivityLinesDetectionBinding

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		binding = ActivityLinesDetectionBinding.inflate(layoutInflater)
		setContentView(binding.root)

		// Request camera permissions
		if (allPermissionsGranted()) {
			startCamera()
		} else {
			ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
		}

		cameraExecutor = Executors.newSingleThreadExecutor()

		// init the paint for drawing the detections
		_paint.color = Color.RED
		_paint.style = Paint.Style.STROKE
		_paint.strokeWidth = 3f
		_paint.textSize = 50f
		_paint.textAlign = Paint.Align.LEFT

		// Set the detections drawings surface transparent
		binding.surfaceView.setZOrderOnTop(true)
		binding.surfaceView.holder.setFormat(PixelFormat.TRANSPARENT)
		binding.lblStatus.text = ""
	}

	private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
		ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
	}

	override fun onRequestPermissionsResult(
		requestCode: Int,
		permissions: Array<String>,
		grantResults: IntArray
	) {
		if (requestCode == REQUEST_CODE_PERMISSIONS) {
			if (allPermissionsGranted()) {
				startCamera()
			} else {
				Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
				finish()
			}
		}
	}

	private fun startCamera() {
		val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

		cameraProviderFuture.addListener(Runnable {
			// Used to bind the lifecycle of cameras to the lifecycle owner
			val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

			val rotation = binding.viewFinder.display.rotation

			// Preview
			val preview = Preview.Builder()
				.setTargetAspectRatio(AspectRatio.RATIO_4_3)
				.setTargetRotation(rotation)
				.build()
				.also {
					it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
				}

			// ImageAnalysis
			imageAnalyzer = ImageAnalysis.Builder()
				.setBackpressureStrategy(STRATEGY_KEEP_ONLY_LATEST)
				.setTargetResolution(Size(768, 1024))
				.setTargetRotation(rotation)
				.setOutputImageRotationEnabled(true)
				.setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
				.build()
				// The analyzer can then be assigned to the instance
				.also {
					it.setAnalyzer(cameraExecutor, this)
				}

			try {
				// Unbind use cases before rebinding
				cameraProvider.unbindAll()

				// Bind use cases to camera
				cameraProvider.bindToLifecycle(
					this,
					CameraSelector.DEFAULT_BACK_CAMERA,
					preview,
					imageAnalyzer
				)
			} catch (exc: Exception) {
				Log.e(TAG, "Use case binding failed", exc)
			}

		}, ContextCompat.getMainExecutor(this))
	}

	override fun analyze(image: ImageProxy) {
		if (image.planes.isEmpty()) {return}
		if (detectorAddr == 0L) {
			detectorAddr = initDetector(this.assets)
		}

		val buffer = image.planes[0].buffer
		val size = buffer.capacity()
		if (!::rgbaFrame.isInitialized) {
			rgbaFrame = ByteArray(size)
		}

		buffer.position(0)
		buffer.get(rgbaFrame, 0, size)

		val start = System.currentTimeMillis()
		val res = detect(detectorAddr, rgbaFrame, image.width, image.height)
		val span = System.currentTimeMillis() - start

		val canvas = binding.surfaceView.holder.lockCanvas()
		if (canvas != null) {
			canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY)

			// Draw the detections, in our case there are only 3
			for (i in 0 until res[0]) {
				this.drawDetection(canvas, image.width, image.height, res, i)
			}

			binding.surfaceView.holder.unlockCanvasAndPost(canvas)
		}

		runOnUiThread {
			binding.lblStatus.text = "$span ms"
		}

		image.close()
	}

	private fun drawDetection(
		canvas: Canvas,
		frameWidth: Int,
		frameHeight: Int,
		detectionsArr: IntArray,
		detectionIdx: Int
	) {

		val pos = detectionIdx * 4 + 1
		var xmin = detectionsArr[pos + 0].toFloat()
		var ymin = detectionsArr[pos + 1].toFloat()
		var xmax = detectionsArr[pos + 2].toFloat()
		var ymax = detectionsArr[pos + 3].toFloat()

		// detection coords are in frame coord system, convert to screen coords
		val scaleX = viewFinder.width.toFloat() / frameWidth
		val scaleY = viewFinder.height.toFloat() / frameHeight

		// The camera view offset on screen
		val xoff = 0 // viewFinder.left.toFloat()
		val yoff = 0 // viewFinder.top.toFloat()

		xmin = xoff + xmin * scaleX
		xmax = xoff + xmax * scaleX
		ymin = yoff + ymin * scaleY
		ymax = yoff + ymax * scaleY

		// Draw the line
		canvas.drawLine(xmin, ymin, xmax, ymax, _paint)
	}

	private external fun initDetector(assetManager: AssetManager?): Long
	private external fun destroyDetector(ptr: Long)
	private external fun detect(ptr: Long, srcAddr: ByteArray, width: Int, height: Int): IntArray
}
