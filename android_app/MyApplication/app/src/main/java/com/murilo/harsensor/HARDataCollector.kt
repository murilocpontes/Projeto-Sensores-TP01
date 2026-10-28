package com.murilo.harsensor

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

class HARDataCollector(
    private val sensorManager: SensorManager,
    private val serverIp: String,
    private val serverPort: Int = 5005
) : SensorEventListener {

    private val windowData = mutableListOf<FloatArray>()

    private var lastAcc = floatArrayOf(0f, 0f, 0f)
    private var lastGyro = floatArrayOf(0f, 0f, 0f)

    private val WINDOW_SIZE = 128
    private val OVERLAP = 64

    // Controle de tempo real
    private var lastSampleTimeNs = 0L
    private val SAMPLE_PERIOD_NS = 20_000_000L // 20ms

    fun startCollection() {
        val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        // SENSOR_DELAY_FASTEST permite que o hardware nos envie os dados assim que os tiver.
        // Nós filtraremos no Kotlin para aceitar apenas 1 amostra a cada 20ms.
        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_FASTEST)
    }

    fun stopCollection() {
        sensorManager.unregisterListener(this)
        windowData.clear()
        lastSampleTimeNs = 0L
    }

    // Método OO puro: disparado automaticamente pelo hardware do Android
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> event.values.copyInto(lastAcc)
            Sensor.TYPE_GYROSCOPE -> event.values.copyInto(lastGyro)
        }

        val currentTimeNs = event.timestamp

        // Só aceita a leitura se já se passaram 20ms desde a última gravação
        if (lastSampleTimeNs == 0L || (currentTimeNs - lastSampleTimeNs) >= SAMPLE_PERIOD_NS) {

            // Grava a leitura física real
            val frame = floatArrayOf(
                lastAcc[0], lastAcc[1], lastAcc[2],
                lastGyro[0], lastGyro[1], lastGyro[2]
            )

            windowData.add(frame)
            lastSampleTimeNs = currentTimeNs

            // Envia via UDP quando encher (2.56 segundos)
            if (windowData.size == WINDOW_SIZE) {
                val windowToProcess = windowData.toTypedArray()
                transmitDataUDP(windowToProcess)

                // Sliding window
                windowData.subList(0, OVERLAP).clear()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun transmitDataUDP(window: Array<FloatArray>) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val payload = window.joinToString(";") { it.joinToString(",") }
                val bytes = payload.toByteArray()

                val address = InetAddress.getByName(serverIp)
                val socket = DatagramSocket()
                val packet = DatagramPacket(bytes, bytes.size, address, serverPort)

                socket.send(packet)
                socket.close()
            } catch (e: Exception) {
                // Erros de rede são ignorados para não quebrar a coleta
            }
        }
    }
}