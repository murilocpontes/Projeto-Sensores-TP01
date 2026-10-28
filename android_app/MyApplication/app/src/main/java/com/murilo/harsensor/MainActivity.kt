package com.murilo.harsensor

import android.content.Context
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var dataCollector: HARDataCollector
    private lateinit var statusTextView: TextView

    // Coloque aqui o IP IPv4 do seu computador na rede Wi-Fi (ex: 192.168.0.10)
    private val IP_DO_COMPUTADOR = "192.168.15.182"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusTextView = findViewById(R.id.tv_status_atividade)

        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        // Inicializa o coletor passando o SensorManager e o IP da rede
        dataCollector = HARDataCollector(sensorManager, IP_DO_COMPUTADOR)

        statusTextView.text = "Status: Pronto para transmitir"
    }

    override fun onResume() {
        super.onResume()
        dataCollector.startCollection()
        statusTextView.text = "Status: Transmitindo dados (50Hz)\nDestino: $IP_DO_COMPUTADOR"
    }

    override fun onPause() {
        super.onPause()
        dataCollector.stopCollection()
        statusTextView.text = "Status: Transmissão Pausada"
    }
}