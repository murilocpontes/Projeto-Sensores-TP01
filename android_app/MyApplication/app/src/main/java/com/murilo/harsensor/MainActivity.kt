package com.murilo.harsensor

import android.content.Context
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Activity principal da aplicação cliente Android.
 *
 * Gerencia o ciclo de vida da interface gráfica e sincroniza o início e a pausa
 * da coleta de sensores inerciais com o estado de visibilidade da tela.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var dataCollector: HARDataCollector
    private lateinit var statusTextView: TextView

    // Endereço IPv4 do servidor de processamento na rede local
    private val IP_DO_COMPUTADOR = "192.168.15.182"

    /**
     * Inicializa a interface de usuário, os serviços de hardware e a instância do coletor HAR.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // setContentView: infla o layout XML principal na tela da Activity
        setContentView(R.layout.activity_main)

        // findViewById: obtém a referência do componente de texto para exibição de status
        statusTextView = findViewById(R.id.tv_status_atividade)

        // getSystemService: recupera o gerenciador nativo de sensores do sistema Android
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager

        // HARDataCollector: instancia o controlador de captura inercial e envio UDP
        dataCollector = HARDataCollector(sensorManager, IP_DO_COMPUTADOR)

        statusTextView.text = "Status: Pronto para transmitir"
    }

    /**
     * Retoma a captura e transmissão de dados quando a Activity entra em primeiro plano.
     */
    override fun onResume() {
        super.onResume()
        // startCollection: registra os listeners dos sensores e inicia a amostragem a 50 Hz
        dataCollector.startCollection()
        statusTextView.text = "Status: Transmitindo dados (50Hz)\nDestino: $IP_DO_COMPUTADOR"
    }

    /**
     * Interrompe a captura de dados quando a Activity perde o foco, otimizando o consumo de bateria.
     */
    override fun onPause() {
        super.onPause()
        // stopCollection: desregistra os sensores e limpa o buffer de amostras pendentes
        dataCollector.stopCollection()
        statusTextView.text = "Status: Transmissão Pausada"
    }
}