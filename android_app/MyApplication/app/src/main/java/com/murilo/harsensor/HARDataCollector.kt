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

/**
 * Gerenciador de captura inercial e transmissão de dados para Reconhecimento de Atividade Humana (HAR).
 *
 * Responsável por amostrar sincronamente os sensores de acelerômetro e giroscópio a 50 Hz,
 * estruturar janelas deslizantes de 128 amostras com 50% de sobreposição e transmiti-las
 * assincronamente via pacotes UDP ao servidor de processamento.
 */
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

    // Controle de amostragem temporal (período de 20 ms equivalente a 50 Hz)
    private var lastSampleTimeNs = 0L
    private val SAMPLE_PERIOD_NS = 20_000_000L

    /**
     * Inicializa a escuta dos sensores de acelerômetro e giroscópio na taxa máxima do hardware.
     */
    fun startCollection() {
        // getDefaultSensor: obtém a referência para o sensor físico padrão do dispositivo
        val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        // registerListener: vincula esta classe para receber eventos contínuos de hardware na menor latência disponível
        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_FASTEST)
    }

    /**
     * Interrompe a leitura dos sensores e reinicializa o buffer da janela temporal.
     */
    fun stopCollection() {
        // unregisterListener: encerra as notificações de eventos de hardware para economizar recursos
        sensorManager.unregisterListener(this)
        windowData.clear()
        lastSampleTimeNs = 0L
    }

    /**
     * Callback acionado a cada atualização de leitura dos sensores registrados.
     * Realiza o controle de taxa (50 Hz), montagem da janela temporal e despacho para rede.
     */
    override fun onSensorChanged(event: SensorEvent) {
        // copyInto: atualiza os vetores de estado com os últimos valores triaxiais lidos pelo sensor
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> event.values.copyInto(lastAcc)
            Sensor.TYPE_GYROSCOPE -> event.values.copyInto(lastGyro)
        }

        val currentTimeNs = event.timestamp

        // Valida se o intervalo mínimo de amostragem (20 ms) foi atingido desde a última captura
        if (lastSampleTimeNs == 0L || (currentTimeNs - lastSampleTimeNs) >= SAMPLE_PERIOD_NS) {

            // Consolida a amostra com 6 eixos [AccX, AccY, AccZ, GyroX, GyroY, GyroZ]
            val frame = floatArrayOf(
                lastAcc[0], lastAcc[1], lastAcc[2],
                lastGyro[0], lastGyro[1], lastGyro[2]
            )

            windowData.add(frame)
            lastSampleTimeNs = currentTimeNs

            // Ao atingir 128 amostras (2,56 s), dispara o envio e aplica o deslocamento da janela
            if (windowData.size == WINDOW_SIZE) {
                val windowToProcess = windowData.toTypedArray()
                // transmitDataUDP: serializa e envia a janela completa ao servidor de forma assíncrona
                transmitDataUDP(windowToProcess)

                // subList.clear: remove as 64 amostras mais antigas para manter sobreposição de 50%
                windowData.subList(0, OVERLAP).clear()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    /**
     * Serializa a janela de dados em formato CSV compacto e envia via protocolo UDP.
     */
    private fun transmitDataUDP(window: Array<FloatArray>) {
        // CoroutineScope.launch(Dispatchers.IO): executa a operação de rede em thread dedicada de I/O
        CoroutineScope(Dispatchers.IO).launch {
            try {
                // joinToString: formata a matriz em string delimitada por ';' (linhas) e ',' (colunas)
                val payload = window.joinToString(";") { it.joinToString(",") }
                val bytes = payload.toByteArray()

                // InetAddress.getByName: resolve o endereço IP de destino do servidor
                val address = InetAddress.getByName(serverIp)
                val socket = DatagramSocket()
                val packet = DatagramPacket(bytes, bytes.size, address, serverPort)

                // socket.send: transmite o datagrama UDP pela interface de rede
                socket.send(packet)
                socket.close()
            } catch (e: Exception) {
                // Suprime falhas pontuais de transmissão para preservar a continuidade da coleta
            }
        }
    }
}