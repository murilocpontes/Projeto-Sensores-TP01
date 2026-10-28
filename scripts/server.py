"""
Servidor de inferência e visualização em tempo real para Reconhecimento de Atividade Humana (HAR).

Recebe janelas de sinais inerciais (128x6) via UDP em uma thread secundária,
extrai atributos nos domínios do tempo e da frequência, classifica a atividade com
um pipeline SVM pré-treinado e atualiza um painel gráfico interativo a 10 FPS.
"""

import socket
import threading
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import rfft
from scipy.signal import butter, filtfilt

# warnings.filterwarnings: suprime alertas de tempo de execução para manter a saída limpa
warnings.filterwarnings("ignore")

# Buffers compartilhados entre a thread de recepção de rede e a thread de renderização gráfica
janela_atual_acc = np.zeros((128, 3))
janela_atual_gyro = np.zeros((128, 3))
atividade_atual = "Aguardando dados da rede..."

def butter_lowpass(data, cutoff, fs=50.0, order=3):
    """
    Aplica um filtro digital Butterworth passa-baixas com fase zero.

    Utilizado para isolar a componente gravitacional de baixa frequência do sinal de aceleração.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # butter: calcula os coeficientes dos polinômios numerador (b) e denominador (a) do filtro IIR
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt: aplica o filtro de forma bidirecional ao longo do eixo temporal (axis=0) sem distorção de fase
    return filtfilt(b, a, data, axis=0)

def extract_advanced_features(window_128x6):
    """
    Extrai o vetor de 126 atributos estatísticos e espectrais a partir de uma janela inercial 128x6.

    Realiza a decomposição gravitacional/corporal, cálculo de derivada temporal (Jerk),
    norma euclidiana 3D, espectro de frequência via FFT e correlações cruzadas entre eixos.
    """
    acc_total = window_128x6[:, 0:3]
    gyro = window_128x6[:, 3:6]

    # Isola a gravidade (< 0.3 Hz) e obtém a aceleração dinâmica do corpo por subtração
    acc_grav = butter_lowpass(acc_total, cutoff=0.3)
    acc_body = acc_total - acc_grav

    # np.diff: calcula a derivada discreta de primeira ordem (Jerk) mantendo a dimensão original
    acc_body_jerk = np.diff(acc_body, axis=0, prepend=acc_body[0:1, :])
    gyro_jerk = np.diff(gyro, axis=0, prepend=gyro[0:1, :])
    
    # np.linalg.norm: computa a magnitude espacial (norma euclidiana L2) para cada instante de tempo
    acc_body_mag = np.linalg.norm(acc_body, axis=1, keepdims=True)
    acc_grav_mag = np.linalg.norm(acc_grav, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)

    # rfft: calcula a Transformada Rápida de Fourier para sinais reais e extrai a magnitude espectral
    acc_body_fft = np.abs(rfft(acc_body, axis=0))
    gyro_fft = np.abs(rfft(gyro, axis=0))

    sinais_tempo = [acc_body, acc_grav, gyro, acc_body_jerk, gyro_jerk, acc_body_mag, acc_grav_mag, gyro_mag]
    sinais_freq = [acc_body_fft, gyro_fft]

    features = []
    # Agrega média, desvio padrão, máximo, mínimo e potência média (energia) de cada sinal
    for sinal in sinais_tempo + sinais_freq:
        features.extend(np.mean(sinal, axis=0))
        features.extend(np.std(sinal, axis=0))
        features.extend(np.max(sinal, axis=0))
        features.extend(np.min(sinal, axis=0))
        features.extend(np.sum(sinal**2, axis=0) / sinal.shape[0])

    def get_correlations(matriz):
        """Calcula os coeficientes de correlação de Pearson entre os pares de eixos (XY, XZ, YZ)."""
        # np.corrcoef: gera a matriz de correlação entre dois vetores unidimensionais
        cor_xy = np.corrcoef(matriz[:,0], matriz[:,1])[0,1]
        cor_xz = np.corrcoef(matriz[:,0], matriz[:,2])[0,1]
        cor_yz = np.corrcoef(matriz[:,1], matriz[:,2])[0,1]
        return [cor_xy, cor_xz, cor_yz]

    # np.nan_to_num: substitui eventuais valores NaN (decorrentes de variância nula) por zero
    features.extend(np.nan_to_num(get_correlations(acc_body)))
    features.extend(np.nan_to_num(get_correlations(gyro)))

    # reshape(1, -1): estrutura o vetor resultante em matriz 2D (1 amostra x N features) para inferência no scikit-learn
    return np.array(features, dtype=np.float32).reshape(1, -1)

def rede_udp_listener():
    """
    Serviço de escuta UDP executado em background.

    Recebe pacotes do dispositivo móvel, reconstrói a matriz 128x6, converte a escala de aceleração
    para múltiplos da gravidade (g) e atualiza o estado de inferência da atividade.
    """
    global janela_atual_acc, janela_atual_gyro, atividade_atual
    
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    BUFFER_SIZE = 65535

    try:
        # joblib.load: desserializa o pipeline contendo o normalizador (StandardScaler) e o classificador SVM
        pipeline_svm = joblib.load("pipeline_svm_meu_celular.pkl")
    except Exception as e:
        atividade_atual = f"ERRO: Modelo não encontrado."
        return

    activity_labels = {
        1: "Walking (Caminhando)", 2: "Walking Upstairs (Subindo)", 3: "Walking Downstairs (Descendo)",
        4: "Sitting (Sentado)", 5: "Standing (Em Pé)", 6: "Laying (Deitado)"
    }

    # socket.socket / bind: inicializa o socket UDP e vincula à porta de escuta local
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    while True:
        try:
            # sock.recvfrom: bloqueia a thread até o recebimento de um datagrama UDP
            data, addr = sock.recvfrom(BUFFER_SIZE)
            payload_str = data.decode("utf-8")
            linhas = payload_str.split(";")
            janela = [list(map(float, linha.split(","))) for linha in linhas if linha]
            window_numpy = np.array(janela)

            # Descarta pacotes cuja dimensão não corresponda a uma janela íntegra (128x6)
            if window_numpy.shape != (128, 6):
                continue

            # Converte a aceleração de m/s² (padrão Android) para unidades de gravidade (g)
            window_numpy[:, 0:3] = window_numpy[:, 0:3] / 9.80665

            # Atualiza os buffers globais consumidos pelo renderizador gráfico
            janela_atual_acc = window_numpy[:, 0:3]
            janela_atual_gyro = window_numpy[:, 3:6]

            # Extrai o vetor de atributos e executa a predição da classe de movimento
            vetor_features = extract_advanced_features(window_numpy)
            # pipeline_svm.predict: aplica normalização seguida de classificação pelo modelo SVM
            predicao = pipeline_svm.predict(vetor_features)
            atividade_atual = activity_labels.get(predicao[0], "Desconhecido")

        except Exception as e:
            pass

# threading.Thread: instancia e inicia a thread de rede em modo daemon para encerramento automático junto à janela
thread_rede = threading.Thread(target=rede_udp_listener, daemon=True)
thread_rede.start()

# plt.subplots: cria a figura principal e os dois eixos cartesianos para acelerômetro e giroscópio
fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(10, 6))
fig.canvas.manager.set_window_title('Painel IoT - Detecção de Atividade Humana')
plt.subplots_adjust(hspace=0.4)

eixo_x_tempo = np.arange(128)

# Configuração das curvas e limites do gráfico de acelerômetro (escala em g)
linha_acc_x, = ax_acc.plot(eixo_x_tempo, np.zeros(128), 'r-', label='Acc X')
linha_acc_y, = ax_acc.plot(eixo_x_tempo, np.zeros(128), 'g-', label='Acc Y')
linha_acc_z, = ax_acc.plot(eixo_x_tempo, np.zeros(128), 'b-', label='Acc Z')
ax_acc.set_ylim(-3, 3)
ax_acc.set_title("Acelerômetro 3D (Escala g)")
ax_acc.legend(loc='upper right')
ax_acc.grid(True)

# Configuração das curvas e limites do gráfico de giroscópio (escala em rad/s)
linha_gyro_x, = ax_gyro.plot(eixo_x_tempo, np.zeros(128), 'r-', label='Gyro X')
linha_gyro_y, = ax_gyro.plot(eixo_x_tempo, np.zeros(128), 'g-', label='Gyro Y')
linha_gyro_z, = ax_gyro.plot(eixo_x_tempo, np.zeros(128), 'b-', label='Gyro Z')
ax_gyro.set_ylim(-5, 5)
ax_gyro.set_title("Giroscópio 3D (Escala rad/s)")
ax_gyro.legend(loc='upper right')
ax_gyro.grid(True)

def atualizar_grafico(frame):
    """
    Callback de animação invocado periodicamente pelo Matplotlib.

    Atualiza os dados das séries temporais nos eixos e exibe o rótulo da atividade inferida.
    """
    # set_ydata: substitui os valores do eixo vertical de cada curva com os dados mais recentes da janela
    linha_acc_x.set_ydata(janela_atual_acc[:, 0])
    linha_acc_y.set_ydata(janela_atual_acc[:, 1])
    linha_acc_z.set_ydata(janela_atual_acc[:, 2])
    
    linha_gyro_x.set_ydata(janela_atual_gyro[:, 0])
    linha_gyro_y.set_ydata(janela_atual_gyro[:, 1])
    linha_gyro_z.set_ydata(janela_atual_gyro[:, 2])
    
    # fig.suptitle: atualiza o cabeçalho superior do painel com a última atividade classificada
    fig.suptitle(f"INFERÊNCIA ATUAL: {atividade_atual}", fontsize=16, fontweight='bold')
    
    return linha_acc_x, linha_acc_y, linha_acc_z, linha_gyro_x, linha_gyro_y, linha_gyro_z

# animation.FuncAnimation: agenda a execução contínua de atualizar_grafico a cada 100 ms (10 FPS)
anim = animation.FuncAnimation(fig, atualizar_grafico, interval=100, blit=False, cache_frame_data=False)

print("[*] Iniciando Dashboard. Feche a janela do gráfico para encerrar o servidor.")
# plt.show: inicia o loop de eventos da interface gráfica
plt.show()