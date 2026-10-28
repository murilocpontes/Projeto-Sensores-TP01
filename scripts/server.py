import socket
import threading
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import rfft
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")

# =========================================================
# 1. VARIÁVEIS GLOBAIS (Ponte entre a Rede e o Gráfico)
# =========================================================
janela_atual_acc = np.zeros((128, 3))
janela_atual_gyro = np.zeros((128, 3))
atividade_atual = "Aguardando dados da rede..."

# =========================================================
# 2. FUNÇÕES DE PROCESSAMENTO (Mantidas intactas)
# =========================================================
def butter_lowpass(data, cutoff, fs=50.0, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def extract_advanced_features(window_128x6):
    acc_total = window_128x6[:, 0:3]
    gyro = window_128x6[:, 3:6]

    acc_grav = butter_lowpass(acc_total, cutoff=0.3)
    acc_body = acc_total - acc_grav

    acc_body_jerk = np.diff(acc_body, axis=0, prepend=acc_body[0:1, :])
    gyro_jerk = np.diff(gyro, axis=0, prepend=gyro[0:1, :])
    
    acc_body_mag = np.linalg.norm(acc_body, axis=1, keepdims=True)
    acc_grav_mag = np.linalg.norm(acc_grav, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)

    acc_body_fft = np.abs(rfft(acc_body, axis=0))
    gyro_fft = np.abs(rfft(gyro, axis=0))

    sinais_tempo = [acc_body, acc_grav, gyro, acc_body_jerk, gyro_jerk, acc_body_mag, acc_grav_mag, gyro_mag]
    sinais_freq = [acc_body_fft, gyro_fft]

    features = []
    for sinal in sinais_tempo + sinais_freq:
        features.extend(np.mean(sinal, axis=0))
        features.extend(np.std(sinal, axis=0))
        features.extend(np.max(sinal, axis=0))
        features.extend(np.min(sinal, axis=0))
        features.extend(np.sum(sinal**2, axis=0) / sinal.shape[0])

    def get_correlations(matriz):
        cor_xy = np.corrcoef(matriz[:,0], matriz[:,1])[0,1]
        cor_xz = np.corrcoef(matriz[:,0], matriz[:,2])[0,1]
        cor_yz = np.corrcoef(matriz[:,1], matriz[:,2])[0,1]
        return [cor_xy, cor_xz, cor_yz]

    features.extend(np.nan_to_num(get_correlations(acc_body)))
    features.extend(np.nan_to_num(get_correlations(gyro)))

    return np.array(features, dtype=np.float32).reshape(1, -1)

# =========================================================
# 3. THREAD DE REDE (Trabalha invisível em background)
# =========================================================
def rede_udp_listener():
    global janela_atual_acc, janela_atual_gyro, atividade_atual
    
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    BUFFER_SIZE = 65535

    try:
        # pipeline_svm = joblib.load("pipeline_svm_servidor.pkl")
        pipeline_svm = joblib.load("pipeline_svm_meu_celular.pkl")
    except Exception as e:
        atividade_atual = f"ERRO: Modelo não encontrado."
        return

    activity_labels = {
        1: "Walking (Caminhando)", 2: "Walking Upstairs (Subindo)", 3: "Walking Downstairs (Descendo)",
        4: "Sitting (Sentado)", 5: "Standing (Em Pé)", 6: "Laying (Deitado)"
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            payload_str = data.decode("utf-8")
            linhas = payload_str.split(";")
            janela = [list(map(float, linha.split(","))) for linha in linhas if linha]
            window_numpy = np.array(janela)

            if window_numpy.shape != (128, 6):
                continue

            # Correção de gravidade (crucial para o dataset UCI)
            window_numpy[:, 0:3] = window_numpy[:, 0:3] / 9.80665

            # Atualiza variáveis globais para o gráfico desenhar
            janela_atual_acc = window_numpy[:, 0:3]
            janela_atual_gyro = window_numpy[:, 3:6]

            # Predição
            vetor_features = extract_advanced_features(window_numpy)
            predicao = pipeline_svm.predict(vetor_features)
            atividade_atual = activity_labels.get(predicao[0], "Desconhecido")

        except Exception as e:
            pass # Continua ouvindo mesmo se um pacote falhar

# =========================================================
# 4. THREAD GRÁFICA (Interface Principal)
# =========================================================
# Inicia a Thread da rede antes de desenhar a tela
thread_rede = threading.Thread(target=rede_udp_listener, daemon=True)
thread_rede.start()

# Configuração da janela do Matplotlib
fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(10, 6))
fig.canvas.manager.set_window_title('Painel IoT - Detecção de Atividade Humana')
plt.subplots_adjust(hspace=0.4)

eixo_x_tempo = np.arange(128)

# Linhas do Acelerômetro
linha_acc_x, = ax_acc.plot(eixo_x_tempo, np.zeros(128), 'r-', label='Acc X')
linha_acc_y, = ax_acc.plot(eixo_x_tempo, np.zeros(128), 'g-', label='Acc Y')
linha_acc_z, = ax_acc.plot(eixo_x_tempo, np.zeros(128), 'b-', label='Acc Z')
ax_acc.set_ylim(-3, 3) # Em 'g' (gravidade)
ax_acc.set_title("Acelerômetro 3D (Escala g)")
ax_acc.legend(loc='upper right')
ax_acc.grid(True)

# Linhas do Giroscópio
linha_gyro_x, = ax_gyro.plot(eixo_x_tempo, np.zeros(128), 'r-', label='Gyro X')
linha_gyro_y, = ax_gyro.plot(eixo_x_tempo, np.zeros(128), 'g-', label='Gyro Y')
linha_gyro_z, = ax_gyro.plot(eixo_x_tempo, np.zeros(128), 'b-', label='Gyro Z')
ax_gyro.set_ylim(-5, 5) # Em rad/s
ax_gyro.set_title("Giroscópio 3D (Escala rad/s)")
ax_gyro.legend(loc='upper right')
ax_gyro.grid(True)

# Função de atualização (roda em loop automático controlando a tela)
def atualizar_grafico(frame):
    # Atualiza as linhas do Acc
    linha_acc_x.set_ydata(janela_atual_acc[:, 0])
    linha_acc_y.set_ydata(janela_atual_acc[:, 1])
    linha_acc_z.set_ydata(janela_atual_acc[:, 2])
    
    # Atualiza as linhas do Gyro
    linha_gyro_x.set_ydata(janela_atual_gyro[:, 0])
    linha_gyro_y.set_ydata(janela_atual_gyro[:, 1])
    linha_gyro_z.set_ydata(janela_atual_gyro[:, 2])
    
    # Atualiza o título mestre com a atividade detectada
    fig.suptitle(f"INFERÊNCIA ATUAL: {atividade_atual}", fontsize=16, fontweight='bold')
    
    return linha_acc_x, linha_acc_y, linha_acc_z, linha_gyro_x, linha_gyro_y, linha_gyro_z

# Configura a animação para rodar a 10 FPS (interval=100ms)
anim = animation.FuncAnimation(fig, atualizar_grafico, interval=100, blit=False, cache_frame_data=False)

print("[*] Iniciando Dashboard. Feche a janela do gráfico para encerrar o servidor.")
plt.show()