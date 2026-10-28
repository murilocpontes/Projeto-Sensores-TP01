"""
Utilitário de aquisição e rotulagem de dados inerciais via rede UDP.

Escuta pacotes enviados pelo dispositivo Android, extrai os 126 atributos de cada
janela válida (128x6) e armazena o conjunto de dados rotulado em arquivo CSV para
treinamento personalizado do modelo.
"""

import socket
import numpy as np
import pandas as pd
import warnings
from scipy.fft import rfft
from scipy.signal import butter, filtfilt

# warnings.filterwarnings: suprime alertas de precisão ou divisão durante a extração
warnings.filterwarnings("ignore")

def butter_lowpass(data, cutoff, fs=50.0, order=3):
    """
    Aplica um filtro Butterworth passa-baixas para separação da componente gravitacional.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # butter: obtém os coeficientes do filtro digital na frequência de corte normalizada
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt: executa a filtragem sem deslocamento de fase ao longo do eixo temporal (axis=0)
    return filtfilt(b, a, data, axis=0)

def extract_advanced_features(window_128x6):
    """
    Converte uma janela temporal 128x6 em um vetor de 126 atributos estatísticos e espectrais.
    """
    acc_total = window_128x6[:, 0:3]
    gyro = window_128x6[:, 3:6]

    # Separação entre aceleração gravitacional e aceleração corporal
    acc_grav = butter_lowpass(acc_total, cutoff=0.3)
    acc_body = acc_total - acc_grav

    # np.diff: calcula a derivada temporal discreta (Jerk) dos sinais lineares e angulares
    acc_body_jerk = np.diff(acc_body, axis=0, prepend=acc_body[0:1, :])
    gyro_jerk = np.diff(gyro, axis=0, prepend=gyro[0:1, :])
    
    # np.linalg.norm: computa a norma euclidiana 3D para obter magnitudes invariantes à orientação
    acc_body_mag = np.linalg.norm(acc_body, axis=1, keepdims=True)
    acc_grav_mag = np.linalg.norm(acc_grav, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)

    # rfft: transforma os sinais temporais para o domínio da frequência
    acc_body_fft = np.abs(rfft(acc_body, axis=0))
    gyro_fft = np.abs(rfft(gyro, axis=0))

    sinais_tempo = [acc_body, acc_grav, gyro, acc_body_jerk, gyro_jerk, acc_body_mag, acc_grav_mag, gyro_mag]
    sinais_freq = [acc_body_fft, gyro_fft]

    features = []
    # Calcula média, desvio padrão, máximo, mínimo e energia média para cada canal
    for sinal in sinais_tempo + sinais_freq:
        features.extend(np.mean(sinal, axis=0))
        features.extend(np.std(sinal, axis=0))
        features.extend(np.max(sinal, axis=0))
        features.extend(np.min(sinal, axis=0))
        features.extend(np.sum(sinal**2, axis=0) / sinal.shape[0])

    def get_correlations(matriz):
        """Retorna as correlações cruzadas entre os pares de eixos X-Y, X-Z e Y-Z."""
        # np.corrcoef: calcula a matriz de correlação de Pearson entre dois eixos
        cor_xy = np.corrcoef(matriz[:,0], matriz[:,1])[0,1]
        cor_xz = np.corrcoef(matriz[:,0], matriz[:,2])[0,1]
        cor_yz = np.corrcoef(matriz[:,1], matriz[:,2])[0,1]
        return [cor_xy, cor_xz, cor_yz]

    # np.nan_to_num: garante estabilidade numérica caso algum eixo apresente variância zero
    features.extend(np.nan_to_num(get_correlations(acc_body)))
    features.extend(np.nan_to_num(get_correlations(gyro)))

    return np.array(features, dtype=np.float32).reshape(1, -1)

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
BUFFER_SIZE = 65535

# Identificador numérico da classe a ser coletada (1: Walking, 2: Up, 3: Down, 4: Sitting, 5: Standing, 6: Laying)
CLASSE_ATUAL = 2
NOME_ARQUIVO = f"meus_dados_classe_{CLASSE_ATUAL}.csv"

# socket.socket / bind: configura o socket UDP para escutar pacotes na porta especificada
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"[*] Gravando dados para a Classe {CLASSE_ATUAL}...")

features_coletadas = []

try:
    while True:
        # sock.recvfrom: recebe o pacote de dados enviado pelo aplicativo móvel
        data, addr = sock.recvfrom(BUFFER_SIZE)
        payload_str = data.decode("utf-8")
        linhas = payload_str.split(";")
        janela = [list(map(float, linha.split(","))) for linha in linhas if linha]
        window_numpy = np.array(janela)

        if window_numpy.shape == (128, 6):
            # Normaliza a aceleração de m/s² para unidades de gravidade (g)
            window_numpy[:, 0:3] = window_numpy[:, 0:3] / 9.80665
            
            # extract_advanced_features: computa o vetor de 126 atributos da janela capturada
            vetor = extract_advanced_features(window_numpy)[0]
            features_coletadas.append(vetor)
            print(f"Janelas gravadas: {len(features_coletadas)} (Pressione Ctrl+C para parar)")

except KeyboardInterrupt:
    print("\n[*] Salvando arquivo CSV...")
    # pd.DataFrame / to_csv: estrutura as amostras coletadas, anexa o rótulo da classe e exporta para disco
    df = pd.DataFrame(features_coletadas)
    df['Label'] = CLASSE_ATUAL
    df.to_csv(NOME_ARQUIVO, index=False)
    print(f"[*] {len(features_coletadas)} janelas salvas em {NOME_ARQUIVO}!")