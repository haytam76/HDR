import time
import numpy as np
import ctypes
import cv2
from arena_api.system import system
from numba import njit, prange
import subprocess
import os
import threading
import queue
import signal

TAB1 = "  "

def create_device_automatically():
    """
    Se connecte automatiquement au premier appareil trouvé.
    S'il n'y en a pas, patiente jusqu'à 'tries_max' tentatives.
    """
    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    while tries < tries_max:
        devices = system.create_device()
        if not devices:
            print(f'{TAB1}Essai {tries+1} sur {tries_max} : '
                  f'attente de {sleep_time_secs} secondes pour qu’un appareil soit connecté !')
            time.sleep(sleep_time_secs)
            tries += 1
        else:
            # On prend simplement le premier appareil
            print(f"\n{TAB1}Appareil détecté : {len(devices)} connecté(s). "
                  "On utilise le premier disponible.\n")
            return devices[0]
    # Si on n'a pas trouvé d'appareil
    raise Exception(f"{TAB1}Aucun appareil trouvé après {tries_max} tentatives.")

def setup(device):
    """
    Configure les dimensions et le format BayerRG24.
    """
    nodemap = device.nodemap
    nodes = nodemap.get_node(['Width', 'Height', 'OffsetX', 'OffsetY', 'PixelFormat'])
    nodes['OffsetX'].value = 0
    nodes['OffsetY'].value = 0
    nodes['Width'].value = 2880
    nodes['Height'].value = 1860
    nodes['PixelFormat'].value = 'BayerRG24'  # Configuration pour BayerRG24
    width = nodes['Width'].value
    height = nodes['Height'].value
    expected_size = width * height * 3  # Chaque pixel utilise 3 octets en BayerRG24
    return width, height, expected_size


def construct_bayer_image(buffer, padded_buffer, buffer_array, height, width):
    """
    Construit une image Bayer brute à partir des données du buffer
    et retourne une représentation en uint32.
    """
    buffer_data = (ctypes.c_uint8 * buffer_array.size).from_address(
        ctypes.addressof(buffer.pdata.contents))
    buffer_array[:] = np.ctypeslib.as_array(buffer_data).reshape((height, width, 3))
    padded_buffer[:, :, :3] = buffer_array
    bayer_image_24bit = padded_buffer.view(dtype=np.uint32).reshape((height, width))
    return bayer_image_24bit


@njit
def apply_gamma_lut_to_image(b_channel, g_channel, r_channel, gamma_lut, rgb_image_8bit, height, width):
    """
    Applique une LUT de correction gamma à chaque canal (B, G, R)
    et retourne une image RGB combinée (height x width x 3).
    """
    for i in prange(height):
        for j in prange(width):
            rgb_image_8bit[i, j, 0] = gamma_lut[b_channel[i, j]]  # Bleu
            rgb_image_8bit[i, j, 1] = gamma_lut[g_channel[i, j]]  # Vert
            rgb_image_8bit[i, j, 2] = gamma_lut[r_channel[i, j]]  # Rouge

    return rgb_image_8bit


def demosaic_bayer_image(bayer_image, r_channel, g_channel, b_channel, kernel_r_b, kernel_g):
    """
    Effectue un dématriçage avancé pour Bayer avec chaque canal en 24 bits.
    """
    r_channel.fill(0)
    g_channel.fill(0)
    b_channel.fill(0)

    # Disposition (RGGB) pour BayerRG
    r_channel[0::2, 0::2] = bayer_image[0::2, 0::2]
    g_channel[0::2, 1::2] = bayer_image[0::2, 1::2]
    g_channel[1::2, 0::2] = bayer_image[1::2, 0::2]
    b_channel[1::2, 1::2] = bayer_image[1::2, 1::2]

    b_channel[:] = cv2.filter2D(b_channel.astype(np.float32), -1, kernel_r_b, borderType=cv2.BORDER_REFLECT)
    g_channel[:] = cv2.filter2D(g_channel.astype(np.float32), -1, kernel_g, borderType=cv2.BORDER_REFLECT)
    r_channel[:] = cv2.filter2D(r_channel.astype(np.float32), -1, kernel_r_b, borderType=cv2.BORDER_REFLECT)

    return b_channel, g_channel, r_channel


def apply_clahe_to_lab_channels(rgb_image):
    """
    Applique CLAHE au canal de luminosité L dans l'espace de couleur LAB.
    """
    # Convertir l'image RGB en espace de couleur LAB
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

    # Extraire le canal L (luminosité), A et B
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Appliquer CLAHE au canal L
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(5, 5))
    l_eq = clahe.apply(l_channel)

    # Recombiner les canaux L égalisés, A et B
    lab_eq_image = cv2.merge((l_eq, a_channel, b_channel))

    # Reconvertir l'image LAB en RGB
    rgb_image_eq = cv2.cvtColor(lab_eq_image, cv2.COLOR_LAB2RGB)

    return rgb_image_eq


# Répertoire de stockage et paramètres
TEMP_DIR = "temp_frames"      # Dossier temporaire pour stocker les images
OUTPUT_VIDEO = "output_ffv1.mkv"  # Nom du fichier vidéo de sortie
FPS = 6                           # Définir les FPS souhaités pour la vidéo


def ensure_temp_dir():
    """
    Crée un dossier temporaire pour stocker les images.
    """
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)


def clean_temp_dir():
    """
    Supprime le dossier temporaire après encodage.
    """
    if os.path.isdir(TEMP_DIR):
        for file in os.listdir(TEMP_DIR):
            os.remove(os.path.join(TEMP_DIR, file))
        os.rmdir(TEMP_DIR)


# --- Queue & Threading Logic for Saving ---

image_queue = queue.Queue()
stop_saving = threading.Event()         # Pour arrêter la sauvegarde des images
stop_saving_event = threading.Event()   # Pour signaler au thread FFmpeg de s’arrêter

def save_images_from_queue():
    """
    Sauvegarde les images depuis la file d'attente dans un thread séparé.
    """
    while not stop_saving.is_set() or not image_queue.empty():
        try:
            frame, frame_count = image_queue.get(timeout=1)
            filename = os.path.join(TEMP_DIR, f"frame_{frame_count:04d}.png")
            cv2.imwrite(filename, frame)
            image_queue.task_done()
        except queue.Empty:
            pass  # File d'attente vide, on attend de nouvelles images


def encode_video_with_ffmpeg_parallel(output_file, fps, stop_event):
    """
    Lance FFmpeg en parallèle pour encoder tous les fichiers frame_XXXX.png
    au fur et à mesure.
    """
    input_pattern = os.path.join(TEMP_DIR, "frame_%04d.png")
    command = [
        "ffmpeg",
        "-y",                   # Écrase la vidéo si elle existe déjà
        "-hide_banner",
        "-loglevel", "warning", # Affiche seulement les avertissements
        "-start_number", "0",   # Commence à chercher les images depuis frame_0000.png
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "ffv1",
        output_file
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Boucle d’attente : tant que l’événement d’arrêt n’est pas levé,
    # on "dort" un peu.
    while not stop_event.is_set():
        time.sleep(0.2)

    # Une fois que 'stop_event' est activé (ESC pressé), on envoie un signal
    # à ffmpeg pour qu’il termine proprement (SIGINT).
    print(f"{TAB1}Arrêt de FFmpeg en cours...")
    try:
        process.send_signal(signal.SIGINT)
    except Exception:
        pass

    process.wait()
    print(f"{TAB1}Vidéo encodée avec succès : {output_file}")


def mouse_callback(event, x, y, flags, param):
    """
    Gère le clic souris pour activer/désactiver CLAHE ou Gamma.
    """
    use_clahe, use_gamma, gamma = param
    # Bouton CLAHE
    if 10 <= x <= 110 and 10 <= y <= 50:
        if event == cv2.EVENT_LBUTTONDOWN:
            use_clahe[0] = not use_clahe[0]
            print(f"{TAB1}CLAHE {'activé' if use_clahe[0] else 'désactivé'}")

    # Bouton Gamma
    if 120 <= x <= 220 and 10 <= y <= 50:
        if event == cv2.EVENT_LBUTTONDOWN:
            use_gamma[0] = not use_gamma[0]
            print(f"{TAB1}{'Sans correction gamma' if use_gamma[0] else f'Avec correction gamma (γ = {gamma})'}")


def stream_and_save_images_parallel():
    """
    1) Démarre la caméra et reçoit les buffers
    2) Applique corrections (Gamma, CLAHE)
    3) Affiche en temps réel
    4) Redimensionne
    5) Enfile dans la queue pour sauvegarde PNG (thread séparé)
    6) L'encodage se fait en parallèle via un autre thread FFmpeg
    """
    # --- On se connecte automatiquement au premier appareil ---
    device = create_device_automatically()

    width, height, _ = setup(device)
    ensure_temp_dir()  # Préparer le dossier temporaire

    # Préparer la LUT gamma par défaut (γ=4)
    max_value = 16777215
    gamma = 4  # Valeur par défaut
    gamma_lut_default = np.array([
        (i / max_value) ** (1 / gamma) * 255 for i in range(max_value + 1)
    ], dtype=np.uint8)

    buffer_array = np.empty((height, width, 3), dtype=np.uint8)
    padded_buffer = np.zeros((height, width, 4), dtype=np.uint8)

    r_channel = np.zeros((height, width), dtype=np.uint32)
    g_channel = np.zeros((height, width), dtype=np.uint32)
    b_channel = np.zeros((height, width), dtype=np.uint32)
    rgb_image_8bit = np.empty((height, width, 3), dtype=np.uint8)

    kernel_r_b = np.array([[1/4, 1/2, 1/4],
                           [1/2, 1,   1/2],
                           [1/4, 1/2, 1/4]], dtype=np.float32)
    kernel_g   = np.array([[0,   1/4, 0  ],
                           [1/4, 1,   1/4],
                           [0,   1/4, 0  ]], dtype=np.float32)

    frame_count = 0

    # Flags CLAHE et gamma
    use_clahe = [False]   # CLAHE désactivé par défaut
    use_gamma = [False]   # Gamma désactivé par défaut (valeur = 4 si activé)

    # Définir la nouvelle taille pour l'affichage
    resized_width = 1000
    resized_height = 800

    # Création fenêtre pour affichage
    cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed Image', resized_width, resized_height)
    cv2.setMouseCallback("Processed Image", mouse_callback, (use_clahe, use_gamma, gamma))

    with device.start_stream():
        try:
            packet_frame_count = 0
            packet_start_time = time.time()

            while True:
                buffer = device.get_buffer()
                bayer_image = construct_bayer_image(
                    buffer, padded_buffer, buffer_array, height, width
                )

                b_channel, g_channel, r_channel = demosaic_bayer_image(
                    bayer_image, r_channel, g_channel, b_channel,
                    kernel_r_b, kernel_g
                )

                # Appliquer la correction gamma si nécessaire
                if use_gamma[0]:
                    gamma_lut = gamma_lut_default
                    rgb_image_8bit = apply_gamma_lut_to_image(
                        b_channel, g_channel, r_channel,
                        gamma_lut, rgb_image_8bit, height, width
                    )
                else:
                    # Pas de correction gamma => extraction directe des bits
                    rgb_image_8bit[:, :, 0] = (r_channel >> 16) & 0xFF  # Rouge
                    rgb_image_8bit[:, :, 1] = (g_channel >> 16) & 0xFF  # Vert
                    rgb_image_8bit[:, :, 2] = (b_channel >> 16) & 0xFF  # Bleu

                # Appliquer CLAHE si activé
                if use_clahe[0]:
                    rgb_image_clahe = apply_clahe_to_lab_channels(rgb_image_8bit)
                else:
                    rgb_image_clahe = rgb_image_8bit

                # Redimensionner l'image pour l'affichage
                resized_image = cv2.resize(rgb_image_clahe, (resized_width, resized_height))

                # Dessiner les boutons "CLAHE" et "Gamma"
                button_img = resized_image.copy()
                cv2.rectangle(button_img, (10, 10), (110, 50),
                              (0, 255, 0) if use_clahe[0] else (0, 0, 255), -1)
                cv2.putText(button_img, 'CLAHE', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.rectangle(button_img, (120, 10), (220, 50),
                              (0, 255, 0) if use_gamma[0] else (0, 0, 255), -1)
                cv2.putText(button_img, 'Gamma', (130, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Affichage
                cv2.imshow('Processed Image', button_img)
                if cv2.waitKey(1) & 0xFF == 27:  # Échap pour quitter
                    break

                # Mettre l'image dans la file d'attente pour la sauvegarde
                image_queue.put((resized_image.copy(), frame_count))

                frame_count += 1
                packet_frame_count += 1

                # Calcul du FPS par paquet de 50 images (optionnel)
                if packet_frame_count == 50:
                    packet_end_time = time.time()
                    packet_duration = packet_end_time - packet_start_time
                    packet_fps = 50 / packet_duration
                    print(f"{TAB1}Paquet de 50 images - FPS : {packet_fps:.2f}")
                    packet_start_time = time.time()
                    packet_frame_count = 0

                device.requeue_buffer(buffer)

        except KeyboardInterrupt:
            print(f"{TAB1}Streaming interrompu par l'utilisateur.")

    device.stop_stream()
    cv2.destroyAllWindows()
    print(f"{TAB1}Fin de la capture. Envoi du signal d’arrêt à FFmpeg...")
    # On signale à ffmpeg qu'on veut s'arrêter
    stop_saving_event.set()


def main():
    # 1) Lancer le thread de sauvegarde
    save_thread = threading.Thread(target=save_images_from_queue)
    save_thread.start()

    # 2) Lancer le thread FFmpeg en parallèle
    encoding_thread = threading.Thread(
        target=encode_video_with_ffmpeg_parallel,
        args=(OUTPUT_VIDEO, FPS, stop_saving_event)
    )
    encoding_thread.start()

    try:
        # 3) Lancer le streaming (capture) + envoi des images à la file
        stream_and_save_images_parallel()
        print(f"{TAB1}Capture terminée. Attente de la fin de l’encodage...")
    finally:
        # 4) Arrêter le thread de sauvegarde d’images
        stop_saving.set()
        # 5) Attendre la fin de l’écriture des PNG
        image_queue.join()
        save_thread.join()

        # 6) Attendre la fin de FFmpeg
        encoding_thread.join()

        # 7) Nettoyer le dossier temporaire
        clean_temp_dir()
        print(f"{TAB1}Dossier temporaire nettoyé.")


if __name__ == "__main__":
    main()
