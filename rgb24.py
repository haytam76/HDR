import time
import numpy as np
import cv2
import ctypes
from arena_api.system import system

TAB1 = "  "
TAB2 = "    "

def create_device_automatically():
    """
    Se connecte automatiquement au premier appareil Arena trouvé.
    Si aucun n'est trouvé après plusieurs tentatives, on lève une exception.
    """
    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    while tries < tries_max:
        devices = system.create_device()
        if not devices:
            print(f'{TAB1}Essai {tries+1} sur {tries_max} : '
                  f'attente de {sleep_time_secs} secondes pour qu’un appareil soit connecté !')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(f'{TAB1}{sec_count + 1 } secondes écoulées ',
                      '.' * sec_count, end='\r')
            tries += 1
        else:
            print(f'\n{TAB1}{len(devices)} appareil(s) détecté(s)\n')
            # On prend directement le premier appareil détecté
            selected_device = devices[0]
            return selected_device
    else:
        raise Exception(f'{TAB1}Aucun appareil trouvé ! '
                        f'Veuillez connecter un appareil et relancer l\'exemple.')

def setup(device):
    """
    Configuration des dimensions du flux et du nodemap du flux.
    Configure la largeur, la hauteur et le format des pixels pour la capture.
    """
    nodemap = device.nodemap
    nodes = nodemap.get_node(['Width', 'Height', 'OffsetX', 'OffsetY', 'PixelFormat'])
    nodes['OffsetX'].value = 0
    nodes['OffsetY'].value = 0
    nodes['Width'].value = 2880
    nodes['Height'].value = 1860
    nodes['PixelFormat'].value = 'RGB24'
    width = nodes['Width'].value
    height = nodes['Height'].value
    expected_size = width * height * 9
    return width, height, expected_size

def precalculate_gamma_lut_24bit(gamma, max_24bit_value=16777215, output_max_value=255):
    """
    Precompute a LUT for 24-bit channel values with gamma correction.
    Maps each 24-bit intensity value to an 8-bit gamma-corrected value.
    """
    lut = np.array([
        (i / max_24bit_value) ** (1 / gamma) * output_max_value
        for i in range(max_24bit_value + 1)
    ], dtype=np.uint8)
    return lut

# Precompute the LUT for gamma = 4
gamma_lut = precalculate_gamma_lut_24bit(gamma=4)

def convert_to_image_data(buffer, width, height, expected_size, gamma_lut):
    """
    Converts raw RGB72 (72 bits per pixel) data into an RGB24 image with gamma correction applied.
    """
    total_convert_start = time.perf_counter()

    # Convert the buffer into a NumPy array
    buffer_data = (ctypes.c_uint8 * expected_size).from_address(
        ctypes.addressof(buffer.pdata.contents)
    )
    buffer_array = np.ctypeslib.as_array(buffer_data)
    raw_array = buffer_array.reshape(-1, 9)

    # Combine 3 bytes per channel to get the 24-bit values
    r = ((raw_array[:, 2].astype(np.uint32) << 16)
         | (raw_array[:, 1].astype(np.uint32) << 8)
         | raw_array[:, 0].astype(np.uint32))
    g = ((raw_array[:, 5].astype(np.uint32) << 16)
         | (raw_array[:, 4].astype(np.uint32) << 8)
         | raw_array[:, 3].astype(np.uint32))
    b = ((raw_array[:, 8].astype(np.uint32) << 16)
         | (raw_array[:, 7].astype(np.uint32) << 8)
         | raw_array[:, 6].astype(np.uint32))

    # Apply the gamma LUT directly for each 24-bit channel
    r_corrected = gamma_lut[r]
    g_corrected = gamma_lut[g]
    b_corrected = gamma_lut[b]

    # Stack channels to form the final RGB24 image
    image_rgb8 = np.stack([r_corrected, g_corrected, b_corrected], axis=-1).reshape((height, width, 3))

    total_convert_end = time.perf_counter()
    print(f"Temps total pour la conversion : {total_convert_end - total_convert_start:.6f} secondes")

    return image_rgb8

def example_entry_point():
    """
    Démontre un flux en direct avec conversion manuelle en image 8 bits
    à l'aide de NumPy, tout en imprimant les FPS moyens toutes les 5 trames.
    Enregistre également deux trames sous format JPEG.
    """
    # Remplace la sélection manuelle par la connexion automatique
    device = create_device_automatically()

    # Configurer et obtenir la largeur, la hauteur et la taille du tampon (calculée une seule fois)
    width, height, expected_size = setup(device)

    curr_frame_time = 0
    prev_frame_time = 0
    frame_count = 0
    fps_sum = 0  # Sum of FPS for calculating the average over 5 frames
    fps_interval = 5  # Number of frames over which to calculate the average FPS

    with device.start_stream():
        """
        Récupérer et afficher les données du tampon indéfiniment
        jusqu'à ce que la touche Échap soit pressée.
        """
        while True:
            total_start_time = time.time()

            # Mesurer le temps pour obtenir un buffer
            buffer_start_time = time.time()
            buffer = device.get_buffer()
            buffer_end_time = time.time()

            try:
                # Mesurer le temps pour convertir les données en image
                convert_start_time = time.time()
                image = convert_to_image_data(buffer, width, height, expected_size, gamma_lut)
                convert_end_time = time.time()

                # Convertir l'image RGB en BGR pour OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Afficher l'image avec OpenCV (image BGR pour OpenCV)
                cv2.imshow('Image RGB', image_bgr)

                # Enregistrer les deux premières trames sous format JPEG
                if frame_count < 1:
                    cv2.imwrite(f'image_frame_{frame_count}.jpg', image_bgr)
                    frame_count += 1
                
                # Quitter si la touche Échap est pressée
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                # Mesurer le temps pour réassigner le buffer
                requeue_start_time = time.time()
                device.requeue_buffer(buffer)
                requeue_end_time = time.time()

                # Calculer les FPS pour la trame courante
                curr_frame_time = time.time()
                fps = 1 / (curr_frame_time - prev_frame_time)
                prev_frame_time = curr_frame_time

                # Accumuler les FPS pour le calcul moyen toutes les 5 trames
                fps_sum += fps
                if (frame_count + 1) % fps_interval == 0:
                    avg_fps = fps_sum / fps_interval
                    # Print highlighted FPS for the last 5 frames
                    print(f"\n{'='*40}")
                    print(f"=== FPS moyen pour les {fps_interval} dernières trames : {avg_fps:.2f} ===")
                    print(f"{'='*40}\n")
                    fps_sum = 0  # Reset fps_sum for the next interval

                # Afficher les FPS et les temps mesurés pour chaque trame
                print(f"Temps pour obtenir un buffer : {buffer_end_time - buffer_start_time:.6f} secondes")
                print(f"Temps pour convertir le buffer en image : {convert_end_time - convert_start_time:.6f} secondes")
                print(f"Temps pour réassigner le buffer : {requeue_end_time - requeue_start_time:.6f} secondes")
                print(f"Temps total pour traiter la trame : {time.time() - total_start_time:.6f} secondes")

                frame_count += 1  # Incrémente le compteur de trames

            except ValueError as e:
                print(f"Erreur lors du traitement de la trame : {e}")

    device.stop_stream()
    cv2.destroyAllWindows()

    system.destroy_device()

    print(f'{TAB1}Tous les appareils créés ont été détruits')

if __name__ == '__main__':
    print('\nExemple démarré\n')
    example_entry_point()
    print('\nExemple terminé avec succès')
