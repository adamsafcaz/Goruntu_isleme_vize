#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import random

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
STAR_SIZE = 5
NUM_STARS = 50
stars = []

def koordinat_getir(landmarks, indeks, h, w):
  landmark = landmarks[indeks]
  return int(landmark.x*w), int(landmark.y*h)

def generate_stars(width, height, num_stars):
  new_stars = []
  for _ in range(num_stars):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Rastgele renk
    new_stars.append((x, y, color))
  return new_stars

def draw_stars(image, current_stars):
  for star_x, star_y, color in current_stars:
    cv2.circle(image, (star_x, star_y), STAR_SIZE, color, -1)

def draw_landmarks_on_image(rgb_image, detection_result):
  global stars
  # eklem boğumlarının listesi
  hand_landmarks_list = detection_result.hand_landmarks
  # sağ el mi sol el mi mevcut göster
  handedness_list = detection_result.handedness # Düzeltme: hAndedness_list -> handedness_list
  annotated_image = np.copy(rgb_image)
  h, w, c = annotated_image.shape

  if not stars:
    stars = generate_stars(w, h, NUM_STARS)

  parmaklar = []
  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    # idx ile belirtilen eldeki boğum noktalarını al
    hand_landmarks = hand_landmarks_list[idx]
    # işaret parmak ucu koordinatları
    x1, y1 = koordinat_getir(hand_landmarks, 8, h, w)
    x5, y5 = koordinat_getir(hand_landmarks, 6, h, w)
    if y1 < y5: # Düzeltme: Parmak yukarıdaysa 1
      parmaklar.append(1)
    else:
      parmaklar.append(0)

    # orta parmak ucu koordinatları
    x1, y1 = koordinat_getir(hand_landmarks, 12, h, w)
    x5, y5 = koordinat_getir(hand_landmarks, 10, h, w)
    if y1 < y5: # Düzeltme: Parmak yukarıdaysa 1
      parmaklar.append(1)
    else:
      parmaklar.append(0)

    # yüzük parmak ucu koordinatları
    x1, y1 = koordinat_getir(hand_landmarks, 16, h, w)
    x5, y5 = koordinat_getir(hand_landmarks, 14, h, w)
    if y1 < y5: # Düzeltme: Parmak yukarıdaysa 1
      parmaklar.append(1)
    else:
      parmaklar.append(0)

    # serçe parmak ucu koordinatları
    x1, y1 = koordinat_getir(hand_landmarks, 20, h, w)
    x5, y5 = koordinat_getir(hand_landmarks, 18, h, w)
    if y1 < y5: # Düzeltme: Parmak yukarıdaysa 1
      parmaklar.append(1)
    else:
      parmaklar.append(0)

    # baş parmak ucu koordinatları
    x, y = koordinat_getir(hand_landmarks, 4, h, w)
    x2, y2 = koordinat_getir(hand_landmarks, 2, h, w)
    if x > x2: # Düzeltme: Başparmak sağdaysa 1 (genellikle)
      parmaklar.append(1)
    else:
      parmaklar.append(0)

    toplam = sum(parmaklar) # Açık parmak sayısını bul
    annotated_image = cv2.putText(annotated_image, str(toplam), (x1, y1 - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2) # Metin pozisyonunu düzelttim

    renk = (255,255,0)
    # işaret parmağı ucunun olduğu yere daire koy
    annotated_image = cv2.circle(annotated_image, (x1,y1), 9, renk , -1) # Dolu daire

    # Yıldızları kontrol et ve güncelle
    new_stars = []
    for star_x, star_y, color in stars:
      distance = ((x1 - star_x)**2 + (y1 - star_y)**2)**0.5
      if distance > 30: # Eğer parmaktan yeterince uzaksa tut
        new_stars.append((star_x, star_y, color))
      else: # Parmağa yakınsa yeni bir konum ve renk oluştur
        new_x = random.randint(0, w - 1)
        new_y = random.randint(0, h - 1)
        new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        new_stars.append((new_x, new_y, new_color))
    stars = new_stars


    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape

    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  draw_stars(annotated_image, stars)
  return annotated_image


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
# kameradan görüntü oku
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Kamera açılamadı!")
    exit()
# kamera açık olduğu sürece
while cam.isOpened():
  # kameradan 1 frame oku
  basari, frame = cam.read()
  # eğer okuma başarılıysa
  if basari:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(mp_image)

    # STEP 5: Process the classification result. In this case, visualize it.
    if detection_result.hand_landmarks:
      annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
      cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    else:
      frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      h, w, _ = frame_bgr.shape
      if not stars:
        stars = generate_stars(w, h, NUM_STARS)
      draw_stars(frame_bgr, stars)
      cv2.imshow("Image", frame_bgr)

    key = cv2.waitKey(1)  # 1 ms bekle
    # q tuşuna basıldıysa programı sonlandır
    if key == ord('q') or key == ord('Q'):
      break
  else:
    print("Kamera okuma hatası!")
    break

cam.release()
cv2.destroyAllWindows()