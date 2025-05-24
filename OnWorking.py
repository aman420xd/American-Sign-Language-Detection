import os
import csv
import copy
import argparse
import itertools
import time
from gtts import gTTS
from pygame import mixer  # Replacing playsound with pygame.mixer
import tempfile
import threading
from queue import Queue

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# Speech queue to prevent overlapping audio
speech_queue = Queue()
is_speaking = False

def speak_text(text):
    """Convert text to speech using gTTS and play it"""
    def _speak():
        global is_speaking
        try:
            is_speaking = True
            # Create a temporary file with a unique name
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Retry mechanism for gTTS
            for attempt in range(3):  # Retry up to 3 times
                try:
                    # Generate speech
                    tts = gTTS(text=text, lang='en')
                    tts.save(temp_path)
                    break
                except Exception as e:
                    if attempt == 2:  # If all retries fail
                        print(f"Error in text-to-speech (gTTS): {e}")
                        return
            
            # Play the audio file using pygame.mixer
            mixer.init()
            mixer.music.load(temp_path)
            mixer.music.play()
            while mixer.music.get_busy():
                time.sleep(0.1)
            mixer.quit()
            
            # Remove the temporary file
            os.unlink(temp_path)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
        finally:
            is_speaking = False
            # Process next item in queue if exists
            if not speech_queue.empty():
                next_text = speech_queue.get()
                _speak(next_text)

    # If currently speaking, add to queue
    if is_speaking:
        speech_queue.put(text)
    else:
        # Start a new thread for speaking to avoid blocking
        threading.Thread(target=_speak).start()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)
    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument("--min_detection_confidence", help="min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help="min_tracking_confidence", type=int, default=0.5)
    return parser.parse_args()

def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # Configuration
    REQUIRED_CONSECUTIVE_FRAMES = 40  # Frames needed to accept a character
    WORD_DELAY = 5                 # Seconds before auto-adding space
    SENTENCE_DELAY = 10             # Seconds before speaking sentence
    RESET_DELAY = 15                # Seconds before resetting everything
    INPUT_COOLDOWN = 5              # Seconds before accepting new input after recognition

    # State variables
    current_sentence = []
    current_word = []
    confirmed_word = ""
    last_character = ""
    current_gesture = None
    consecutive_frames = 0
    last_gesture_time = time.time()
    last_handled_time = time.time()
    reset_timer = time.time()
    pending_sentence = ""
    input_cooldown = 0               # Time when input will be accepted again
    last_spoken_word = ""            # Track last spoken word to avoid repetition
    gesture_detected = False         # Track if a gesture is detected

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    classifier = KeyPointClassifier()
    
    with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
        labels = [row[0] for row in csv.reader(f)]
    
    fps_calc = CvFpsCalc(buffer_len=10)
    mode = 0

    while True:
        fps = fps_calc.get()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
            
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        current_time = time.time()
        time_since_last_gesture = current_time - last_gesture_time
        time_since_last_handled = current_time - last_handled_time
        time_since_reset = current_time - reset_timer

        # Process hand landmarks
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # Reset gesture_detected flag
        gesture_detected = False

        if results.multi_hand_landmarks and current_time >= input_cooldown:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)
                gesture_id = classifier(pre_processed)
                gesture = labels[gesture_id]

                # Check gesture consistency
                if gesture == current_gesture:
                    consecutive_frames += 1
                else:
                    current_gesture = gesture
                    consecutive_frames = 1

                # Mark gesture as detected
                gesture_detected = True

                # Only process if we have enough consecutive frames
                if consecutive_frames >= REQUIRED_CONSECUTIVE_FRAMES:
                    # Only add character if it's different or enough time has passed
                    if (gesture != last_character or 
                        time_since_last_gesture > 0.5):
                        
                        if gesture == "Space":
                            if current_word:
                                confirmed_word = ''.join(current_word)
                                current_word = []
                                input_cooldown = current_time + INPUT_COOLDOWN  # Add cooldown
                        elif gesture == "Del":
                            if current_word:
                                current_word = current_word[:-1]
                            elif confirmed_word:
                                confirmed_word = confirmed_word[:-1]
                            elif current_sentence:
                                current_sentence.pop()
                            input_cooldown = current_time + INPUT_COOLDOWN  # Add cooldown
                        elif gesture not in ["Space", "Del"]:
                            current_word.append(gesture)
                            input_cooldown = current_time + INPUT_COOLDOWN  # Add cooldown
                        
                        last_character = gesture
                        last_gesture_time = current_time
                        last_handled_time = current_time
                        reset_timer = current_time

        # Handle timing-based actions only when no gesture is detected
        if not gesture_detected:
            # Sentence completion logic
            if time_since_last_handled > SENTENCE_DELAY and (current_sentence or confirmed_word or current_word):
                # Add any pending word to sentence
                if current_word:
                    confirmed_word = ''.join(current_word)
                    current_word = []
                
                if confirmed_word:
                    current_sentence.append(confirmed_word)
                    confirmed_word = ""
                
                # Prepare the complete sentence for speaking
                if current_sentence:
                    pending_sentence = ' '.join(current_sentence)
                    speak_text(pending_sentence)
                    current_sentence = []
                
                last_handled_time = current_time
                reset_timer = current_time
            
            # Word completion logic
            elif time_since_last_handled > WORD_DELAY and (confirmed_word or current_word):
                if current_word:
                    confirmed_word = ''.join(current_word)
                    current_word = []
                
                if confirmed_word and confirmed_word != last_spoken_word:
                    current_sentence.append(confirmed_word)
                    speak_text(confirmed_word)  # Speak the completed word
                    last_spoken_word = confirmed_word
                    confirmed_word = ""
                
                last_handled_time = current_time
                reset_timer = current_time

        # Reset timers if gesture is detected
        if gesture_detected:
            last_handled_time = current_time
            reset_timer = current_time

        # Handle reset logic
        if time_since_reset > RESET_DELAY:
            if current_sentence or confirmed_word or current_word:
                pending_sentence = ' '.join(current_sentence + [confirmed_word] + [''.join(current_word)]).strip()
                if pending_sentence:
                    speak_text(pending_sentence)
            current_sentence = []
            current_word = []
            confirmed_word = ""
            pending_sentence = ""
            last_character = ""
            current_gesture = None
            consecutive_frames = 0
            reset_timer = current_time

        # Draw hand landmarks and info
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                debug_image = draw_landmarks(debug_image, landmark_list)
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_info_text(
                    debug_image, brect, 
                    results.multi_handedness[0], 
                    f"{current_gesture or 'None'} ({consecutive_frames}/{REQUIRED_CONSECUTIVE_FRAMES})"
                )

        # Display current state
        display_height = debug_image.shape[0]
        display_width = debug_image.shape[1]

        # --- Bottom left: Sentence and Current Word ---
        margin = 20
        line_gap = 10
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        sentence_text = f"Sentence: {' '.join(current_sentence)}"
        word_text = f"Current Word: {''.join(current_word)}"

        # Calculate text sizes for proper stacking
        (sentence_width, sentence_height), _ = cv.getTextSize(sentence_text, font, font_scale, thickness)
        (word_width, word_height), _ = cv.getTextSize(word_text, font, font_scale, thickness)

        # Sentence at bottom left
        sentence_org = (margin, display_height - margin)
        # Current Word just above Sentence
        word_org = (margin, display_height - margin - sentence_height - line_gap)

        cv.putText(debug_image, word_text, word_org, font, font_scale, (255, 255, 0), thickness)
        cv.putText(debug_image, sentence_text, sentence_org, font, font_scale, (255, 255, 255), thickness)
        # --- End bottom left ---

        # --- Show Word Delay and Sentence Delay at Top Right ---
        word_delay_left = max(0, int(WORD_DELAY - (current_time - last_handled_time)))
        sentence_delay_left = max(0, int(SENTENCE_DELAY - (current_time - last_handled_time)))
        text1 = f"Word Delay: {word_delay_left}s"
        text2 = f"Sentence Delay: {sentence_delay_left}s"
        font_delay = cv.FONT_HERSHEY_SIMPLEX
        font_scale_delay = 0.7
        thickness_delay = 2

        # Calculate text sizes
        (text1_width, _), _ = cv.getTextSize(text1, font_delay, font_scale_delay, thickness_delay)
        (text2_width, _), _ = cv.getTextSize(text2, font_delay, font_scale_delay, thickness_delay)

        margin_right = 20
        x1 = display_width - text1_width - margin_right
        x2 = display_width - text2_width - margin_right
        y1 = margin_right + 30
        y2 = margin_right + 60

        cv.putText(debug_image, text1, (x1, y1), font_delay, font_scale_delay, (0, 255, 255), thickness_delay)
        cv.putText(debug_image, text2, (x2, y2), font_delay, font_scale_delay, (0, 255, 0), thickness_delay)
        # --- End of delay display ---

        cv.imshow("ASL Sentence Builder", debug_image)

    cap.release()
    cv.destroyAllWindows()

def select_mode(key, mode):
    number = -1
    if 65 <= key <= 90:  # A ~ B
        number = key - 65
    if key == 110:  # n (Inference Mode)
        mode = 0
    if key == 107:  # k (Capturing Landmark From Camera Mode)
        mode = 1
    if key == 100:  # d (Capturing Landmarks From Provided Dataset Mode)
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if (mode == 1 or mode == 2) and (0 <= number <= 35):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (255, 255, 255),
            2,
        )

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[6]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_point[7]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_point[8]),
            (255, 255, 255),
            2,
        )

        # Middle finger
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[10]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[10]),
            tuple(landmark_point[11]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[11]),
            tuple(landmark_point[12]),
            (255, 255, 255),
            2,
        )

        # Ring finger
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 255, 255),
            2,
        )

        # Little finger
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (255, 255, 255),
            2,
        )

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (255, 255, 255),
            2,
        )

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    return image

def draw_info(image, fps, mode, number):
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    mode_string = [
        "Logging Key Point",
        "Capturing Landmarks From Provided Dataset Mode",
    ]
    if 1 <= mode <= 2:
        cv.putText(
            image,
            "MODE:" + mode_string[mode - 1],
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                "NUM:" + str(number),
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    return image

if __name__ == "__main__":
    main()