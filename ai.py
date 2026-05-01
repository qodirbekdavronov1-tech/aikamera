from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# Kamera ochish
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Kamera sifatini yaxshilash
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

if not camera.isOpened():
    print("Kamera ochilmadi. Kamerani tekshir.")
    exit()

translations = {
    "person": "odam",
    "cell phone": "telefon",
    "laptop": "noutbuk",
    "mouse": "sichqoncha",
    "keyboard": "klaviatura",
    "book": "kitob",
    "bottle": "butilka",
    "cup": "piyola",
    "chair": "stul",
    "couch": "divan",
    "tv": "televizor",
    "car": "mashina",
    "bus": "avtobus",
    "truck": "yuk mashinasi",
    "bicycle": "velosiped",
    "dog": "it",
    "cat": "mushuk",
    "apple": "olma",
    "banana": "banan",
    "orange": "apelsin",
    "clock": "soat",
    "backpack": "ryukzak",
    "scissors": "qaychi"
}

cv2.namedWindow("AI Kamera - Buyum tanish", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Kamera - Buyum tanish", 1280, 720)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Kameradan rasm olinmadi.")
        break

    # Tasvirni biroz tiniqlashtirish
    frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=10)

    # AI orqali obyektlarni aniqlash
    results = model(frame, imgsz=960, verbose=False)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])

            # 0.35 qilsak, oldingidan ko'proq obyektlarni ko'rsatadi
            if confidence < 0.35:
                continue

            class_id = int(box.cls[0])
            english_name = model.names[class_id]
            uzbek_name = translations.get(english_name, english_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Ramka chizish
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Matn uchun qora fon
            text = f"{uzbek_name} {confidence * 100:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2

            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size

            text_x = x1
            text_y = max(y1 - 10, text_height + 10)

            cv2.rectangle(
                frame,
                (text_x, text_y - text_height - 8),
                (text_x + text_width + 10, text_y + 5),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                text,
                (text_x + 5, text_y),
                font,
                font_scale,
                (0, 255, 0),
                thickness
            )

    cv2.imshow("AI Kamera - Buyum tanish", frame)

    # q tugmasini bossang chiqadi
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()