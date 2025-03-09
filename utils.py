import urllib
import numpy as np
import cv2


def url_to_image(url):
    # download the image, convert it to a NumPy array, 
    # and then read it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,  cv2.COLOR_BGR2RGB )
    # return the image
    return image

def print_result(result):
    for res in result:
        coord=res[0]
        text=res[1]
        conf=res[2]
        print(text)   

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def group_boxes_by_lines(results, y_threshold=10):
    # Сортируем все результаты по Y-координате сверху вниз
    sorted_results = sorted(results, key=lambda x: np.mean([x[0][0][1], x[0][2][1]]))

    if not sorted_results:
        return []

    # Группируем боксы по строкам
    lines = []
    current_line = [sorted_results[0]]
    current_y = np.mean([sorted_results[0][0][0][1], sorted_results[0][0][2][1]])

    for box in sorted_results[1:]:
        y_center = np.mean([box[0][0][1], box[0][2][1]])
        if abs(y_center - current_y) <= y_threshold:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
            current_y = y_center

    if current_line:
        lines.append(current_line)

    merged_lines = []
    for line in lines:
        # Сортируем слова в строке по X-координате (слева направо)
        line_sorted = sorted(line, key=lambda item: min([point[0] for point in item[0]]))

        # Собираем данные для объединения
        all_boxes = [item[0] for item in line_sorted]
        all_texts = [item[1] for item in line_sorted]
        all_probs = [item[2] for item in line_sorted]

        # Вычисляем общий bounding box для строки
        all_x = [point[0] for box in all_boxes for point in box]
        all_y = [point[1] for box in all_boxes for point in box]
        merged_bbox = [
            [min(all_x), min(all_y)],
            [max(all_x), min(all_y)],
            [max(all_x), max(all_y)],
            [min(all_x), max(all_y)]
        ]

        # Объединяем текст и усредняем вероятность
        merged_text = " ".join(all_texts)
        avg_prob = np.mean(all_probs)

        merged_lines.append((merged_bbox, merged_text, avg_prob))

    return merged_lines

def put_results_on_image(result, img):
    image = img.copy()
    for (bbox, text, prob) in result:
    
        # display the OCR'd text and associated probability
        #print("[INFO] {:.4f}: {}".format(prob, text))
    
        # unpack the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
    
        # cleanup the text and draw the box surrounding the text along
        # with the OCR'd text itself
        text = cleanup_text(text)
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, text, (tl[0], tl[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image
