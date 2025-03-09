import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cv2
    import easyocr
    import matplotlib.pyplot as plt
    import urllib

    return cv2, easyocr, mo, np, plt, urllib


@app.cell
def _(cv2, np, urllib):
    def url_to_image(url):
        # download the image, convert it to a NumPy array, 
        # and then read it into OpenCV format
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
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



    return cleanup_text, group_boxes_by_lines, print_result, url_to_image


@app.cell
def _(easyocr, print_result, url_to_image):


    #image = url_to_image('https://diafilmy.su/uploads/posts/2019-07/1564381098_pict-jimkpp-49.jpg')
    image = url_to_image('https://yandex-images.clstorage.net/hi9J5f450/ae523cX-S/5OcSLy37rFR7UE4sW47XGpXJHUPvQDoYir8bRrQNiRyrT8Y9R0rzdkteUly7i01wAQqAe1vEN2zdTQTZuIQSjqHro3clkuFVJkkbBSKS1w7E5TRzh3B6TbrKFy79bbnERr9b0P_-IrbCmv4ssvoUASA3kWbufFYRJA0l2ZVtR_o8o-sSdJ-QvMMIYoSO-rNaOKVRz8McdtVuK8v_RMC9sCqPY5A_pnL6zn7iUHrW5HVNQ51utjGeocANEQkZwZdOdL-HHgzvCLDfzOagEiJ_KkwloSPTRKbBWptTN7SI4ShuWwPQj0r2GkIqJkR6npCp0a_pmiINWpk8iUldJT335ozjm1rUn8woa1BSTFIWc1rQydmjDwQCNYqjL3ewvN188jfvkCcaCroSggLABiJgDRXr-Z4e3WqpAE0NXfUtS06AD49qhP_oBKOo_mCGXnMinJmhox8cciWKP5-vAAxtXA6XjxCTxlJ2ngZukC4WoPERgxW68o2GMbCpJVVFJXvSIM9D8ixPmDxDmFrETpJrbpDVuWuHnOLlhu-Xr7AAuQAmvweYFxoyhsJiygweekgxGUNlktpdEtnQqanB4XWDqiD_Bx4gdzywbwyGhNoGVxK4YeWn28RO8XI3xxOsdGVctn-b7EemJg4OntbMqiooyemrkXK6UZI5xEmdWZlFb94kq7_yyAfAhONEEthe0gMaBOmxXxdomsFuQxe_GARREP4_C3Cr7mb24q5a6HbumHXN6w2KutVCkbg5ad29cZ-qEL_PQnTraFjbCB60vnq7BphZNSOXjAIt4l_T_0CkOei-63NcYyouOpYmJtA-flgN8af1hiJlcm2kEbnRwSXvcmQbF_ow4wiQN7gObHKePzb8RSknQ2gqGX4Pb5OcaJ2Ifq9riN8KetpOJlKE_qZczeW34W52YXJR3HEVkQW5V1psCwsutMME7CPIFmBS9kdK4CW1y5OMOsUG50M33Ihc')
    #image = cv2.imread('b0442d569c9af24bf1830f833c4f886e.jpg')
    reader = easyocr.Reader(['ru'], recog_network='cyrillic_g1') # this needs to run only once to load the model into memory
    result = reader.readtext(image) #'b0442d569c9af24bf1830f833c4f886e.jpg')
    print_result(result)
    return image, reader, result


@app.cell
def _(cleanup_text, cv2, group_boxes_by_lines, image, plt, result):

    lines = group_boxes_by_lines(result)
    print(len(lines), len(result))
    for (bbox, text, prob) in lines:
    
        # display the OCR'd text and associated probability
        print("[INFO] {:.4f}: {}".format(prob, text))
    
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
    plt.imshow(image)
    return bbox, bl, br, lines, prob, text, tl, tr


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
