import cv2
from PIL import Image
import uuid
import time

def images_acquisition(labels_list: list, nb_images_to_acquire: int):
    """ """

    cap = cv2.VideoCapture(0)

    chrono = time.time()
    for label in labels_list:
        print(label)

        for image_number in range(nb_images_to_acquire):
            ret, frame = cap.read()
            if ret :
                # effet miroir
                frame = cv2.flip(frame, 1)
                frame_with_label = cv2.putText(img=frame,
                                               text=label+" : "+str(image_number),
                                               org=(50, 50),
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=1,
                                               color=(0, 0, 255),
                                               lineType=2)

                cv2.imshow('Acquisition des images', frame_with_label)
                time.sleep(3)
                cv2.imwrite(label+str(image_number)+".png", frame)

                # enregistrer l'image

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    labels_list = ["Ok", "Hello", "Double Hand"]
    nb_images_to_acquire = 3
    images_acquisition(labels_list=labels_list,
                       nb_images_to_acquire=nb_images_to_acquire)