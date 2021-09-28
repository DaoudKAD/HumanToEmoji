import cv2
import time
import uuid

def images_acquisition(labels_list: list, nb_images_to_acquire: int, path_to_save_images: str):
    """
        permet d'acquérir des images pour constituer un dataset selon certaines classes
            Params:
                labels_list (list): liste de labels correspondants aux classes
                nb_images_to_acquire (int): nombre d'images à acquérir par classe
                path_to_save_images (str) : chemin menant au repertoire de sauvegarde du dataset
            Return:
                None
    """

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('Acquisition des images', frame)
    time.sleep(3)

    for label in labels_list:

        print("-- Classe : " + label)
        for image_number in range(nb_images_to_acquire):
            ret, frame = cap.read()

            if ret:
                # effet miroir
                frame = cv2.flip(frame, 1)
                frame_with_label = cv2.putText(img=frame,
                                               text=label + " : " + str(image_number),
                                               org=(50, 50),
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=1,
                                               color=(0, 0, 255),
                                               lineType=2)

                cv2.imshow('Acquisition des images', frame_with_label)
                time.sleep(3)

                # enregistrer l'image
                cv2.imwrite(path_to_save_images + label + str(uuid.uuid1()) + ".png", frame)
                print(path_to_save_images + label + str(image_number) + ".png : saved !")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    labels_list = ["Peace", "Ok", "Hello", "Fist", "Heart"]
    nb_images_to_acquire = 15
    path_to_save_images = "/Users/daoud.kadoch/Documents/HumanToEmoji_dataset/acquired_images/"
    images_acquisition(labels_list=labels_list,
                       nb_images_to_acquire=nb_images_to_acquire,
                       path_to_save_images=path_to_save_images)
