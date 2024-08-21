import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path) #Crea un objeto para el vídeo en concreto
    frames = [] #Inicializamos un vector vacío
    while True: #Mientras sea true --> significará que tenemos frames de vídeo
        ret, frame = cap.read() #Se añade el cuadro de vídeo (frame) a la lista.
        if not ret: #Cuando es false significa que ya no tenemos frames de vídeo y, por lo tanto, el vídeo ha terminado.
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #Definimos el formato de salida --> XVID
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0])) 
    #A través de la ruta, el formato de salida definido, los frames (24) y las dimensiones (altura y ancho), creamos una variable.
    for frame in ouput_video_frames:
        out.write(frame) #Escribimos cada frame en cada video writer.
    out.release()