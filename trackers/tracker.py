
from ultralytics import YOLO #Esta librería es necesaria para cargar el modelo
import supervision as sv #Esta librería servirá para cargar el tracking
import pickle #Esta librería sirve para guardar los resultados de una forma concreta
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path): #Se inicializa cuando iniciamos la clase
        self.model = YOLO(model_path) #Inicializamos el modelo
        self.tracker = sv.ByteTrack() #Inicializamos el tracking

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions): #Con esta función nos servirá para detectar la pelota en aquellos frames en los que no la detecta
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions] #Creamos una lista con las posiciones de la pelota
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolar valores faltantes
        df_ball_positions = df_ball_positions.interpolate() #Con la función interpolate() sirve para detectar casi todas las detecciones que faltan
        df_ball_positions = df_ball_positions.bfill() #Con la función bfill() es para los primeros frames

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()] #Devolvemos todos los valores en un diccionario.

        return ball_positions

    def detect_frames(self, frames): #Con esta función nos servirá para detectar los fotogramas.
        batch_size=20 #Se tomarán 20 datos a la vez
        detections = [] #Vector vacío donde se irán añadiendo todos los objetos que se detecten
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch #Añade todo lo que detecta a la lista
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None): #Con esta función nos servirá para detectar las diferentes pistas
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames) #Sirve para detectar los fotogramas

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        } #Creamos un diccionario para guardar las referencias de cada objeto

        for frame_num, detection in enumerate(detections): #Lo que hará este bucle será recorrer `detections` (todos los jugadores u objetos) y les asignará un número
            cls_names = detection.names #Guarda todas las clases (0=jugador, 1=pelota...)
            cls_names_inv = {v:k for k,v in cls_names.items()} #Invierte el nombre de las clases, para trabajar más fácilmente

            #Convertir al formato de detección de supervisión
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convertir GoalKeeper en objeto jugador (porque lo detectaba como otro objeto aparte)
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            #Rastrear objetos
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            #Cada diccionario tendrá una clave para cada ID (tracker_id=array....) y los valores serán el bounding box
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist() #Nos detectará el bounding box del frame correspondiente
                cls_id = frame_detection[3] #Nos detectará el ID de la clase del frame correspondiente
                track_id = frame_detection[4] #Nos detectará el ID del track del frame correspondiente

                #Las siguientes 4 líneas de código servirán para que cada bounding box tenga su propia información, dependiendo de si es
                #un jugador o un árbitro. Para la pelota no es necesario, ya que solo hay una.
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            #Ahora haremos lo mismo para la pelota. Como solo hay una, no es necesario rastrearla y buscamos en detection_supervision
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox} #Ponemos un [1] ya que solo hay una pelota.

        #Sirve para guardar los resultados
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_traingle(self,frame,bbox,color): #Con esta función nos servirá para dibujar un triángulo sobre la pelota
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_ellipse(self,frame,bbox,color,track_id=None): #Con esta función nos servirá para dibujar las elipses (círculos un poco diferentes)
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox) #Calcula el centro del círculo
        width = get_bbox_width(bbox) #Calcula el ancho del círculo

        #Función para calcular la elipse
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        #Código para definir el rectángulo (que cada jugador tendrá)
        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        #Código para crear el rectángulo en el jugador correspondiente
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            #Código para escribir el número sobre el rectángulo de cada jugador
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control): #Con esta función nos servirá para dibujar la información sobre quién está dominando la posesión de la pelota
        ##Dibuja un rectángulo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 ) #Dibuja el rectángulo, donde dentro estará la información
        alpha = 0.4 #Transparencia del rectángulo
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        #El siguiente código sirve para calcular el % de posesión de cada equipo
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        #Obtén el número de veces que cada equipo tuvo el control del balón
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control): #Con esta función nos servirá para agregar anotaciones
        output_video_frames= [] #Creamos un vector vacío donde se irá añadiendo la información de cada frame
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() #Guardamos el frame correspondiente

            player_dict = tracks["players"][frame_num] #Guardaremos el track de cada jugador en un diccionario
            ball_dict = tracks["ball"][frame_num] #Guardaremos el track de la pelota en un diccionario
            referee_dict = tracks["referees"][frame_num] #Guardaremos el track de cada árbitro en un diccionario

            #Dibuja a los jugadores
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id) #Dibujará un círculo del color del equipo

                if player.get('has_ball',False): #Sirve para marcar con un triángulo al jugador que tiene el balón
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255)) 

            #Dibuja a los árbitros
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Dibuja a la pelota
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            #Dibuja el control del balón del equipo
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames