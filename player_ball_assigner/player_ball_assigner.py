import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70 #Si la pelota está a 70 píxeles o menos del jugador, se marcará al jugador en cuestión; si no, no.

    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999 #Nos servirá para marcar al jugador más cercano a la pelota
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position) #Calcula la distancia del pie izquierdo
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position) #Calcula la distancia del pie derecho
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player