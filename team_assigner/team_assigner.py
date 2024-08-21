from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image): #Función para realizar el clustering (k-means)
        # Reconfigura la imagen a una matriz 2D
        image_2d = image.reshape(-1,3)

        # Realiza K-means con 2 grupos
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox): #Función para definir a cada jugador con un color y así diferenciar los dos equipos
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        #Nos quedamos con solo la mitad superior de la imagen, ya que nos interesa la camiseta.
        top_half_image = image[0:int(image.shape[0]/2),:]

        #Obtener el modelo de clustering
        kmeans = self.get_clustering_model(top_half_image)

        # Obtener las etiquetas de clúster para cada píxel
        labels = kmeans.labels_

        # Reconfigura las etiquetas a la forma de la imagen
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Obtén el clúster del jugador
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self,frame, player_detections): #Función para asignar un color a cada equipo.
        
        player_colors = [] #Creamos una lista donde tendremos todos los colores diferentes de los jugadores (serán solo 2, uno para cada equipo)
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0] #El color del primer equipo será el primer clúster del k-means
        self.team_colors[2] = kmeans.cluster_centers_[1] #El color del segundo equipo será el segundo clúster del k-means

    def get_player_team(self,frame,player_bbox,player_id): #Función para asignar a cada jugador a un equipo.
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        #Esto sirve para que la posesión del portero se cuente correctamente, ya que se detecta como si el portero fuera del otro equipo.
        if player_id ==92: #CUIDADO PORQUE ESTO DEPENDE DEL NÚMERO DEL PORTERO
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id