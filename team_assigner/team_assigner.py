from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self, team1_color=[255, 53, 0], team2_color=[255, 6, 0]):
        self.team_colors = {}
        self.player_team_dict = {}
        self.box_colors = {}
        
        # Initialize with custom colors if provided
        if team1_color is not None:
            self.team_colors[1] = np.array(team1_color)
        if team2_color is not None:
            self.team_colors[2] = np.array(team2_color)
        
        self.custom_colors = team1_color is not None and team2_color is not None
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        # Skip automatic color assignment if custom colors are set
        if self.custom_colors:
            return
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        if self.custom_colors:
            # When using custom colors, compute distance to each team color
            dist1 = np.linalg.norm(player_color - self.team_colors[1])
            dist2 = np.linalg.norm(player_color - self.team_colors[2])
            team_id = 1 if dist1 < dist2 else 2
        else:
            # Use kmeans clustering as before
            team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
            team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id
    
    def set_team_colors(self, team1_color, team2_color):
        """
        Manually set the team colors
        
        Args:
            team1_color: RGB color for team 1 as a list/tuple [r, g, b]
            team2_color: RGB color for team 2 as a list/tuple [r, g, b]
        """
        self.team_colors[1] = np.array(team1_color)
        self.team_colors[2] = np.array(team2_color)
        self.custom_colors = True
        
        # Reset player team assignments when colors change
        self.player_team_dict = {}
    
    def set_box_colors(self, team1_box_color, team2_box_color):
        """
        Manually set the box colors for player numbers
        
        Args:
            team1_box_color: RGB color for team 1 boxes as a list/tuple [r, g, b]
            team2_box_color: RGB color for team 2 boxes as a list/tuple [r, g, b]
        """
        self.box_colors[1] = np.array(team1_box_color)
        self.box_colors[2] = np.array(team2_box_color)
        
        # The player team assignments don't need to be reset when just changing box colors
    
    def update_player_colors(self, player_track, team):
        """
        Update player track with appropriate team and box colors
        
        Args:
            player_track: The player track dictionary to update
            team: The assigned team (1 or 2)
            
        Returns:
            Updated player track dictionary with team_color and box_color
        """
        # Add team assignment
        player_track['team'] = team
        
        # Add team color (convert numpy array to tuple for OpenCV compatibility)
        team_color = self.team_colors[team]
        if hasattr(team_color, 'tolist'):
            team_color = tuple(team_color.tolist())
        player_track['team_color'] = team_color
        
        # Add box color if available
        if team in self.box_colors:
            box_color = self.box_colors[team]
            if hasattr(box_color, 'tolist'):
                box_color = tuple(box_color.tolist())
            player_track['box_color'] = box_color
            
        return player_track
