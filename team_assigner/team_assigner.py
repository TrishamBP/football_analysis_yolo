from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
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


    def assign_team_color(self,frames, player_tracks, sample_every=25):
        # Sample players across the whole video: a single frame has only ~20 players,
        # so one goalkeeper or occluded player can hijack a cluster
        player_colors = []
        for frame_num in range(0, len(frames), sample_every):
            for _, player_detection in player_tracks[frame_num].items():
                bbox = player_detection["bbox"]
                player_color =  self.get_player_color(frames[frame_num],bbox)
                player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10,random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        team_id = self.predict_team(frame,player_bbox)
        self.player_team_dict[player_id] = team_id

        return team_id

    def predict_team(self,frame,player_bbox):
        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        return team_id

    def assign_player_teams(self,frames, player_tracks, sample_every=5):
        # Majority vote over each player's appearances across the video. A single
        # crop can be motion-blurred or occluded (e.g. grass bleeding into a white kit)
        votes = {}
        for frame_num, players in enumerate(player_tracks):
            for player_id, track in players.items():
                if frame_num % sample_every != 0 and player_id in votes:
                    continue
                team_id = self.predict_team(frames[frame_num], track['bbox'])
                votes.setdefault(player_id, []).append(team_id)

        for player_id, player_votes in votes.items():
            self.player_team_dict[player_id] = max(set(player_votes), key=player_votes.count)