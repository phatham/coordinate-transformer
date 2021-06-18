from geographiclib import geodesic
import numpy as np
from sklearn.manifold import MDS

class CoordinateTransformer:
    def __init__(self, points):
        self.points = points
        n = len(points)
        self.D = np.zeros((n, n))
        self.points_matrix = np.zeros((n, 2))
        self.heading = 0
        for i in range(n):
            self.points_matrix[i, :] = np.array(points[i]['local'])
            for j in range(n):
                ret_obj = geodesic.Geodesic.WGS84.Inverse(points[i]['geo'][0], points[i]['geo'][1], points[j]['geo'][0], points[j]['geo'][1])
                self.D[i, j] = ret_obj['s12']
                if j > i:
                    local_angle = np.rad2deg(np.arctan2(points[j]['local'][1] - points[i]['local'][1], points[j]['local'][0] - points[i]['local'][0]))
                    self.heading += (ret_obj['azi1'] + ret_obj['azi2'])/2 + local_angle
        self.heading /= n * (n - 1) / 2
        
        embedding = MDS(dissimilarity='precomputed', metric=True, n_init=200, eps=1e-3, max_iter=1000).fit_transform(self.D)
        answer1 = self.sub_fit_matrix(embedding)
        embedding[:, 0] *= -1 #sometimes mds might flip the axis
        answer2 = self.sub_fit_matrix(embedding)
        if np.sum(np.power(answer1-self.points_matrix, 2)) < np.sum(np.power(answer2-self.points_matrix, 2)):
            self.best_fit_coordinate = answer1
        else:
            self.best_fit_coordinate = answer2
            
        #find centroid as reference
        self.centroid = np.mean(self.best_fit_coordinate, axis=0)
        self.centroid_x = self.centroid[0]
        self.centroid_y = self.centroid[1]
        self.centroid_lat = 0
        self.centroid_lon = 0
        for i in range(n):
            diff = self.centroid - self.points_matrix[i, :] # travel from corner to centroid, referenced at corner
            dist = np.sqrt(np.sum(np.power(diff, 2)))
            local_angle = np.rad2deg(np.arctan2(diff[1], diff[0]))
            az = self.heading - local_angle
            if az < -360: az += 360
            if az > 360: az -= 360
            ret_obj = geodesic.Geodesic.WGS84.Direct(points[i]['geo'][0], points[i]['geo'][1], az, dist)
            self.centroid_lat += ret_obj['lat2']
            self.centroid_lon += ret_obj['lon2']
        self.centroid_lat /= n
        self.centroid_lon /= n
        
    def local2geo(self, p):
        x, y = p[0], p[1]
        dist = np.sqrt(np.power(x - self.centroid_x, 2) + np.power(y - self.centroid_y, 2))
        local_angle = np.rad2deg(np.arctan2(y - self.centroid_y, x - self.centroid_x))
        az = self.heading - local_angle
        if az < -360: az += 360
        if az > 360: az -= 360
        ret_obj = geodesic.Geodesic.WGS84.Direct(self.centroid_lat, self.centroid_lon, az, dist)
        return ret_obj['lat2'], ret_obj['lon2']
    
    def geo2local(self, p):
        lat, lon = p[0], p[1]
        ret_obj = geodesic.Geodesic.WGS84.Inverse(self.centroid_lat, self.centroid_lon, lat, lon)
        dist = ret_obj['s12']
        az = (ret_obj['azi1']+ret_obj['azi1'])/2
        local_angle = self.heading - az
        return self.centroid_x + dist * np.cos(np.deg2rad(local_angle)), self.centroid_y + dist * np.sin(np.deg2rad(local_angle))
    
    def sub_fit_matrix(self, new_in):
        original_out = self.points_matrix
        new_in_mean = np.mean(new_in, axis=1).reshape(-1, 1)
        original_out_mean = np.mean(original_out, axis=1).reshape(-1, 1)
        H = np.matmul((new_in - new_in_mean), (original_out - original_out_mean).T)
        U, S, Vt = np.linalg.svd(H)
        R = np.matmul(Vt.T, U.T)
        # if rotation matrix is reflected
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.matmul(Vt.T, U.T)
        translation_mat = original_out_mean - np.matmul(R, new_in_mean)
        new_out = np.matmul(R, new_in) + translation_mat
        return new_out
