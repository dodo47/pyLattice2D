from pyLattice2D.lattices.base import Base
import numpy as np
import scipy as sp
import scipy.spatial
from shapely.geometry import box, Polygon, Point #(NOTE: Need to install shapely via conda)
import random
import sys
import copy
eps = sys.float_info.epsilon
import scipy.spatial as sci
import math

class VoronoiLattice(Base):
    def __init__(self, delta_ratio, **kwargs):
        '''
        Implementation of Voronoi tiling with different degrees of disorder.
        
        Produces a Voronoi diagram with nodes that intersect a bounding box. This is needed for clean box edges for compression test.
        Inspired by: https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells
        '''
        self.delta_ratio = delta_ratio # amount of order in the tiling
        super(VoronoiLattice, self).__init__(**kwargs)

    def create_base_lattice(self):
        self.num_points = self.num_layers
        bounding_box = np.array([0., 1., 0., 1.])
        points = disorder_delta_voronoi(self.delta_ratio, self.num_points, bounding_box[1], bounding_box[3])
        
        vor = bounded_voronoi(points, bounding_box)

        vor_vertices=np.empty(shape=(1,2))
        vor_edges=np.empty(shape=(1,2))
        for region in vor.filtered_regions:
            vertices = vor.vertices[region, :]
            vor_vertices = np.concatenate((vor_vertices,vertices),axis=0)

        vor_vertices = np.unique(vor_vertices, axis=0)

        ## REMAP ALL IDs --- Convert from filtered vertices indices to standalone indices (0,1,2...)
        all_ids = np.sort(np.array(list(set(np.array(vor.filtered_ridges).flatten()))))

        remap = {}
        new_coords = []
        for i in range(len(all_ids)):
            remap[all_ids[i]] = i
            new_coords.append(vor.vertices[all_ids[i]])

        new_ridges = []
        for i,j in vor.filtered_ridges:
            new_ridges.append([remap[i], remap[j]])

        new_coords = np.array(new_coords)

        ## REMOVE THE VERTICAL EDGES IN AXIS OF COMPRESSION TEST
        new_ids = np.sort(np.array(list(set(np.array(new_ridges).flatten()))))

        self.lattice.edges = new_ridges
        self.lattice.coordinates = new_coords

        self._rescale_width()
        self._rescale_height()


## csbinproc function (MATLAB analogue)
'''
% [X,Y] = CSBINPROC(XP,YP,N) This function generates a
% homogeneous 2-D Poisson process. Conditional on the number
% of data points N, this is uniformly distributed over the
% study region. The vectors XP and YP correspond to the x and y
% vertices of the study region. The vectors X and Y contain
% the locations for the generated events.
%
% See also CSPOISSPROC, CSCLUSTPROC, CSINHIBPROC, CSSTRAUSPROC
% W. L. and A. R. Martinez, 9/15/01
% Computational Statistics Toolbox
'''

def csbinproc(xp, yp, n):
    x = []
    y = []
    i = 0
    # find the maximum and the minimum for a 'box' around the region. 
    # Will generate uniform on this,and throw out those points that are not inside the region.
    minx = min(xp)
    maxx = max(xp)
    miny = min(yp)
    maxy = max(yp)
    
    pgon = box(minx,miny,maxx,maxy)
    
    cx = maxx - minx
    cy = maxy - miny
    while i < n:
        xt = np.random.uniform(0,1,1)*cx + minx
        yt = np.random.uniform(0,1,1)*cy + miny
        k = pgon.contains(Point(xt,yt))
        if k==True:
            # it is in the region!
            x.append(xt.tolist())
            y.append(yt.tolist())
            i = i+1
    return x,y

###########################################################################################
def disorder_delta_voronoi(delta_ratio,n,H,L):
    #Defines the enlosing area:
    A = H*L            #Basal area
    #Generate vertices for the regions - Units in mm
    rx = np.array([0, L, L, 0, 0])
    ry = np.array([0, 0, H, H, 0])  
   
    #(!!)
    #Defines the geometric disorder parameters based on desired delta
    r = math.sqrt((2*A)/(math.sqrt(3)*n))  #Defines r_hex distance in perfect order
    s = delta_ratio*r    # (!!) Defines minimum inhibition distance
    
    '''
    Now we move into the simple sequential inhibition script:
    '''
    #Generate the first event:
    X = []
    dist = []
    fx,fy = csbinproc(rx,ry,1)
    X.append(fx[0]+fy[0])

    j = 1
    #Generate subsequent events:
    while j < n:
        sx, sy = csbinproc(rx,ry,1)
        S_j = sx[0]+sy[0]
        xt = np.concatenate([np.array([S_j]),X])

        #Find distance between the events:
        dist = sci.distance.pdist(xt, 'euclidean')

        #Find distance between candidate event and others that have been 
        #generated already, and make sure no less than inhibition dist s.
        bool_dist = dist <=s
        output = np.where(bool_dist)[0]

        if not output.tolist():
            j = j+1
            X.append(S_j) 

        #Printing progress of loop:
        sys.stdout.write("\r{0} %".format(np.round(((j/n)*100),2)))
        sys.stdout.flush()
    coords = np.array(X)
    
    #Verify that indeed all points in X are no closer than inhibition distance:
    dist_check = sci.distance.pdist(X, 'euclidean')
    delhat = min(dist_check)
    print("")
    print("delhat = ", np.round(delhat,5), "vs. s =",np.round(s,5))
    if delhat > s:
        print("Min. distance check passed!")
        print("Actual delta value = {one}".format(one=delhat/r))
    else:
        print("check failed, please re-run!")
    return coords 

    
# Generates a bounded voronoi diagram with finite regions in the bounding box
def bounded_voronoi(points, bounding_box):
    # Select points inside the bounding box
    i = in_box(points, bounding_box)

    # Mirror points left, right, above, and under to provide finite regions for the
    # edge regions of the bounding box
    points_center = points[i, :]

    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])

    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])

    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])

    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])

    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)

    # Compute FULL Voronoi
    vor = sp.spatial.Voronoi(points)

    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                       bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)

    ridges = []
    for ridge in vor.ridge_vertices:
        flag = True
        for index in ridge:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                       bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if ridge != [] and flag:
            ridges.append(ridge)

    vor.filtered_points = points_center
    vor.filtered_regions = regions
    vor.filtered_ridges = ridges

    return vor

# Returns a new np.array of points that within the bounding_box
def in_box(points, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= points[:, 0],
                                         points[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= points[:, 1],
                                         points[:, 1] <= bounding_box[3]))

# Finds the centroid of a region. First and last point should be the same.
def centroid_region(vertices):
    # Polygon's signed area
    A = 0
    # Centroid's x
    C_x = 0
    # Centroid's y
    C_y = 0
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return np.array([[C_x, C_y]])

# Performs x iterations of loyd's algorithm to calculate a centroidal vornoi diagram
def generate_CVD(points, iterations, bounding_box):
    p = copy.copy(points)

    for i in range(iterations):
        vor = bounded_voronoi(p, bounding_box)
        centroids = []

        for region in vor.filtered_regions:
            # grabs vertices for the region and adds a duplicate
            # of the first one to the end
            vertices = vor.vertices[region + [region[0]], :]
            centroid = centroid_region(vertices)
            centroids.append(list(centroid[0, :]))

        p = np.array(centroids)

    return bounded_voronoi(p, bounding_box)
