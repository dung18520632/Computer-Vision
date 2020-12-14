import argparse
import cv2
import numpy as np
import math
import os
from objloader_simple import *
import time


MIN_MATCHES = 10
DEFAULT_COLOR = (0, 0, 0)
def main():
    homography = None 
    resizeFactor = 0.25
    resizeFactorImage = 0.50
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
    orb = cv2.xfeatures2d.SIFT_create()
    # create BFMatcher object based on hamming distance  
    bf = cv2.cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    # load the reference surface that will be searched in the video stream
    # model = cv2.imread('reference/Test1.jpg', 0)
    # # Compute model keypoints and its descriptors
    # kp_model, des_model = orb.detectAndCompute(model, None)
    # # Load 3D model from OBJ file
    # obj = OBJ( 'models/'+objectName+'.obj', swapyz=True)  
    # # init video capture
    # cap = cv2.VideoCapture(0)
    # url="http://192.168.137.143:8080/video"
    # cap.open(url)
    # while True:
    #     # read the current frame
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Unable to capture video")
    #         return 
    #     # find and draw the keypoints of the frame
    #     kp_frame, des_frame = orb.detectAndCompute(frame, None)
    #     # match frame descriptors with model descriptors
    #     matches = bf.match(des_model, des_frame)
    #     # sort them in the order of their distance
    #     # the lower the distance, the better the match
    #     matches = sorted(matches, key=lambda x: x.distance)

    #     # compute Homography if enough matches are found
    #     if len(matches) > MIN_MATCHES:
    #         # differenciate between source points and destination points
    #         src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    #         dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    #         # compute Homography
    #         homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #         if args.rectangle:
    #             # Draw a rectangle that marks the found model in the frame
    #             h, w = model.shape
    #             pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #             # project corners into frame
    #             dst = cv2.perspectiveTransform(pts, homography)
    #             # connect them with lines  
    #             frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
    #         # if a valid homography matrix was found render cube on model plane
    #         if homography is not None:
    #             try:
    #                 # obtain 3D projection matrix from homography matrix and camera parameters
    #                 projection = projection_matrix(camera_parameters, homography)  
    #                 # project cube or model
    #                 frame = render(frame, obj, projection, model,objectName, False)
    #                 #frame = render(frame, model, projection)
    #             except:
    #                 pass
    #         # draw first 10 matches.
    #         if args.matches:
    #             frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
    #         # show result
    #         cv2.imshow('frame', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #     else:
    #         print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

    # cap.release()
    # cv2.destroyAllWindows()
    # return 0
    
    objectNames = ["House","Tree"]
    models = []
    modelPipe = cv2.imread('reference/Test1.jpg', 0)
    modelPipe = cv2.resize(modelPipe,None,fx=resizeFactor,fy=resizeFactor, interpolation = cv2.INTER_AREA)
    models.append(modelPipe)
    modelCafe = cv2.imread('reference/Test2.jpg', 0)
    modelCafe = cv2.resize(modelCafe,None,fx=resizeFactor,fy=resizeFactor, interpolation = cv2.INTER_AREA) 
    models.append(modelCafe)

    obj = []
    obj.append(OBJ('models/house.obj', swapyz=True))
    obj.append(OBJ('models/tree.obj', swapyz=True))

    cap = cv2.VideoCapture(0)
    # url="http://192.168.137.213:8080/video"
    # cap.open(url)
    while True:
        start = time.time()
        ret,frame = cap.read()
        originalFrame = frame.copy()
        height, width = originalFrame.shape[:2]

        if not ret:
            print("Error with Video")
            break

        for i,model in enumerate(models):
            try:
                kp_model, des_model = orb.detectAndCompute(model, None)
                kp_frame, des_frame = orb.detectAndCompute(frame, None)

                matches = bf.match(des_model, des_frame)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > MIN_MATCHES:
                    src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if homography is not None:
                        try:
                            projection = projection_matrix(camera_parameters, homography)  
                            frame = render(frame, obj[i], projection, model, objectNames[i], False)
                        except:
                            pass
            except:
                pass
        cv2.putText(originalFrame,"Original",(int(width*0.05),int(height*0.95)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(150,0,0),5, cv2.LINE_8)
        cv2.putText(frame,"Augmented Reality",(int(width*0.05),int(height*0.95)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,150,0),5, cv2.LINE_8)

        combinedImage = np.zeros((height,width*2,3), np.uint8)
        combinedImage[0:height,0:width] = originalFrame
        combinedImage[0:height,width:2*width] = frame
        cv2.line(combinedImage,(width,0),(width,height),(30,30,30),3,8)
        cv2.imshow("Live", combinedImage)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    print("")
    cap.release()
    cv2.destroyAllWindows()

def render(img, obj, projection, model,objectName, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    # vertices = obj.vertices
    # if objectName=='alduin-dragon':
    #     scale_matrix = np.eye(3) * 0.65
    # else:
    #     scale_matrix = np.eye(3) * 3
    
    # h, w = model.shape
    # numberOfPoints = len(obj.faces)
    # for counter,face in enumerate(obj.faces):
    #     face_vertices = face[0]
    #     points = np.array([vertices[vertex - 1] for vertex in face_vertices])
    #     points = np.dot(points, scale_matrix)
    #     # render model in the middle of the reference surface. To do so,
    #     # model points must be displaced
    #     points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
    #     dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
    #     imgpts = np.int32(dst)
    #     if color is False and counter < len(obj.faces):
    #         if(counter < 1*numberOfPoints/32):
    #             cv2.fillConvexPoly(img, imgpts, (89,71,53))
    #         elif(counter < 2*numberOfPoints/8):
    #             cv2.fillConvexPoly(img, imgpts, (53,80,66))
    #         elif(counter < 13*numberOfPoints/16):
    #             cv2.fillConvexPoly(img, imgpts, (50,77,95))
    #         else:
    #             cv2.fillConvexPoly(img, imgpts, (92,67,82))
    #     else:
    #         color = hex_to_rgb(face[-1])
    #         color = color[::-1]  # reverse
    #         cv2.fillConvexPoly(img, imgpts, color)

    # return img
    vertices = obj.vertices

    if(objectName == "Tree"):
        scale_matrix = np.eye(3) * 0.03
    elif(objectName == "House"):
        scale_matrix = np.eye(3) * 0.65

    h, w = model.shape

    numberOfPoints = len(obj.faces)
    for counter,face in enumerate(obj.faces):
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        cv2.imshow('anh',img)
        if color is False and counter < len(obj.faces):
            if(objectName == "Tree"):
                if(counter < numberOfPoints/1.5):
                    cv2.fillConvexPoly(img, imgpts, (27,211,50))
                else:
                    cv2.fillConvexPoly(img, imgpts, (33,67,101))
            elif(objectName == "House"):
                if(counter < 1*numberOfPoints/32):
                    cv2.fillConvexPoly(img, imgpts, (226,219,50))
                elif(counter < 2*numberOfPoints/8):
                    cv2.fillConvexPoly(img, imgpts, (250,250,250))
                elif(counter < 13*numberOfPoints/16):
                    cv2.fillConvexPoly(img, imgpts, (28,186,249))
                else:
                    cv2.fillConvexPoly(img, imgpts, (14,32,130))

        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)

    return img
def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)

    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
# TODO jgallostraa -> add support for model specification
#parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
