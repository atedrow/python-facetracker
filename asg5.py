import platform, sys, os, math, subprocess
import cv2
from collections import deque

# helper function to computer distance between features
def featureDist(feat1, feat2):

    return math.sqrt(((feat1.x - feat2.x)**2) + ((feat1.y - feat2.y)**2))

def featureSizeDiff(feat1, feat2):
    size1 = feat1.w * feat1.h
    size2 = feat2.w * feat2.h
    return math.sqrt((size1 - size2)**2)

# class for facial features: face, mouth, Right eye, Left eye
class Feature:

    def __init__(self, x, y, w, h, roi):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        # region of interest, 
        # area we are detecting feature in
        self.roi = roi

    def getROI(self):
        return self.roi

    def setROI(self, n_roi):
       self.roi = n_roi


# class for a single face with facial features 
# has method to display face on video frame with openCV
class Face:

    def __init__(self, face, Reye, Leye, smile, num):
        self.face = face
        self.Reye = Reye
        self.Leye = Leye
        self.smile = smile
        self.num = num

    def displayFace(self):
        # tuples for easier shape creataion
        face_tup_pos = (self.face.x, self.face.y)
        face_tup_siz = (self.face.x  + self.face.w, self.face.y + self.face.h)
        smile_tup_pos = (self.smile.x, self.smile.y)
        smile_tup_siz = (self.smile.x + self.smile.w, self.smile.y + self.smile.h)
        Reye_tup_pos = (self.Reye.x, self.Reye.y)
        Reye_tup_siz = (self.Reye.x + self.Reye.w, self.Reye.y + self.Reye.h)
        Leye_tup_pos = (self.Leye.x, self.Leye.y)
        Leye_tup_siz = (self.Leye.x + self.Leye.w, self.Leye.y + self.Leye.h)
        # for placeing text in proper region of interest 
        x = self.face.x
        y = self.face.y
        h = self.face.h
        w = self.face.w
        # display face box
        cv2.rectangle(self.face.roi, face_tup_pos, face_tup_siz, (255,255,255),3)
        cv2.putText(
                self.face.roi,
                'Face_'+str(self.num),
                (face_tup_pos[0]-10,face_tup_pos[1]-10),
                1,1,(255,255,255),1
        )

        # display smile box
        cv2.rectangle(self.smile.roi, smile_tup_pos, smile_tup_siz, (0,255,0),1)
        cv2.putText(
                self.smile.roi,
                'smile',
                (smile_tup_pos[0],smile_tup_pos[1]-5),
                1,1,(0,255,0),1
        )

        # display Right Eye box
        cv2.rectangle(self.Reye.roi, Reye_tup_pos, Reye_tup_siz, (0,0,255),1)
        cv2.putText(
                self.Reye.roi,
                'R_eye',
                (Reye_tup_pos[0], Reye_tup_pos[1]-5),
                1,1,(0,0,255),1
        )

        # display Left  Eye box
        cv2.rectangle(self.Leye.roi, Leye_tup_pos, Leye_tup_siz, (0,0,255),1)
        cv2.putText(
                self.Leye.roi,
                'L_eye',
                (Leye_tup_pos[0], Leye_tup_pos[1]-5),
                1,1,(0,0,255),1
        )


# class for holding all faces captured within a frame
# initalization ensures cascade file paths and cascade classifiers
# are set for each facial feature 
# has method to detect faces and features in the current frame
# it also filters false positives and tracks features through 
# collection.deque old_faces. uses tracking to persists old faces
# if detection is temporarily lost
class Faces:

    def __init__(self):
        # set cascade paths
        # update cascade paths for current system if incorrect 
        self.cascPath = "./haarcascades/haarcascade_frontalface_alt2.xml"
        self.ReyePath = "./haarcascades/haarcascade_righteye_2splits.xml"
        self.LeyePath = "./haarcascades/haarcascade_lefteye_2splits.xml"
        self.smilePath = "./haarcascades/haarcascade_smile.xml"
        # set cascades
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.ReyeCascade = cv2.CascadeClassifier(self.ReyePath)
        self.LeyeCascade = cv2.CascadeClassifier(self.LeyePath)
        self.smileCascade = cv2.CascadeClassifier(self.smilePath)


    def getFeatures(self, frame, grey, old_faces):
        # deque of all detected faces
        all_faces = deque()
        past_faces = deque()
        # initalize temporary feature variables 
        face = Feature(0,0,0,0,0)
        Reye = Feature(0,0,0,0,0)
        Leye = Feature(0,0,0,0,0)
        smile = Feature(0,0,0,0,0)

        # get last detected faces if they exsist 
        if len(old_faces) > 0:
            past_faces = old_faces.pop()
            old_faces.append(past_faces)
        # Detect faces save to self.face
        self.faces = self.faceCascade.detectMultiScale(
                    grey,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    flags=cv2.CASCADE_SCALE_IMAGE
            )
        numFaces = 0
        # if no faces detected display last face
        if len(self.faces) == 0:
            if len(old_faces) == 0:
                pass
            else:
                for past_face in past_faces:
                    x = past_face.face.x
                    y = past_face.face.y
                    h = past_face.face.h
                    w = past_face.face.w
                    # ajust roi to current frame
                    past_face.face.setROI(frame)
                    past_face.Reye.setROI(frame[y:y+(h-int(h/2)), x+int(w*(1/5)):x+(w-int(w*(1/5)))])
                    past_face.Leye.setROI(frame[y:y+(h-int(h/2)), x+int(w*(1/5)):x+(w-int(w*(1/5)))])
                    past_face.smile.setROI(frame[y+int(h/2):y+h, x:x+w])
                    # display old face
                    past_face.displayFace()

        else:
            for (x, y, w, h) in self.faces:
                if w > 25:
                    numFaces += 1
                    face = Feature(x,y,w,h,frame)
                    # set the roi for features 
                    roi_grey = grey[y:y+h, x:x+h]
                    roi_color = frame[y:y+h, x:x+h]
                    roi_grey_eye = grey[y:y+(h-int(h/2)), x+int(w*(1/5)):x+(w-int(w*(1/5)))]
                    roi_color_eye = frame[y:y+(h-int(h/2)), x+int(w*(1/5)):x+(w-int(w*(1/5)))]
                    roi_grey_mouth = grey[y+int(h/2):y+h, x:x+w]
                    roi_color_mouth = frame[y+int(h/2):y+h, x:x+w]

                    self.smiles = self.smileCascade.detectMultiScale(
                        roi_grey_mouth,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    for (sx, sy, sw, sh) in self.smiles:
                        smile = Feature(sx,sy,sw,sh,roi_color_mouth)

                    if len(old_faces) > 0:
                        for past_face in past_faces:
                            if past_face.num == numFaces:
                                if featureDist(smile, past_face.smile) > 80 or \
                                   (smile.x == 0 and smile.y == 0):
                                    smile = past_face.smile
                                    smile.setROI(roi_color_mouth)

                    self.Reyes = self.ReyeCascade.detectMultiScale(
                            roi_grey_eye,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    for (sx, sy, sw, sh) in self.Reyes:
                        if sx > int(len(roi_grey_eye)/2):
                            Reye = Feature(sx,sy,sw,sh,roi_color_eye)

                    if len(old_faces) > 0:
                        for past_face in past_faces:
                            if past_face.num == numFaces:
                                if featureDist(Reye, past_face.Reye) > 80 or \
                                   (Reye.x == 0 and Reye.y == 0):
                                    Reye = past_face.Reye
                                    Reye.setROI(roi_color_eye)

                    self.Leyes = self.LeyeCascade.detectMultiScale(
                            roi_grey_eye,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    for (sx, sy, sw, sh) in self.Leyes:
                        if sx < int(len(roi_grey_eye)/2):
                            Leye = Feature(sx,sy,sw,sh,roi_color_eye)

                    if len(old_faces) > 0:
                        for past_face in past_faces:
                            if past_face.num == numFaces:
                                if featureDist(Leye, past_face.Leye) > 80 or \
                                   Leye == (0,0,0,0):
                                    Leye = past_face.Leye
                                    Leye.setROI(roi_color_eye)

                    new_face = Face(face, Reye, Leye, smile, numFaces)
                    all_faces.append(new_face)
                    new_face.displayFace()
                old_faces.append(all_faces)


def main(argv):
    # get video file
    cap = cv2.VideoCapture("Video.mp4")
    if not cap:
        print("Error video file failed to open")

    # get video size
    max_size = (1280, 720)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # resize image 
    if size[0] > max_size[0] or size[1] > max_size[1]:
        if(size[0] > size[1]):
            ratio = size[0] / size[1]
            diff = size[0] - max_size[0]
            size = (size[0] - diff, size[1] - int(diff / ratio))

        else:
            ratio = size[1] / size[0]
            diff = size[1] - max_size[1]
            size = (size[0] - int(diff / ratio), size[1] - diff)
    # get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    # set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mov', fourcc, fps, size, True)
    face_que = deque()
    # loop through frames
    while(cap.isOpened()):
        # read frame
        ret, frame = cap.read()
        if ret == True:
            # write frame
            r_frame = cv2.resize(
                    frame,
                    size,
                    fx=0,fy=0,
                    interpolation = cv2.INTER_CUBIC
            )
            grey = cv2.cvtColor(r_frame, cv2.COLOR_BGR2GRAY)
            curr_faces = Faces()
            curr_faces.getFeatures(r_frame, grey, face_que)
            # delete oldest face from que
            if len(face_que) > 300:
                face_que.popleft()

            out.write(r_frame)
            # display frame
            cv2.imshow('frame', r_frame)
            # stop transcodeing with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # release video capture
    cap.release()
    out.release()
    # destroy window
    cv2.destroyAllWindows()

    # use ffmpeg to convert output to mp4
    ffmpeg = "/bin/ffmpeg"
    command = [ffmpeg, "-i", "output.mov", "output.mp4"]
    if subprocess.run(command).returncode == 0:
        print("File saved as output.mp4")
        rm = "/bin/rm"
        command = [rm, "-f", "output.mov"]
    else:
        print("Issue converting to mp4")

if __name__ == '__main__':
    main(sys.argv)
