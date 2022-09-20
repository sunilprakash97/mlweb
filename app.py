from flask import Flask,render_template,Response
import cv2

app = Flask(__name__, 
            static_url_path = '', 
            static_folder = 'static')

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB = False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection [0,0,i,2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(50, 168, 151),  2)
            cv2.line (frame, (x1,y1), (x1+30, y1), (50, 168, 151), 6)
            cv2.line (frame, (x1,y1), (x1, y1+30), (50, 168, 151), 6)
            cv2.line (frame, (x2,y1), (x2-30, y1), (50, 168, 151), 6)
            cv2.line (frame, (x2,y1), (x2, y1+30), (50, 168, 151), 6)
            cv2.line (frame, (x1,y2), (x1+30, y2), (50, 168, 151), 6)
            cv2.line (frame, (x1,y2), (x1, y2-30), (50, 168, 151), 6)
            cv2.line (frame, (x2,y2), (x2-30, y2), (50, 168, 151), 6)
            cv2.line (frame, (x2,y2), (x2, y2-30), (50, 168, 151), 6)
    return frame, bboxs
    
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
faceProto = "faceDetector.pbtxt"
faceModel = "faceDetector_.pb"
ageProto = "age.prototxt"
ageModel = "age_model.caffemodel"
genderProto = "gender.prototxt"
genderModel = "gender_model.caffemodel"

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            frameFace, bboxes = faceBox(faceNet, frame)
            for bbox in bboxes:
                face = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                label = "{},{}".format(gender, age)
                cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            ret,buffer = cv2.imencode('.jpg',frameFace)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug = True)    