import mediapipe as mp
import cv2
import argparse, os, random
import torch
import torch.nn.functional as F
from torchvision import transforms
import onnxruntime as ort
import pandas as pd
import numpy as np
from model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color


parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
parser.add_argument('--face', type=str, help='face detection file path. mediapipe face detector is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--onnx', type=str, help='path to ONNX model (auto-exported from model_weight if missing)', default='data/model.onnx')
parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
parser.add_argument('-save_vis', help='saves output as video', action='store_true')
parser.add_argument('-save_text', help='saves output as text', action='store_true')
parser.add_argument('-display_off', help='do not display frames', action='store_true')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MediaPipe face detection (replaces dlib CNN detector for ~10x CPU speedup)


def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right+bbox_left)/2.0
    cy = (bbox_bottom+bbox_top)/2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right-cx)*scale + cx
    bbox_left = (bbox_left-cx)*scale + cx
    bbox_top = (bbox_top-cy)*scale + cy
    bbox_bottom = (bbox_bottom-cy)*scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def export_onnx(model_weight, onnx_path):
    """Export PyTorch model to ONNX (runs once, then cached on disk)."""
    print(f"Exporting model to ONNX: {onnx_path} ...")
    model = model_static(model_weight)
    model.train(False)
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("ONNX export done.")


def run(video_path, face_path, model_weight, onnx_path, jitter, vis, display_off, save_text):
    # set up vis settings
    red = Color("red")
    colors = list(red.range_to(Color("green"),10))
    font = ImageFont.truetype("data/arial.ttf", 40)

    # set up video source
    if video_path is None:
        cap = cv2.VideoCapture(1)
        video_path = 'live.avi'
    else:
        cap = cv2.VideoCapture(video_path)

    # set up output file
    if save_text:
        outtext_name = os.path.basename(video_path).replace('.avi','_output.txt')
        f = open(outtext_name, "w")
    if vis:
        outvis_name = os.path.basename(video_path).replace('.avi','_output.avi')
        imwidth = int(cap.get(3)); imheight = int(cap.get(4))
        outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(5), (imwidth,imheight))

    # set up face detection mode
    if face_path is None:
        facemode = 'MEDIAPIPE'
    else:
        facemode = 'GIVEN'
        column_names = ['frame', 'left', 'top', 'right', 'bottom']
        df = pd.read_csv(face_path, names=column_names, index_col=0)
        df['left'] -= (df['right']-df['left'])*0.2
        df['right'] += (df['right']-df['left'])*0.2
        df['top'] -= (df['bottom']-df['top'])*0.1
        df['bottom'] += (df['bottom']-df['top'])*0.1
        df['left'] = df['left'].astype('int')
        df['top'] = df['top'].astype('int')
        df['right'] = df['right'].astype('int')
        df['bottom'] = df['bottom'].astype('int')

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        exit()

    if facemode == 'MEDIAPIPE':
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    frame_cnt = 0

    # set up data transformation
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load ONNX model (export from PyTorch weights on first run)
    if not os.path.exists(onnx_path):
        export_onnx(model_weight, onnx_path)
    print(f"Loading ONNX model from {onnx_path} ...")

    # Configure ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_cpu_mem_arena = True
    sess_options.enable_mem_pattern = True
    
    # Enable thread optimization for CPU
    if device == 'cpu':
        # Set intra-op threads (how many threads to use for an operation)
        sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)
        # Set inter-op threads (how many operations to run in parallel)
        sess_options.inter_op_num_threads = min(8, os.cpu_count() or 4)
    
    # Set up providers based on device
    if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # CUDA Provider options
        provider_options = [
            {
                'device_id': 0,  # Use first GPU
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB GPU Memory limit
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            },
            {}  # CPUExecutionProvider doesn't need options
        ]

        # session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers, provider_options=provider_options)
    else:
        session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])  
    
    input_name = session.get_inputs()[0].name
    print("ONNX model ready.")

    # video reading loop
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            height, width, channels = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_cnt += 1
            bbox = []
            if facemode == 'MEDIAPIPE':
                results = face_detector.process(frame)
                if results.detections:
                    for detection in results.detections:
                        rb = detection.location_data.relative_bounding_box
                        l = rb.xmin * width
                        t = rb.ymin * height
                        r = l + rb.width * width
                        b = t + rb.height * height
                        # expand a bit (same margins as before)
                        l -= (r-l)*0.2
                        r += (r-l)*0.2
                        t -= (b-t)*0.2
                        b += (b-t)*0.2
                        bbox.append([int(l), int(t), int(r), int(b)])
            elif facemode == 'GIVEN':
                if frame_cnt in df.index:
                    bbox.append([df.loc[frame_cnt,'left'],df.loc[frame_cnt,'top'],df.loc[frame_cnt,'right'],df.loc[frame_cnt,'bottom']])

            frame = Image.fromarray(frame)
            for b in bbox:
                face = frame.crop((b))
                img = test_transforms(face)
                img.unsqueeze_(0)
                if jitter > 0:
                    for i in range(jitter):
                        bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                        bj = [bj_left, bj_top, bj_right, bj_bottom]
                        facej = frame.crop((bj))
                        img_jittered = test_transforms(facej)
                        img_jittered.unsqueeze_(0)
                        img = torch.cat([img, img_jittered])

                # forward pass via ONNX Runtime
                output = session.run(None, {input_name: img.numpy()})[0]  # numpy array
                output = torch.from_numpy(output)
                if jitter > 0:
                    output = torch.mean(output, 0)
                score = torch.sigmoid(output).item()

                coloridx = min(int(round(score*10)),9)
                draw = ImageDraw.Draw(frame)
                drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
                draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=font)
                if save_text:
                    f.write("%d,%f\n"%(frame_cnt,score))

            if not display_off:
                frame = np.asarray(frame) # convert PIL image back to opencv format for faster display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow('',frame)
                if vis:
                    outvid.write(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        else:
            break

    if vis:
        outvid.release()
    if save_text:
        f.close()
    if key == ord('q'):
        print ('Exiting ...')
    cap.release()
    print ('DONE!')


if __name__ == "__main__":
    run(args.video, args.face, args.model_weight, args.onnx, args.jitter, args.save_vis, args.display_off, args.save_text)
