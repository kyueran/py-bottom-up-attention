import os
import io
import detectron2
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import cv2
import torch
import PIL.Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

def save_image(array, filename):
    array = np.uint8(np.clip(array, 0, 255))
    img = PIL.Image.fromarray(array)
    img.save(filename)

# Load VG Classes
data_path = 'data/genome/1600-400-20'

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())

MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs

cfg = get_cfg()
cfg.merge_from_file("./configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.WEIGHTS = "./weights/model_weights_attr.pkl"
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

NUM_OBJECTS = 36

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

def doit(raw_image):
    start_time = time.time()
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        # Predict classes and boxes for each proposal
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        end_time = time.time()  # End timing
        processing_time = end_time - start_time
        return instances, roi_features, processing_time

def process_image(image_path, output_image_folder):
    image_name = os.path.basename(image_path)
    im = cv2.imread(image_path)
    instances, features, processing_time = doit(im)

    # Save visualized image
    pred = instances.to('cpu')
    v = Visualizer(im[:, :, :], MetadataCatalog.get("vg"), scale=1.2)
    v = v.draw_instance_predictions(pred)
    output_image_path = os.path.join(output_image_folder, image_name)
    save_image(v.get_image()[:, :, ::-1], output_image_path)

    return features.numpy(), image_name, processing_time

def load_existing_data(npy_file_path):
    if os.path.exists(npy_file_path):
        data = np.load(npy_file_path, allow_pickle=True)
        return data['features'].tolist(), data['names'].tolist()
    return [], []

def save_data_incrementally(output_npy_path, image_features_list, image_names):
    structured_array = np.array([(image_features_list[i], image_names[i]) for i in range(len(image_features_list))],
                                dtype=[('features', 'O'), ('names', 'O')])
    np.save(output_npy_path, structured_array)

def process_images(image_folder, output_npy_path, output_image_folder, batch_size=32):
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.jpg', '.png', '.jpeg'))]

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    existing_features, existing_names = load_existing_data(output_npy_path)
    processed_images = set(existing_names)

    image_features_list = existing_features
    image_names = existing_names

    processing_times = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}")
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(process_image, image_path, output_image_folder): image_path for image_path in batch_paths if os.path.basename(image_path) not in processed_images}
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    features, image_name, processing_time = future.result()
                    image_features_list.append(features)
                    image_names.append(image_name)
                    processed_images.add(image_name)
                    processing_times.append(processing_time)
                except Exception as e:
                    print(f"Error processing image {futures[future]}: {e}")

        save_data_incrementally(output_npy_path, image_features_list, image_names)
    
    print(f"Average processing time per image: {np.mean(processing_times)} seconds")
    print(f"Total processing time: {np.sum(processing_times)} seconds")

if __name__ == '__main__':
    # Set multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn')

    #image_folder = '/home/kyueran/caption-generation/BLIP/annotation/output'  # Folder containing the images
    image_folder = '/home/kyueran/caption-generation/BLIP/merlion/'
    output_npy_path = '../shared_data/merlion.npy'  # Path to save the numpy file
    output_image_folder = '../shared_data/merlion'  # Folder to save the processed images

    process_images(image_folder, output_npy_path, output_image_folder)
