#!/usr/bin/env python3
"""
Inferencia de YOLOv4-leaky (formato YOLOv3-style output) usando VART directo.
Evita vitis-ai-library por completo -- solo usa vart + xir (libvart 2.5.0),
que ya sabemos consistente en esta Kria.

Uso:
    python3 run_yolov4_vart.py <ruta_al_xmodel> <ruta_a_imagen.jpg>
"""
import sys
import subprocess
import numpy as np
import cv2
import xir
import vart

DPU_APP_NAME = "kv260-smartcam"  # ajustar si se usa otro overlay

# --- Parámetros tomados del yolov4_leaky_416_tf.prototxt ---
INPUT_SIZE = 416
NUM_CLASSES = 80
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.6
# 9 anchors, agrupados de 3 en 3: [chicos (52x52), medianos (26x26), grandes (13x13)]
ANCHORS = np.array([
    [10, 13], [16, 30], [33, 23],       # para el mapa 52x52
    [30, 61], [62, 45], [59, 119],      # para el mapa 26x26
    [116, 90], [156, 198], [373, 326],  # para el mapa 13x13
], dtype=np.float32)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensure_dpu_loaded(app_name=DPU_APP_NAME):
    """Verifica que el overlay del DPU esté activo antes de tocar VART.
    Si no lo está, ofrece cargarlo -- evita que el proceso se cuelgue
    intentando hablar con un DPU que no existe en el FPGA todavía."""
    try:
        result = subprocess.run(
            ["sudo", "xmutil", "listapps"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        print(f"ADVERTENCIA: no se pudo verificar el estado del overlay ({e}).")
        print("Si el script se congela a continuación, es probable que el DPU no esté cargado.")
        return

    active = False
    for line in result.stdout.splitlines():
        if app_name in line:
            # la última columna (Active_slot) es distinta de "-1" cuando está cargado
            active = "-1" not in line.split(app_name)[-1].split(",")[0]
            break

    if active:
        return

    print(f"El overlay '{app_name}' no está activo. Cargarlo AHORA es necesario "
          "para continuar -- si sigues sin cargarlo, el script podría congelarse "
          "al intentar acceder al DPU.")
    resp = input(f"¿Cargar '{app_name}' ahora? [s/N]: ").strip().lower()
    if resp != "s":
        print("Cancelado por el usuario. Corre 'sudo xmutil loadapp "
              f"{app_name}' manualmente y vuelve a intentar.")
        sys.exit(1)

    load = subprocess.run(
        ["sudo", "xmutil", "loadapp", app_name],
        capture_output=True, text=True, timeout=30,
    )
    print(load.stdout.strip())
    if load.returncode != 0 or "Error" in load.stdout:
        print(f"No se pudo cargar el overlay automáticamente: {load.stdout}")
        print(f"Intenta manualmente: sudo xmutil unloadapp && sudo xmutil loadapp {app_name}")
        sys.exit(1)


def get_child_subgraph_dpu(graph):
    root = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()
    return [c for c in children if c.has_attr("device") and c.get_attr("device").upper() == "DPU"]


def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")
    orig_h, orig_w = img.shape[:2]

    # Letterbox: redimensiona manteniendo aspect ratio y rellena con gris (114)
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    norm = rgb.astype(np.float32) / 255.0  # scale: 0.00390625 del prototxt
    return img, norm, scale, pad_x, pad_y


def decode_output(output, anchors, input_size, conf_threshold):
    """output: [1, gh, gw, 3*(5+num_classes)] ya en float (descuantizado)"""
    _, gh, gw, _ = output.shape
    output = output.reshape(1, gh, gw, 3, 5 + NUM_CLASSES)

    grid_x, grid_y = np.meshgrid(np.arange(gw), np.arange(gh))
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, gh, gw, 1, 2).astype(np.float32)

    box_xy = (sigmoid(output[..., 0:2]) + grid) / np.array([gw, gh], dtype=np.float32)
    box_wh = (np.exp(output[..., 2:4]) * anchors.reshape(1, 1, 1, 3, 2)) / input_size
    obj_conf = sigmoid(output[..., 4:5])
    class_probs = sigmoid(output[..., 5:])

    scores = obj_conf * class_probs
    mask = scores.max(axis=-1) > conf_threshold

    boxes_xy = box_xy[mask]
    boxes_wh = box_wh[mask]
    scores_masked = scores[mask]

    boxes = np.concatenate([boxes_xy - boxes_wh / 2, boxes_xy + boxes_wh / 2], axis=-1)
    class_ids = scores_masked.argmax(axis=-1)
    confidences = scores_masked.max(axis=-1)
    return boxes, confidences, class_ids


def nms(boxes, confidences, class_ids, iou_threshold):
    if len(boxes) == 0:
        return [], [], []
    idxs = cv2.dnn.NMSBoxes(
        boxes.tolist(), confidences.tolist(), CONF_THRESHOLD, iou_threshold
    )
    if len(idxs) == 0:
        return [], [], []
    idxs = np.array(idxs).flatten()
    return boxes[idxs], confidences[idxs], class_ids[idxs]


def main():
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} <xmodel> <imagen.jpg>")
        sys.exit(1)

    xmodel_path, image_path = sys.argv[1], sys.argv[2]

    ensure_dpu_loaded()

    graph = xir.Graph.deserialize(xmodel_path)
    subgraphs = get_child_subgraph_dpu(graph)
    assert len(subgraphs) == 1, f"Se esperaba 1 subgrafo DPU, se encontraron {len(subgraphs)}"
    runner = vart.Runner.create_runner(subgraphs[0], "run")

    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    orig_img, norm_img, scale, pad_x, pad_y = preprocess(image_path)

    in_fixpos = input_tensors[0].get_attr("fix_point")
    in_scale = 2 ** in_fixpos
    input_data = np.clip(np.round(norm_img * in_scale), -128, 127).astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)

    output_buffers = []
    for ot in output_tensors:
        shape = tuple(ot.dims)
        output_buffers.append(np.empty(shape, dtype=np.int8, order="C"))

    job_id = runner.execute_async([input_data], output_buffers)
    runner.wait(job_id)

    all_boxes, all_conf, all_cls = [], [], []
    for ot, buf in zip(output_tensors, output_buffers):
        fixpos = ot.get_attr("fix_point")
        out_float = buf.astype(np.float32) / (2 ** fixpos)
        gh = out_float.shape[1]
        if gh == 13:
            anchors = ANCHORS[6:9]
        elif gh == 26:
            anchors = ANCHORS[3:6]
        else:
            anchors = ANCHORS[0:3]
        boxes, conf, cls = decode_output(out_float, anchors, INPUT_SIZE, CONF_THRESHOLD)
        all_boxes.append(boxes)
        all_conf.append(conf)
        all_cls.append(cls)

    all_boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.empty((0, 4))
    all_conf = np.concatenate(all_conf, axis=0) if all_conf else np.empty((0,))
    all_cls = np.concatenate(all_cls, axis=0) if all_cls else np.empty((0,))

    # boxes están en formato [x1,y1,x2,y2] normalizado 0-1 sobre el canvas 416x416
    boxes_pix = all_boxes * INPUT_SIZE
    boxes_xywh = np.stack([
        boxes_pix[:, 0], boxes_pix[:, 1],
        boxes_pix[:, 2] - boxes_pix[:, 0], boxes_pix[:, 3] - boxes_pix[:, 1],
    ], axis=-1) if len(boxes_pix) else boxes_pix

    final_boxes, final_conf, final_cls = nms(boxes_xywh, all_conf, all_cls, NMS_THRESHOLD)

    print(f"\n{len(final_boxes)} detección(es):\n")
    out_img = orig_img.copy()
    for box, conf, cls_id in zip(final_boxes, final_conf, final_cls):
        x, y, w, h = box
        # deshacer el letterbox para volver a coordenadas de la imagen original
        x1 = (x - pad_x) / scale
        y1 = (y - pad_y) / scale
        x2 = (x + w - pad_x) / scale
        y2 = (y + h - pad_y) / scale
        cls_name = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else str(int(cls_id))
        print(f"  {cls_name:15s}  conf={conf:.3f}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
        cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(out_img, f"{cls_name} {conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_path = "resultado_deteccion.jpg"
    cv2.imwrite(out_path, out_img)
    print(f"\nImagen con detecciones guardada en: {out_path}")


if __name__ == "__main__":
    main()
