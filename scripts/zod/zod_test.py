from matplotlib import pyplot as plt
from zod import ZodSequences
import zod.constants as constants
from zod.constants import AnnotationProject
from zod.visualization.object_visualization import overlay_object_3d_box_on_image

dataset_root = "/mnt/sdb/zod"
version = "full"  # "mini" or "full"

zod_sequences = ZodSequences(dataset_root=dataset_root, version=version)
validation_sequences = zod_sequences.get_split(constants.VAL)

seq = zod_sequences[list(validation_sequences)[0]]

print(f"Number of camera frames: {len(seq.info.get_camera_frames())}")
print(f"Timespan: {(seq.info.end_time - seq.info.start_time).total_seconds()}")

key_camera_frame = seq.info.get_key_camera_frame()

try:
    annotations = seq.get_annotation(AnnotationProject.OBJECT_DETECTION)
except:
    annotations = []

image = key_camera_frame.read()

for annotation in annotations:
    if annotation.box3d:
        image = overlay_object_3d_box_on_image(
            image, annotation.box3d, seq.calibration, color=(0, 100, 0), line_thickness=10
        )

plt.axis("off")
plt.imshow(image)
plt.savefig("/home/erik/Downloads/something.png")
