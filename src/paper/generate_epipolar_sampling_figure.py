'''
python -m src.paper.generate_epipolar_sampling_figure
'''

from math import atan2
from pathlib import Path

import svg
import torch
from einops import rearrange, repeat
from jaxtyping import install_import_hook

from src.dataset import get_dataset
from src.dataset.dataset_re10k import DatasetRE10kCfg
from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
from src.geometry.epipolar_lines import lift_to_3d, project_rays
from src.geometry.projection import get_world_rays
from src.paper.common import encode_image, save_svg
from torch.utils.data import DataLoader
from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import to_tensor

SCENE = "train"
INDICES = (0, 10)
RAY_POSITION = (0.135, 0.44)  # xy
FIGURE_WIDTH = 240
IMAGE_SHAPE = (545, 980)
MARGIN = 4
NUM_MARKERS = 8
TEXT_SIZE = 5

def convert_poses(poses):
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics

def convert_images(images):
    torch_images = []
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes()))
        torch_images.append(to_tensor(image))
    return torch.stack(torch_images)

def generate_epipolar_sampling_figure():
    data_root = Path("datasets/...")
    chunk_files = list(data_root.glob("*.torch"))
    
    if not chunk_files:
        print(f"error: no .torch files found in {data_root}")
        return
    
    scene_found = False
    for chunk_file in chunk_files:
        print(f"check {chunk_file}")
        chunk = torch.load(chunk_file)
        
        for example in chunk:
            if SCENE in example["key"]:
                scene_found = True

                num_views = len(example["images"])
                
                if max(INDICES) >= num_views:
                    print(f"Wrong number of views in {example['key']}: {num_views}, expected at least {max(INDICES) + 1}")
                    return
                
                context_indices = torch.tensor(list(INDICES))
                context_images = convert_images([example["images"][i] for i in INDICES])
                extrinsics, intrinsics = convert_poses(example["cameras"])
                
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                    },
                    "scene": example["key"]
                }
    left_image, right_image = example["context"]["image"]

    # Project the epipolar line onto the other image.
    xy = torch.tensor(RAY_POSITION, dtype=torch.float32)
    origin, direction = get_world_rays(
        xy,
        example["context"]["extrinsics"][0],
        example["context"]["intrinsics"][0],
    )
    projection = project_rays(
        origin,
        direction,
        example["context"]["extrinsics"][1],
        example["context"]["intrinsics"][1],
    )

    # Create an SVG canvas.
    image_width = (FIGURE_WIDTH - MARGIN) / 2
    image_height = image_width * IMAGE_SHAPE[0] / IMAGE_SHAPE[1]
    fig = svg.SVG(
        width=FIGURE_WIDTH,
        height=image_height,
        elements=[],
        viewBox=svg.ViewBoxSpec(0, 0, FIGURE_WIDTH, image_height),
    )

    # Draw the left image.
    left_image = svg.Image(
        width=image_width,
        height=image_height,
        href=encode_image(left_image, "jpeg"),
    )
    fig.elements.append(left_image)

    # Draw the right image.
    right_image = svg.Image(
        width=image_width,
        height=image_height,
        href=encode_image(right_image, "jpeg"),
        x=image_width + MARGIN,
    )
    fig.elements.append(right_image)

    # Create a mask for the epipolar line.
    mask = svg.Mask(
        elements=[svg.Rect(width=FIGURE_WIDTH, height=image_height, fill="white")],
        id="mask",
    )
    fig.elements.append(mask)

    # Compute the epipolar line's position in pixel space.
    scale = torch.tensor((image_width, image_height), dtype=torch.float32)
    start = (projection["xy_min"] * scale).tolist()
    start[0] += image_width + MARGIN
    end = (projection["xy_max"] * scale).tolist()
    end[0] += image_width + MARGIN
    ray_xy = (xy * scale).tolist()

    def draw_samples(r: int | float, fill: str):
        for i in range(1, NUM_MARKERS):
            t = i / NUM_MARKERS
            sample = svg.Circle(
                cx=start[0] * t + (1 - t) * end[0],
                cy=start[1] * t + (1 - t) * end[1],
                r=r,
                fill=fill,
            )
            fig.elements.append(sample)

    # Draw the backers.
    epipolar_line_backer = svg.Line(
        x1=2 * start[0] - end[0],  # extra length that gets clipped
        y1=2 * start[1] - end[1],  # extra length that gets clipped
        x2=end[0],
        y2=end[1],
        stroke="white",
        stroke_width="4",
        mask="url(#mask)",
    )
    fig.elements.append(epipolar_line_backer)
    infinity_backer = svg.Circle(
        cx=end[0],
        cy=end[1],
        r=4,
        fill="white",
    )
    fig.elements.append(infinity_backer)
    ray_backer = svg.Circle(
        cx=ray_xy[0],
        cy=ray_xy[1],
        r=4,
        fill="white",
    )
    fig.elements.append(ray_backer)
    draw_samples(3.5, "white")

    # Draw the epipolar line.
    epipolar_line = svg.Line(
        x1=2 * start[0] - end[0],  # extra length that gets clipped
        y1=2 * start[1] - end[1],  # extra length that gets clipped
        x2=end[0],
        y2=end[1],
        stroke="#4263eb",
        stroke_width="2",
        mask="url(#mask)",
    )
    fig.elements.append(epipolar_line)
    infinity = svg.Circle(
        cx=end[0],
        cy=end[1],
        r=3,
        fill="#4263eb",
    )
    fig.elements.append(infinity)
    ray = svg.Circle(
        cx=ray_xy[0],
        cy=ray_xy[1],
        r=3,
        fill="#4263eb",
    )
    fig.elements.append(ray)
    draw_samples(2.5, "#4263eb")

    # Draw depth labels.
    angle = 90 - atan2(abs(start[1] - end[1]), abs(start[0] - end[0])) * 180 / 3.14159
    for i in range(NUM_MARKERS):
        t = i / NUM_MARKERS

        # Compute depth for the label.
        xyz = lift_to_3d(
            origin,
            direction,
            projection["xy_min"] * t + (1 - t) * projection["xy_max"],
            example["context"]["extrinsics"][1],
            example["context"]["intrinsics"][1],
        )
        depth = (xyz - example["context"]["extrinsics"][0, :3, 3]).norm(dim=-1)

        x = start[0] * t + (1 - t) * end[0]
        y = start[1] * t + (1 - t) * end[1]

        # Draw the backer.
        backer = svg.Rect(
            fill="white",
            width=21.5 if i > 0 else 15,
            height=TEXT_SIZE,
            rx=1,
            ry=1,
            transform=f"translate({x}, {y}) rotate({angle}) translate(5, {-TEXT_SIZE * 0.5})",  # noqa: E501
        )
        fig.elements.append(backer)

        # Draw the label. Transforms are applied right to left.
        extra = "" if i > 0 else "translate(0, -1)"
        label = svg.Text(
            elements=[f"{depth:.2f}" if i > 0 else "$\infty$"],
            transform=f"translate({x}, {y}) rotate({angle}) translate(5, {-TEXT_SIZE * 0.5}) translate(2, {TEXT_SIZE * 0.825}) {extra}",  # noqa: E501
            font_size=TEXT_SIZE,
        )
        fig.elements.append(label)

    # Draw the ray label.
    backer = svg.Rect(
        fill="white",
        x=ray_xy[0] - 10,
        y=ray_xy[1] - TEXT_SIZE - 6,
        width=20,
        height=TEXT_SIZE,
        rx=1,
        ry=1,
    )
    fig.elements.append(backer)
    label = svg.Text(
        elements=["Ray"],
        x=ray_xy[0],
        y=ray_xy[1] - 8,
        font_size=TEXT_SIZE,
        text_anchor="middle",
    )
    fig.elements.append(label)

    save_svg(fig, Path(f"epipolar_sampling{INDICES}.svg"))


if __name__ == "__main__":
    generate_epipolar_sampling_figure()