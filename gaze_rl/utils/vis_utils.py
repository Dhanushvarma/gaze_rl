"""
This file contains utility functions for visualizing image observations in the training pipeline.
These functions can be a useful debugging tool.
"""

from enum import Enum
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk")
plt.rc("text", usetex=True)
colors = sns.color_palette("colorblind", 10)


class Colors(Enum):
    LIME = (0, 255, 0)
    BLUE = (0, 165, 255)
    RED = (255, 0, 0)
    PURPLE = (120, 20, 100)
    YELLOW = (255, 255, 0)


RENDER_META = {
    "current_gaze": dict(
        color=Colors.LIME.value,
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=5,
    ),
    "goal": dict(
        color=Colors.BLUE.value,
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=5,
    ),
    "other_block": dict(
        color=Colors.PURPLE.value,
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=5,
    ),
    "all_sampled_subgoals": dict(
        color=Colors.YELLOW.value,
        markerType=cv2.MARKER_DIAMOND,
        markerSize=5,
        thickness=2,
    ),
    "selected_subgoal": dict(
        color=Colors.RED.value,
        markerType=cv2.MARKER_DIAMOND,
        markerSize=5,
        thickness=2,
    ),
}


def annotate_single_video(
    video: np.ndarray,
    annotations: Dict,
    label: str = "",
    subgoal_vars_per_dim: np.ndarray = None,
):
    # load a nice big readable font
    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 13)

    subgoal_dim_labels = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]

    if subgoal_vars_per_dim:
        # create a new image on the side and add plt.plot of subgoal vars
        fig, ax = plt.subplots(figsize=(10, 6))
        subgoal_vars_per_dim = np.array(subgoal_vars_per_dim)
        T, subgoal_dim = subgoal_vars_per_dim.shape
        min_var = subgoal_vars_per_dim.min()
        max_var = subgoal_vars_per_dim.max()

        sgv_lines = [
            ax.plot([], [], label=subgoal_dim_labels[dim_indx])[0]
            for dim_indx in range(subgoal_dim)
        ]
        ax.legend()

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Var")
        ax.set_title("Subgoal Var per Dimension")
        ax.set_xlim([0, T])
        ax.set_ylim([min_var, max_var])

    annotated_imgs = []

    for step, frame in enumerate(video):
        frame = Image.fromarray(frame)

        # add border on top of image
        extra_border_height = 100
        annotated_img = Image.new(
            "RGB",
            (frame.width, frame.height + extra_border_height),
            color=(255, 255, 255),
        )
        annotated_img.paste(frame, (0, extra_border_height))
        draw = ImageDraw.Draw(annotated_img)

        count = 0
        lines = []
        to_display = ""
        num_keys_per_line = 2

        for key, values in annotations.items():
            if isinstance(values[step], np.ndarray):
                values = np.round(values[step], 2)
                to_add = f"{key}: {values}  "
            elif isinstance(values[step], float) or isinstance(
                values[step], np.float32
            ):
                to_add = f"{key}: {values[step]:.5f}  "
            else:
                to_add = f"{key}: {values[step]}  "

            if count < num_keys_per_line:
                to_display += to_add
                count += 1
            else:
                lines.append(to_display)
                count = 1
                to_display = to_add

        # add the last line
        if lines:
            lines.append(to_display)

        # add label to beginning
        if label:
            lines.insert(0, label)

        for i, line in enumerate(lines):
            # make font size bigger
            draw.text((10, 10 + i * 20), line, fill="black", font=font)

        if subgoal_vars_per_dim:
            for dim_indx, sgv_line in enumerate(sgv_lines):
                sgv_line.set_data(range(step), subgoal_vars_per_dim[:step, dim_indx])

            # Redraw only the updated plot
            fig.canvas.draw_idle()

            # Convert plot to image only once at the desired size
            plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

            # add plot to image
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plot_img = Image.fromarray(plot_img)
            ratio = plot_img.width / plot_img.height
            plot_img = plot_img.resize((int(frame.height * ratio), frame.height))

            # Create the combined annotated image with subplot
            annotated_img_and_subplots = Image.new(
                "RGB",
                (annotated_img.width + plot_img.width, annotated_img.height),
                color=(255, 255, 255),
            )
            annotated_img_and_subplots.paste(annotated_img, (0, 0))
            annotated_img_and_subplots.paste(plot_img, (frame.width, 0))
            annotated_img = np.array(annotated_img_and_subplots)

        # convert to numpy array
        annotated_imgs.append(np.array(annotated_img))

    # close the plot
    plt.close("all")
    return np.array(annotated_imgs)
