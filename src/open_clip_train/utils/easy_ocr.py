from dataclasses import dataclass

import cv2
import numpy as np
from easyocr import Reader
from omegaconf import OmegaConf
from PIL import Image


class CustomReader(Reader):
    "overwrite to deal with the color conversion issue https://github.com/JaidedAI/EasyOCR/issues/1186"

    def readtext(
        self,
        img,
        img_cv_grey,
        decoder="greedy",
        beamWidth=5,
        batch_size=1,
        workers=0,
        allowlist=None,
        blocklist=None,
        detail=1,
        rotation_info=None,
        paragraph=False,
        min_size=20,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        filter_ths=0.003,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=1.0,
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        y_ths=0.5,
        x_ths=1.0,
        add_margin=0.1,
        threshold=0.2,
        bbox_min_score=0.2,
        bbox_min_size=3,
        max_candidates=0,
        output_format="standard",
    ):
        """
        Parameters:
        image: file path or numpy-array or a byte stream object
        """
        horizontal_list, free_list = self.detect(
            img,
            min_size=min_size,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            slope_ths=slope_ths,
            ycenter_ths=ycenter_ths,
            height_ths=height_ths,
            width_ths=width_ths,
            add_margin=add_margin,
            reformat=False,
            threshold=threshold,
            bbox_min_score=bbox_min_score,
            bbox_min_size=bbox_min_size,
            max_candidates=max_candidates,
        )
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        result = self.recognize(
            img_cv_grey,
            horizontal_list,
            free_list,
            decoder,
            beamWidth,
            batch_size,
            workers,
            allowlist,
            blocklist,
            detail,
            rotation_info,
            paragraph,
            contrast_ths,
            adjust_contrast,
            filter_ths,
            y_ths,
            x_ths,
            False,
            output_format,
        )

        return result


@dataclass
class TestEasyOCRConfig:
    input_path: str = "/home/junyuc/dataset/TextOCR/test_images/004d6068f43bc436.jpg"


def main():
    cfg: TestEasyOCRConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TestEasyOCRConfig), OmegaConf.from_cli())
    )
    reader = CustomReader(["en"])
    img = np.array(Image.open(cfg.input_path))
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = reader.readtext(img, img_grey)

    for bbox, text, score in result:
        top_left_corner = tuple(map(int, bbox[0]))
        bottom_right_corner = tuple(map(int, bbox[2]))
        color = (0, 0, 255)  # BGR
        thickness = 2
        cv2.rectangle(img, top_left_corner, bottom_right_corner, color, thickness)
    Image.fromarray(img).save("easy_ocr_result.jpg")


if __name__ == "__main__":
    main()

"""
python -m efficientvit.apps.utils.easy_ocr input_path=/home/junyuc/dataset/TextOCR/test_images/004d6068f43bc436.jpg
python -m efficientvit.apps.utils.easy_ocr input_path=/home/junyuc/dataset/TextOCR/test_images/01d3c94699ea4613.jpg
python -m efficientvit.apps.utils.easy_ocr input_path=/home/junyuc/dataset/TextOCR/test_images/05d6b805449696c3.jpg
"""
