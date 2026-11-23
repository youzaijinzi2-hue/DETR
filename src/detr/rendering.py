"""検出枠と、画像外の読みやすいクラス・スコア表示。"""

from PIL import Image, ImageDraw, ImageFont


def draw_predictions(image, predictions, ground_truth=None, *, scale=1):
    """左に画像、右に凡例を配置する。ボックスの入力座標は元画像のxyxy。"""
    image = image.resize(
        (image.width * scale, image.height * scale), Image.Resampling.NEAREST
    )
    font = ImageFont.load_default(size=18)
    small_font = ImageFont.load_default(size=14)
    probe = ImageDraw.Draw(image)
    names = [record["class_name"] for record in predictions]
    names += [record["class_name"] for record in ground_truth or []]
    panel_width = max(
        [260] + [int(probe.textlength(name, font=font)) + 90 for name in names]
    )
    panel_height = 72 + 38 * len(ground_truth or []) + 60 * max(len(predictions), 1)
    canvas = Image.new(
        "RGB", (image.width + panel_width, max(image.height, panel_height)), "white"
    )
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line((image.width, 0, image.width, canvas.height), fill=(210, 210, 210))
    left, row = image.width + 18, 18

    def box_and_tag(record, tag, color, line_width):
        x1, y1, x2, y2 = [value * scale for value in record["box_xyxy"]]
        x1, x2 = [min(max(value, 0), image.width - 1) for value in (x1, x2)]
        y1, y2 = [min(max(value, 0), image.height - 1) for value in (y1, y2)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        # 枠の上に、白文字の短い番号だけを描く。
        tag_width = int(draw.textlength(tag, font=small_font)) + 8
        tx = max(0, min(x1, image.width - tag_width))
        ty = max(0, y1 - 23)
        draw.rectangle((tx, ty, tx + tag_width, ty + 21), fill=color)
        draw.text((tx + 4, ty + 2), tag, fill="white", font=small_font)

    if ground_truth is not None:
        draw.text((left, row), "Ground truth", fill="black", font=font)
        row += 30
        for index, record in enumerate(ground_truth, 1):
            box_and_tag(record, f"G{index}", (0, 130, 60), 3)
            draw.text(
                (left, row),
                f"G{index}  {record['class_name']}",
                fill="black",
                font=font,
            )
            row += 38
    draw.text((left, row), "Predictions", fill="black", font=font)
    row += 30
    if not predictions:
        draw.text((left, row), "No detections", fill="black", font=font)
    for index, record in enumerate(predictions, 1):
        box_and_tag(record, f"P{index}", (210, 30, 30), 2)
        draw.text(
            (left, row), f"P{index}  {record['class_name']}", fill="black", font=font
        )
        draw.text(
            (left, row + 25), f"Score: {record['score']:.1%}", fill="black", font=font
        )
        row += 60
    return canvas
