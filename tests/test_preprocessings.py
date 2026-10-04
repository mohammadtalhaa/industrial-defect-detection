from PIL import Image

from src.preprocessing import ResizeKeepAspectAndPad, build_eval_transform


def test_resize_pad_output_shape():
    img = Image.new("RGB", (630, 230), color=(128, 128, 128))
    out = ResizeKeepAspectAndPad((256, 704))(img)
    assert out.size == (704, 256)


def test_eval_transform_tensor_shape():
    img = Image.new("RGB", (630, 230))
    t = build_eval_transform((256, 704))(img)
    assert t.shape == (3, 256, 704)
    assert t.dtype.is_floating_point