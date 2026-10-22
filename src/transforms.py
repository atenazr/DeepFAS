"""Image transforms used by the pipeline."""

from torchvision import transforms

# ImageNet statistics (the DeepLabV3 backbone was pre-trained with them).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INV_MEAN = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
INV_STD = [1 / 0.229, 1 / 0.224, 1 / 0.225]


def get_image_transform(image_size: int = 520) -> transforms.Compose:
    """Resize to ``image_size`` x ``image_size``, convert to tensor in [0, 1], normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class InvTransform:
    """Undo the normalization of an image tensor (used for visualization only)."""

    def __init__(self, inverse_mean=INV_MEAN, inverse_std=INV_STD):
        self.inverse_transform = transforms.Normalize(mean=inverse_mean, std=inverse_std)

    def __call__(self, image):
        return self.inverse_transform(image)


class Binarize:
    """Pixels above ``threshold`` become 1, all others 0."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, mask):
        return (mask > self.threshold).float()


inv_transform = InvTransform()
binarize = Binarize(threshold=0.5)
