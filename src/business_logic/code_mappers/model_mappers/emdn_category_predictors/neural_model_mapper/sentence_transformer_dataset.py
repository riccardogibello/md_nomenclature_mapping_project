from sentence_transformers import SentenceTransformer
from torch import long, no_grad, device, cuda
from torch.utils.data import Dataset
from torch import tensor

from src.business_logic.utilities.os_utilities import convert_to_device


class SentenceTransformerDataset(Dataset):
    def __init__(
            self,
            _texts,
            _labels,
            _transformer_model: SentenceTransformer
    ):
        super().__init__()

        self.texts = _texts
        self.labels = _labels
        # Move the model to the GPU, if available
        self.transformer_model = convert_to_device(
            _transformer_model,
            'cuda'
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        This method is required to be implemented in order to be able to use the PyTorch DataLoader.
        It returns the text embedding and the label of the given index.

        :param idx: The index of the text and label to be returned.

        :return: The text embedding and the label of the given index.
        """
        # Get the text that corresponds to the given index
        text = self.texts[idx]


        # Get the label and ensure it is cast to an integer
        label = tensor(self.labels[idx], dtype=long)

        # Set the boolean to avoid tracking the gradients
        with no_grad():
            # Check if GPU is available and set the device accordingly
            _device = device(
                "cuda"
                if cuda.is_available()
                else "cpu"
            )
            # Encode the text using the SentenceTransformer model
            text_embedding = self.transformer_model.encode(
                text,
                convert_to_tensor=True,
                device=_device
            )

        return text_embedding, label
