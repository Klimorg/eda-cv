from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as rt
import umap
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

from app.config import settings
from app.pydantic_models import ClusteringMode, EmbeddingsModel, Providers


class EmbeddingEngine:
    """_summary_"""

    def __init__(
        self, model: EmbeddingsModel, provider: Providers, *args, **kwargs
    ) -> None:
        """_summary_

        Args:
            model (EmbeddingsModel): _description_
            provider (Providers): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        super().__init__(*args, **kwargs)
        if model == EmbeddingsModel.resnet50v2:
            self.model = settings.resnet50v2
        else:
            raise NotImplementedError

        if provider == Providers.cpu:
            self.provider = [settings.cpu]
        else:
            logger.warning("GPU support not implemented, please chose cpu support.")
            raise NotImplementedError

        self.loaded_model = rt.InferenceSession(self.model, providers=self.provider)

    def infer(self, images_paths) -> np.ndarray:
        """_summary_

        Args:
            images_paths (_type_): _description_

        Returns:
            np.ndarray: _description_
        """

        images_list = [Image.open(image).resize((224, 224)) for image in images_paths]

        images = [np.asarray(image, dtype="float32") / 255 for image in images_list]

        images = np.reshape(images, (-1, 224, 224, 3))

        logits = self.loaded_model.run(["avg_pool"], {"input": images})
        return logits[0]

    def compute_clustering(
        self,
        logits: np.ndarray,
        mode: ClusteringMode,
    ) -> np.ndarray:
        """_summary_

        Args:
            logits (np.ndarray): _description_
            mode (ClusteringMode): _description_

        Returns:
            np.ndarray: _description_
        """

        if mode == ClusteringMode.tsne:
            clusters = TSNE(
                n_components=2,
                learning_rate="auto",
                init="random",
            ).fit_transform(logits)
        elif mode == ClusteringMode.umap:
            reducer = umap.UMAP()
            clusters = reducer.fit_transform(logits)

        return clusters

    def plot(
        self,
        logits: np.ndarray,
        images_paths: List[Path],
        timestamp: str,
        mode: ClusteringMode,
    ) -> Path:
        """_summary_

        Args:
            logits (np.ndarray): _description_
            images_paths (List[Path]): _description_
            timestamp (str): _description_
            mode (ClusteringMode): _description_

        Returns:
            Path: _description_
        """

        images_labels = [Path(image_path).parent.stem for image_path in images_paths]
        labels_dict = {
            label: idx for idx, label in enumerate(sorted(set(images_labels)))
        }
        tags = [labels_dict[image_label] for image_label in images_labels]

        N = len(set(images_labels))

        # define the colormap
        cmap = plt.cm.jet
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)

        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(logits[:, 0], logits[:, 1], c=tags, cmap=cmap)

        if mode == ClusteringMode.tsne:
            plt.title("tSNE scatterplot with ResNet50v2 preprocess")
        if mode == ClusteringMode.umap:
            plt.title("UMAP scatterplot with ResNet50v2 preprocess")

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=sorted(set(images_labels)),
            title="Labels",
        )

        saved_image_path = Path(
            f"{settings.scatterplots_dir}/tSNE_scatter_{timestamp}.png",
        ).resolve()

        plt.savefig(saved_image_path)

        return saved_image_path
