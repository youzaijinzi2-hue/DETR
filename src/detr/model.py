"""画像からクラス・ボックス予測までをつなぐDETR本体。"""

from collections.abc import Sequence
from typing import NotRequired, TypedDict

from torch import Tensor, nn

from detr.backbone import ResNet50Backbone
from detr.batching import batch_images
from detr.decoder import ObjectQueries, TransformerDecoder
from detr.encoder import TransformerEncoder
from detr.input_projection import InputProjection
from detr.position_encoding import PositionEmbeddingSine
from detr.prediction_head import PredictionHead


class Prediction(TypedDict):
    pred_logits: Tensor
    pred_boxes: Tensor


class DETROutput(Prediction):
    aux_outputs: NotRequired[list[Prediction]]


class DETR(nn.Module):
    """ResNet-50とEncoder/Decoder、共有予測ヘッドを組み合わせる。

    入力は各要素が(3, H, W)の浮動小数点画像。同じdtype・deviceを使う。
    モデル自体は画素値の標準化・リサイズを行わない。ImageNet重みを使う
    場合は[0, 1]画像をmean=(0.485, 0.456, 0.406)、
    std=(0.229, 0.224, 0.225)で標準化してから渡す。
    pretrained_backbone=TrueでもTransformerと予測ヘッドは未学習。
    """

    def __init__(
        self,
        num_classes: int,
        *,
        num_queries: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        pretrained_backbone: bool = False,
        aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(d_model)
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.input_projection = InputProjection(self.backbone.num_channels, d_model)
        self.encoder = TransformerEncoder(
            d_model, nhead, dim_feedforward, dropout, num_encoder_layers
        )
        self.object_queries = ObjectQueries(num_queries, d_model)
        self.decoder = TransformerDecoder(
            d_model, nhead, dim_feedforward, dropout, num_decoder_layers
        )
        self.prediction_head = PredictionHead(num_classes, d_model)
        self.aux_loss = aux_loss
        # 公式Transformerと同じ初期化。Backboneやヘッドの重みは変更しない。
        for module in (self.encoder, self.decoder):
            for parameter in module.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, images: Sequence[Tensor]) -> DETROutput:
        """最終予測と、任意で最後以外のDecoder層の予測を返す。

        pred_logits: (B, N, num_classes+1)。最後のクラスがno-object。
        pred_boxes: (B, N, 4)。各画像のパディング前サイズに対するcxcywh。
        aux_outputs: aux_loss=True時、最後以外の層のPredictionのリスト。
        ここでは予測のみを返し、マッチング・損失計算は行わない。
        """
        pixels, padding_mask = batch_images(images)
        features, feature_mask = self.backbone(pixels, padding_mask)
        tokens, token_mask = self.input_projection(features, feature_mask)
        positions = self.position_encoding(feature_mask).to(dtype=tokens.dtype)
        memory = self.encoder(tokens, positions, token_mask)
        targets, query_positions = self.object_queries(batch_size=pixels.shape[0])
        decoded = self.decoder(targets, query_positions, memory, positions, token_mask)
        predictions = self.prediction_head(decoded)

        output: DETROutput = {
            "pred_logits": predictions["pred_logits"][-1],
            "pred_boxes": predictions["pred_boxes"][-1],
        }
        if self.aux_loss:
            output["aux_outputs"] = [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(
                    predictions["pred_logits"][:-1], predictions["pred_boxes"][:-1]
                )
            ]
        return output
