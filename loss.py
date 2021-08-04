import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1,self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(
            predictions[..., 2:6], target[..., 2:6])
        iou_b2 = intersection_over_union(
            predictions[..., 7:11], target[..., 2:6])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 1].unsqueeze(3)

        box_predictions = exists_box*(
            (
                bestbox*predictions[..., 7:11]
                + (1-bestbox)*predictions[..., 2:6]
            )
        )

        box_targets = exists_box*target[..., 2:6]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4])*torch.sqrt(
            torch.abs(box_predictions[..., 2:4]+1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2))

        pred_box = (
            bestbox*predictions[..., 6:7]+(1-bestbox)*predictions[..., 1:2]
        )

        object_loss = self.mse(
            torch.flatten(exists_box*pred_box),
            torch.flatten(exists_box*target[..., 1:2])
        )

        no_object_loss = self.mse(
            torch.flatten((1-exists_box)*predictions[..., 1:2], start_dim=1),
            torch.flatten((1-exists_box)*target[..., 1:2], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1-exists_box)*predictions[..., 1:2], start_dim=1),
            torch.flatten((1-exists_box)*target[..., 1:2], start_dim=1)
        )

        class_loss = self.mse(
            torch.flatten(exists_box*predictions[..., :1], end_dim=-2),
            torch.flatten(exists_box*target[..., :1], end_dim=-2)
        )

        loss = (
            self.lambda_coord*box_loss
            + object_loss
            + self.lambda_noobj*no_object_loss
            + class_loss
        )

        return loss
