import torch
import torch.nn as nn

from parrotletml.utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1.0
        self.lambda_noobj = 1.0  ## Reduce it 10 orig, 0.5 works fine
        self.lambda_obj = 1.0
        self.lambda_box = 10 ## Increase it 10 orig, 5.0 works fine

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]),
            (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # # Original Anchoors
        # anchors = anchors.reshape(1, 3, 1, 1, 2)

        # A hack to try for all different size images This Expects Scalled Anchors for each Prediction/Target
        anchors = anchors.reshape(predictions.shape[0], 3, 1, 1, 2)

        box_preds = torch.cat(
            [
                self.sigmoid(predictions[..., 1:3]),
                torch.exp(predictions[..., 3:5]) * anchors,
            ],
            dim=-1,
        )
        ious = intersection_over_union(
            box_preds[obj], target[..., 1:5][obj]
        ).detach()  ## Re-enabled it, if needed disabe detach
        object_loss = self.mse(
            self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj]
        )

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]),
            (target[..., 5][obj].long()),
        )

        # print("__________________________________")
        # print(self.lambda_box * box_loss)
        # print(self.lambda_obj * object_loss)
        # print(self.lambda_noobj * no_object_loss)
        # print(self.lambda_class * class_loss)
        # print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
