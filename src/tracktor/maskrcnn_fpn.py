from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.roi_heads import maskrcnn_inference
from torchvision.models.detection.roi_heads import paste_masks_in_image


class MaskRCNN_FPN(MaskRCNN):

    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(MaskRCNN_FPN, self).__init__(backbone, num_classes)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach(), detections['masks'].detach()

    def predict_boxes_and_masks(self, boxes):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        # using this command we already filter some boxes, but let keep it this way for now
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()

        mask_proposals = [pred_boxes]
        labels = [torch.ones((len(mask_proposals[0]),), dtype=torch.int64)]

        # predict masks
        mask_features = self.roi_heads.mask_roi_pool(self.features, mask_proposals,
                                                           self.preprocessed_images.image_sizes)
        mask_features = self.roi_heads.mask_head(mask_features)
        mask_logits = self.roi_heads.mask_predictor(mask_features)

        # inference
        masks_probs = maskrcnn_inference(mask_logits, labels)

        # postprocressing
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0],
                                  self.original_image_sizes[0])
        pred_masks = paste_masks_in_image(masks_probs[0], pred_boxes, self.original_image_sizes[0])

        return pred_boxes, pred_scores, pred_masks

    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
