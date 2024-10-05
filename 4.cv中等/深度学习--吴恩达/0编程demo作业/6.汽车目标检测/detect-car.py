import torch
import torch.nn.functional as F

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    """

    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = box_scores.argmax(dim=-1)
    box_class_scores = box_scores.max(dim=-1).values
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold"
    filtering_mask = box_class_scores >= threshold
    
    # Step 4: Apply the mask to scores, boxes, and classes
    scores = box_class_scores[filtering_mask]
    boxes = boxes[filtering_mask]
    classes = box_classes[filtering_mask]
    
    return scores, boxes, classes

# Example usage
box_confidence = torch.normal(mean=1, std=4, size=(19, 19, 5, 1))
boxes = torch.normal(mean=1, std=4, size=(19, 19, 5, 4))
box_class_probs = torch.normal(mean=1, std=4, size=(19, 19, 5, 80))

scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)

print("scores[2] =", scores[2].item())
print("boxes[2] =", boxes[2].numpy())
print("classes[2] =", classes[2].item())
print("scores.shape =", scores.shape)
print("boxes.shape =", boxes.shape)
print("classes.shape =", classes.shape)
