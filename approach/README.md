# CAPdroid Approach

<p align="center">
<img src="../figures/overview.png" width="60%"/> 
</p>

Given an input GUI recording, we propose an automated approach to segment the recording into a sequence of clips based on user actions and subsequently localize the action positions to generate natural language descriptions.
Therefore, we divide into:

(i) the [Action Segmentation](action_segmentation.md) phase, which segments user actions from GUI recording into a sequence of clips,

(ii) the [Action Attribute Inference](action_attribute_inference.md) phase, which infers touch location, moving offset, and input text from action clips,

(iii) the [Description Generation](description_generation.md) phase, which utilizes the off-the-shelf GUI understanding models to generate high-level semantic descriptions.
