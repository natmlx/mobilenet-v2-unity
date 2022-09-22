# MobileNet v2

![classifier](Media/classifier.gif)

Realtime object classification machine learning model optimized for mobile applications. This package requires [NatML](https://github.com/natmlx/NatML).

## Classifying an Image
First, create the MobileNet v2 predictor:
```csharp
// Fetch the model data from NatML Hub
var modelData = await MLModelData.FromHub("@natsuite/mobilenet-v2");
// Deserialize the model
var model = modelData.Deserialize();
// Create the MobileNet v2 predictor
var predictor = new MobileNetv2Predictor(model, modelData.labels);
```

Then create an image feature:
```csharp
// Create image feature
Texture2D image = ...;  // Can also be a `WebCamTexture` or pixel buffer
var input = new MLImageFeature(image);
// Set the normalization and aspect mode
(input.mean, input.std) = modelData.normalization;
input.aspectMode = modelData.aspectMode;
```

Finally, classify the image:
```csharp
// Classify the image
(string label, float confidence) result = predictor.Predict(input);
```

## Requirements
- Unity 2020.3+
- [NatML 1.0.11+](https://github.com/natmlx/NatML)

## Quick Tips
- Discover more ML models on [NatML Hub](https://hub.natml.ai).
- See the [NatML documentation](https://docs.natml.ai/unity).
- Join the [NatML community on Discord](https://discord.gg/y5vwgXkz2f).
- Discuss [NatML on Unity Forums](https://forum.unity.com/threads/natml-machine-learning-runtime.1109339/).
- Contact us at [hi@natml.ai](mailto:hi@natml.ai).

Thank you very much!