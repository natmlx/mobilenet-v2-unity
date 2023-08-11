# MobileNet v2
Realtime object classification machine learning model optimized for mobile applications.

## Installing MobileNet v2
Add the following items to your Unity project's `Packages/manifest.json`:
```json
{
  "scopedRegistries": [
    {
      "name": "NatML",
      "url": "https://registry.npmjs.com",
      "scopes": ["ai.natml"]
    }
  ],
  "dependencies": {
    "ai.natml.vision.mobilenet-v2": "1.0.4"
  }
}
```

## Classifying an Image
First, create the MobileNet v2 predictor:
```csharp
// Create the MobileNet v2 predictor
var predictor = await MobileNetv2Predictor.Create();
```

Then make predictions on images:
```csharp
// Given an image...
Texture2D image = ...;
// Classify the image
MobileNetv2Predictor.Label result = predictor.Predict(image);
// Use the result
Debug.Log($"Model predicted {result.label} with confidence {result.confidence}");
```

## Requirements
- Unity 2022.3+

## Quick Tips
- Join the [NatML community on Discord](https://natml.ai/community).
- Discover more ML models on [NatML Hub](https://hub.natml.ai).
- See the [NatML documentation](https://docs.natml.ai/unity).
- Contact us at [hi@natml.ai](mailto:hi@natml.ai).

Thank you very much!