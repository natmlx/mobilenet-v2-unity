/*
*   MobileNet v2
*   Copyright (c) 2022 NatML Inc. All Rights Reserved.
*/

namespace NatML.Examples {

    using UnityEngine;
    using UnityEngine.UI;
    using NatML.Devices;
    using NatML.Devices.Outputs;
    using NatML.Features;
    using NatML.Vision;

    public class MobileNetv2Sample : MonoBehaviour {

        [Header(@"Camera Preview")]
        public RawImage rawImage;
        public AspectRatioFitter aspectFitter;

        [Header(@"Predictions")]
        public Text labelText;
        public Text scoreText;

        private CameraDevice cameraDevice;
        private TextureOutput cameraTextureOutput;

        private MLModelData modelData;
        private MLModel model;
        private MobileNetv2Predictor predictor;

        async void Start () {
            // Request camera permissions
            var permissionStatus = await MediaDeviceQuery.RequestPermissions<CameraDevice>();
            if (permissionStatus != PermissionStatus.Authorized) {
                Debug.LogError(@"User did not grant camera permissions");
                return;
            }
            // Get the default camera device
            var query = new MediaDeviceQuery(MediaDeviceCriteria.CameraDevice);
            cameraDevice = query.current as CameraDevice;
            // Start the camera preview
            cameraDevice.previewResolution = (1280, 720);
            cameraTextureOutput = new TextureOutput();
            cameraDevice.StartRunning(cameraTextureOutput);
            // Display the camera preview
            var previewTexture = await cameraTextureOutput;
            rawImage.texture = previewTexture;
            aspectFitter.aspectRatio = (float)previewTexture.width / previewTexture.height;
            // Create the MobileNet v2 predictor
            Debug.Log("Fetching model from NatML...");
            modelData = await MLModelData.FromHub("@natsuite/mobilenet-v2");
            model = modelData.Deserialize();
            predictor = new MobileNetv2Predictor(model, modelData.labels);
        }

        void Update () {
            // Check that the predictor has been created
            if (predictor == null)
                return;
            // Create an image feature
            var imageFeature = new MLImageFeature(cameraTextureOutput.texture);
            (imageFeature.mean, imageFeature.std) = modelData.normalization;
            // Predict
            var (label, score) = predictor.Predict(imageFeature);
            // Display
            labelText.text = label;
            scoreText.text = $"{score:0.##}";
        }

        void OnDisable () {
            // Dispose model
            model?.Dispose();
        }
    }
}