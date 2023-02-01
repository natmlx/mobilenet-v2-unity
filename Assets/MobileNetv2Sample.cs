/*
*   MobileNet v2
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Examples {

    using UnityEngine;
    using UnityEngine.UI;
    using NatML.Vision;
    using NatML.VideoKit;

    public class MobileNetv2Sample : MonoBehaviour {

        [Header(@"Camera Preview")]
        public VideoKitCameraManager cameraManager;

        [Header(@"Predictions")]
        public Text labelText;
        public Text scoreText;

        private MLEdgeModel model;
        private MobileNetv2Predictor predictor;

        private async void Start () {
            // Create the model
            model = await MLEdgeModel.Create("@natsuite/mobilenet-v2");
            // Create the predictor
            predictor = new MobileNetv2Predictor(model);
            // Start listening for camera stream
            cameraManager.OnCameraFrame.AddListener(OnCameraFrame);
        }

        private void OnCameraFrame (CameraFrame cameraFrame) {
            // Predict
            var (label, score) = predictor.Predict(cameraFrame);
            // Display
            labelText.text = label;
            scoreText.text = $"{score:0.##}";
        }

        void OnDisable () {
            // Stop listening for camera frames
            cameraManager.OnCameraFrame.RemoveListener(OnCameraFrame);
            // Dispose model
            model?.Dispose();
        }
    }
}