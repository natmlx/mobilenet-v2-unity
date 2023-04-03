/* 
*   MobileNet v2
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Vision {

    using System;
    using System.Threading.Tasks;
    using NatML.Features;
    using NatML.Internal;
    using NatML.Types;

    /// <summary>
    /// MobileNet v2 classification predictor.
    /// This predictor classifies an image with the ImageNet labels.
    /// </summary>
    public sealed class MobileNetv2Predictor : IMLPredictor<MobileNetv2Predictor.Label> {

        #region --Types--
        /// <summary>
        /// Class label with confidence score.
        /// </summary>
        public struct Label {

            /// <summary>
            /// Class label.
            /// </summary>
            public string label;

            /// <summary>
            /// Unnormalized confidence score.
            /// </summary>
            public float confidence;

            public void Deconstruct (out string label, out float confidence) {
                label = this.label;
                confidence = this.confidence;
            }
        }
        #endregion


        #region --Client API--
        /// <summary>
        /// Predictor tag.
        /// </summary>
        public const string Tag = "@natsuite/mobilenet-v2";

        /// <summary>
        /// Classify an image.
        /// </summary>
        /// <param name="inputs">Input image feature.</param>
        /// <returns>Output label with unnormalized confidence value.</returns>
        public Label Predict (params MLFeature[] inputs) {
            // Apply image pre-processing
            var input = inputs[0];
            if (input is MLImageFeature imageFeature) {
                (imageFeature.mean, imageFeature.std) = model.normalization;
                imageFeature.aspectMode = model.aspectMode;
            }
            // Predict
            using var inputFeature = (input as IMLEdgeFeature).Create(model.inputs[0]);
            using var outputFeatures = model.Predict(inputFeature);
            // Find label
            var logits = new MLArrayFeature<float>(outputFeatures[0]);
            var argMax = 0;
            for (int i = 1, ilen = logits.shape[1]; i < ilen; ++i)
                argMax = logits[0,i] > logits[0,argMax] ? i : argMax;
            // Return
            var result = new Label { label = model.labels[argMax], confidence = logits[argMax] };
            return result;
        }

        /// <summary>
        /// Dispose the predictor and release resources.
        /// </summary>
        public void Dispose () => model.Dispose();

        /// <summary>
        /// Create the MobileNet v2 predictor.
        /// </summary>
        /// <param name="configuration">Edge model configuration.</param>
        /// <param name="accessKey">NatML access key.</param>
        public static async Task<MobileNetv2Predictor> Create (
            MLEdgeModel.Configuration configuration = null,
            string accessKey = null
        ) {
            var model = await MLEdgeModel.Create(Tag, configuration, accessKey);
            var predictor = new MobileNetv2Predictor(model);
            return predictor;
        }
        #endregion


        #region --Operations--
        private readonly MLEdgeModel model;

        private MobileNetv2Predictor (MLEdgeModel model) => this.model = model as MLEdgeModel;
        #endregion
    }
}