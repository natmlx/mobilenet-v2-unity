/* 
*   MobileNet v2
*   Copyright © 2023 NatML Inc. All Rights Reserved.
*/

namespace NatML.Vision {

    using System;
    using NatML.Features;
    using NatML.Internal;
    using NatML.Types;

    /// <summary>
    /// MobileNet v2 classification predictor.
    /// This predictor classifies an image with the ImageNet labels.
    /// </summary>
    public sealed class MobileNetv2Predictor : IMLPredictor<(string label, float confidence)> {

        #region --Client API--
        /// <summary>
        /// Create the MobileNet v2 classification predictor.
        /// </summary>
        /// <param name="model">MobileNet v2 model.</param>
        public MobileNetv2Predictor (MLEdgeModel model) => this.model = model as MLEdgeModel;

        /// <summary>
        /// Classify an image.
        /// </summary>
        /// <param name="inputs">Input image feature.</param>
        /// <returns>Output label with unnormalized confidence value.</returns>
        public (string label, float confidence) Predict (params MLFeature[] inputs) {
            // Check
            if (inputs.Length != 1)
                throw new ArgumentException(@"MobileNet v2 predictor expects a single feature", nameof(inputs));
            // Check type
            var input = inputs[0];
            if (!MLImageType.FromType(input.type))
                throw new ArgumentException(@"MobileNet v2 predictor expects an an array or image feature", nameof(inputs));
            // Apply image pre-processing
            if (input is MLImageFeature imageFeature) {
                (imageFeature.mean, imageFeature.std) = model.normalization;
                imageFeature.aspectMode = model.aspectMode;
            }
            // Predict
            var inputType = model.inputs[0];
            using var inputFeature = (input as IMLEdgeFeature).Create(inputType);
            using var outputFeatures = model.Predict(inputFeature);
            // Find label
            var logits = new MLArrayFeature<float>(outputFeatures[0]);
            var argMax = 0;
            for (int i = 1, ilen = logits.shape[1]; i < ilen; ++i)
                argMax = logits[0,i] > logits[0,argMax] ? i : argMax;
            // Return
            var result = (model.labels[argMax], logits[argMax]);
            return result;
        }
        #endregion


        #region --Operations--
        private readonly MLEdgeModel model;

        void IDisposable.Dispose () { } // Nop
        #endregion
    }
}