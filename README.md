# Project: Survivor AI - Can Survivor Bias Help Us to Identify Features or Improve Representation Learning?
This repository explores a novel machine learning concept: an algorithm inspired by the statistical phenomenon of survivor bias.

## Core Idea: Learning from the "Hard Positives"
The project's foundation comes from the classic WWII story about reinforcing airplanes. Instead of analyzing the planes that were shot down, researchers studied those that returned. The insight was that the areas without bullet holes (like the engine) were the most critical, as a hit there meant the plane wouldn't return at all.

We apply this logic to machine learning. Instead of solely focusing on misclassified samples ("failures"), we propose to learn from the "hard positives": samples that the model classifies correctly, but with very low confidence or high loss. These are the "survivors" that nearly failed.

Our hypothesis is that these "survivors" managed to be classified correctly because they relied on certain indispensable "core features". Our algorithm aims to automatically identify and reinforce these core features, making the model more robust, intelligent, and secure.

## Proof of Concept
1. Are core features exists?
 - use pretrained models to extract feature from cifar10
 - find out which samples are "back safely" and which are "barely make it"
 - - apply clustering to generate sample confidence
 - - seperate based on confidence
 -> resnet50+kmeans has really low prediction accuracy, use orginal resnet prediction to rank sample "damage" instead.


## The "Reinforce the Core" Algorithm
We introduce a new algorithmic approach that, unlike traditional approaches that "fortify the borders" (focusing on the decision boundary), aims to "reinforce the core" (strengthening the model's understanding of a class's essential identity).

### Algorithm Flow:
1. Filter for "Survivors": During training, identify the "hard positive" samples.
2. Define the "Blueprint": For each class, calculate a feature-space "prototype" that represents the most typical, ideal sample.
3. Identify Core vs. Flaw: Compare the internal features of each "survivor" to the class prototype.
 - "Bullet Hole" Features: Features that deviate significantly from the prototype. These are the noisy or distracting parts that made the sample "hard."
 - "Core" Features: Features that remain highly consistent with the prototype. These are the key elements that allowed the sample to "survive."
4. Intelligent Gradient Modulation: During backpropagation, modulate the gradient signals:
 - Amplify signals from "core features" to encourage the model to rely on these stable cues.
 - Suppress signals from "bullet hole features" to teach the model to ignore unreliable noise.

### Relationship to Existing Techniques
This method is designed to be complementary to, not a replacement for, existing technologies:
 - vs. Margin-based Loss (e.g., SVM, ArcFace): They focus on pushing classes apart at the decision boundary. We focus on improving the internal representation of each class. Combining them could lead to models that are both robust externally and coherent internally.
 - vs. Attention Mechanisms: Traditional attention is self-referential. Our approach is prototype-referential, introducing global, class-level information to guide the model's focus.
 - vs. Focal Loss: Focal Loss up-weights hard samples at the sample level. We go deeper, analyzing within the sample at the feature level to understand why a sample is hard and fix it at the source.

### Potential Applications
The core idea is highly versatile and can be extended to several domains:
1. Semantic Anomaly Detection: By modeling the "core" of what's normal, we can become highly sensitive to subtle anomalies that occur in these critical areas. Useful for industrial defect detection or financial fraud.
2. XAI & Representation Learning: Use the algorithm to discover the "core dimensions" a model uses for a specific task. This provides a deeper, prototype-based explanation of the model's reasoning and can lead to more efficient fine-tuning.
3. Privacy-Preserving AI:
 - Precision Anonymization: Identify the "core features" an AI uses for facial recognition and apply minimal perturbations only to those areas, preserving visual quality while breaking identification.
 - Information Enveloping: In an Encoder-Decoder architecture, explicitly allow the decoder to access only task-relevant "core features," obfuscating other potentially private information.

This project aims to formalize this idea, implement a proof-of-concept, and test its effectiveness on standard benchmarks. Contributions and ideas are welcome!
