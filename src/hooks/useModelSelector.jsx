import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
// import * as vgg from '@tensorflow-models/vgg16'; // You'll need to install this
// import * as resnet from '@tensorflow-models/resnet50'; // You'll need to install this

// Models configuration with their loading functions
const MODEL_CONFIGS = {
  mobilenet: {
    name: 'MobileNet v2',
    load: async () => {
      return await mobilenet.load({
        version: 2,
        alpha: 1.0
      });
    },
    getEmbedding: (model, tensor) => {
      // For MobileNet, use infer with second param true to get the embedding
      return model.infer(tensor, true);
    }
  },
  vgg16: {
    name: 'VGG16',
    load: async () => {
      return await vgg.load();
    },
    getEmbedding: (model, tensor) => {
      // For VGG16, get the penultimate layer
      return model.predict(tensor);
    }
  },
  resnet50: {
    name: 'ResNet50',
    load: async () => {
      return await resnet.load();
    },
    getEmbedding: (model, tensor) => {
      // For ResNet50, get the penultimate layer
      return model.predict(tensor);
    }
  }
};

export default function useModelSelector() {
  const [selectedModelKey, setSelectedModelKey] = useState('mobilenet');
  const [model, setModel] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [embeddings, setEmbeddings] = useState([]);
  const [isGeneratingEmbeddings, setIsGeneratingEmbeddings] = useState(false);
  
  // Load the selected model
  useEffect(() => {
    let isActive = true;
    
    async function loadModel() {
      try {
        setIsModelLoading(true);
        // Clear previous model from memory if exists
        if (model) {
          model.dispose();
        }
        
        // Load the selected model
        const newModel = await MODEL_CONFIGS[selectedModelKey].load();
        
        if (isActive) {
          setModel(newModel);
          setIsModelLoading(false);
          
          // Clear embeddings when changing models
          setEmbeddings([]);
        }
      } catch (error) {
        console.error('Failed to load model:', error);
        if (isActive) {
          setIsModelLoading(false);
        }
      }
    }
    
    loadModel();
    
    // Clean up function
    return () => {
      isActive = false;
    };
  }, [selectedModelKey]);
  
  // Load embeddings from localStorage if available
  useEffect(() => {
    const storedEmbeddings = localStorage.getItem(`embeddings_${selectedModelKey}`);
    if (storedEmbeddings) {
      try {
        setEmbeddings(JSON.parse(storedEmbeddings));
      } catch (err) {
        console.error('Failed to parse stored embeddings:', err);
      }
    }
  }, [selectedModelKey]);
  
  // Function to generate embeddings for all images in src/assets
  const generateEmbeddings = async () => {
    if (!model || isGeneratingEmbeddings) return;
    
    try {
      setIsGeneratingEmbeddings(true);
      
      // This would normally be a server-side operation, but for client-side demo:
      // Fetch the list of images from a predefined JSON or API endpoint
      const response = await fetch('/api/listAssets');
      const imageList = await response.json();
      
      const newEmbeddings = [];
      
      for (const imageData of imageList) {
        try {
          // Load the image
          const img = new Image();
          img.crossOrigin = 'anonymous';
          await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
            img.src = imageData.imageUrl;
          });
          
          // Process image to tensor
          const tensor = tf.browser.fromPixels(img)
            .resizeBilinear([224, 224])
            .div(255)
            .expandDims(0);
            
          // Get embedding based on the selected model
          const activation = MODEL_CONFIGS[selectedModelKey].getEmbedding(model, tensor);
          const features = Array.from(await activation.data());
          
          // Clean up
          tensor.dispose();
          activation.dispose();
          
          // Add to embeddings array
          newEmbeddings.push({
            ...imageData,
            features
          });
        } catch (error) {
          console.error(`Failed to process ${imageData.imageUrl}:`, error);
        }
      }
      
      // Update state with new embeddings
      setEmbeddings(newEmbeddings);
      
      // Store in localStorage for persistence
      localStorage.setItem(`embeddings_${selectedModelKey}`, JSON.stringify(newEmbeddings));
      
    } catch (error) {
      console.error('Error generating embeddings:', error);
    } finally {
      setIsGeneratingEmbeddings(false);
    }
  };
  
  // Get embedding for a single image using current model
  const getImageEmbedding = async (img) => {
    if (!model) throw new Error('Model not loaded');
    
    const tensor = tf.browser.fromPixels(img)
      .resizeBilinear([224, 224])
      .div(255)
      .expandDims(0);
      
    // Use the model-specific embedding function
    const activation = MODEL_CONFIGS[selectedModelKey].getEmbedding(model, tensor);
    const embedding = await activation.data();
    
    // Clean up tensors
    tensor.dispose();
    activation.dispose();
    
    return Array.from(embedding);
  };
  
  // Available models for the dropdown
  const availableModels = Object.entries(MODEL_CONFIGS).map(([key, config]) => ({
    key,
    name: config.name
  }));

  return {
    model,
    selectedModelKey,
    setSelectedModelKey,
    isModelLoading,
    embeddings,
    isGeneratingEmbeddings,
    generateEmbeddings,
    getImageEmbedding,
    availableModels
  };
}