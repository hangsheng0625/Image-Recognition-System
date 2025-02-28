import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

/**
 * Custom hook for managing model selection, loading, and embedding operations
 * Handles model loading, embedding fetching, and provides functions for image analysis
 */
export default function useModelSelector() {
  const [model, setModel] = useState(null);
  const [embeddings, setEmbeddings] = useState([]);
  const [modelLoading, setModelLoading] = useState(true);
  const [selectedModelKey, setSelectedModelKey] = useState('mobilenet');
  const [generatingEmbeddings, setGeneratingEmbeddings] = useState(false);
  
  // Available model configurations
  const availableModels = [
    { key: 'mobilenet', name: 'MobileNet v2' },
    // Add other models if implemented
  ];

  // Load model when component mounts or model selection changes
  useEffect(() => {
    async function loadModel() {
      try {
        setModelLoading(true);
        
        // Unload previous model if exists to free memory
        if (model) {
          // In a real app you might need to properly dispose of the model
        }
        
        // Load the selected model
        let loadedModel;
        if (selectedModelKey === 'mobilenet') {
          loadedModel = await mobilenet.load({
            version: 2,
            alpha: 1.0
          });
        }
        // Add other model loading logic for different models
        
        setModel(loadedModel);
        console.log(`${selectedModelKey} model loaded successfully`);
      } catch (error) {
        console.error('Failed to load model:', error);
      } finally {
        setModelLoading(false);
      }
    }
    
    loadModel();
  }, [selectedModelKey]);

  // Load embeddings when model changes or component mounts
  useEffect(() => {
    async function loadEmbeddings() {
      try {
        // Get the appropriate embeddings filename based on selected model
        const filename = `${selectedModelKey}_embeddings.json`;
        console.log(`Attempting to load embeddings from: ${filename}`);
        
        // Import the embeddings directly
        // This works with Vite/webpack and assumes the JSON files are in src/data
        const embeddingsModule = await import(`../data/${filename}`);
        const data = embeddingsModule.default || embeddingsModule;
        
        console.log(`Loaded ${data.length} embeddings from ${filename}`);
        setEmbeddings(data);
      } catch (error) {
        console.error('Error loading embeddings:', error);
        setEmbeddings([]);
      }
    }
    
    if (!modelLoading && model) {
      loadEmbeddings();
    }
  }, [selectedModelKey, model, modelLoading]);

  /**
   * Extract embedding features from an image element using the loaded model
   * @param {HTMLImageElement} imgElement - The image to process
   * @returns {Promise<number[]>} Array of embedding features
   */
  const getImageEmbedding = async (imgElement) => {
    if (!model) {
      throw new Error('Model not loaded');
    }
    
    // Convert image to tensor
    const tensor = tf.browser.fromPixels(imgElement)
      .resizeBilinear([224, 224]) // Resize to model input size
      .div(255.0)                 // Normalize to [0,1]
      .expandDims(0);             // Add batch dimension
      
    // Get embedding (penultimate layer activation)
    // For MobileNet, use infer with second param true to get the embedding
    const embedding = model.infer(tensor, true);
    
    // Convert to array
    const features = Array.from(await embedding.data());
    
    // Clean up tensors to prevent memory leaks
    tensor.dispose();
    embedding.dispose();
    
    return features;
  };

  /**
   * Generate embeddings for all images (normally would be done server-side)
   * This would typically call an API endpoint that runs your Node.js script
   */
  const generateEmbeddings = async () => {
    try {
      setGeneratingEmbeddings(true);
      
      // In a real app, this would be an API call to your backend
      // that runs the generate-embeddings.cjs script
      
      // Mock implementation - in reality you wouldn't do this client-side
      console.log('Generating embeddings would be a server-side operation');
      
      // After generation, reload the embeddings
      const filename = `${selectedModelKey}_embeddings.json`;
      const embeddingsModule = await import(`../data/${filename}?timestamp=${Date.now()}`);
      const data = embeddingsModule.default || embeddingsModule;
      
      setEmbeddings(data);
      return true;
    } catch (error) {
      console.error('Error generating embeddings:', error);
      return false;
    } finally {
      setGeneratingEmbeddings(false);
    }
  };

  return {
    model,
    selectedModelKey,
    setSelectedModelKey,
    isModelLoading: modelLoading,
    embeddings,
    isGeneratingEmbeddings: generatingEmbeddings,
    generateEmbeddings,
    getImageEmbedding,
    availableModels
  };
}