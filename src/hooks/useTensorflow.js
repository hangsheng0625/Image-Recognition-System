import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import embeddings from '../data/embeddings.json';

export default function useTensorFlow() {
  const [model, setModel] = useState(null);
  const [companyEmbeddings] = useState(embeddings);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadModel() {
      try {
        setLoading(true);
        // Ensure TensorFlow backend is initialized
        await tf.ready();
        console.log('TensorFlow backend ready');
        
        // Load the model
        const net = await mobilenet.load({
          version: 2,
          alpha: 1.0,
        });
        
        console.log('MobileNet model loaded successfully');
        setModel(net);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load model:', err);
        setError('Failed to load TensorFlow model');
        setLoading(false);
      }
    }
    
    loadModel();
    
    // Cleanup function
    return () => {
      // Nothing to clean up with browser version
    };
  }, []);

  return { model, companyEmbeddings, loading, error };
}