import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button, message } from 'antd';
import * as tf from '@tensorflow/tfjs';

export default function ImageUpload({ model, embeddings, setResults, setLoading }) {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [imageName, setImageName] = useState('');

  const { getRootProps, getInputProps } = useDropzone({
    accept: {'image/*': []},
    multiple: false,
    onDrop: async ([file]) => {
      if (!model) {
        message.error('Model not loaded yet');
        return;
      }
      
      try {
        setLoading(true);
        
        // Store the image preview and name
        setUploadedImage(URL.createObjectURL(file));
        setImageName(file.name);
        
        const img = await loadImage(file);
        const embedding = await getEmbedding(model, img);
        const results = findSimilar(embedding, embeddings);
        setResults(results);
      } catch (error) {
        console.error('Error processing image:', error);
        message.error('Error processing image');
      } finally {
        setLoading(false);
      }
    }
  });
  
  return (
    <div>
      <div {...getRootProps()} style={{
        border: '2px dashed #1890ff',
        borderRadius: '8px',
        padding: '2rem',
        textAlign: 'center',
        cursor: 'pointer',
        marginBottom: '2rem'
      }}>
        <input {...getInputProps()} />
        <Button type="primary" size="large">
          Upload Tile Image
        </Button>
        <p style={{ marginTop: '1rem', color: '#666' }}>
          Drag & drop or click to select
        </p>
      </div>
      
      {uploadedImage && (
        <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
          <h3>Uploaded Image:</h3>
          <img 
            src={uploadedImage}
            alt="Uploaded tile"
            style={{
              maxWidth: '100%',
              maxHeight: '200px',
              objectFit: 'contain',
              borderRadius: '4px',
              marginTop: '1rem'
            }}
          />
          <p style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>{imageName}</p>
        </div>
      )}
    </div>
  );
}

// Keep the existing helper functions
async function loadImage(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.src = e.target.result;
      img.onload = () => resolve(img);
    };
    reader.readAsDataURL(file);
  });
}

async function getEmbedding(model, img) {
  const tensor = tf.browser.fromPixels(img)
    .resizeBilinear([224, 224])
    .div(255)
    .expandDims(0);
    
  // For MobileNet, use infer with second param true to get the embedding
  const activation = model.infer(tensor, true);
  const embedding = await activation.data();
  
  // Clean up tensors
  tensor.dispose();
  activation.dispose();
  
  return Array.from(embedding);
}

function findSimilar(inputEmbedding, embeddings, topK = 5) {
  if (!embeddings || !embeddings.length) {
    console.error('No embeddings available');
    return [];
  }
  
  const similarityArray = embeddings.map(emb => {
    // Add more validation
    if (!emb.features || !Array.isArray(emb.features)) {
      console.error('Invalid embedding features for', emb.id);
      return { ...emb, similarity: 0 };
    }
    
    const similarity = cosineSimilarity(inputEmbedding, emb.features);
    
    // More detailed logging
    console.log(`Comparing to ${emb.id || 'unknown'}: similarity=${similarity.toFixed(4)}`);
    
    return { ...emb, similarity };
  });
  
  // Sort by similarity descending and take topK
  const results = similarityArray
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
  
  return results;
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) {
    console.error(`Invalid vectors for similarity calculation: a=${a?.length}, b=${b?.length}`);
    return 0;
  }
  
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}