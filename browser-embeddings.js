import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as vgg from '@tensorflow-models/vgg16'; // You'll need to install this
import * as resnet from '@tensorflow-models/resnet50'; // You'll need to install this

// Model configurations with load and embedding extraction functions
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

// Load a specific model
export async function loadModel(modelKey = 'mobilenet') {
  if (!MODEL_CONFIGS[modelKey]) {
    throw new Error(`Unknown model: ${modelKey}`);
  }
  
  try {
    console.log(`Loading ${MODEL_CONFIGS[modelKey].name} model...`);
    return await MODEL_CONFIGS[modelKey].load();
  } catch (error) {
    console.error(`Failed to load ${MODEL_CONFIGS[modelKey].name}:`, error);
    throw error;
  }
}

// Get embeddings for an image using a specific model
export async function getEmbedding(model, img, modelKey = 'mobilenet') {
  if (!model) throw new Error('Model not provided');
  if (!MODEL_CONFIGS[modelKey]) throw new Error(`Unknown model: ${modelKey}`);
  
  const tensor = tf.browser.fromPixels(img)
    .resizeBilinear([224, 224])
    .div(255)
    .expandDims(0);
    
  // Use model-specific embedding function
  const activation = MODEL_CONFIGS[modelKey].getEmbedding(model, tensor);
  const embedding = await activation.data();
  
  // Clean up
  tensor.dispose();
  activation.dispose();
  
  return Array.from(embedding);
}

// Compare features using cosine similarity
export function cosineSimilarity(a, b) {
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

// Find similar images based on embedding
export function findSimilar(inputEmbedding, embeddings, topK = 5) {
  if (!embeddings || !embeddings.length) {
    console.error('No embeddings available');
    return [];
  }
  
  const similarityArray = embeddings.map(emb => {
    if (!emb.features || !Array.isArray(emb.features)) {
      console.error('Invalid embedding features for', emb.id);
      return { ...emb, similarity: 0 };
    }
    
    const similarity = cosineSimilarity(inputEmbedding, emb.features);
    return { ...emb, similarity };
  });
  
  // Sort by similarity descending and take topK
  return similarityArray
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
}

// Get list of available models for UI
export function getAvailableModels() {
  return Object.entries(MODEL_CONFIGS).map(([key, config]) => ({
    key,
    name: config.name
  }));
}