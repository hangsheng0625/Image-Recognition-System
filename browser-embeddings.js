// browser-embeddings.js
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as tf from '@tensorflow/tfjs'; // Use the regular tfjs package
import '@tensorflow/tfjs-backend-cpu'; // Use CPU backend
import * as mobilenet from '@tensorflow-models/mobilenet';
import { createCanvas, loadImage } from 'canvas'; // Need to install: npm install canvas

// ESM-friendly __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ASSETS_DIR = path.join(__dirname, 'src', 'assets');
const OUTPUT_FILE = path.join(__dirname, 'src', 'data', 'embeddings.json');

// Utility to load and resize an image using canvas
async function processImage(filePath) {
  // Load the image using canvas
  const image = await loadImage(filePath);
  
  // Create a canvas with MobileNet's expected dimensions
  const canvas = createCanvas(224, 224);
  const ctx = canvas.getContext('2d');
  
  // Draw and resize the image to the canvas
  ctx.drawImage(image, 0, 0, 224, 224);
  
  // Convert to tensor
  const tensor = tf.browser.fromPixels(canvas)
    .div(255) // Normalize [0,1]
    .expandDims(0); // Add batch dimension
    
  return tensor;
}

async function generateEmbeddings() {
  if (!fs.existsSync(ASSETS_DIR)) {
    console.error(`Assets directory not found at ${ASSETS_DIR}`);
    return;
  }

  // Set up TensorFlow.js to use CPU backend
  await tf.setBackend('cpu');
  
  // Load MobileNet
  const model = await mobilenet.load({
    version: 2,
    alpha: 1.0
  });

  // Find images in assets directory
  const imageFiles = fs.readdirSync(ASSETS_DIR)
    .filter(file => {
      const ext = path.extname(file).toLowerCase();
      return ['.jpg', '.jpeg', '.png', '.gif', '.bmp'].includes(ext);
    });

  const embeddings = [];

  for (const file of imageFiles) {
    console.log(`Processing: ${file}`);
    const fullPath = path.join(ASSETS_DIR, file);
    
    const tensor = await processImage(fullPath);
    // Get embedding from an intermediate layer
    const activation = model.infer(tensor, true);
    // Convert to JS array
    const features = Array.from(await activation.data());
    
    // Dispose to free memory
    tensor.dispose();
    activation.dispose();
    
    embeddings.push({
      id: path.basename(file, path.extname(file)),
      imageUrl: `/assets/${file}`, // Adjust to where your images are actually served
      features
    });
  }

  // Ensure the target directory exists
  const dataDir = path.dirname(OUTPUT_FILE);
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(embeddings, null, 2));
  console.log(`Saved embeddings for ${embeddings.length} images to ${OUTPUT_FILE}`);
}

// Run the script
generateEmbeddings()
  .then(() => console.log('Done generating embeddings!'))
  .catch(err => console.error(err));