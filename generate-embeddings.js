import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { createCanvas, loadImage } from 'canvas';
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';

// Polyfill `require()` for CommonJS modules
const require = createRequire(import.meta.url);

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const ASSETS_DIR = path.join(__dirname, 'src', 'assets');
const OUTPUT_FILE = path.join(__dirname, 'src', 'data', 'embeddings.json');

// Create data directory if it doesn't exist
const dataDir = path.join(__dirname, 'src', 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

// Function to load an image and convert it to a tensor
async function processImage(imagePath) {
  try {
    // Load the image
    const image = await loadImage(imagePath);
    
    // Create a canvas and draw the image
    const canvas = createCanvas(224, 224);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, 224, 224);
    
    // Convert to tensor
    const tensor = tf.browser.fromPixels(canvas)
      .toFloat()
      .div(tf.scalar(255))
      .expandDims(0);
    
    return tensor;
  } catch (error) {
    console.error(`Error processing image ${imagePath}:`, error);
    return null;
  }
}

// Main function
async function generateEmbeddings() {
  console.log('Starting embeddings generation...');
  
  // Check if assets directory exists
  if (!fs.existsSync(ASSETS_DIR)) {
    console.error(`Error: Assets directory not found at ${ASSETS_DIR}`);
    console.log('Creating assets directory. Please add images to this folder.');
    fs.mkdirSync(ASSETS_DIR, { recursive: true });
    return;
  }
  
  // Get all image files from assets directory
  const imageFiles = fs.readdirSync(ASSETS_DIR)
    .filter(file => {
      const ext = path.extname(file).toLowerCase();
      return ['.jpg', '.jpeg', '.png', '.gif', '.bmp'].includes(ext);
    });
  
  if (imageFiles.length === 0) {
    console.log('No image files found in assets directory.');
    // Create empty embeddings file
    fs.writeFileSync(OUTPUT_FILE, '[]');
    console.log(`Created empty embeddings file at ${OUTPUT_FILE}`);
    return;
  }
  
  console.log(`Found ${imageFiles.length} image files`);
  
  // Load MobileNet model
  console.log('Loading MobileNet model...');
  const model = await mobilenet.load({
    version: 2,
    alpha: 1.0
  });
  console.log('Model loaded successfully');
  
  // Process each image
  const embeddings = [];
  
  for (const file of imageFiles) {
    const imagePath = path.join(ASSETS_DIR, file);
    console.log(`Processing ${file}...`);
    
    try {
      // Load and preprocess the image
      const tensor = await processImage(imagePath);
      if (!tensor) continue;
      
      // Get embeddings (features) from the model
      const activation = model.infer(tensor, true);
      const features = Array.from(await activation.data());
      
      // Add to embeddings array
      embeddings.push({
        id: path.basename(file, path.extname(file)),
        name: path.basename(file, path.extname(file)).replace(/[-_]/g, ' '),
        imageUrl: `/src/assets/${file}`,
        features: features
      });
      
      // Clean up tensors
      tensor.dispose();
      activation.dispose();
      
      console.log(`Successfully processed ${file}`);
    } catch (error) {
      console.error(`Error processing ${file}:`, error);
    }
  }
  
  // Save embeddings to JSON file
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(embeddings, null, 2));
  console.log(`Saved ${embeddings.length} embeddings to ${OUTPUT_FILE}`);
}

// Run the main function
generateEmbeddings()
  .then(() => {
    console.log('Embeddings generation complete!');
    process.exit(0);
  })
  .catch(error => {
    console.error('Error generating embeddings:', error);
    process.exit(1);
  });