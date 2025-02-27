// generate-embeddings.js
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// For Node.js TensorFlow
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';

// ESM-friendly __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ASSETS_DIR = path.join(__dirname, 'src', 'assets');
const OUTPUT_FILE = path.join(__dirname, 'src', 'data', 'embeddings.json');

// Utility to load and resize an image
async function processImage(filePath) {
  const imageBuffer = fs.readFileSync(filePath);
  // Decode image to a tensor
  const decoded = tf.node.decodeImage(imageBuffer, 3); // 3 channels (RGB)
  // Resize to MobileNet's expected input
  const resized = tf.image.resizeBilinear(decoded, [224, 224]);
  // Normalize [0,1]
  const normalized = resized.div(255);
  // Expand dims so shape is [1, 224, 224, 3]
  const batched = normalized.expandDims(0);

  // Clean up intermediate tensors
  decoded.dispose();
  resized.dispose();
  normalized.dispose();

  return batched;
}

async function generateEmbeddings() {
  if (!fs.existsSync(ASSETS_DIR)) {
    console.error(`Assets directory not found at ${ASSETS_DIR}`);
    return;
  }

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

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(embeddings, null, 2));
  console.log(`Saved embeddings for ${embeddings.length} images to ${OUTPUT_FILE}`);
}

// Run the script
generateEmbeddings()
  .then(() => console.log('Done generating embeddings!'))
  .catch(err => console.error(err));
