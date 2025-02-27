// Use ES module syntax
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Convert __dirname to work in ES modules
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

// Main function
function generateSampleEmbeddings() {
  console.log('Starting sample embeddings generation...');

  if (!fs.existsSync(ASSETS_DIR)) {
    console.error(`Error: Assets directory not found at ${ASSETS_DIR}`);
    console.log('Creating assets directory. Please add images to this folder.');
    fs.mkdirSync(ASSETS_DIR, { recursive: true });
    return;
  }

  const imageFiles = fs.readdirSync(ASSETS_DIR)
    .filter(file => ['.jpg', '.jpeg', '.png', '.gif', '.bmp'].includes(path.extname(file).toLowerCase()));

  if (imageFiles.length === 0) {
    console.log('No image files found in assets directory.');
    fs.writeFileSync(OUTPUT_FILE, '[]');
    console.log(`Created empty embeddings file at ${OUTPUT_FILE}`);
    return;
  }

  console.log(`Found ${imageFiles.length} image files`);

  const embeddings = imageFiles.map(file => ({
    id: path.basename(file, path.extname(file)),
    name: path.basename(file, path.extname(file)).replace(/[-_]/g, ' '),
    imageUrl: `/src/assets/${file}`,
    features: Array.from({ length: 1024 }, () => Math.random()) // Fake embeddings
  }));

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(embeddings, null, 2));
  console.log(`Saved ${embeddings.length} embeddings to ${OUTPUT_FILE}`);
}

// Run
generateSampleEmbeddings();
console.log('Sample embeddings generation complete!');
