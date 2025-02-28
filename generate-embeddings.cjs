const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs'); // CPU-only usage with canvas
require('@tensorflow/tfjs-backend-cpu');
const mobilenet = require('@tensorflow-models/mobilenet');
// const vgg = require('@tensorflow-models/vgg16'); 
// const resnet = require('@tensorflow-models/resnet50');
const { createCanvas, loadImage } = require('canvas');

const ASSETS_DIR = path.join(__dirname, 'src', 'assets');
const OUTPUT_DIR = path.join(__dirname, 'src', 'data');

// Model configurations with load and embedding extraction functions
const MODEL_CONFIGS = {
  mobilenet: {
    name: 'MobileNet v2',
    filename: 'mobilenet_embeddings.json',
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
    filename: 'vgg16_embeddings.json',
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
    filename: 'resnet50_embeddings.json',
    load: async () => {
      return await resnet.load();
    },
    getEmbedding: (model, tensor) => {
      // For ResNet50, get the penultimate layer
      return model.predict(tensor);
    }
  }
};

// 1. Recursively collect all image file paths
function getAllImageFiles(dirPath, allowedExts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']) {
  let results = [];
  const files = fs.readdirSync(dirPath);

  for (const file of files) {
    const fullPath = path.join(dirPath, file);
    const stat = fs.statSync(fullPath);

    if (stat.isDirectory()) {
      // Recurse into subfolder
      results = results.concat(getAllImageFiles(fullPath, allowedExts));
    } else {
      // Check file extension
      const ext = path.extname(fullPath).toLowerCase();
      if (allowedExts.includes(ext)) {
        results.push(fullPath);
      }
    }
  }
  return results;
}

// 2. Convert an image file to a Tensor using canvas - with better error handling
async function processImage(filePath) {
  try {
    // Read the file into a buffer first
    const buffer = fs.readFileSync(filePath);
    
    // Load image from buffer instead of from file path
    const image = await loadImage(buffer);

    // Create a 224×224 canvas (common input size for most models)
    const canvas = createCanvas(224, 224);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, 224, 224);

    // Convert canvas to tensor, normalize [0,1], expand dims to [1,224,224,3]
    return {
      tensor: tf.browser.fromPixels(canvas).div(255).expandDims(0),
      success: true
    };
  } catch (error) {
    console.error(`Failed to process image: ${filePath}`);
    console.error(`Error details: ${error.message}`);
    return {
      tensor: null,
      success: false
    };
  }
}

// Helper function to slugify filenames for safe URLs
function slugify(text) {
  return text
    .toString()
    .normalize('NFD')                // Split accented characters into base character and accent
    .replace(/[\u0300-\u036f]/g, '') // Remove accent marks
    .toLowerCase()
    .replace(/[^\w\-]+/g, '-')       // Replace non-word chars with dash
    .replace(/\-\-+/g, '-')          // Replace multiple dashes with single dash
    .replace(/^-+/, '')              // Trim dashes from start
    .replace(/-+$/, '');             // Trim dashes from end
}

// Main function to generate embeddings for a specific model
async function generateEmbeddings(modelKey = 'mobilenet') {
  if (!MODEL_CONFIGS[modelKey]) {
    console.error(`Unknown model: ${modelKey}`);
    return;
  }
  
  const modelConfig = MODEL_CONFIGS[modelKey];
  const OUTPUT_FILE = path.join(OUTPUT_DIR, modelConfig.filename);
  
  console.log(`Generating embeddings using ${modelConfig.name}...`);
  
  if (!fs.existsSync(ASSETS_DIR)) {
    console.error(`Assets directory not found at ${ASSETS_DIR}`);
    return;
  }

  // Force CPU backend (optional, but ensures consistency)
  await tf.setBackend('cpu');

  // 3. Load the selected model
  console.log(`Loading ${modelConfig.name}...`);
  const model = await modelConfig.load();
  console.log(`${modelConfig.name} loaded successfully`);

  // 4. Recursively get all image paths from src/assets
  const allImagePaths = getAllImageFiles(ASSETS_DIR);
  console.log(`Found ${allImagePaths.length} images to process`);

  const embeddings = [];
  let successCount = 0;
  let failureCount = 0;

  for (const fullPath of allImagePaths) {
    console.log(`Processing: ${fullPath}`);

    // Create a relative path (so subfolders appear in /assets/ URLs)
    const relativePath = path.relative(ASSETS_DIR, fullPath).replace(/\\/g, '/');

    // Separate the folder part and the file part
    const folderPath = path.dirname(relativePath).replace(/\\/g, '/'); 
    const baseName = path.basename(relativePath, path.extname(relativePath)); 
    const extension = path.extname(relativePath);

    // Build a "display name" with folder + baseName
    const displayName = folderPath ? `${folderPath}/${baseName}` : baseName;

    // Create a URL-safe version of the filename for the browser
    const safeBaseName = slugify(baseName) + extension;
    const safeRelativePath = folderPath ? `${folderPath}/${safeBaseName}` : safeBaseName;
    
    // Convert image to tensor
    const { tensor, success } = await processImage(fullPath);

    if (!success || tensor === null) {
      console.error(`Skipping ${fullPath} due to processing error`);
      failureCount++;
      continue;
    }

    // Extract embedding using the model-specific function
    const activation = modelConfig.getEmbedding(model, tensor);
    const features = Array.from(await activation.data());

    // Clean up
    tensor.dispose();
    activation.dispose();

    // 5. Push to array (store subfolder structure in imageUrl)
    embeddings.push({
      folder: folderPath,             
      id: baseName,                   
      displayName,                    
      originalPath: relativePath,     
      imageUrl: `/assets/${relativePath}`, 
      features,
      modelUsed: modelKey  // Add the model identifier to each embedding
    });
    
    successCount++;
  }

  // Ensure the data folder exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Write embeddings to JSON
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(embeddings, null, 2));
  console.log(`Saved embeddings for ${successCount} images to ${OUTPUT_FILE} using ${modelConfig.name}`);
  if (failureCount > 0) {
    console.warn(`Failed to process ${failureCount} images`);
  }
  
  // Create a manifest file to track available models and their embedding files
  updateManifestFile(modelKey);
}

// Update the manifest file that tracks which models have embeddings
function updateManifestFile(modelKey) {
  const manifestPath = path.join(OUTPUT_DIR, 'embeddings_manifest.json');
  
  let manifest = {};
  if (fs.existsSync(manifestPath)) {
    try {
      manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    } catch (error) {
      console.error('Error reading manifest file:', error);
    }
  }
  
  // Update the manifest with this model's information
  manifest[modelKey] = {
    name: MODEL_CONFIGS[modelKey].name,
    filename: MODEL_CONFIGS[modelKey].filename,
    lastUpdated: new Date().toISOString()
  };
  
  // Write the updated manifest
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`Updated manifest file with ${MODEL_CONFIGS[modelKey].name} information`);
}

// Process command line arguments
async function main() {
  // Get model from command line argument or use default
  const modelArg = process.argv[2];
  
  if (modelArg && !MODEL_CONFIGS[modelArg]) {
    console.error(`Unknown model: ${modelArg}`);
    console.log(`Available models: ${Object.keys(MODEL_CONFIGS).join(', ')}`);
    process.exit(1);
  }
  
  const selectedModel = modelArg || 'mobilenet';
  
  try {
    console.log(`Starting embedding generation for model: ${selectedModel}`);
    await generateEmbeddings(selectedModel);
    console.log('Embedding generation completed successfully');
  } catch (error) {
    console.error('Error during embedding generation:', error);
    process.exit(1);
  }
}

// If this is the main module being run
if (require.main === module) {
  main().catch(console.error);
} else {
  // Export functions for use in other modules
  module.exports = { generateEmbeddings, MODEL_CONFIGS };
}