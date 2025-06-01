/* --------------------------------------------------
   Simple HTTP Server for AutoClipEval Client App
   -------------------------------------------------- */

const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Enable CORS for API calls
app.use(cors());

// Serve static files from the current directory
app.use(express.static('./'));

// Route for the main app
app.get('/', (req, res) => {
  res.sendFile(path.resolve('./index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`🚀 AutoClipEval Client App running at:`);
  console.log(`   Local:    http://localhost:${PORT}`);
  console.log(`   Network:  http://0.0.0.0:${PORT}`);
  console.log('');
  console.log('📂 Serving files from:', path.resolve('./'));
  console.log('🎬 Ready to evaluate clips!');
});
