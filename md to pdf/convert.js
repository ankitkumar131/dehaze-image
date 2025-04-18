const markdownpdf = require('markdown-pdf');
const fs = require('fs');
const path = require('path');

// Define input and output file paths
const inputFile = path.join(__dirname, 'thesis.md');
const outputFile = path.join(__dirname, 'thesis.pdf');

// Check if the input file exists
if (!fs.existsSync(inputFile)) {
  console.error(`Error: Input file '${inputFile}' does not exist.`);
  process.exit(1);
}

console.log(`Converting '${inputFile}' to PDF...`);

// PDF configuration options
const pdfOptions = {
  // Formatting options
  paperFormat: 'A4',
  paperOrientation: 'portrait',
  paperBorder: '1cm',
  
  // Styling with CSS
  cssPath: path.join(__dirname, 'thesis-style.css'),
  
  // Header and footer
  runningsPath: path.join(__dirname, 'thesis-headerfooter.js')
};

// Convert markdown to PDF
fs.createReadStream(inputFile)
  .pipe(markdownpdf(pdfOptions))
  .pipe(fs.createWriteStream(outputFile))
  .on('finish', function() {
    console.log(`PDF successfully created at: ${outputFile}`);
  })
  .on('error', function(err) {
    console.error('Error converting markdown to PDF:', err);
  });