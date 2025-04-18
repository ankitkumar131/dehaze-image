const markdownpdf = require('markdown-pdf');
const fs = require('fs');
const path = require('path');

// Define input and output file paths
const expandedPart1 = path.join(__dirname, 'expanded-thesis.md');
const expandedPart2 = path.join(__dirname, 'expanded-thesis-part2.md');
const expandedPart3 = path.join(__dirname, 'expanded-thesis-part3.md');
const outputFile = path.join(__dirname, 'expanded-thesis.pdf');

// Check if the input files exist
if (!fs.existsSync(expandedPart1)) {
  console.error(`Error: Input file '${expandedPart1}' does not exist.`);
  process.exit(1);
}

if (!fs.existsSync(expandedPart2)) {
  console.error(`Error: Input file '${expandedPart2}' does not exist.`);
  process.exit(1);
}

if (!fs.existsSync(expandedPart3)) {
  console.error(`Error: Input file '${expandedPart3}' does not exist.`);
  process.exit(1);
}

// Combine the expanded thesis parts into a single file
console.log('Combining expanded thesis parts...');
const combinedContent = fs.readFileSync(expandedPart1, 'utf8') + 
                       fs.readFileSync(expandedPart2, 'utf8') + 
                       fs.readFileSync(expandedPart3, 'utf8');
const combinedFile = path.join(__dirname, 'combined-thesis.md');
fs.writeFileSync(combinedFile, combinedContent, 'utf8');

console.log(`Converting combined thesis to PDF...`);

// PDF configuration options
const pdfOptions = {
  // Formatting options
  paperFormat: 'A4',
  paperOrientation: 'portrait',
  paperBorder: '0',  // We'll control margins in CSS
  
  // Styling with CSS - using the new CSS file
  cssPath: path.join(__dirname, 'new-thesis-style.css'),
  
  // Header and footer - using the new header/footer file
  runningsPath: path.join(__dirname, 'new-thesis-headerfooter.js'),

  // Additional options for better PDF quality
  remarkable: {
    html: true,
    breaks: true
  }
  
  // Removed the problematic preProcessHtml function
};

// Convert markdown to PDF
fs.createReadStream(combinedFile)
  .pipe(markdownpdf(pdfOptions))
  .pipe(fs.createWriteStream(outputFile))
  .on('finish', function() {
    console.log(`PDF successfully created at: ${outputFile}`);
    // Clean up the combined file
    fs.unlinkSync(combinedFile);
    console.log('Temporary combined file removed.');
  })
  .on('error', function(err) {
    console.error('Error converting markdown to PDF:', err);
  });