// Header and footer for thesis PDF

exports.header = {
  height: '1cm',
  contents: function(pageNum, numPages) {
    // Only show header from page 2 onwards
    if (pageNum === 1) return '';
    
    // Get the current chapter title
    // This is a simplified approach - in a real implementation, you would need to track
    // the current chapter based on page numbers or other markers
    return '<div style="text-align: center; font-family: \'Times New Roman\', Times, serif; font-size: 10pt; color: #333;">Image Dehazing System</div>';
  }
};

exports.footer = {
  height: '1cm',
  contents: function(pageNum, numPages) {
    // Center-aligned page numbers as specified in requirements
    return '<div style="text-align: center; font-family: \'Times New Roman\', Times, serif; font-size: 10pt; color: #333;">' + pageNum + '</div>';
  }
};