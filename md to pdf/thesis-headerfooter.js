// Header and footer for thesis PDF

exports.header = {
  height: '1cm',
  contents: function(pageNum, numPages) {
    // Only show header from page 2 onwards
    if (pageNum === 1) return '';
    return '<div style="text-align: center; font-size: 10pt; color: #666;">Image Dehazing System</div>';
  }
};

exports.footer = {
  height: '1cm',
  contents: function(pageNum, numPages) {
    return '<div style="text-align: center; font-size: 10pt; color: #666;">' + pageNum + ' / ' + numPages + '</div>';
  }
};