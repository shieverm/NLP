// app.js

document.addEventListener('DOMContentLoaded', function () {
    var divElement = document.getElementById('correct-text');
  
    if (divElement) {
        var text = divElement.textContent.trim();
        var capitalizedText = text.charAt(0).toUpperCase() + text.slice(1);
        var newText = capitalizedText.endsWith('.') ? capitalizedText : capitalizedText + '.';
    
        divElement.textContent = newText;
        divElement.style.color = 'green';
        divElement.textContent = newText;
    }
  });