//app.js

document.addEventListener('DOMContentLoaded', function () {
    var divElement = document.getElementById('snt');

    if (divElement) {
        var sentiment = divElement.textContent.trim();

        if (sentiment === 'positive') {
            divElement.classList.add('positive');
        } else if (sentiment === 'negative') {
            divElement.classList.add('negative');
        } else if (sentiment === 'neutral') {
            divElement.classList.add('neutral');
        }
    }
  });