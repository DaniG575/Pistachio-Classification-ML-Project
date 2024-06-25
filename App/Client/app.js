const dropbox = document.getElementById('dropbox');
const fileInput = document.getElementById('file-input');
const chooseFileBtn = document.getElementById('choose-file-btn');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predict-btn');
const kirmiziProb = document.getElementById('kirmizi-prob');
const siirtProb = document.getElementById('siirt-prob');
const finalResult = document.getElementById('final-result');

let base64Image = '';

dropbox.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropbox.classList.add('dragover');
});

dropbox.addEventListener('dragleave', () => {
    dropbox.classList.remove('dragover');
});

dropbox.addEventListener('drop', (event) => {
    event.preventDefault();
    dropbox.classList.remove('dragover');
    const files = event.dataTransfer.files;

    if (files.length > 0) {
        const file = files[0];
        previewFile(file);
    } else {
        const imageUrl = event.dataTransfer.getData('text/uri-list');
        if (imageUrl) {
            displayImagePreviewFromUrl(imageUrl);
            convertImageUrlToBase64(imageUrl);
        }
    }
});

chooseFileBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        previewFile(file);
    }
});

function previewFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        base64Image = e.target.result.split(',')[1];  // Get the base64 string
        preview.innerHTML = `<img src="${e.target.result}" alt="Image Preview">`;
        console.log('Base64 Image (File):', base64Image);
    };
    reader.readAsDataURL(file);
}

function convertImageUrlToBase64(url) {
    fetch(url)
        .then(response => response.blob())
        .then(blob => {
            const reader = new FileReader();
            reader.onload = (e) => {
                base64Image = e.target.result.split(',')[1];  // Get the base64 string
                preview.innerHTML = `<img src="${e.target.result}" alt="Image Preview">`;
                console.log('Base64 Image (URL):', base64Image);
            };
            reader.readAsDataURL(blob);
        })
        .catch(error => console.error('Error converting image URL to base64:', error));
}

function displayImagePreviewFromUrl(url) {
    preview.innerHTML = `<img src="${url}" alt="Image Preview">`;
}

predictBtn.addEventListener('click', () => {
    // Handle predict button click
    if (base64Image) {
        fetch('http://127.0.0.1:8000/predict', {  // Update with your backend URL and endpoint
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image })
        })
        .then(response => response.json())
        .then(data => {
            const probabilities = data.probabilities;
            const prediction = data.prediction;

            document.getElementById('kirmizi-prob').textContent = probabilities[0].toFixed(2);
            document.getElementById('siirt-prob').textContent = probabilities[1].toFixed(2);
            document.getElementById('final-result').textContent = `Final Prediction: ${prediction}`;
        })
        .catch(error => console.error('Error:', error));
    } else {
        alert('Please upload or drag and drop an image first.');
    }
});







