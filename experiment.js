let images = [];
let shuffledImages = [];
let currentIndex = 0;
let csvData = [];
let studentId = '';
let imageScores = {}; // Track scores for each image

const IMAGES_INDEX_FILE = 'images.json';    // Name of the JSON file containing image URLs (created using the py script)

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('start-button').addEventListener('click', startExperiment);
    document.addEventListener('keydown', handleKeyPress);
});

async function loadAllImages() {
    try {
        const response = await fetch(IMAGES_INDEX_FILE);
        const data = await response.json();
        images = data || [];
    } catch (error) {
        console.error('Error loading ' + IMAGES_INDEX_FILE +' :', error);
        images = [];
    }
}

async function startExperiment() {
    studentId = document.getElementById('student-id').value;
    
    csvData = ['Image,Score1,Score2\n'];
    imageScores = {};
    
    // Load all images from JSON file
    await loadAllImages();
    
    if (images.length === 0) {
        alert('No images found in ' + IMAGES_INDEX_FILE + ' file!');
        return;
    }
    
    // Duplicate and shuffle images
    shuffledImages = [...images, ...images];
    shuffledImages.sort(() => Math.random() - 0.5);
    
    // Show first image
    currentIndex = 0;
    showImage();
}

function showImage() {
    document.body.innerHTML = `
        <img src="${shuffledImages[currentIndex]}"><br>
        <p>Rate this image (1-5): Press 1, 2, 3, 4, or 5</p>
        <p>Image ${currentIndex + 1} of ${shuffledImages.length}</p>
    `;
}

function handleKeyPress(event) {
    if (currentIndex >= shuffledImages.length) return;
    
    let score = parseInt(event.key);
    if (score >= 1 && score <= 5) {
        const currentImage = shuffledImages[currentIndex];
        
        // Track scores for each image
        if (!imageScores[currentImage]) {
            imageScores[currentImage] = [];
        }
        imageScores[currentImage].push(score);
        
        nextImage();
    }
}

function nextImage() {
    currentIndex++;
    if (currentIndex < shuffledImages.length) {
        showImage();
    } else {
        downloadCSV();
    }
}

function downloadCSV() {
    // Build CSV data from imageScores
    csvData = ['Image,Score1,Score2\n'];
    
    for (const [image, scores] of Object.entries(imageScores)) {
        const score1 = scores[0] || '';
        const score2 = scores[1] || '';
        csvData.push(`${image},${score1},${score2}\n`);
    }
    
    let csvContent = csvData.join('');
    let blob = new Blob([csvContent], { type: 'text/csv' });
    let url = window.URL.createObjectURL(blob);
    let a = document.createElement('a');
    a.href = url;
    a.download = `${studentId}.csv`;
    a.click();
    
    document.body.innerHTML = '<p>Experiment complete! CSV file downloaded.</p>';
}