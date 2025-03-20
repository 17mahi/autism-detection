// JavaScript for the Emotion Analysis page

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const mediaInput = document.getElementById('mediaInput');
    const imagePreview = document.getElementById('imagePreview');
    const videoPreview = document.getElementById('videoPreview');
    const emotionForm = document.getElementById('emotionForm');
    const resultsSection = document.querySelector('.results-section');
    
    // File selection handler
    mediaInput.addEventListener('change', function() {
        // Reset preview containers
        imagePreview.innerHTML = '<i class="fas fa-image"></i><p>Image preview will appear here</p>';
        videoPreview.innerHTML = '<i class="fas fa-video"></i><p>Video preview will appear here</p>';
        
        // Hide both previews initially
        imagePreview.style.display = 'none';
        videoPreview.style.display = 'none';
        
        if (this.files && this.files[0]) {
            const file = this.files[0];
            const fileName = document.querySelector('.file-name');
            if (fileName) {
                fileName.textContent = file.name;
            }
            
            // Check if file is an image
            if (file.type.match('image.*')) {
                showImagePreview(file);
            } 
            // Check if file is a video
            else if (file.type.match('video.*')) {
                showVideoPreview(file);
            }
        }
    });
    
    // Form submission handler
    emotionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        const loading = document.createElement('div');
        loading.className = 'loading';
        loading.innerHTML = '<div class="spinner"></div><p>Analyzing emotions... This may take a moment.</p>';
        emotionForm.appendChild(loading);
        
        // Disable submit button
        const submitButton = emotionForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
        }
        
        try {
            // Create form data
            const formData = new FormData(emotionForm);
            
            // Make API request
            const response = await fetch(emotionForm.action, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Remove loading indicator
            loading.remove();
            
            // Enable submit button
            if (submitButton) {
                submitButton.disabled = false;
            }
            
            // Display results
            if (data.status === 'success') {
                displayResults(data.result);
            } else {
                showNotification('Error', data.message || 'An error occurred during analysis.', 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            
            // Remove loading indicator
            loading.remove();
            
            // Enable submit button
            if (submitButton) {
                submitButton.disabled = false;
            }
            
            showNotification('Error', 'Failed to analyze emotions. Please try again.', 'error');
        }
    });
    
    // Function to show image preview
    function showImagePreview(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Image Preview">`;
            imagePreview.style.display = 'block';
        };
        
        reader.readAsDataURL(file);
    }
    
    // Function to show video preview
    function showVideoPreview(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            videoPreview.innerHTML = `
                <video controls>
                    <source src="${e.target.result}" type="${file.type}">
                    Your browser does not support the video tag.
                </video>
            `;
            videoPreview.style.display = 'block';
        };
        
        reader.readAsDataURL(file);
    }
    
    // Function to display analysis results
    function displayResults(results) {
        // Make sure results section exists
        if (!resultsSection) {
            console.error('Results section not found');
            return;
        }
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Get result container
        const resultsContent = resultsSection.querySelector('.results-content');
        if (!resultsContent) {
            console.error('Results content container not found');
            return;
        }
        
        // Get dominant emotion
        const dominant = results.emotions ? getMaxEmotion(results.emotions) : { emotion: 'neutral', score: 0 };
        
        // Create HTML for results
        let html = `
            <div class="result-card">
                <h3>Dominant Emotion</h3>
                <div class="emotion-display">
                    <i class="${getEmotionIcon(dominant.emotion)}"></i>
                    <span class="emotion-text">${formatEmotionName(dominant.emotion)}</span>
                </div>
                <div class="emotion-confidence">
                    <div class="progress-bar">
                        <div class="progress" style="width: ${dominant.score * 100}%"></div>
                    </div>
                    <span>${Math.round(dominant.score * 100)}% Confidence</span>
                </div>
            </div>
        `;
        
        // Add emotion distribution chart
        if (results.emotions) {
            html += `
                <div class="result-card">
                    <h3>Emotion Distribution</h3>
                    <div class="emotion-chart">
                        <div class="result-summary">
            `;
            
            // Add emotion badges
            Object.entries(results.emotions).forEach(([emotion, score]) => {
                if (score > 0.05) { // Only show emotions with significant scores
                    html += `
                        <span class="emotion-badge badge-${emotion}">
                            <i class="${getEmotionIcon(emotion)}"></i>
                            ${formatEmotionName(emotion)}: ${Math.round(score * 100)}%
                        </span>
                    `;
                }
            });
            
            html += `
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Add emotion timeline if available
        if (results.timeline) {
            html += `
                <div class="result-card">
                    <h3>Emotion Timeline</h3>
                    <div class="timeline-chart">
                        <p>Emotion changes over time will be displayed here.</p>
                        <!-- Timeline chart would be rendered here -->
                    </div>
                </div>
            `;
        }
        
        // Add insights
        html += `
            <div class="result-card">
                <h3>Key Insights</h3>
                <ul class="insights-list">
        `;
        
        if (results.insights && results.insights.length > 0) {
            results.insights.forEach(insight => {
                html += `<li>${insight}</li>`;
            });
        } else {
            // Default insights
            html += `
                <li>${formatEmotionName(dominant.emotion)} is the predominant emotional response</li>
                <li>Emotional intensity is ${dominant.score > 0.7 ? 'high' : (dominant.score > 0.4 ? 'moderate' : 'low')}</li>
                <li>This analysis is based on facial expressions and may not capture all emotional nuances</li>
            `;
        }
        
        html += `
                </ul>
            </div>
        `;
        
        // Set HTML content
        resultsContent.innerHTML = html;
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to get the emotion with the highest score
    function getMaxEmotion(emotions) {
        if (!emotions) return { emotion: 'neutral', score: 0 };
        
        return Object.entries(emotions).reduce(
            (max, [emotion, score]) => score > max.score ? { emotion, score } : max,
            { emotion: 'neutral', score: 0 }
        );
    }
    
    // Function to get icon class for an emotion
    function getEmotionIcon(emotion) {
        const icons = {
            'happy': 'fas fa-smile',
            'sad': 'fas fa-frown',
            'angry': 'fas fa-angry',
            'surprise': 'fas fa-surprise',
            'fear': 'fas fa-grimace',
            'disgust': 'fas fa-dizzy',
            'neutral': 'fas fa-meh',
            // Add more emotion-icon mappings as needed
        };
        
        return icons[emotion] || 'fas fa-meh';
    }
    
    // Function to format emotion name
    function formatEmotionName(emotion) {
        return emotion.charAt(0).toUpperCase() + emotion.slice(1);
    }
    
    // Placeholder function for downloading report
    window.downloadReport = function() {
        showNotification('Coming Soon', 'Report download feature will be available soon.', 'info');
    };
    
    // Placeholder function for sharing results
    window.shareResults = function() {
        showNotification('Coming Soon', 'Result sharing feature will be available soon.', 'info');
    };
    
    // Function to show notification
    function showNotification(title, message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <h4>${title}</h4>
                <p>${message}</p>
            </div>
            <button class="notification-close">&times;</button>
        `;
        
        document.body.appendChild(notification);
        
        // Add close button functionality
        const closeButton = notification.querySelector('.notification-close');
        closeButton.addEventListener('click', () => {
            notification.remove();
        });
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}); 