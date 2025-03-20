// JavaScript for the Autism Screening page

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const form = document.getElementById('autism-screening-form');
    const videoInput = document.getElementById('video-file');
    const fileName = document.getElementById('file-name');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultsContent = document.getElementById('results-content');
    const recommendations = document.getElementById('recommendations');

    // Update file name when file is selected
    videoInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileName.textContent = this.files[0].name;
        } else {
            fileName.textContent = 'No file chosen';
        }
    });

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading indicator
        loading.style.display = 'block';
        results.style.display = 'none';
        
        try {
            // Create form data
            const formData = new FormData(form);
            
            // Add video file if selected
            if (videoInput.files.length > 0) {
                formData.append('video', videoInput.files[0]);
            }
            
            // Make API request
            const response = await fetch('/api/v1/analyze/screening', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Hide loading indicator
            loading.style.display = 'none';
            
            // Display results
            results.style.display = 'block';
            
            if (data.status === 'success') {
                displayResults(data.result);
            } else {
                displayError(data.message);
            }
        } catch (error) {
            console.error('Error:', error);
            loading.style.display = 'none';
            results.style.display = 'block';
            displayError('Unable to process assessment. Please try again later.');
        }
    });

    // Function to display results
    function displayResults(result) {
        const prediction = result.prediction_summary;
        const behavioral = result.behavioral_analysis;
        
        // Create results HTML
        let resultsHTML = `
            <div class="result-summary">
                <p><strong>Screening Result:</strong> ${prediction.interpretation}</p>
                <p><strong>Confidence Level:</strong> ${prediction.confidence}</p>
                <p><strong>Probability Score:</strong> ${(prediction.probability * 100).toFixed(1)}%</p>
            </div>
            
            <div class="behavioral-analysis">
                <h4>Behavioral Analysis</h4>
        `;
        
        if (behavioral.areas_of_concern.length > 0) {
            resultsHTML += `
                <p><strong>Areas of Potential Concern:</strong></p>
                <ul>
            `;
            
            behavioral.areas_of_concern.forEach(area => {
                resultsHTML += `<li>${formatFeatureName(area)}</li>`;
            });
            
            resultsHTML += `</ul>`;
        }
        
        if (behavioral.strengths.length > 0) {
            resultsHTML += `
                <p><strong>Strengths:</strong></p>
                <ul>
            `;
            
            behavioral.strengths.forEach(strength => {
                resultsHTML += `<li>${formatFeatureName(strength)}</li>`;
            });
            
            resultsHTML += `</ul>`;
        }
        
        resultsHTML += `</div>`;
        
        resultsContent.innerHTML = resultsHTML;
        
        // Create recommendations HTML
        let recommendationsHTML = `
            <div class="recommendations">
                <h4>Recommendations</h4>
                <p>${result.recommendations.primary_recommendation}</p>
        `;
        
        if (result.recommendations.suggested_interventions.length > 0) {
            recommendationsHTML += `
                <p><strong>Suggested Activities:</strong></p>
                <ul>
            `;
            
            result.recommendations.suggested_interventions.forEach(intervention => {
                recommendationsHTML += `<li>${intervention}</li>`;
            });
            
            recommendationsHTML += `</ul>`;
        }
        
        recommendationsHTML += `
            <p class="disclaimer">${result.disclaimer}</p>
        </div>
        `;
        
        recommendations.innerHTML = recommendationsHTML;
    }

    // Function to display error message
    function displayError(message) {
        resultsContent.innerHTML = `<p class="error">Error: ${message}</p>`;
        recommendations.innerHTML = '';
    }

    // Helper function to format feature names
    function formatFeatureName(name) {
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // Add form validation
    const requiredFields = form.querySelectorAll('[required]');
    requiredFields.forEach(field => {
        field.addEventListener('change', function() {
            if (this.value) {
                this.classList.remove('invalid');
                this.classList.add('valid');
            } else {
                this.classList.remove('valid');
                this.classList.add('invalid');
            }
        });
    });

    // Add progress tracking
    let answeredQuestions = 0;
    const totalQuestions = form.querySelectorAll('.question').length;

    function updateProgress() {
        const progress = (answeredQuestions / totalQuestions) * 100;
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        progressBar.style.width = `${progress}%`;
        
        const progressContainer = document.querySelector('.progress-container');
        if (!progressContainer) {
            const container = document.createElement('div');
            container.className = 'progress-container';
            container.appendChild(progressBar);
            form.insertBefore(container, form.firstChild);
        } else {
            progressContainer.innerHTML = '';
            progressContainer.appendChild(progressBar);
        }
    }

    form.querySelectorAll('input[type="radio"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                const question = this.closest('.question');
                if (!question.classList.contains('answered')) {
                    question.classList.add('answered');
                    answeredQuestions++;
                    updateProgress();
                }
            }
        });
    });

    // Initialize progress bar
    updateProgress();
}); 