class ECGViewer {
    constructor(canvasId, data) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.data = data; // Array of numbers
        
        // Configuration
        this.scaleX = 1.0; // Zoom level
        this.scaleY = 1.0;
        this.rows = 1;     // Number of rows/strips
        this.offsetY = 0;  // Pan offset (optional)
        
        // Constants for Grid (Assuming standard ECG: 25mm/s, 10mm/mV)
        this.gridColorMajor = '#f0a1a1'; // Darker pink/red
        this.gridColorMinor = '#fce0e0'; // Lighter pink
        this.signalColor = '#000000';    // Black signal
        
        // Initialize
        this.fitToScreen();
        this.draw();
        
        // Event Listeners for Interaction
        this.setupEvents();
    }

    fitToScreen() {
        // Auto scale to fit data in width
        this.scaleX = this.canvas.width / (this.data.length / this.rows);
        // Auto scale Y based on data min/max
        const min = Math.min(...this.data);
        const max = Math.max(...this.data);
        const range = max - min || 1;
        this.scaleY = (this.canvas.height / this.rows) / (range * 1.5); // 1.5 padding
    }

    setRows(n) {
        this.rows = parseInt(n);
        this.fitToScreen(); // Reset zoom when changing layout
        this.draw();
    }

    setZoom(factor) {
        this.scaleX *= factor;
        this.draw();
    }

    drawGrid(width, height) {
        this.ctx.lineWidth = 1;
        
        // Minor Grid (1mm boxes - represented as approx 10px here for visibility)
        const gridSize = 15 * this.scaleX; // Dynamic grid size based on zoom? Or fixed?
        // Let's use fixed visual grid for "paper look" regardless of zoom for now, 
        // or sync it. Standard ECG viewer usually has fixed background.
        // For simplicity: Simple Grid Background
        
        const boxSize = 20; // 1 large box
        
        this.ctx.strokeStyle = this.gridColorMinor;
        this.ctx.beginPath();
        for (let x = 0; x < width; x += boxSize/5) { this.ctx.moveTo(x, 0); this.ctx.lineTo(x, height); }
        for (let y = 0; y < height; y += boxSize/5) { this.ctx.moveTo(0, y); this.ctx.lineTo(width, y); }
        this.ctx.stroke();

        this.ctx.lineWidth = 1.5;
        this.ctx.strokeStyle = this.gridColorMajor;
        this.ctx.beginPath();
        for (let x = 0; x < width; x += boxSize) { this.ctx.moveTo(x, 0); this.ctx.lineTo(x, height); }
        for (let y = 0; y < height; y += boxSize) { this.ctx.moveTo(0, y); this.ctx.lineTo(width, y); }
        this.ctx.stroke();
    }

    draw() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        // Clear
        this.ctx.clearRect(0, 0, w, h);
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, w, h);
        
        // Draw Grid
        this.drawGrid(w, h);
        
        // Draw Signal
        this.ctx.strokeStyle = this.signalColor;
        this.ctx.lineWidth = 1.5;
        this.ctx.lineJoin = 'round';
        
        const pointsPerRow = Math.ceil(this.data.length / this.rows);
        const rowHeight = h / this.rows;
        
        // Calculate Y range for normalization
        const minVal = Math.min(...this.data);
        const maxVal = Math.max(...this.data);
        const range = maxVal - minVal || 1;
        const midVal = (maxVal + minVal) / 2;

        this.ctx.beginPath();
        
        for (let r = 0; r < this.rows; r++) {
            const startIdx = r * pointsPerRow;
            const endIdx = Math.min(startIdx + pointsPerRow, this.data.length);
            const rowCenterY = (r * rowHeight) + (rowHeight / 2);
            
            for (let i = startIdx; i < endIdx; i++) {
                const val = this.data[i];
                
                // X Position: relative to row start
                const x = ((i - startIdx) * this.scaleX);
                
                // Y Position: centered in row, scaled inverted (canvas y goes down)
                // (val - midVal) centers the wave around 0
                const y = rowCenterY - ((val - midVal) * this.scaleY); 
                
                if (i === startIdx) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
        }
        this.ctx.stroke();
        
        // Draw Row Separators
        if (this.rows > 1) {
            this.ctx.strokeStyle = '#333';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            for (let r = 1; r < this.rows; r++) {
                const y = r * rowHeight;
                this.ctx.moveTo(0, y);
                this.ctx.lineTo(w, y);
            }
            this.ctx.stroke();
        }
    }

    setupEvents() {
        // Zoom on Scroll
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            this.setZoom(factor);
        });
    }
}