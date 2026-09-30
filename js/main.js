/**
 * ThinkProprio Project Website Scripts
 */

// ========== Copy BibTeX ==========
function copyBibtex() {
    const bibtex = document.querySelector('.bibtex code').textContent;
    const btn = document.querySelector('.copy-btn');

    navigator.clipboard.writeText(bibtex).then(() => {
        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        btn.classList.add('copied');

        setTimeout(() => {
            btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            btn.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = bibtex;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);

        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        btn.classList.add('copied');

        setTimeout(() => {
            btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            btn.classList.remove('copied');
        }, 2000);
    });
}

// ========== Coming Soon Toast ==========
function showComingSoon() {
    const existing = document.querySelector('.coming-soon-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'coming-soon-toast';
    toast.textContent = 'Will release soon';
    toast.style.cssText = `
        position: fixed;
        top: 2rem;
        left: 50%;
        transform: translateX(-50%);
        background: #2c3e50;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 600;
        z-index: 10000;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    document.body.appendChild(toast);

    requestAnimationFrame(() => { toast.style.opacity = '1'; });

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

// ========== Video Grid ==========
class VideoGrid {
    constructor(containerId, videosBasePath) {
        this.container = document.getElementById(containerId);
        this.basePath = videosBasePath;
        this.videos = [];
    }

    // Load videos from a predefined list
    loadVideos(videoList) {
        this.videos = videoList;
        this.render();
    }

    render() {
        if (this.videos.length === 0) {
            return; // Keep placeholder if no videos
        }

        this.container.innerHTML = '';

        this.videos.forEach((video, index) => {
            const card = document.createElement('div');
            card.className = 'video-card';
            card.innerHTML = `
                <video controls preload="metadata" poster="${video.poster || ''}">
                    <source src="${this.basePath}/${video.src}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="video-title">${video.title}</div>
            `;
            this.container.appendChild(card);
        });
    }
}

// ========== Video Carousel ==========
class VideoCarousel {
    constructor(container) {
        this.container = container;
        this.track = container.querySelector('.carousel-track');
        this.prevBtn = container.querySelector('.carousel-btn.prev');
        this.nextBtn = container.querySelector('.carousel-btn.next');
        this.timeline = container.querySelector('.timeline');
        this.cards = Array.from(this.track.querySelectorAll('.video-card'));
        this.currentIndex = 0;
        this.totalCards = this.cards.length;

        this.init();
    }

    init() {
        this.prevBtn.addEventListener('click', () => this.prev());
        this.nextBtn.addEventListener('click', () => this.next());

        // Handle resize for responsive behavior
        window.addEventListener('resize', () => this.updateView());

        // Initial state
        this.updateView();
    }

    getVisibleCount() {
        const viewportWidth = this.container.querySelector('.carousel-viewport').offsetWidth;
        if (viewportWidth <= 600) return 1;
        if (viewportWidth < 900) return 2;
        return 3;
    }

    prev() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            this.updateView();
        }
    }

    next() {
        const visibleCount = this.getVisibleCount();
        const maxIndex = Math.max(0, this.totalCards - visibleCount);
        if (this.currentIndex < maxIndex) {
            this.currentIndex++;
            this.updateView();
        }
    }

    updateView() {
        const visibleCount = this.getVisibleCount();
        const maxIndex = Math.max(0, this.totalCards - visibleCount);

        // Clamp current index
        this.currentIndex = Math.min(this.currentIndex, maxIndex);

        // Calculate card width from viewport
        const viewportWidth = this.container.querySelector('.carousel-viewport').offsetWidth;
        const computedStyle = getComputedStyle(this.track);
        const gap = parseFloat(computedStyle.gap) || 24;
        const cardWidth = (viewportWidth - (visibleCount - 1) * gap) / visibleCount;

        // Set explicit card widths
        this.cards.forEach(card => {
            card.style.flex = `0 0 ${cardWidth}px`;
        });

        // Slide the track
        const offset = this.currentIndex * (cardWidth + gap);
        this.track.style.transform = `translateX(-${offset}px)`;

        // Update buttons
        this.prevBtn.disabled = this.currentIndex === 0;
        this.nextBtn.disabled = this.currentIndex >= maxIndex;

        // Update timeline
        this.updateTimeline(visibleCount);
    }

    updateTimeline(visibleCount) {
        if (!this.timeline) return;

        const steps = this.timeline.querySelectorAll('.timeline-step');
        const connectors = this.timeline.querySelectorAll('.timeline-connector');

        // Update step highlights
        steps.forEach((step, index) => {
            const isVisible = index >= this.currentIndex && index < this.currentIndex + visibleCount;
            step.classList.toggle('active', isVisible);
        });

        // Update connector highlights
        connectors.forEach((connector, index) => {
            const isActive = index >= this.currentIndex && index < this.currentIndex + visibleCount - 1;
            connector.classList.toggle('active', isActive);
        });
    }
}

// Initialize all carousels on the page
function initCarousels() {
    const containers = document.querySelectorAll('.trajectory-container');
    containers.forEach(container => {
        new VideoCarousel(container);
    });
}

// ========== Smooth Scroll ==========
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ========== Scroll to Top ==========
function setupScrollToTop() {
    const scrollBtn = document.createElement('button');
    scrollBtn.className = 'scroll-to-top';
    scrollBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
    scrollBtn.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: var(--accent-color, #3498db);
        color: white;
        border: none;
        cursor: pointer;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s, visibility 0.3s;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    `;
    document.body.appendChild(scrollBtn);

    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            scrollBtn.style.opacity = '1';
            scrollBtn.style.visibility = 'visible';
        } else {
            scrollBtn.style.opacity = '0';
            scrollBtn.style.visibility = 'hidden';
        }
    });

    scrollBtn.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// ========== Initialize ==========
document.addEventListener('DOMContentLoaded', () => {
    setupSmoothScroll();
    setupScrollToTop();
    initCarousels();
});
