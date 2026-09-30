function copyBibtex(button) {
    const text = document.querySelector('.bibtex code').textContent;
    navigator.clipboard.writeText(text).then(() => {
        button.textContent = 'Copied';
        setTimeout(() => { button.textContent = 'Copy'; }, 1500);
    });
}

function setupTabs() {
    document.querySelectorAll('.tabs').forEach((group) => {
        const buttons = group.querySelectorAll('[role="tab"]');
        buttons.forEach((button) => {
            button.addEventListener('click', () => {
                buttons.forEach((other) => {
                    const selected = other === button;
                    other.setAttribute('aria-selected', String(selected));
                    const panel = document.getElementById(other.getAttribute('aria-controls'));
                    panel.hidden = !selected;
                    panel.querySelectorAll('video').forEach((video) => {
                        if (!selected) {
                            video.pause();
                        } else if (!video.controls) {
                            video.play().catch(() => {});
                        }
                    });
                });
            });
        });
    });
}

function setupVideos() {
    const videos = document.querySelectorAll('video');

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        videos.forEach((video) => { video.controls = true; });
        return;
    }

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.play().catch(() => {});
            } else {
                entry.target.pause();
            }
        });
    }, { threshold: 0.25 });

    videos.forEach((video) => observer.observe(video));
}

document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupVideos();
});
