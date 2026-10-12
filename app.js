(() => {
  'use strict';

  const videos = [...document.querySelectorAll('video')];
  videos.forEach(video => {
    video.addEventListener('play', () => {
      videos.forEach(other => { if (other !== video) other.pause(); });
    });
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) videos.forEach(video => video.pause());
  });

  const walkthrough = document.querySelector('#upload-walkthrough');
  if (walkthrough && 'IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.intersectionRatio >= .25 && !document.hidden) {
        if (!videos.some(video => video !== walkthrough && !video.paused)) {
          walkthrough.play().catch(() => {});
        }
      } else {
        walkthrough.pause();
      }
    }, { threshold: [0, .25] });
    observer.observe(walkthrough);
  }
})();
