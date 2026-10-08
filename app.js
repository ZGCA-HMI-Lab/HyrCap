(() => {
  'use strict';

  const toggle = document.querySelector('#language-toggle');
  let language = 'en';
  try {
    if (localStorage.getItem('hyrcap-language') === 'zh') language = 'zh';
  } catch {}

  function applyLanguage() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-en][data-zh]').forEach(node => {
      node.textContent = node.dataset[language];
    });
    document.querySelectorAll('[data-label-en][data-label-zh]').forEach(node => {
      node.setAttribute('aria-label', language === 'zh' ? node.dataset.labelZh : node.dataset.labelEn);
    });
    toggle.textContent = language === 'en' ? '中文' : 'English';
    toggle.setAttribute('aria-label', language === 'en' ? 'Switch to Chinese' : 'Switch to English');
  }

  toggle.addEventListener('click', () => {
    language = language === 'en' ? 'zh' : 'en';
    try { localStorage.setItem('hyrcap-language', language); } catch {}
    applyLanguage();
  });

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

  applyLanguage();
})();
