// Keyboard-operated tabs with shareable URLs; original-figure enlargement.
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(tab) {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    item.classList.toggle(`active-${item.dataset.color}`, selected);
    const panel = document.getElementById(item.getAttribute('aria-controls'));
    panel.hidden = !selected;
    panel.classList.toggle('active', selected);
  }
}
function activateTab(tab) {
  selectTab(tab);
  history.replaceState(null, '', `#${tab.getAttribute('aria-controls')}`);
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault();
    activateTab(tabs[next]);
    tabs[next].focus();
  });
});
function revealLinkedContent() {
  const id = location.hash.slice(1);
  const tab = tabs.find(item => item.getAttribute('aria-controls') === id);
  if (tab) {
    selectTab(tab);
    document.getElementById(id).scrollIntoView({block:'start'});
  }
  const target = document.getElementById(id);
  if (target && target.tagName === 'DETAILS') target.open = true;
}
window.addEventListener('hashchange', revealLinkedContent);
revealLinkedContent();
const lightbox = document.querySelector('.cw-lightbox');
for (const link of document.querySelectorAll('.cw-zoom')) {
  link.addEventListener('click', event => {
    event.preventDefault();
    const img = lightbox.querySelector('img');
    img.src = link.href;
    img.alt = link.querySelector('img').alt;
    lightbox.querySelector('figcaption').textContent = link.dataset.caption;
    lightbox.querySelector('.cw-original-pdf').href = link.dataset.pdf;
    lightbox.querySelector('.cw-original-image').href = link.href;
    lightbox.showModal();
  });
}
lightbox.querySelector('button').addEventListener('click', () => lightbox.close());
lightbox.addEventListener('click', event => { if (event.target === lightbox) lightbox.close(); });
// Both desktop and mobile section navigation follow the reading position.
const navigation = [...document.querySelectorAll('.cw-nav-links a, .cw-mobile-nav a')];
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    for (const link of navigation) {
      if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }
}, {rootMargin:'-15% 0px -65% 0px'});
for (const id of ['highlights','data','benchmarks','training']) observer.observe(document.getElementById(id));
