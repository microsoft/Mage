document.addEventListener('DOMContentLoaded', function () {
  var switcher = document.querySelector('.rf-project-switcher');
  var trigger = document.querySelector('.rf-project-trigger');
  if (!switcher || !trigger) return;

  trigger.addEventListener('click', function () {
    var open = !switcher.classList.contains('is-open');
    switcher.classList.toggle('is-open', open);
    trigger.setAttribute('aria-expanded', String(open));
  });

  document.addEventListener('click', function (event) {
    if (!switcher.contains(event.target)) closeSwitcher();
  });

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
      closeSwitcher();
      trigger.blur();
    }
  });

  function closeSwitcher() {
    switcher.classList.remove('is-open');
    trigger.setAttribute('aria-expanded', 'false');
  }
});
