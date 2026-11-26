document.querySelectorAll('[data-loading-form]').forEach(form=>form.addEventListener('submit',()=>form.classList.add('is-loading')));
