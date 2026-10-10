/* MeowScope service worker — cache-first shell for offline use (PWA) */
const CACHE='meowscope-v6';
const ASSETS=[
  './','./index.html','./features.js','./i18n.js','./demos.js',
  './forest.json','./metrics.json','./manifest.webmanifest','./icon-192.png','./icon-512.png'
];
self.addEventListener('install', e=>{
  e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)).then(()=>self.skipWaiting()));
});
self.addEventListener('activate', e=>{
  e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));
});
self.addEventListener('fetch', e=>{
  if(e.request.method!=='GET') return;
  e.respondWith(
    caches.match(e.request).then(hit=>hit || fetch(e.request).then(res=>{
      // cache same-origin GETs opportunistically
      if(e.request.url.startsWith(self.location.origin)){
        const cl=res.clone();
        caches.open(CACHE).then(c=>c.put(e.request, cl));
      }
      return res;
    }).catch(()=>hit))
  );
});
