/* MeowScope service worker — network-first so users always get the latest page;
   cache only as offline fallback. */
const CACHE='meowscope-v7';
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
    fetch(e.request).then(res=>{
      if(e.request.url.startsWith(self.location.origin)){
        const cl=res.clone();
        caches.open(CACHE).then(c=>c.put(e.request, cl));
      }
      return res;
    }).catch(()=>caches.match(e.request).then(hit=>hit || caches.match('./')))
  );
});
