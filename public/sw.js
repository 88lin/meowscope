/* MeowScope service worker — network-first so users always get the latest page;
   cache only as offline fallback. Bump CACHE whenever public/ assets change. */
const CACHE='meowscope-v10';
const ASSETS=[
  './','./index.html','./features.js','./i18n.js',
  './forest.json','./metrics.json','./manifest.webmanifest',
  './icon-192.png','./icon-512.png','./favicon.svg','./og-image.png'
];
self.addEventListener('install', e=>{
  e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)).then(()=>self.skipWaiting()));
});
self.addEventListener('activate', e=>{
  e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));
});
self.addEventListener('fetch', e=>{
  if(e.request.method!=='GET') return;
  const url=new URL(e.request.url);
  if(url.origin!==self.location.origin) return;
  e.respondWith(
    fetch(e.request).then(res=>{
      // 只缓存同源的成功响应；opaque / 错误响应会让离线回退拿到坏数据
      if(res && res.ok && res.type==='basic'){
        const cl=res.clone();
        caches.open(CACHE).then(c=>c.put(e.request, cl));
      }
      return res;
    }).catch(()=>caches.match(e.request).then(hit=>hit || caches.match('./index.html')))
  );
});
