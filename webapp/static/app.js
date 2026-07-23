// นาฬิกา
const TH_MONTHS=['','มกราคม','กุมภาพันธ์','มีนาคม','เมษายน','พฤษภาคม','มิถุนายน','กรกฎาคม','สิงหาคม','กันยายน','ตุลาคม','พฤศจิกายน','ธันวาคม'];
function tick(){
  const d=new Date();
  const ymd=d.toLocaleDateString('en-CA',{timeZone:'Asia/Bangkok'});
  const [Y,M,D]=ymd.split('-');
  const t=d.toLocaleTimeString('en-GB',{timeZone:'Asia/Bangkok',hour12:false});
  const el=document.getElementById('clock');
  if(el) el.textContent='วันที่ '+(+D)+' '+TH_MONTHS[+M]+' พ.ศ. '+((+Y)+543)+' เวลา '+t+' น.';
}
setInterval(tick,1000); tick();

// ตัวกรอง
const state={status:'all',q:''};
function applyFilters(){
  const q=state.q.trim().toLowerCase();
  let shown=0;
  document.querySelectorAll('#rows .row').forEach(tr=>{
    let ok=true;
    if(state.status!=='all') ok = tr.dataset.status===state.status;
    if(ok && q) ok = tr.dataset.s.toLowerCase().includes(q);
    tr.style.display = ok?'':'none';
    if(ok) shown++;
  });
  document.getElementById('count').textContent='แสดง '+shown+' รายการ';
}
document.querySelectorAll('#seg-status button').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('#seg-status button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); state.status=b.dataset.v; syncKpi(); applyFilters();
}));
document.getElementById('q').addEventListener('input',e=>{state.q=e.target.value;applyFilters();});

// คลิก KPI = กรองสถานะ
document.querySelectorAll('.kpi').forEach(k=>{
  if(!k.dataset.f) return;
  k.addEventListener('click',()=>{
    state.status=k.dataset.f;
    document.querySelectorAll('#seg-status button').forEach(x=>x.classList.toggle('on',x.dataset.v===state.status));
    syncKpi(); applyFilters();
  });
});
function syncKpi(){document.querySelectorAll('.kpi').forEach(k=>k.classList.toggle('active',k.dataset.f===state.status));}

// drawer
const bg=document.getElementById('drawerbg'), dr=document.getElementById('drawer'), body=document.getElementById('drawerbody');
async function openDrawer(ref){
  body.innerHTML='<div style="padding:40px;text-align:center;color:#6b7a8d">กำลังโหลด...</div>';
  bg.classList.add('show'); dr.classList.add('show');
  const r=await fetch('/d/'+encodeURIComponent(ref));
  body.innerHTML = r.ok ? await r.text() : '<div style="padding:40px">ไม่พบข้อมูล</div>';
}
function closeDrawer(){bg.classList.remove('show');dr.classList.remove('show');}
bg.addEventListener('click',closeDrawer);
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeDrawer();});
document.querySelectorAll('#rows .row').forEach(tr=>tr.addEventListener('click',()=>openDrawer(tr.dataset.ref)));

applyFilters();
