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
// ผูก event แบบ delegation เพราะแถวถูกเติมเข้า tbody ทีละแถวตอนไล่ประมวลผล ไม่ได้มีตั้งแต่โหลดหน้า
document.getElementById('rows').addEventListener('click',e=>{
  const tr=e.target.closest('.row');
  if(tr) openDrawer(tr.dataset.ref);
});

// ไล่ประมวลผลทีละ transaction (auto-play) — จำลองว่า transaction เข้ามาให้ระบบตรวจทีละใบ
const STATUS_LABEL={red:'Undervalue',orange:'Overvalue',green:'Normal',unknown:'Unknown'};
const GUARD_MSG={
  red:'🚩 สงสัยว่าสำแดงราคาต่ำผิดปกติ',
  orange:'🔶 สงสัยว่าสำแดงราคาสูงผิดปกติ',
  green:'✓ ไม่พบความผิดปกติ',
  unknown:'⚪ ไม่มีข้อมูลอ้างอิงให้ตัดสิน',
};
const PROCESS_MS=650, GAP_MS=250;
const ROWS=JSON.parse(document.getElementById('rows-data').textContent);
const rowsBody=document.getElementById('rows');
const counts={red:0,orange:0,green:0,unknown:0};
let idx=0, skipped=false;

function sleep(ms){return new Promise(res=>setTimeout(res,ms));}

function rowHtml(r){
  return `<tr class="row ch-${r.status}" data-ref="${r.decl_id}" data-status="${r.status}"
      data-s="${(r.decl_no+' '+r.importer+' '+r.importer_eng+' '+r.trfcls+' '+r.gdsdscth).toLowerCase()}">
    <td class="mono">${r.decl_no}</td>
    <td class="mono dim">${r.date_disp}</td>
    <td><span class="kind import">ขาเข้า</span></td>
    <td class="l"><div class="tname">${r.importer}</div><div class="tprofile">${r.importer_eng}</div></td>
    <td class="mono">${r.trfcls}</td>
    <td class="mono">${r.topic ?? '-'}</td>
    <td class="r mono">${r.price_per_kg}</td>
    <td class="r mono">${r.group_mean_kg ?? '-'}</td>
    <td class="l"><div class="tprofile">${r.gdsdscth}</div></td>
    <td><span class="status ${r.status}">${STATUS_LABEL[r.status]}</span></td>
  </tr>`;
}

function showProcessing(r){
  document.getElementById('stagebadge').className='stage-badge processing';
  document.getElementById('stagebadge').textContent='กำลังประมวลผล...';
  document.getElementById('stageref').textContent=r.decl_no;
  document.getElementById('stagedesc').textContent=r.gdsdscth||r.gdsdsc||'';
}

function commitRow(r){
  rowsBody.insertAdjacentHTML('beforeend',rowHtml(r));
  counts[r.status]++;
  document.getElementById('kpi-'+r.status).textContent=counts[r.status];
  document.getElementById('kpi-total').textContent=(idx+1);
  applyFilters();
}

function revealResult(r){
  const badge=document.getElementById('stagebadge');
  badge.className='stage-badge '+r.status;
  badge.textContent=STATUS_LABEL[r.status];
  document.getElementById('stagedesc').textContent=GUARD_MSG[r.status];
  commitRow(r);
  document.getElementById('stagecount').textContent=(idx+1)+' / '+ROWS.length;
  document.getElementById('stagefill').style.width=(100*(idx+1)/ROWS.length)+'%';
}

async function playNext(){
  if(skipped || idx>=ROWS.length){ finish(); return; }
  const r=ROWS[idx];
  showProcessing(r);
  await sleep(PROCESS_MS);
  if(skipped) return;
  revealResult(r);
  idx++;
  await sleep(GAP_MS);
  playNext();
}

function finish(){
  if(skipped){
    for(;idx<ROWS.length;idx++) commitRow(ROWS[idx]);
    document.getElementById('stagecount').textContent=ROWS.length+' / '+ROWS.length;
    document.getElementById('stagefill').style.width='100%';
  }
  document.getElementById('stagebadge').className='stage-badge done';
  document.getElementById('stagebadge').textContent='เสร็จสิ้น';
  document.getElementById('stageref').textContent='—';
  document.getElementById('stagedesc').textContent='ประมวลผลครบ '+ROWS.length+' รายการแล้ว';
  document.getElementById('skipbtn').disabled=true;
}

document.getElementById('skipbtn').addEventListener('click',()=>{skipped=true;finish();});

applyFilters();
if(ROWS.length) playNext(); else finish();
