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
// สีตอนนี้บอก "ความรุนแรง" ของส่วนต่างราคาอย่างเดียว (ไม่บอกทิศทาง undervalue/overvalue แล้ว — ดูทิศทางได้
// ตอนเปิดรายละเอียด/drawer แทน — ดู webapp/main.py._severity/_row_view)
const STATUS_LABEL={red:'แดง',yellow:'เหลือง',green:'เขียว',no_model:'No Model',new_cluster:'New Cluster'};
const GUARD_MSG={
  red:'🚩 แดง',
  yellow:'🔶 เหลือง',
  green:'✓ เขียว',
  no_model:'⚪ ยังไม่มีพิกัดนี้ในข้อมูล train',
  new_cluster:'🔵 train แล้ว แต่เจอ cluster ใหม่',
};
const PROCESS_MS=650, GAP_MS=250;
const ROWS=JSON.parse(document.getElementById('rows-data').textContent);
const rowsBody=document.getElementById('rows');
let idx=0, skipped=false;

// อัปเดตตัวเลข KPI (ประมวลผลแล้ว/Undervalue/Overvalue/...) จาก summary ที่ server คำนวณมาให้ (ทั้งระบบ ไม่
// ผูกกับหน้า pagination ใดๆ — ดู pipeline._system_wide_stats) ไม่ใช่นับจากแถวที่ commit ในหน้านี้เอง เพราะ
// หน้านี้อาจมีแค่บางส่วนของทั้งระบบ (ดู webapp/main.py, webapp/pipeline.py)
function updateKpisFromSummary(summary){
  document.getElementById('kpi-total').textContent=summary.n_processed_total;
  document.getElementById('kpi-red').textContent=summary.n_red_total;
  document.getElementById('kpi-yellow').textContent=summary.n_yellow_total;
  document.getElementById('kpi-green').textContent=summary.n_green_total;
  document.getElementById('kpi-no_model').textContent=summary.n_no_model_total;
  document.getElementById('kpi-new_cluster').textContent=summary.n_new_cluster_total;
  const allX=document.querySelector('.kpi[data-f="all"] .kpi-x');
  if(allX) allX.textContent='จาก '+summary.total_rows+' รายการ';
}

function sleep(ms){return new Promise(res=>setTimeout(res,ms));}

function rowHtml(r){
  return `<tr class="row ch-${r.status}" data-ref="${r.decl_id}" data-status="${r.status}"
      data-s="${(r.decl_no+' '+r.importer+' '+r.importer_eng+' '+r.trfcls+' '+r.gdsdscth).toLowerCase()}">
    <td class="mono">${r.decl_no}</td>
    <td class="mono dim">${r.date_disp}</td>
    <td><span class="kind import">ขาเข้า</span></td>
    <td class="l"><div class="tname">${r.importer}</div><div class="tprofile">${r.importer_eng}</div></td>
    <td class="mono">${r.trfcls}</td>
    <td class="r mono">${r.price_per_kg}</td>
    <td class="r mono">${r.group_mean_kg ?? '-'}</td>
    <td class="l"><div class="tprofile">${r.gdsdscth}</div></td>
    <td class="r">${r.diff_score === '-' ? '-' : `<span class="score-badge ${r.status}">${r.diff_score}</span>`}</td>
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

// แถวที่ sync มาตั้งแต่ตอนโหลดหน้าครั้งแรก (ของเก่า/รอบ sync แรก) โชว์ทันทีเลย ไม่ต้องรอ animation ไล่
// ทีละแถว — เก็บ animation "กำลังประมวลผล..." ไว้ใช้กับแถวที่โผล่มาใหม่จริงๆผ่าน polling หลังจากนี้เท่านั้น
// (skipped ยังเป็น false อยู่ — pollNew() จะเห็นว่า idx>=ROWS.length ตั้งแต่แรกแล้วเรียก playNext() ให้
// แถวใหม่ที่ push เข้ามาทีหลัง animate ตามปกติ)
applyFilters();
for(;idx<ROWS.length;idx++) commitRow(ROWS[idx]);
if(ROWS.length){
  document.getElementById('stagecount').textContent=ROWS.length+' / '+ROWS.length;
  document.getElementById('stagefill').style.width='100%';
}
document.getElementById('stagebadge').className='stage-badge done';
document.getElementById('stagebadge').textContent='เสร็จสิ้น';
document.getElementById('stageref').textContent='—';
document.getElementById('stagedesc').textContent='ประมวลผลครบ '+ROWS.length+' รายการแล้ว';
document.getElementById('skipbtn').disabled=true;

// Polling หาแถวใหม่จาก Oracle เป็นระยะ (ไม่ใช่ WebSocket/SSE — เรียบง่ายกว่าสำหรับ demo นี้) stateless
// ฝั่ง server เต็มที่ (ดู webapp/main.py /api/poll) — client (ตัวนี้เอง) เป็นฝ่ายจำ lastLoadTs (LOAD_TS
// ล่าสุดที่ตัวเองมีอยู่แล้ว) แล้วส่งไปเป็น ?since= ทุกครั้ง เพื่อขอแค่แถวที่ใหม่กว่านั้น — ROWS/idx เป็นตัว
// เดียวกับที่ playNext() ใช้อยู่ แค่ push แถวใหม่เข้า ROWS แล้วให้ playNext() เดินมาถึงเอง (ถ้ากำลังเล่นอยู่)
// หรือสั่งเริ่มใหม่เอง (ถ้า idle ไปแล้วก่อนหน้า)
//
// เริ่มต้น lastLoadTs จาก INITIAL_MAX_LOAD_TS ที่ server คำนวณมาให้ (LOAD_TS สูงสุดของ "ทั้งระบบ" ไม่ใช่
// แค่หน้านี้ — ดู webapp/main.py _run_and_load) ถ้า derive จากแค่ ROWS ของหน้านี้ (page 1 cap ที่
// DEFAULT_PAGE_SIZE) แถวที่ตกไปอยู่หน้าอื่นแต่ LOAD_TS ใหม่กว่าทุกแถวที่โชว์อยู่ในหน้านี้จะ "หลุด" เข้ามา
// ผ่าน poll หลังรีเฟรชไม่กี่วินาที เหมือนเป็นแถวใหม่ทั้งที่จริงมีอยู่ในระบบตั้งแต่ต้นแล้ว (แค่ตกหน้าอื่นไป
// เพราะ pagination) — ยังวน ROWS ซ้ำไว้เป็น fallback เผื่อ INITIAL_MAX_LOAD_TS เป็น null (ระบบยังไม่มีข้อมูล
// สักแถวเลย)
let lastLoadTs = (typeof INITIAL_MAX_LOAD_TS !== 'undefined') ? INITIAL_MAX_LOAD_TS : null;
for(const r of ROWS){ if(r.load_ts && (!lastLoadTs || r.load_ts>lastLoadTs)) lastLoadTs=r.load_ts; }

const POLL_MS=4000;
async function pollNew(){
  try{
    const url=lastLoadTs?('/api/poll?since='+encodeURIComponent(lastLoadTs)):'/api/poll';
    const r=await fetch(url);
    if(r.ok){
      const data=await r.json();
      if(data.rows && data.rows.length){
        const wasIdle = idx>=ROWS.length;
        ROWS.push(...data.rows);
        for(const row of data.rows){ if(row.load_ts && (!lastLoadTs || row.load_ts>lastLoadTs)) lastLoadTs=row.load_ts; }
        if(data.summary) updateKpisFromSummary(data.summary);
        if(skipped){
          for(;idx<ROWS.length;idx++) commitRow(ROWS[idx]);
          document.getElementById('stagecount').textContent=ROWS.length+' / '+ROWS.length;
          document.getElementById('stagefill').style.width='100%';
        }else if(wasIdle){
          document.getElementById('skipbtn').disabled=false;
          playNext();
        }
      }
    }
  }catch(e){/* พลาดรอบเดียวเงียบไว้ได้ — ลองใหม่รอบถัดไปเอง ไม่ต้องรบกวนผู้ใช้ */}
  setTimeout(pollNew,POLL_MS);
}
setTimeout(pollNew,POLL_MS);
