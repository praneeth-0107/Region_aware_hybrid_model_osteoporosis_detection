/**
 * BoneHealth AI — Frontend Application
 */

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const uploadPreview = document.getElementById('uploadPreview');
const previewImg = document.getElementById('previewImg');
const previewName = document.getElementById('previewName');
const previewSize = document.getElementById('previewSize');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('results-section');

let selectedFile = null;
let ensembleChartInstance = null;
let comparisonChartInstance = null;
let lastResultData = null;

// ── File Upload ─────────────────────────────────────
uploadBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault(); uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(file) {
    if (!file.name.match(/\.(png|jpg|jpeg|bmp|tiff|tif)$/i)) {
        alert('Please upload a valid image file.'); return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => { previewImg.src = e.target.result; };
    reader.readAsDataURL(file);
    previewName.textContent = file.name;
    previewSize.textContent = file.size < 1024*1024 ? (file.size/1024).toFixed(1)+' KB' : (file.size/1024/1024).toFixed(1)+' MB';
    uploadPreview.classList.add('show');
    analyzeBtn.classList.add('show');
    analyzeBtn.disabled = false;
}

removeBtn.addEventListener('click', () => {
    selectedFile = null; fileInput.value = '';
    uploadPreview.classList.remove('show'); analyzeBtn.classList.remove('show'); analyzeBtn.disabled = true;
});

// ── Analyze ─────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    analyzeBtn.disabled = true; showLoading();
    const fd = new FormData(); fd.append('image', selectedFile);
    try {
        const resp = await fetch('/api/predict', { method: 'POST', body: fd });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Prediction failed');
        hideLoading(); renderResults(data);
    } catch (err) { hideLoading(); alert('Error: ' + err.message); analyzeBtn.disabled = false; }
});

// ── Loading ─────────────────────────────────────────
function showLoading() {
    loadingOverlay.classList.add('show');
    ['step1','step2','step3'].forEach(s => { document.getElementById(s).className = 'loading-step'; });
    setTimeout(() => document.getElementById('step1').classList.add('active'), 300);
    setTimeout(() => { document.getElementById('step1').classList.replace('active','done'); document.getElementById('step2').classList.add('active'); }, 2000);
    setTimeout(() => { document.getElementById('step2').classList.replace('active','done'); document.getElementById('step3').classList.add('active'); }, 4000);
}
function hideLoading() {
    document.getElementById('step3').classList.replace('active','done');
    setTimeout(() => loadingOverlay.classList.remove('show'), 500);
}

// ── Render Results ──────────────────────────────────
function renderResults(data) {
    lastResultData = data;
    const e = data.ensemble;
    const riskClass = e.risk_level === 'HIGH RISK' ? 'high-risk' : e.risk_level === 'MEDIUM RISK' ? 'medium-risk' : 'healthy';
    const badgeClass = e.risk_level === 'HIGH RISK' ? 'high' : e.risk_level === 'MEDIUM RISK' ? 'medium' : 'low';

    document.getElementById('resultHero').className = 'result-hero ' + riskClass;
    document.getElementById('resultBadge').className = 'result-badge ' + badgeClass;
    document.getElementById('resultBadge').textContent = '● ' + e.risk_level;
    document.getElementById('resultDiagnosis').textContent = e.prediction;
    document.getElementById('resultConfidence').textContent = e.confidence + '%';

    document.getElementById('resultMeta').innerHTML = `
        <div class="result-meta-item"><div class="label">Timestamp</div><div class="value">${data.timestamp}</div></div>
        <div class="result-meta-item"><div class="label">Femur Weight</div><div class="value">${e.femur_weight}%</div></div>
        <div class="result-meta-item"><div class="label">Tibia Weight</div><div class="value">${e.tibia_weight}%</div></div>
        <div class="result-meta-item"><div class="label">Segmentation</div><div class="value">${data.segmentation.mode}</div></div>`;

    renderProbBars('femurBars', data.femur);
    renderProbBars('tibiaBars', data.tibia);
    renderEnsembleChart(data);
    renderComparisonChart(data);
    renderImages(data.images);
    resultsSection.classList.add('show');
    setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 200);
    analyzeBtn.disabled = false;
}

function renderProbBars(id, siteData) {
    const c = document.getElementById(id); c.innerHTML = '';
    for (const [cls, val] of Object.entries(siteData.probabilities)) {
        const isPred = cls === siteData.prediction;
        const item = document.createElement('div'); item.className = 'prob-bar-item';
        item.innerHTML = `<div class="prob-bar-header"><span class="prob-bar-label" style="${isPred?'color:var(--accent)':''}">${cls} ${isPred?'◄':''}</span><span class="prob-bar-value" style="color:${isPred?'var(--accent)':'var(--text-secondary)'}">${val.toFixed(1)}%</span></div><div class="prob-bar-track"><div class="prob-bar-fill ${cls}" style="width:0%"></div></div>`;
        c.appendChild(item);
        setTimeout(() => { item.querySelector('.prob-bar-fill').style.width = val + '%'; }, 100);
    }
}

function renderEnsembleChart(data) {
    const ctx = document.getElementById('ensembleChart').getContext('2d');
    if (ensembleChartInstance) ensembleChartInstance.destroy();
    const p = data.ensemble.probabilities;
    ensembleChartInstance = new Chart(ctx, {
        type: 'doughnut', data: { labels: Object.keys(p).map(s=>s.charAt(0).toUpperCase()+s.slice(1)),
            datasets: [{ data: Object.values(p), backgroundColor: ['rgba(239,68,68,0.8)','rgba(245,158,11,0.8)','rgba(34,197,94,0.8)'],
                borderColor: ['#ef4444','#f59e0b','#22c55e'], borderWidth: 2, hoverOffset: 10 }] },
        options: { responsive: true, maintainAspectRatio: true, cutout: '60%',
            plugins: { legend: { position:'bottom', labels:{color:'#94a3b8',padding:16,font:{family:'Inter',size:12}}},
                tooltip:{backgroundColor:'#1e293b',titleColor:'#f1f5f9',bodyColor:'#94a3b8',callbacks:{label:c=>` ${c.label}: ${c.parsed.toFixed(1)}%`}}}}
    });
}

function renderComparisonChart(data) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    if (comparisonChartInstance) comparisonChartInstance.destroy();
    const L = ['Osteoporosis','Osteopenia','Normal'];
    comparisonChartInstance = new Chart(ctx, {
        type: 'bar', data: { labels: L, datasets: [
            {label:'Femur',data:L.map(l=>data.femur.probabilities[l.toLowerCase()]),backgroundColor:'rgba(6,182,212,0.7)',borderColor:'#06b6d4',borderWidth:1,borderRadius:6},
            {label:'Tibia',data:L.map(l=>data.tibia.probabilities[l.toLowerCase()]),backgroundColor:'rgba(167,139,250,0.7)',borderColor:'#a78bfa',borderWidth:1,borderRadius:6},
            {label:'Ensemble',data:L.map(l=>data.ensemble.probabilities[l.toLowerCase()]),backgroundColor:'rgba(20,184,166,0.7)',borderColor:'#14b8a6',borderWidth:1,borderRadius:6}]},
        options:{responsive:true,maintainAspectRatio:true,
            plugins:{legend:{position:'bottom',labels:{color:'#94a3b8',padding:16,font:{family:'Inter',size:12}}}},
            scales:{x:{ticks:{color:'#64748b'},grid:{color:'rgba(255,255,255,0.04)'}},y:{ticks:{color:'#64748b',callback:v=>v+'%'},grid:{color:'rgba(255,255,255,0.04)'},max:100}}}
    });
}

function renderImages(images) {
    document.getElementById('resultImages').innerHTML = [
        {src:images.original,label:'Original X-Ray'},{src:images.annotated,label:'Annotated (Joint Line Detected)'},
        {src:images.femur_crop,label:'Femur Region (Segmented)'},{src:images.tibia_crop,label:'Tibia Region (Segmented)'}
    ].map(i=>`<div class="image-item"><img src="${i.src}" alt="${i.label}"><div class="caption">${i.label}</div></div>`).join('');
}

// ── Scroll Animations ───────────────────────────────
const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.1 });
document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

// ── Recommendations Data ────────────────────────────
function getRecs(pred) {
    const R = {
        osteoporosis: [
            'Urgently consult an orthopedic specialist or endocrinologist.',
            'Request a DEXA scan to confirm bone mineral density T-score.',
            'Begin prescribed bisphosphonate therapy as recommended by physician.',
            'Increase calcium intake to 1200mg/day (dairy, leafy greens, supplements).',
            'Take Vitamin D supplements (800-1000 IU/day).',
            'Engage in supervised weight-bearing exercises.',
            'Implement fall prevention measures at home.',
            'Avoid smoking and limit alcohol consumption.',
            'Schedule follow-up bone density scans every 1-2 years.',
            'Consider hormone replacement therapy if post-menopausal (consult doctor).'
        ],
        osteopenia: [
            'Schedule appointment with primary care physician for bone health assessment.',
            'Consider a baseline DEXA scan for accurate bone mineral density measurement.',
            'Increase daily calcium intake to 1000-1200mg through calcium-rich foods.',
            'Ensure adequate Vitamin D (600-800 IU/day); request blood test.',
            'Begin regular weight-bearing exercises (brisk walking, jogging) 30 min/day.',
            'Add resistance training 2-3 times per week.',
            'Quit smoking if applicable.',
            'Limit alcohol to 1-2 drinks per day maximum.',
            'Maintain balanced diet rich in fruits, vegetables, lean protein.',
            'Monitor bone density with follow-up scans every 2-3 years.'
        ],
        normal: [
            'Continue maintaining your healthy lifestyle.',
            'Ensure daily calcium intake of 1000mg through balanced diet.',
            'Maintain Vitamin D levels with 600 IU/day and sun exposure.',
            'Continue regular physical activity including weight-bearing exercises.',
            'Schedule routine bone density screening after age 50.',
            'Maintain a healthy body weight.',
            'Stay hydrated and consume adequate protein.',
            'Limit excessive caffeine which may affect calcium absorption.'
        ]
    };
    return R[pred] || R.normal;
}

// ── PDF Download (jsPDF — no overlap) ───────────────
function downloadPDF() {
    if (!lastResultData) { alert('No results yet.'); return; }
    const btn = document.getElementById('downloadPdfBtn');
    btn.classList.add('generating'); btn.textContent = 'Generating...';

    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF('portrait','mm','a4');
        const d = lastResultData, ens = d.ensemble;
        const W=210, M=15, CW=W-2*M;
        let y=0;

        // Single text helper — no duplicates
        function T(s,x,yy,sz,st,col,al){
            doc.setFontSize(sz||10); doc.setFont('helvetica',st||'normal'); doc.setTextColor(col||'#1a1a2e');
            if(al) doc.text(s,x,yy,{align:al}); else doc.text(s,x,yy);
        }
        function L(yy){doc.setDrawColor(200,210,220);doc.setLineWidth(0.3);doc.line(M,yy,W-M,yy);}
        function B(x,yy,v,mw,c){doc.setFillColor(241,245,249);doc.roundedRect(x,yy,mw,4,2,2,'F');if(v>0){doc.setFillColor(c[0],c[1],c[2]);doc.roundedRect(x,yy,Math.max(v/100*mw,1),4,2,2,'F');}}
        function PG(n){if(y+n>280){doc.addPage();y=M;}}

        const CC={osteoporosis:[239,68,68],osteopenia:[245,158,11],normal:[34,197,94]};
        const rc=ens.risk_level==='HIGH RISK'?[220,38,38]:ens.risk_level==='MEDIUM RISK'?[217,119,6]:[22,163,74];

        // HEADER
        doc.setFillColor(6,182,212); doc.rect(0,0,W,26,'F');
        T('BoneHealth AI - Diagnostic Report',M,11,16,'bold','#ffffff');
        T(d.timestamp+' | ID: '+d.run_id,W-M,11,8,'normal','#ccfbf1','right');
        T('Region-Aware Hybrid Ensemble Model',M,19,8,'normal','#ccfbf1');
        y=34;

        // DIAGNOSIS
        doc.setFillColor(rc[0],rc[1],rc[2]); doc.roundedRect(M,y,CW,24,3,3,'F');
        T('FINAL DIAGNOSIS',W/2,y+7,8,'normal','#ffffff','center');
        T(ens.prediction.toUpperCase(),W/2,y+15,14,'bold','#ffffff','center');
        T(ens.risk_level+' | Confidence: '+ens.confidence+'%',W/2,y+21,8,'normal','#ffffff','center');
        y+=32;

        // META
        doc.setFillColor(248,250,252); doc.roundedRect(M,y,CW,10,2,2,'F');
        T('Mode: '+d.segmentation.mode+'   |   Size: '+d.segmentation.image_width+'x'+d.segmentation.image_height+'   |   Femur Wt: '+ens.femur_weight+'%   |   Tibia Wt: '+ens.tibia_weight+'%',W/2,y+6.5,7,'normal','#475569','center');
        y+=16;

        // REGION TABLE
        T('PER-REGION SUMMARY',M,y,10,'bold','#0e7490'); y+=5; L(y); y+=5;
        doc.setFillColor(241,245,249); doc.rect(M,y-2,CW,7,'F');
        T('Region',M+3,y+3,8,'bold','#334155'); T('Prediction',M+50,y+3,8,'bold','#334155');
        T('Confidence',M+100,y+3,8,'bold','#334155'); T('Risk Level',M+140,y+3,8,'bold','#334155');
        y+=9;
        [{n:'Femur',p:d.femur.prediction,c:d.femur.confidence,r:d.femur.risk_level},
         {n:'Tibia',p:d.tibia.prediction,c:d.tibia.confidence,r:d.tibia.risk_level},
         {n:'Ensemble',p:ens.prediction,c:ens.confidence,r:ens.risk_level}].forEach((r,i)=>{
            if(i===2){doc.setFillColor(240,249,255);doc.rect(M,y-3,CW,8,'F');}
            const s=i===2?'bold':'normal';
            T(r.n,M+3,y+2,8,s,'#1e293b'); T(r.p.charAt(0).toUpperCase()+r.p.slice(1),M+50,y+2,8,s,'#1e293b');
            T(r.c+'%',M+100,y+2,8,s,'#1e293b'); T(r.r,M+140,y+2,8,s,i===2?'#0e7490':'#475569');
            y+=8;
        });
        y+=6;

        // PROB BARS
        PG(45); T('PROBABILITY BREAKDOWN',M,y,10,'bold','#0e7490'); y+=5; L(y); y+=6;
        const bW=45;
        const SS=[{l:'FEMUR',p:d.femur.probabilities},{l:'TIBIA',p:d.tibia.probabilities},{l:'ENSEMBLE',p:d.ensemble.probabilities}];
        SS.forEach((s,i)=>T(s.l,M+i*62,y,8,'bold','#475569')); y+=5;
        ['osteoporosis','osteopenia','normal'].forEach(cls=>{
            PG(10); T(cls.charAt(0).toUpperCase()+cls.slice(1),M,y+3,7,'normal','#64748b'); y+=1;
            SS.forEach((s,i)=>{B(M+i*62,y,s.p[cls],bW,CC[cls]); T(s.p[cls].toFixed(1)+'%',M+i*62+bW+2,y+3,7,'bold','#334155');});
            y+=8;
        });
        y+=6;

        // RECOMMENDATIONS
        PG(40); T('RECOMMENDATIONS BASED ON DIAGNOSIS',M,y,10,'bold','#0e7490'); y+=5; L(y); y+=5;
        const sev=ens.prediction;
        const bg2=sev==='osteoporosis'?[254,242,242]:sev==='osteopenia'?[255,251,235]:[240,253,244];
        const bc2=sev==='osteoporosis'?[252,165,165]:sev==='osteopenia'?[252,211,77]:[134,239,172];
        const tc2=sev==='osteoporosis'?'#991b1b':sev==='osteopenia'?'#92400e':'#166534';
        const tag=sev==='osteoporosis'?'HIGH PRIORITY - Immediate Action Required':sev==='osteopenia'?'MODERATE PRIORITY - Preventive Action Advised':'MAINTAIN - Continue Healthy Habits';
        doc.setFillColor(bg2[0],bg2[1],bg2[2]); doc.setDrawColor(bc2[0],bc2[1],bc2[2]); doc.setLineWidth(0.5);
        doc.roundedRect(M,y,CW,7,2,2,'FD');
        T(tag,M+4,y+5,7,'bold',tc2); y+=11;
        getRecs(sev).forEach((rec,i)=>{
            PG(8); T((i+1)+'.',M+2,y,7,'bold','#475569');
            const ln=doc.splitTextToSize(rec,CW-12);
            doc.setFontSize(7); doc.setFont('helvetica','normal'); doc.setTextColor('#334155');
            doc.text(ln,M+9,y); y+=ln.length*3.5+2;
        });
        y+=4;

        // IMAGES
        PG(70); T('SEGMENTATION ANALYSIS',M,y,10,'bold','#0e7490'); y+=5; L(y); y+=5;
        const iW=(CW-6)/2, iH=45;
        [{s:d.images.original,l:'Original'},{s:d.images.annotated,l:'Annotated'},
         {s:d.images.femur_crop,l:'Femur'},{s:d.images.tibia_crop,l:'Tibia'}].forEach((img,i)=>{
            const col=i%2, x=M+col*(iW+6);
            if(i===2){y+=iH+10; PG(iH+15);}
            try{doc.addImage(img.s,'JPEG',x,y,iW,iH);doc.setDrawColor(200,210,220);doc.setLineWidth(0.3);doc.rect(x,y,iW,iH);}catch(e){}
            T(img.l,x+iW/2,y+iH+5,7,'normal','#64748b','center');
        });
        y+=iH+12;

        // DISCLAIMER
        PG(20);
        doc.setFillColor(255,251,235); doc.setDrawColor(252,211,77); doc.setLineWidth(0.5);
        doc.roundedRect(M,y,CW,14,2,2,'FD');
        T('MEDICAL DISCLAIMER',M+4,y+5,7,'bold','#92400e');
        doc.setFontSize(6); doc.setFont('helvetica','normal'); doc.setTextColor('#92400e');
        doc.text(doc.splitTextToSize('This AI report is NOT a substitute for professional medical diagnosis. Consult a qualified healthcare provider for advice.',CW-8),M+4,y+9);
        y+=18;

        // FOOTER
        PG(8); L(y);
        T('(c) 2026 BoneHealth AI | Region-Aware Hybrid Model | EfficientNet V2S Ensemble',W/2,y+5,7,'normal','#94a3b8','center');

        doc.save('BoneHealth_Report_'+d.run_id+'.pdf');
        btn.classList.remove('generating');
        btn.innerHTML='<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg> Download PDF Report';
    } catch(err) {
        console.error(err);
        btn.classList.remove('generating');
        btn.innerHTML='<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg> Download PDF Report';
        alert('PDF failed: '+err.message);
    }
}
