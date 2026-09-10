// Rebuild from the project root with pptxgenjs available in NODE_PATH.
const fs = require('fs');
const pptxgen = require('pptxgenjs');
const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Shankara Rama Sharma';
pptx.subject = 'CPMG J-coupling correction: validation and experimental planning';
pptx.title = 'CPMG PI Update, 11 September 2026';
pptx.company = 'IISc';
pptx.lang = 'en-IN';
pptx.theme = {headFontFace:'Arial',bodyFontFace:'Arial',lang:'en-IN'};
const C={navy:'232B60',teal:'247E70',orange:'C27022',gray:'566171',light:'E7ECEF',white:'FFFFFF'};
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const ladder=read('docs/ml_results/ladder500k_clean.json');
const audit=read('docs/ml_results/audited_cnn_evaluation.json');
const fit=read('docs/ml_results/fit_expansion_20260910_summary.json');
if (!fit.complete) throw Error('The fitting expansion must complete before building the deck.');
function text(s,t,x,y,w,h,size=20,color=C.navy,bold=false){
  s.addText(t,{x,y,w,h,fontFace:'Arial',fontSize:size,color,bold,margin:0,breakLine:false,valign:'mid',paraSpaceAfterPt:8});
}
function base(title,sub,num,dark=false){
  const s=pptx.addSlide();s.background={color:dark?C.navy:C.white};
  text(s,title,.55,.35,12.15,.62,34,dark?C.white:C.navy,true);
  if(sub)text(s,sub,.57,1.08,12.15,.6,18,dark?'DFE5EF':C.gray);
  text(s,`11 September 2026   •   ${num}/6`,.57,7.04,11.9,.26,16,dark?'DFE5EF':C.gray);
  return s;
}
function chartOpt(x,y,w,h){return {x,y,w,h,showTitle:false,showLegend:true,legendPos:'b',legendFontFace:'Arial',legendFontSize:16,
  catAxisLabelFontFace:'Arial',catAxisLabelFontSize:16,valAxisLabelFontFace:'Arial',valAxisLabelFontSize:16,
  showValue:true,dataLabelFontSize:16,dataLabelPosition:'outEnd',dataLabelColor:C.navy,dataLabelFormatCode:'0.0',dataLabelBkgrdColor:C.white,
  showBorder:false,showCatName:false,showSerName:false,
  chartColors:[C.orange,C.teal,C.gray],
  valGridLine:{color:'E3E7EA',width:1},catGridLine:{style:'none'},
  showValAxisTitle:true,valAxisTitle:'',valAxisTitleFontSize:18,
  showMarker:false,showLine:true,showShadow:false,
  chartArea:{fill:{color:C.white},border:{color:C.white}},plotArea:{fill:{color:C.white},border:{color:C.white}},
  showPercent:false,fontFace:'Arial',fontSize:16};}
function note(s,t){s.addNotes(t);}

// 1. Brief context and an actual correction example, no separate cover slide.
{
 const s=base('CPMG J-coupling correction','Weekly PI update: progress since the August group meeting and the unpresented September update',1);
 text(s,'Goal',.6,1.95,4.0,.35,23,C.teal,true);
 text(s,'Predict the J=0 curve from a coupled CPMG-RD curve.',.6,2.42,4.0,1.15,28,C.navy,true);
 text(s,'Preserve the exchange information used to estimate kex, pB and |Δω|.',.6,3.86,3.9,1.12,23);
 text(s,'Current status: synthetic proof of concept with downstream fitting tests.',.6,5.35,3.9,.95,20,C.gray);
 const folder='/private/tmp/cpmg-expand-20260910-k600-s99101';
 const examplePath='docs/ml_results/pi_update_11Sep2026_example.json';
 const series=fs.existsSync(examplePath)?read(examplePath).series:[['Noisy coupled','noisy_coupled'],['CNN corrected','cnn_corrected'],['Clean J=0 target','clean_j0']].map(([name,arm])=>{
   const lines=fs.readFileSync(`${folder}/${arm}_start0/600_2.out`,'utf8').trim().split('\n').slice(1).map(l=>l.trim().split(/\s+/).map(Number));
   return {name,labels:lines.map(r=>String(r[0]/.04)),values:lines.map(r=>-Math.log(r[1])/.04)};
 });
 if(!fs.existsSync(examplePath))fs.writeFileSync(examplePath,JSON.stringify({source:'fit_expansion_20260910/k600_seed99101.json, 600 MHz residue 2',kex:600,pb:.05,dw:3,noise_seed:99101,series},null,2));
 s.addChart(pptx.ChartType.line,series,{...chartOpt(4.85,2.12,7.85,3.9),showValue:false,catAxisLabelFrequency:8,
  valAxisTitle:'R₂,eff (s⁻¹)',showCatName:false,showCatAxisTitle:true,catAxisTitle:'νCPMG (Hz)',catAxisTitleFontSize:18,
  lineSize:2.5,showMarker:false,legendPos:'b'});
 text(s,'600 MHz, kex = 600 s⁻¹, pB = 0.05, |Δω| = 3 ppm\nJ = 92 Hz input, T = 40 ms, reference SNR = 100',4.95,6.18,7.5,.57,16,C.gray);
 note(s,'About 90 seconds. Our objective is computational J-coupling correction while retaining the information about exchange. The target is the same simulator and pulse sequence with J set to zero, not automatically a physically decoupled experimental acquisition. This illustration uses the 600 MHz, shift=3 ppm profile from the new fixed-geometry fitting expansion, kex=600, pB=.05, noise seed 99101. B1=5555 Hz, tau_m=5 ns, tau_e=50 ps, S2=.85, r_IS=1.02 A, r_eff=1.86 A, CSA=-160 ppm, theta=22 degrees. The input is noisy, while the target is clean, so the learned mapping includes denoising. Source: cpmg_ml/fit_validation_pilot.py and docs/ml_results/fit_expansion_20260910/k600_seed99101.json. Context sources: CPMG_GroupMeeting_v2_with_Jblind_slide.pptx and CPMG_PI_Update_3Sep2026.pptx.');
}
// 2. Unpresented data/model progress plus qualified simulator check.
{
 const s=base('The CNN learns the synthetic correction','500,000 training profiles, 20,000 validation profiles and 20,000 original test profiles',2);
 const names=['identity','linear','mlp','transformer_nopos','cnn'];
 const vals=names.map(n=>ladder.find(r=>r.name===n).rmse);
 s.addChart(pptx.ChartType.bar,[{name:'Clean test RMSE',labels:['Unchanged','Linear','MLP','Transformer','FiLM-CNN'],values:vals}],
  {...chartOpt(.55,2.03,7.65,4.48),barDir:'bar',showLegend:false,chartColors:[C.teal],valAxisTitle:'RMSE (s⁻¹)',valAxisMinVal:0,valAxisMaxVal:.95,valAxisMajorUnit:.2,dataLabelFormatCode:'0.000',dataLabelPosition:'outEnd',dataLabelColor:C.navy});
 text(s,'0.05324 s⁻¹',8.6,2.1,4.0,.6,34,C.teal,true);
 text(s,'CNN RMSE after excluding 7 test profiles duplicated in training.',8.6,2.84,4.0,1.02,21);
 text(s,'Matrix comparison: max difference 0',8.6,4.17,4.05,.66,23,C.navy,true);
 text(s,'30×30 homogeneous block, with shared supplied relaxation rates. Independent rate checks remain.',8.6,4.98,4.05,1.17,20,C.gray);
 text(s,'CNN inputs: curve, frequency, B₀, B₁ and J. No ground-truth exchange parameters.',.62,6.54,12,.35,17,C.gray);
 note(s,'About 2 minutes. This includes progress from the unpresented September deck. The clean ladder is the original 20k test benchmark: identity .8190, linear .5412, MLP .2058, Transformer without positional embedding .0711, CNN .0532 s^-1. The positional Transformer scored .0808. This establishes the CNN as the best tested configuration, not a universal architecture winner. Training seeds and budgets need fuller comparison. The right-hand re-evaluation excludes seven exact test/train duplicates and changes the CNN RMSE negligibly to .05323586 on 19,993 profiles. Four validation overlaps and 59 training duplicate occurrences were also found. Existing checkpoints retain their original selection history. The matrix assertion passes, but rates are taken from our matrix and supplied to ChemEx and identity/source terms are excluded. Avoid saying all physical formulas are independently validated. Sources: ladder500k_clean.json, benchmark_audit_500k.json, audited_cnn_evaluation.json, tests/compare_matrix_vs_chemex_dqzq.py.');
}
// 3. Equal mask and defined metrics, without a claimed physical ceiling.
{
 const s=base('Noise handling needs a more careful benchmark','Noise enters peak intensities. Taking the logarithm amplifies errors at weak intensities.',3);
 const clean=audit.results[0].metrics['expected_intensity_above_0.05'];
 const mean=m=>audit.results.slice(1).reduce((a,r)=>a+r.metrics['expected_intensity_above_0.05'][m].rmse,0)/3;
 s.addChart(pptx.ChartType.bar,[
  {name:'Unchanged input',labels:['Clean input','SNR 100 input'],values:[clean.unchanged.rmse,mean('unchanged')]},
  {name:'CNN',labels:['Clean input','SNR 100 input'],values:[clean.cnn.rmse,mean('cnn')]}
 ],{...chartOpt(.55,2.05,7.6,4.4),barDir:'col',catAxisLabelFontSize:18,valAxisTitle:'RMSE (s⁻¹)',valAxisMinVal:0,valAxisMaxVal:1.6,valAxisMajorUnit:.4,dataLabelFormatCode:'0.000'});
 text(s,'Same evaluation subset',8.6,2.18,4,.42,23,C.navy,true);
 text(s,'Expected intensity ≥ 0.05 I₀\n19,663 profiles with retained points\nThree new noise draws at SNR 100',8.6,2.9,4,1.42,19);
 text(s,'Revised interpretation',8.6,4.65,4,.43,22,C.teal,true);
 text(s,'Shared-reference covariance checked. The earlier “hard recovery ceiling” claim is withdrawn.',8.6,5.23,4,1.12,20);
 text(s,'Truth-based mask for this retrospective check. Full-point noisy CNN RMSE is 3.369 s⁻¹.',.62,6.57,12.0,.32,17,C.gray);
 note(s,'About 2 minutes. Both chart groups use the same expected-intensity mask, computed using the clean coupled curve. This is a diagnostic subset, not an experimental quality filter that can use unknown truth. Clean scores are .6332 unchanged and .01019 CNN. Noisy scores are 1.30357 and .45869, averaged over three new noise realisations. CNN standard deviation across these draws is .00162 on this subset, not a training-seed uncertainty or population confidence interval. Full-point noisy CNN RMSE is 3.369 +/- .034 across draws. Around 1.53-1.55% of intensities become nonpositive before historical clipping. We retained the old preprocessing to evaluate existing checkpoints. A simple validation-tuned smoothing control was weaker, but a dedicated denoiser and uncertainty-aware controls remain necessary. Noise in the shared reference correlates R2 errors. The old maximum-noise visibility heuristic does not prove any recovery ceiling. No definite averaging or acquisition recommendation follows yet. Source: audited_cnn_evaluation.json, noise_statistics.py, tests/test_noise_statistics.py.');
}
// 4. New expanded downstream evidence, with failures visible.
{
 const s=base('Correction improves fitting, but residual bias remains','New exploratory check: kex = 300, 600 and 1800 s⁻¹, two noise draws each, pB = 0.05',4);
 const keys=['KEX_AB','PB','DW_AB'];
 const series=[['Noisy J=0','noisy_j0'],['Noisy coupled','noisy_coupled'],['CNN corrected','cnn_corrected']].map(([name,arm])=>({name,labels:['kex','pB','|Δω|'],values:keys.map(k=>fit.aggregate[arm][k].mean_absolute_percent_error)}));
 s.addChart(pptx.ChartType.bar,series,{...chartOpt(.52,2.02,8.1,4.55),chartColors:[C.gray,C.orange,C.teal],barDir:'col',valAxisTitle:'Mean absolute relative error (%)',valAxisMinVal:0,valAxisMaxVal:27,valAxisMajorUnit:5,catAxisLabelFontSize:20});
 text(s,'Improved after correction',8.95,2.14,3.8,.65,23,C.teal,true);
 text(s,`kex: ${fit.paired_improvement.KEX_AB.improved}/6 cases\npB: ${fit.paired_improvement.PB.improved}/6 cases\n|Δω|: ${fit.paired_improvement.DW_AB.improved}/6 cases`,8.95,3.04,3.6,1.27,24);
 text(s,'At kex = 300 s⁻¹, corrected kex still has about 20–22% error.',8.95,4.68,3.75,1.1,22,C.orange,true);
 text(s,'600 + 800 MHz, 3 shifts, two fitting starts. Known R₁ fixed. Six cases, not population statistics.',.62,6.61,12.1,.35,16,C.gray);
 note(s,'About 3 minutes. This extends the September 9 single-case pilot. Three physical exchange rates, each with two noise draws, give six exploratory cases, not six independent biological systems. Each case uses three shifts (1,3,5 ppm) at two fields (600/800 MHz), four arms, and two starts: 48 fits completed. We fix R1 to simulator values and fit R2, making this an optimistic controlled benchmark. The clean J=0 reference has mean absolute relative errors .0176% kex, .3137% pB and .0672% shift magnitude. The displayed noisy reference errors are 1.656%, 1.495%, 1.095%. Coupled errors are 18.470%, 21.961%, 4.094%. Corrected errors are 11.037%, 11.441%, 1.585%. Shifts are scored by magnitude, with signed fitted values preserved. kex and shift errors improve in all six cases; pB improves in five. Corrected errors at kex=300 remain substantial: kex about -20 to -22%, pB about +24 to +27%. Do not hide these failures in an average. The small pB worsening at kex=1800/seed99101 is .469% to .733%. Corrected intensities are pseudo-data with fitting weights, not calibrated uncertainties. Sources: fit_expansion_20260910_summary.json and its six component JSON files.');
}
// 5. The pre-experimental work is ordered and bounded.
{
 const s=base('Remaining computational work','The next milestone is reliable parameter recovery across a declared operating range.',5);
 const rows=[
  ['Priority','Work package','Evidence needed'],
  ['1','Residual fitting bias','More exchange groups and noise draws.\nCoupled-target denoising control.'],
  ['2','Weak-signal handling','Intensity or mask-aware model.\nCalibrated uncertainty and rejection rules.'],
  ['3','Fresh benchmark','Disjoint seeds and ≥3 training runs.\nJ=0 / no-exchange and held-out regimes.'],
  ['4','Experimental compatibility','B₁ / J errors and missing frequencies.\nRate checks and consistent inference.']
 ];
 s.addTable(rows,{x:.62,y:2.0,w:12.05,h:3.85,colW:[1.15,4.0,6.9],rowH:.77,fontFace:'Arial',fontSize:18,color:C.navy,
  border:{type:'solid',color:'DCE3E8',pt:.8},fill:C.white,margin:.08,
  bold:false,autoPage:false,align:'left',valign:'mid',
 });
 text(s,'Experimental planning can start in parallel. Broad parameter ranges remain unchanged.',.66,6.38,11.8,.42,20,C.teal,true);
 note(s,'About 2 minutes. We can proceed computationally without waiting for all experimental details. Priority one is explaining the bias at kex=300 and separating denoising from J correction. We should not add millions more training profiles before understanding this. Then update preprocessing and train a denoising control and correction model across seeds. A fresh benchmark should be frozen before viewing outcomes and use separate seed ranges, with held-out acquisition/regime tests separate from random interpolation tests. Changing input preprocessing requires retraining. Include already-J-free and no-exchange controls to detect invented dispersion. Quantify input uncertainty in B1 and J. Independently derive rate and source terms rather than relying only on a shared-rate matrix comparison. Deployment must use the actual successful checkpoint/preprocessing path. No parameter narrowing has been applied pending PI discussion. This is a proposed work sequence, not a claim these items are complete.');
}
// 6. The PI discussion has concrete decisions.
{
 const s=base('Decisions for the first experimental validation','Suggested discussion points for tomorrow',6,true);
 const entries=[
  ['01','Which experiment should version 1 support?','Exact pulse programme, field strengths, T and frequency grid.'],
  ['02','Which sample and reference can we measure?','Repeated coupled data and a suitable artefact-suppressed comparison.'],
  ['03','Which parameter and SNR range matters most?','Agree the target application before changing the broad training range.'],
  ['04','What counts as scientifically useful correction?','Acceptable bias in kex, pB and |Δω|, plus failure and uncertainty limits.'],
  ['05','What is the intended paper scope?','Computational methods first, or an experimentally validated application?']
 ];
 entries.forEach(([n,q,d],i)=>{const y=1.93+i*.9;text(s,n,.65,y,.6,.35,22,'85CDBE',true);text(s,q,1.48,y,11.15,.35,22,C.white,true);text(s,d,1.48,y+.4,11.15,.32,18,'DFE5EF');});
 note(s,'Allow discussion. Ask for the actual pulse-program file rather than only an experiment name: physical decoupling can change more than J and is not automatically identical to setting J=0 in a simulation. Identify a sample with interpretable exchange, repeat measurements, and an independent comparison. Agree realistic field strengths, pulse widths, frequency sampling, signal-to-noise and exchange ranges. Ask what parameter bias or failure rate would prevent using a corrected curve and whether a correction should sometimes be withheld. Choose thresholds before the final test, rather than selecting them after seeing performance. Finally agree whether experimental validation is required for the first manuscript and who can provide/acquire the data. The existing CNN remains the working baseline; a new architecture is not a prerequisite for this discussion.');
}
fs.mkdirSync('docs',{recursive:true});
pptx.writeFile({fileName:'docs/CPMG_PI_Update_11Sep2026.pptx'});
