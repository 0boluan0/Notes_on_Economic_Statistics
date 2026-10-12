/* A single-query attention story. All displayed model values come from result.data. */
(function (root) {
  'use strict';
  const names = ['A', 'B', 'C', 'D'];
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const steps = [
    {title:'匹配：我在找什么？',caption:'选一个 Query，看它与四个 Key 的点积。分数越大，表示这组向量越匹配。',action:'切换 A、B、C、D，比较四个匹配分数。',focus:['query','strength'],params:{causal:0}},
    {title:'分配：每个位置读多少？',caption:'Softmax 把四个分数变成非负权重。条越长，权重越大；四条的数值加起来是 1。',action:'降低温度，观察原本领先的分数怎样得到更多权重。',focus:['temperature','strength'],params:{causal:0}},
    {title:'读取：把内容加起来',caption:'权重决定读多少，Value 决定读什么。将每个 Value 乘自己的权重，再逐坐标相加。',action:'放大 Value，观察权重不动、贡献和输出按相同比例缩放。',focus:['valueScale'],params:{causal:0}},
    {title:'遮挡：未来还不能读',caption:'因果遮挡排除当前 Query 右侧的位置。允许位置重新分配全部权重，输出也随之改变。',action:'选 A 再选 C，检查哪些位置被挡住，以及余下权重怎样相加为 1。',focus:['query','causal'],params:{causal:1}}
  ];

  function render({el,params,result,step=0,onParam,format}) {
    const data = result && result.data;
    if (!data || !data.Q || !data.K || !data.V || !data.scores || !data.weights || !data.outputs) {
      el.textContent = '等待当前 Query 的计算结果…';
      return;
    }
    const stage = Math.max(0, Math.min(3, Number.isInteger(step) ? step : steps.indexOf(step)));
    const query = data.selectedQuery == null ? Number(params.query) : data.selectedQuery;
    const q = data.Q[query], scores = data.scores[query], weights = data.weights[query], output = data.outputs[query];
    const allowed = data.mask ? data.mask[query] : names.map((_,i) => !params.causal || i <= query);
    const contributions = data.V.map((v,j) => v.map(x => x * weights[j]));
    const fmt = value => {
      if (!Number.isFinite(value)) return '未定义';
      if (value === 0) return '0';
      if (Math.abs(value) < .0001) return value.toExponential(2).replace('-', '−');
      const rendered = typeof format === 'function' ? format(value,4) : Number(value.toFixed(4)).toString();
      return String(rendered).replace(/^-/, '−');
    };
    const n = value => escape(fmt(value));
    const vector = v => `(${v.map(n).join(', ')})`;
    const percent = value => value > 0 && value < .0001 ? '&lt;0.01%' : `${Number((value*100).toFixed(2))}%`;
    const visibleNames = names.filter((_,j) => allowed[j]);
    const sum = weights.reduce((a,b)=>a+b,0);
    const causal = Boolean(params.causal);
    const activeElement = el.ownerDocument && el.ownerDocument.activeElement;
    const focusedQuery = activeElement && el.contains(activeElement) && activeElement.getAttribute('data-attention-query');
    const token = (j) => `<span class="sa-token ${j===query?'is-query':''}">${names[j]}</span>`;
    const weightBar = (j) => `<div class="sa-weight-bar" role="img" aria-label="${names[j]} 的权重 ${escape(fmt(weights[j]))}${allowed[j]?'':'，被遮挡'}"><span style="width:${weights[j]*100}%"></span></div>`;
    const stamp = (j) => !allowed[j] ? '<span class="sa-blocked-label" aria-label="未来位置，不参与读取">×</span>' : '';

    function rows() {
      return names.map((name,j) => {
        const blocked = !allowed[j];
        let body;
        if (stage===0) {
          const dot = q.reduce((a,v,k)=>a+v*data.K[j][k],0);
          body = `<div class="sa-match-key"><span class="sa-mobile-label">Key ${name}</span><strong>${vector(data.K[j])}</strong></div><div class="sa-dot"><span class="sa-mobile-label">Q · K</span><span>${q.map((v,k)=>`${n(v)} × ${n(data.K[j][k])}`).join(' + ')} = <b>${n(dot)}</b></span></div><div class="sa-score"><span class="sa-mobile-label">分数 s</span><strong>${n(scores[j])}</strong></div>`;
        } else if (stage===1) {
          body = `<div class="sa-score"><span class="sa-mobile-label">分数 s</span><strong>${n(scores[j])}</strong></div><div class="sa-weight"><div class="sa-weight-numbers"><b>${n(weights[j])}</b><span>${percent(weights[j])}</span></div>${weightBar(j)}</div>`;
        } else {
          body = `<div class="sa-reading-weight"><div class="sa-weight-numbers"><span class="sa-mobile-label">权重</span><b>${n(weights[j])}</b></div>${weightBar(j)}</div><div class="sa-contribution"><span class="sa-vector-equation"><span>${n(weights[j])} × ${vector(data.V[j])}</span><strong>= ${vector(contributions[j])}</strong></span></div>`;
        }
        const maskDetail = stage===3 && blocked ? `<div class="sa-mask-detail">未来位置：屏蔽前分数 ${n(scores[j])} → 不进入 Softmax</div>` : '';
        return `<div class="sa-token-row sa-row-${stage}${blocked?' is-blocked':''}" data-token="${name}"><div class="sa-token-identity">${token(j)}${stamp(j)}</div><div class="sa-row-body">${body}${maskDetail}</div></div>`;
      }).join('');
    }

    function outputRuler(value,lo,hi,label) {
      const zero = (0-lo)/(hi-lo)*100, end = (value-lo)/(hi-lo)*100;
      return `<div class="sa-output-coordinate"><div class="sa-ruler-label"><span>${label}</span><strong>${n(value)}</strong></div><div class="sa-ruler" role="img" aria-label="${label} ${escape(fmt(value))}，固定刻度从 ${lo} 到 ${hi}"><span class="sa-ruler-zero" style="left:${zero}%"></span><span class="sa-ruler-fill" style="left:${Math.min(zero,end)}%;width:${Math.abs(end-zero)}%"></span><span class="sa-ruler-end" style="left:${end}%"></span></div><div class="sa-ruler-ticks"><span>${lo}</span><span>${(lo+hi)/2}</span><span>${hi}</span></div></div>`;
    }

    let conclusion;
    if (stage===0) {
      const best = scores.reduce((m,s,j)=>s>scores[m]?j:m,0), tied = names.filter((_,j)=>Math.abs(scores[j]-scores[best])<1e-12);
      conclusion = `<div class="sa-takeaway sa-takeaway-match"><span class="sa-result-label">最大分数</span><strong>${tied.join('、')} · ${n(scores[best])}</strong><p>得到分数，还未读取 Value。</p></div>`;
    } else if (stage===1) {
      conclusion = `<div class="sa-takeaway"><span class="sa-result-label">权重之和</span><div class="sa-sum-equation">${weights.map(n).join(' + ')} <strong>≈ ${n(sum)}</strong></div></div>`;
    } else {
      const equation = [0,1].map(k => `<div class="sa-coordinate-sum"><span>坐标 ${k+1}</span><div>${contributions.map(v=>`<span>${v[k]<0?'('+n(v[k])+')':n(v[k])}</span>`).join('<i> + </i>')} <strong>≈ ${n(output[k])}</strong></div></div>`).join('');
      const weightTotal = stage===3 ? `<div class="sa-mask-total"><span>允许位置的权重和</span><b>${weights.filter((_,j)=>allowed[j]).map(n).join(' + ')} ≈ ${n(sum)}</b></div>` : '';
      conclusion = `${weightTotal}<section class="sa-readout" aria-label="加权读取结果"><div class="sa-readout-heading"><h3>相加 → O<sub>${names[query]}</sub> = ${vector(output)}</h3><span class="sa-readout-tag">O = Σ wV</span></div><div class="sa-sums">${equation}</div><div class="sa-output-rulers">${outputRuler(output[0],-3,3,'坐标 1')}${outputRuler(output[1],0,6,'坐标 2')}</div><p class="sa-scale-note">固定刻度，每格 3；放大 Value 会改变条长。显示取整，相加使用 ≈。</p></section>`;
    }

    const formula = stage===0
      ? `<b>s = (Q · K) α / (√2 T)</b><span>α=${n(params.strength)}，T=${n(params.temperature)}；二维向量</span>`
      : stage===1
        ? '<b>wⱼ = exp(sⱼ) / Σ exp(sₖ)</b><span>只在可读位置归一化</span>'
        : stage===2
          ? `<b>权重 × Value → 一份贡献</b><span>Value 共同尺度 ${n(params.valueScale)}</span>`
          : `<b>${causal?'只允许：':'全部可见：'}${visibleNames.join('、')}</b><span>${causal?'× 未来位置；剩余权重重新归一化':'因果遮挡已关闭'}</span>`;

    const columns = stage===0 ? '<span>Key 向量</span><span>点积 Q · K</span><span>分数 s</span>' : stage===1 ? '<span>分数 s</span><span>权重 w（条长 0–1）</span>' : '<span>权重 w</span><span>Value 乘权重 = 写入的贡献</span>';
    el.innerHTML = `<section class="sa-story" data-stage="${stage}" aria-label="单 Query 注意力计算"><header class="sa-query-header"><div class="sa-query-context"><span>当前 Query</span><strong>${names[query]} = ${vector(q)}</strong></div><div class="sa-query-picker" role="group" aria-label="选择 Query">${names.map((name,j)=>`<button type="button" data-attention-query="${j}" aria-pressed="${j===query}" aria-label="选择 Query ${name}">${name}</button>`).join('')}</div></header><div class="sa-formula">${formula}</div><div class="sa-column-head sa-head-${stage}"><span>位置</span><div>${columns}</div></div><div class="sa-token-stage">${rows()}</div>${conclusion}<p class="sa-footnote">${causal?'× 表示未来位置，权重为 0。':'A–D 是手工向量，无语言语义。'}${stage>0?' 数值显示取整；条长使用真实权重。':''}</p></section>`;
    el.querySelectorAll('[data-attention-query]').forEach(button => {
      button.addEventListener('click',()=>{if(typeof onParam==='function')onParam('query',Number(button.getAttribute('data-attention-query')));});
    });
    if (focusedQuery !== null && focusedQuery !== false && focusedQuery !== undefined) {
      const button = el.querySelector(`[data-attention-query="${focusedQuery}"]`);
      if (button) button.focus({preventScroll:true});
    }
  }

  root.LabStories ||= {};
  root.LabStories.attention = {steps,render};
})(typeof globalThis !== 'undefined' ? globalThis : this);
