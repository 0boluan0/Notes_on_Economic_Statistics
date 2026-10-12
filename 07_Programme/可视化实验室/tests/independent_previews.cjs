'use strict';
const path=require('path'),fs=require('fs');
const {pathToFileURL}=require('url');
const {chromium}=require(process.env.PLAYWRIGHT_PATH || 'playwright');
(async()=>{
  const root=path.resolve(__dirname,'..'),dir=path.join(root,'python','previews'),qa=process.env.QA_OUTPUT||path.join(require('os').tmpdir(),'math-lab-preview-qa');fs.mkdirSync(qa,{recursive:true});
  const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
  const report=[];
  for(const name of fs.readdirSync(dir).filter(x=>x.endsWith('.html'))){
    const page=await browser.newPage({viewport:{width:1280,height:960}});const errors=[],network=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.route('**/*',route=>{const u=route.request().url();if(/^https?:/.test(u)){network.push(u);return route.abort()}return route.continue()});
    await page.goto(pathToFileURL(path.join(dir,name)).href);await page.waitForLoadState('load');
    const state=await page.evaluate(()=>({math:document.querySelectorAll('.math').length,rendered:document.querySelectorAll('.katex').length,mathErrors:[...document.querySelectorAll('.katex-error')].map(x=>({text:x.textContent,title:x.title})),figures:document.images.length,brokenImages:[...document.images].filter(x=>!x.complete||!x.naturalWidth).length,overflow:document.documentElement.scrollWidth>innerWidth}));
    if(errors.length||network.length||state.mathErrors.length||state.math!==state.rendered||state.brokenImages||state.overflow)throw Error(name+' '+JSON.stringify({state,errors,network}));
    await page.screenshot({path:path.join(qa,name.replace('.html','-desktop.png'))});
    await page.setViewportSize({width:390,height:844});
    const narrow=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,mainWidth:document.querySelector('main').getBoundingClientRect().width}));
    if(narrow.overflow)throw Error(name+' narrow horizontal overflow');
    await page.screenshot({path:path.join(qa,name.replace('.html','-mobile.png'))});
    report.push({name,...state,narrow,errors,network,pass:true});await page.close();
  }
  await browser.close();fs.writeFileSync(path.join(dir,'preview-report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e);process.exit(1)});
