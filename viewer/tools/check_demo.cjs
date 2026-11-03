// Headed multi-scene acceptance. Usage: node check_demo.cjs URL OUTDIR BROWSER '["id",...]'
const fs = require('node:fs'), path = require('node:path');
const {chromium} = require('playwright-core');
const [base, outArg, executablePath, idsArg] = process.argv.slice(2);
const out = path.resolve(outArg);
fs.mkdirSync(out, {recursive:true});
const result = {passed:false, headless:false, scenes:[], errors:[]};
(async () => {
  let browser;
  try {
    const ids = JSON.parse(idsArg);
    if (!ids.length) throw Error('No scenes supplied');
    browser = await chromium.launch({headless:false, executablePath,
      env:{...process.env, __NV_PRIME_RENDER_OFFLOAD:'1', __GLX_VENDOR_LIBRARY_NAME:'nvidia',
        __EGL_VENDOR_LIBRARY_FILENAMES:'/usr/share/glvnd/egl_vendor.d/10_nvidia.json'},
      args:['--use-angle=gl-egl', '--window-size=1600,900']});
    for (const [i, id] of ids.entries()) {
      const page = await browser.newPage({viewport:{width:1600,height:900}});
      const entry = {scene:id, passed:false, errors:[]};
      result.scenes.push(entry);
      page.on('pageerror', e => entry.errors.push(String(e)));
      page.on('console', m => {if (m.type() === 'error') entry.errors.push(m.text());});
      try {
        const url = new URL(base); url.searchParams.set('scene', id);
        // Keep the real mode/quality; do not force guard off through the test API.
        entry.url = url.href;
        const guided = url.searchParams.get('guided') !== '0';
        await page.goto(url.href);
        await page.waitForFunction(() => window.viewerReady === true || window.viewerBlocked ||
          document.querySelector('#error')?.hidden === false, null, {timeout:180000});
        if (!await page.evaluate(() => window.viewer?.state?.ready)) throw Error('Scene failed to load');
        await page.waitForFunction(() => !window.viewer.state.intro, null, {timeout:30000});
        await page.waitForTimeout(800);
        entry.start = await page.evaluate(() => window.viewer.state);
        entry.device = await page.evaluate(() => window.viewerDevice);
        entry.title = await page.locator('#title').textContent();
        entry.startScreenshot = path.join(out, `scene-${i + 1}-start.png`);
        await page.screenshot({path:entry.startScreenshot});
        await page.mouse.click(800, 450);
        for (const [key, ms] of [['KeyW', 3000], ['KeyR', 2000]]) {
          await page.keyboard.down(key);
          await page.waitForTimeout(ms);
          await page.keyboard.up(key);
          if (key === 'KeyW') entry.afterForward = await page.evaluate(() => window.viewer.state.position);
        }
        await page.waitForTimeout(800);
        entry.moved = await page.evaluate(() => window.viewer.state);
        entry.movedScreenshot = path.join(out, `scene-${i + 1}-moved.png`);
        await page.screenshot({path:entry.movedScreenshot});
        entry.movedDistance = Math.hypot(...entry.moved.position.map((v,j) => v-entry.start.position[j]));
        entry.corridorDistance = entry.moved.corridorDistance;
        if (entry.moved.points) {
          await page.keyboard.press('KeyV');
          await page.waitForFunction(() => window.viewer.state.ready && window.viewer.state.representation === 'points',
            null, {timeout:60000});
          await page.waitForTimeout(800);
          entry.pointsScreenshot = path.join(out, `scene-${i + 1}-points.png`);
          await page.screenshot({path:entry.pointsScreenshot});
          entry.pointsToggled = true;
        }
        entry.checks = {
          loaded: entry.start.ready && entry.moved.ready && entry.start.scene === id && entry.start.gaussianCount > 0,
          hardwareGPU: entry.device.software === false && !/swiftshader|llvmpipe|software|subzero/i.test(entry.start.gpuName),
          mode: entry.start.guard === guided && entry.moved.guard === guided,
          moved: entry.movedDistance > .05,
          forwardMoved: Math.hypot(...entry.afterForward.map((v,j) => v-entry.start.position[j])) > .05,
          // Guided mode deliberately stays in the corridor; this condition is free-roam only.
          leftCorridor: guided ? null : entry.corridorDistance > .3,
          pointsToggle: entry.moved.points ? entry.pointsToggled === true : null,
          noErrors: entry.errors.length === 0,
        };
        entry.passed = Object.values(entry.checks).every(v => v === true || v === null);
      } catch (e) {
        entry.errors.push(String(e));
      } finally {
        await page.close();
        if (entry.errors.length) entry.passed = false;
      }
    }
    result.passed = result.scenes.every(s => s.passed);
  } catch (e) {
    result.errors.push(String(e));
  } finally {
    await browser?.close();
    fs.writeFileSync(path.join(out, 'check.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result.scenes.map(s => ({scene:s.scene, passed:s.passed,
      gpu:s.start?.gpuName, moved:s.movedDistance, corridor:s.corridorDistance, errors:s.errors})), null, 2));
    process.exitCode = result.passed ? 0 : 1;
  }
})().catch(e => {console.error(e); process.exitCode=1;});
