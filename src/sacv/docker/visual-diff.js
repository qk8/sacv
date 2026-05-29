#!/usr/bin/env node
/**
 * visual-diff.js
 * ==============
 * Playwright-based visual regression checker.
 * Takes a screenshot of the app, compares it with the stored baseline,
 * and writes a JSON result to --output.
 *
 * Usage: node /sacv/visual-diff.js --task <task_id> --output <path>
 * Environment: SACV_APP_URL (default: http://localhost:3000)
 */
const { chromium } = require('playwright');
const fs   = require('fs');
const path = require('path');

const args  = process.argv.slice(2);
const get   = (flag) => { const i = args.indexOf(flag); return i >= 0 ? args[i+1] : null; };
const taskId = get('--task') || 'unknown';
const output = get('--output') || `/tmp/vd-${taskId}.json`;
const appUrl = process.env.SACV_APP_URL || 'http://localhost:3000';

const BASELINE_DIR  = path.join('/workspace', '.workflow', 'visual-baselines');
const SNAPSHOT_DIR  = path.join('/workspace', '.workflow', 'visual-snapshots');

(async () => {
  fs.mkdirSync(BASELINE_DIR,  { recursive: true });
  fs.mkdirSync(SNAPSHOT_DIR,  { recursive: true });

  const browser = await chromium.launch();
  const page    = await browser.newPage({ viewport: { width: 1280, height: 800 } });

  await page.goto(appUrl, { waitUntil: 'networkidle', timeout: 30000 });

  const snapshotPath = path.join(SNAPSHOT_DIR, `${taskId}.png`);
  await page.screenshot({ path: snapshotPath, fullPage: true });
  await browser.close();

  const baselinePath = path.join(BASELINE_DIR, `${taskId}.png`);
  if (!fs.existsSync(baselinePath)) {
    // No baseline yet — treat current snapshot as baseline (first run)
    fs.copyFileSync(snapshotPath, baselinePath);
    fs.writeFileSync(output, JSON.stringify({
      passed: true, baseline_created: true, diff_pixels: 0
    }));
    process.exit(0);
  }

  // Compare with pixelmatch
  const { PNG }       = require('pngjs');
  const pixelmatch    = require('pixelmatch');
  const baseline      = PNG.sync.read(fs.readFileSync(baselinePath));
  const snapshot      = PNG.sync.read(fs.readFileSync(snapshotPath));
  const { width, height } = baseline;
  const diff          = new PNG({ width, height });

  const mismatchedPixels = pixelmatch(
    baseline.data, snapshot.data, diff.data, width, height,
    { threshold: 0.1 }
  );
  const totalPixels   = width * height;
  const diffRatio     = mismatchedPixels / totalPixels;
  const passed        = diffRatio < 0.01;   // <1% pixel difference = pass

  const result = {
    passed,
    diff_pixels: mismatchedPixels,
    total_pixels: totalPixels,
    diff_ratio: diffRatio,
    snapshot: snapshotPath,
    baseline: baselinePath,
  };

  fs.writeFileSync(output, JSON.stringify(result, null, 2));
  process.exit(passed ? 0 : 1);
})().catch(err => {
  fs.writeFileSync(output, JSON.stringify({ passed: false, error: String(err) }));
  process.exit(1);
});
