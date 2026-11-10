/**
 * Where is the camera's view blocked by something the detector cannot name?
 *
 * A plain wall, a door, a tree trunk or a hand over the lens fills part of
 * the frame with a surface that has almost no visible detail, or with
 * darkness. The object detector only knows named classes, and monocular depth
 * is least reliable on exactly these featureless surfaces. So this looks at
 * the pixels directly: it splits the eye-level band of the frame into columns
 * and marks a column blocked when most of it is featureless or dark and no
 * detected object covers it.
 *
 * The result says *which directions* look blocked, never how far away the
 * surface is. The live map draws it as "distance unknown".
 */
import type { LocalDetection } from './detector.types';

export type BlockedView = {
  /** One entry per column, left to right: true when that direction looks blocked. */
  columns: boolean[];
  /** Horizontal center of each column across the image, 0 (left) to 1 (right). */
  centers: number[];
  /** Most of the whole frame is featureless or dark: lens covered or pressed against something. */
  covered: boolean;
  /** Most of the frame is dark: the camera cannot see, whatever is in front. */
  dark: boolean;
};

type Region = { x: number; y: number; width: number; height: number };

const COLUMNS = 12;
const ROWS = 9;
// Eye-level band (rows 2..6 of 9): walls and trunks show here, while the sky
// and ceiling sit above and the floor below.
const BAND_FIRST_ROW = 2;
const BAND_LAST_ROW = 6;
// Mean luminance step between neighbouring samples (0-255). Textured scenes
// sit far above this; a plain painted wall or a covered lens sits below it.
const FEATURELESS_GRADIENT = 3.2;
const DARK_LUMINANCE = 24;
const COLUMN_BLOCKED_SHARE = 0.6;
const COVERED_SHARE = 0.85;
const SAMPLES_PER_SIDE = 14;

function luminance(rgba: ArrayLike<number>, offset: number) {
  return 0.299 * rgba[offset] + 0.587 * rgba[offset + 1] + 0.114 * rgba[offset + 2];
}

function coveredByDetection(column: number, row: number, detections: LocalDetection[]) {
  const left = column / COLUMNS;
  const right = (column + 1) / COLUMNS;
  const top = row / ROWS;
  const bottom = (row + 1) / ROWS;
  const area = (right - left) * (bottom - top);
  return detections.some((item) => {
    const overlapX = Math.max(0, Math.min(right, item.x + item.width) - Math.max(left, item.x));
    const overlapY = Math.max(0, Math.min(bottom, item.y + item.height) - Math.max(top, item.y));
    return overlapX * overlapY >= area * 0.5;
  });
}

/**
 * Analyse an RGBA buffer. `region` is the part of the buffer that holds the
 * camera image (the web detector letterboxes it inside a square); detection
 * boxes are normalized to that region.
 */
export function analyzeBlockedView(
  rgba: ArrayLike<number>,
  bufferWidth: number,
  region: Region,
  detections: LocalDetection[] = [],
): BlockedView | null {
  if (region.width < COLUMNS * 4 || region.height < ROWS * 4) return null;
  const cellWidth = region.width / COLUMNS;
  const cellHeight = region.height / ROWS;
  const step = Math.max(1, Math.floor(Math.min(cellWidth, cellHeight) / SAMPLES_PER_SIDE));
  const featureless: boolean[][] = [];
  let darkCells = 0;
  let blankCells = 0;

  for (let row = 0; row < ROWS; row += 1) {
    const rowFlags: boolean[] = [];
    for (let column = 0; column < COLUMNS; column += 1) {
      const x0 = Math.floor(region.x + column * cellWidth);
      const y0 = Math.floor(region.y + row * cellHeight);
      const x1 = Math.floor(region.x + (column + 1) * cellWidth) - step - 1;
      const y1 = Math.floor(region.y + (row + 1) * cellHeight) - step - 1;
      let sum = 0;
      let gradient = 0;
      let samples = 0;
      for (let y = y0; y <= y1; y += step) {
        for (let x = x0; x <= x1; x += step) {
          const here = luminance(rgba, (y * bufferWidth + x) * 4);
          const right = luminance(rgba, (y * bufferWidth + x + step) * 4);
          const below = luminance(rgba, ((y + step) * bufferWidth + x) * 4);
          sum += here;
          gradient += Math.abs(right - here) + Math.abs(below - here);
          samples += 1;
        }
      }
      if (!samples) {
        rowFlags.push(false);
        continue;
      }
      const dark = sum / samples < DARK_LUMINANCE;
      const blank = dark || gradient / (samples * 2) < FEATURELESS_GRADIENT;
      if (dark) darkCells += 1;
      if (blank) blankCells += 1;
      rowFlags.push(blank);
    }
    featureless.push(rowFlags);
  }

  const totalCells = ROWS * COLUMNS;
  const covered = blankCells >= totalCells * COVERED_SHARE;
  const dark = darkCells >= totalCells * COVERED_SHARE;
  const bandRows = BAND_LAST_ROW - BAND_FIRST_ROW + 1;
  let columns = Array.from({ length: COLUMNS }, (_, column) => {
    if (covered) return true;
    let blocked = 0;
    let open = 0;
    for (let row = BAND_FIRST_ROW; row <= BAND_LAST_ROW; row += 1) {
      // A detected object explains this cell; it has its own marker.
      if (coveredByDetection(column, row, detections)) {
        open += 1;
        continue;
      }
      if (featureless[row][column]) blocked += 1;
    }
    return open < bandRows && blocked >= (bandRows - open) * COLUMN_BLOCKED_SHARE;
  });
  // A single blank column is usually a door frame or a gap, not a blocked way.
  if (!covered) {
    columns = columns.map((value, index) => value && (columns[index - 1] === true || columns[index + 1] === true));
  }
  return {
    columns,
    centers: Array.from({ length: COLUMNS }, (_, column) => (column + 0.5) / COLUMNS),
    covered,
    dark,
  };
}
