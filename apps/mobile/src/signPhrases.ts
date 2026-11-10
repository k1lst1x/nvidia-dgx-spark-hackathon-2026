/**
 * Turn sign-model labels into the text a partner reads and hears.
 *
 * The word models emit compact labels ("thankyou", "callonphone", "hesheit").
 * Spoken aloud or shown in large type those read badly, so the bridge maps
 * them to plain phrases before sending.
 */
const PHRASES: Record<string, string> = {
  thankyou: 'THANK YOU',
  ily: 'I LOVE YOU',
  callonphone: 'CALL ON THE PHONE',
  call: 'CALL ME',
  frenchfries: 'FRENCH FRIES',
  glasswindow: 'WINDOW',
  haveto: 'HAVE TO',
  hesheit: 'HE / SHE',
  icecream: 'ICE CREAM',
  minemy: 'MINE',
  weus: 'WE',
  owie: 'IT HURTS',
  pain: 'IT HURTS',
  shhh: 'QUIET, PLEASE',
  okay: 'OKAY',
  stop: 'STOP',
  help: 'HELP',
  safe: "I'M SAFE",
  done: "I'M DONE",
  more: 'MORE, PLEASE',
  tv: 'TV',
};

export function signLabelToPhrase(label: string): string {
  const key = label.trim().toLowerCase().replace(/[\s_-]+/g, '');
  if (!key) return '';
  return PHRASES[key] ?? label.trim().replace(/[_-]+/g, ' ').toUpperCase();
}
