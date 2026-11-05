export type OfflinePhrase = {
  trigger: string;
  speech: string;
  urgency: 'normal' | 'urgent';
};

const PHRASES: Record<string, Omit<OfflinePhrase, 'trigger'>> = {
  HELP: { speech: 'I need help.', urgency: 'urgent' },
  SOS: { speech: 'I need emergency help.', urgency: 'urgent' },
  STOP: { speech: 'Please stop.', urgency: 'urgent' },
  DANGER: { speech: 'There is danger. Please help me.', urgency: 'urgent' },
  WATER: { speech: 'I need water.', urgency: 'normal' },
  FOOD: { speech: 'I need food.', urgency: 'normal' },
  BATHROOM: { speech: 'I need the bathroom.', urgency: 'normal' },
  DOCTOR: { speech: 'I need a doctor.', urgency: 'urgent' },
  HOSPITAL: { speech: 'Please take me to a hospital.', urgency: 'urgent' },
  YES: { speech: 'Yes.', urgency: 'normal' },
  NO: { speech: 'No.', urgency: 'normal' },
  THANKS: { speech: 'Thank you.', urgency: 'normal' },
  HELLO: { speech: 'Hello.', urgency: 'normal' },
  HOME: { speech: 'I want to go home.', urgency: 'normal' },
};

export function resolveOfflinePhrase(spelledText: string): OfflinePhrase | null {
  const words = spelledText.toUpperCase().trim().split(/\s+/).filter(Boolean);
  const trigger = words.at(-1);
  if (!trigger || !PHRASES[trigger]) return null;
  return { trigger, ...PHRASES[trigger] };
}

export const offlinePhraseCount = Object.keys(PHRASES).length;
