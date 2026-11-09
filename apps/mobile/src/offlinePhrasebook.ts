export type OfflinePhrase = {
  trigger: string;
  speech: string;
  urgency: 'normal' | 'urgent';
};

const PHRASES: Record<string, Omit<OfflinePhrase, 'trigger'>> = {
  // Emergency & Urgent
  HELP: { speech: 'I need help.', urgency: 'urgent' },
  SOS: { speech: 'I need emergency help.', urgency: 'urgent' },
  STOP: { speech: 'Stop.', urgency: 'normal' },
  DANGER: { speech: 'There is danger. Please help me.', urgency: 'urgent' },
  DOCTOR: { speech: 'I need a doctor.', urgency: 'urgent' },
  HOSPITAL: { speech: 'Please take me to a hospital.', urgency: 'urgent' },
  POLICE: { speech: 'Please call the police.', urgency: 'urgent' },
  AMBULANCE: { speech: 'Please call an ambulance.', urgency: 'urgent' },
  FIRE: { speech: 'There is a fire.', urgency: 'urgent' },
  MEDICINE: { speech: 'I need my medicine.', urgency: 'urgent' },
  PAIN: { speech: 'I am in severe pain.', urgency: 'urgent' },
  SICK: { speech: 'I feel sick.', urgency: 'urgent' },
  HURT: { speech: 'I am hurt.', urgency: 'urgent' },
  ALLERGY: { speech: 'I am having an allergic reaction.', urgency: 'urgent' },

  // Daily Needs & Care
  WATER: { speech: 'I need water.', urgency: 'normal' },
  FOOD: { speech: 'I need food.', urgency: 'normal' },
  BATHROOM: { speech: 'I need the bathroom.', urgency: 'normal' },
  REST: { speech: 'I need to rest.', urgency: 'normal' },
  TIRED: { speech: 'I am tired.', urgency: 'normal' },
  HUNGRY: { speech: 'I am hungry.', urgency: 'normal' },
  THIRSTY: { speech: 'I am thirsty.', urgency: 'normal' },
  COLD: { speech: 'I feel cold.', urgency: 'normal' },
  HOT: { speech: 'I feel hot.', urgency: 'normal' },
  HOME: { speech: 'I want to go home.', urgency: 'normal' },
  FAMILY: { speech: 'Please contact my family.', urgency: 'normal' },
  CALL: { speech: 'Please make a call for me.', urgency: 'normal' },
  WAIT: { speech: 'Please wait a moment.', urgency: 'normal' },
  REPEAT: { speech: 'Could you please repeat that?', urgency: 'normal' },

  // Conversational
  YES: { speech: 'Yes.', urgency: 'normal' },
  NO: { speech: 'No.', urgency: 'normal' },
  PLEASE: { speech: 'Please.', urgency: 'normal' },
  THANKS: { speech: 'Thank you.', urgency: 'normal' },
  THANKYOU: { speech: 'Thank you.', urgency: 'normal' },
  HELLO: { speech: 'Hello.', urgency: 'normal' },
  MORE: { speech: 'I want more, please.', urgency: 'normal' },
  ILY: { speech: 'I love you.', urgency: 'normal' },
  GOODBYE: { speech: 'Goodbye.', urgency: 'normal' },
  OKAY: { speech: 'Okay.', urgency: 'normal' },
  OK: { speech: 'Okay.', urgency: 'normal' },
  PEACE: { speech: 'Peace.', urgency: 'normal' },
  GOOD: { speech: 'Good.', urgency: 'normal' },
  BAD: { speech: 'Bad.', urgency: 'normal' },
  HI: { speech: 'Hello.', urgency: 'normal' },
  BYE: { speech: 'Goodbye.', urgency: 'normal' },
  FINE: { speech: 'I am fine.', urgency: 'normal' },
  SORRY: { speech: 'I am sorry.', urgency: 'normal' },
  SAFE: { speech: 'I am safe.', urgency: 'normal' },
  READY: { speech: 'I am ready.', urgency: 'normal' },
  UNDERSTAND: { speech: 'I understand.', urgency: 'normal' },
  WELCOME: { speech: 'You are welcome.', urgency: 'normal' },
  DRINK: { speech: 'I need a drink.', urgency: 'normal' },
  EAT: { speech: 'I want to eat.', urgency: 'normal' },
  SLEEP: { speech: 'I want to sleep.', urgency: 'normal' },
  LIKE: { speech: 'I like this.', urgency: 'normal' },
  DISLIKE: { speech: 'I dislike this.', urgency: 'normal' },
  LOVE: { speech: 'Love.', urgency: 'normal' },
  FRIEND: { speech: 'You are my friend.', urgency: 'normal' },
  TOILET: { speech: 'I need the toilet.', urgency: 'normal' },
  DONE: { speech: 'All finished.', urgency: 'normal' },
  FINISH: { speech: 'All finished.', urgency: 'normal' },
  LOOK: { speech: 'Please look here.', urgency: 'normal' },
  SEE: { speech: 'I see.', urgency: 'normal' },
  LISTEN: { speech: 'Please listen.', urgency: 'normal' },
  HEAR: { speech: 'I hear you.', urgency: 'normal' },
  COME: { speech: 'Please come here.', urgency: 'normal' },
  GO: { speech: 'Let us go.', urgency: 'normal' },
  ONE: { speech: 'One.', urgency: 'normal' },
  TWO: { speech: 'Two.', urgency: 'normal' },
  THREE: { speech: 'Three.', urgency: 'normal' },
  FOUR: { speech: 'Four.', urgency: 'normal' },
  FIVE: { speech: 'Five.', urgency: 'normal' },
};

export function resolveOfflinePhrase(spelledText: string): OfflinePhrase | null {
  const words = spelledText.toUpperCase().trim().split(/\s+/).filter(Boolean);
  const trigger = words.at(-1);
  if (!trigger || !PHRASES[trigger]) return null;
  return { trigger, ...PHRASES[trigger] };
}

/**
 * Predicts up to 4 autocompletions for the currently typed/fingerspelled word prefix.
 * e.g. "WAT" -> [{ trigger: "WATER", speech: "I need water." }]
 */
export function predictOfflineCompletions(currentWord: string): OfflinePhrase[] {
  const clean = currentWord.toUpperCase().trim();
  if (clean.length < 2) return [];

  const matches: OfflinePhrase[] = [];
  for (const [key, value] of Object.entries(PHRASES)) {
    if (key.startsWith(clean) && key !== clean) {
      matches.push({ trigger: key, ...value });
      if (matches.length >= 4) break;
    }
  }
  return matches;
}

export const offlinePhraseCount = Object.keys(PHRASES).length;
