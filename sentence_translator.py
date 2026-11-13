import re

class FSLSentenceTranslator:
    """
    Intelligent FSL (Filipino Sign Language) Gloss-to-Sentence Translator.
    Transforms raw, disjointed signed glosses into fluent, grammatically correct English sentences.
    
    Examples:
        - "You, I, work, now." -> "Let's start working."
        - "You I work"         -> "Let's work together."
        - "Where bathroom"     -> "Where is the bathroom?"
        - "What name you"      -> "What is your name?"
        - "I hungry"           -> "I am hungry."
        - "She doctor"         -> "She is a doctor."
        - "Thank you help"     -> "Thank you for your help."
    """

    # Precompiled at class definition time: identical pattern strings, identical
    # order and first-match semantics as the original re.match loop.
    IDIO_PATTERNS = [(re.compile(pattern), replacement) for pattern, replacement in [
        # ── Conjoint You + I / We Actions ──
        (r'^(you i|i you|you and i|you me|me you) work now$', "Let's start working."),
        (r'^(you i|i you|you and i|you me|me you) work$', "Let's work together."),
        (r'^(you i|i you|you and i) eat now$', "Let's eat now."),
        (r'^(you i|i you|you and i) eat$', "Let's eat."),
        (r'^(you i|i you|you and i) go now$', "Let's go now."),
        (r'^(you i|i you|you and i) go$', "Let's go."),
        (r'^(you i|i you|you and i) talk now$', "Let's talk now."),
        (r'^(you i|i you|you and i) talk$', "Let's talk."),
        (r'^(you i|i you|you and i) study now$', "Let's study now."),
        (r'^(you i|i you|you and i) study$', "Let's study together."),
        (r'^(you i|i you|you and i) practice$', "Let's practice."),
        (r'^(you i|i you|you and i) start$', "Let's get started."),
        (r'^(you i|i you|you and i) leave$', "Let's leave now."),
        (r'^(you i|i you|you and i) friend$', "We are friends."),
        (r'^(you i|i you|you and i) team$', "We are a team."),

        # ── Questions (FSL Topic-Comment & Wh-inversion) ──
        (r'^(what name you|what your name|you name what|name you what)$', "What is your name?"),
        (r'^(where bathroom|bathroom where|where toilet|toilet where)$', "Where is the bathroom?"),
        (r'^(where you go|you go where|you where go)$', "Where are you going?"),
        (r'^(where hospital|hospital where)$', "Where is the nearest hospital?"),
        (r'^(how you|you how)$', "How are you?"),
        (r'^(how you feel|you feel how)$', "How are you feeling?"),
        (r'^(who that|that who)$', "Who is that?"),
        (r'^(who you|you who)$', "Who are you?"),
        (r'^(what this|this what)$', "What is this?"),
        (r'^(what happen|what happen now)$', "What happened?"),
        (r'^(what time|time what)$', "What time is it?"),
        (r'^(when we meet|when meet)$', "When will we meet?"),
        (r'^(when you leave|when leave)$', "When are you leaving?"),
        (r'^(when class start)$', "When does class start?"),
        (r'^(why you late|why late)$', "Why are you late?"),
        (r'^(why you sad|why sad)$', "Why are you sad?"),
        (r'^(how much|how much this)$', "How much is this?"),
        (r'^(can you help|you help me|help me please|help please)$', "Can you please help me?"),
        (r'^(can i ask|i ask question)$', "Can I ask a question?"),
        (r'^(you understand|understand you)$', "Do you understand?"),

        # ── Greetings, Courtesies & Social Signs ──
        (r'^(hello who you|hi who you|hey who you)$', "Hello, who are you?"),
        (r'^(hello what name you|hi what name you|hello your name what)$', "Hello! What is your name?"),
        (r'^(hello how you|hello you how|hi how you)$', "Hello! How are you?"),
        (r'^(hello where you go|hi where you go)$', "Hello, where are you going?"),
        (r'^(hello everyone|hi everyone)$', "Hello everyone!"),
        (r'^(nice meet you|meet you nice)$', "Nice to meet you."),
        (r'^(nice see you|see you nice)$', "It is nice to see you."),
        (r'^(thank you help|thank_you help|thanks help)$', "Thank you for your help."),
        (r'^(thank you very much|thank_you very much)$', "Thank you very much!"),
        (r'^(thank you|thank_you|thanks)$', "Thank you!"),
        (r'^(you welcome|welcome)$', "You are welcome."),
        (r'^(no problem)$', "No problem."),
        (r'^(sorry late|i sorry late)$', "I am sorry for being late."),
        (r'^(i sorry|sorry)$', "I am sorry."),
        (r'^(excuse me|excuse)$', "Excuse me, please."),
        (r'^(good morning)$', "Good morning!"),
        (r'^(good afternoon)$', "Good afternoon!"),
        (r'^(good evening)$', "Good evening!"),
        (r'^(good night)$', "Good night!"),
        (r'^(have good day|good day)$', "Have a good day!"),
        (r'^(see you later)$', "See you later!"),
        (r'^(see you tomorrow)$', "See you tomorrow!"),
        (r'^(take care)$', "Take care!"),
        (r'^(goodbye|bye)$', "Goodbye!"),
        (r'^(i love you)$', "I love you."),
        (r'^(i miss you)$', "I miss you."),

        # ── Common Personal State & Actions ──
        (r'^(i work now)$', "I am working now."),
        (r'^(i eat now)$', "I am eating now."),
        (r'^(i study now)$', "I am studying now."),
        (r'^(i leave now)$', "I am leaving now."),
        (r'^(i hungry)$', "I am hungry."),
        (r'^(i thirsty)$', "I am thirsty."),
        (r'^(i tired)$', "I am tired."),
        (r'^(i sick)$', "I am feeling sick."),
        (r'^(i fine|i good|i okay)$', "I am doing well."),
        (r'^(i ready)$', "I am ready."),
        (r'^(i happy)$', "I am happy."),
        (r'^(i sad)$', "I am feeling sad."),
        (r'^(i deaf)$', "I am deaf."),
        (r'^(i hearing)$', "I am hearing."),
        (r'^(i learn fsl|i learn sign)$', "I am learning Filipino Sign Language."),
        (r'^(i want water)$', "I would like some water, please."),
        (r'^(i want food|i want eat)$', "I want to eat."),
        (r'^(i need help)$', "I need help."),
        (r'^(i need bathroom|i go bathroom)$', "I need to use the bathroom."),
        (r'^(i don\'t know|i not know)$', "I don't know."),
        (r'^(i understand)$', "I understand."),
        (r'^(i not understand)$', "I don't understand."),
    ]]

    PRONOUNS_SUBJ = {'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    BE_FORMS = {'am', 'is', 'are', 'was', 'were', 'be', 'being', 'been'}

    ADJECTIVES = {
        'hungry', 'tired', 'thirsty', 'happy', 'sad', 'angry', 'fine', 'good',
        'bad', 'ready', 'busy', 'sick', 'hot', 'cold', 'beautiful', 'late',
        'early', 'excited', 'deaf', 'hearing', 'afraid', 'clean', 'dirty', 'nice',
        'smart', 'kind', 'friendly', 'sleepy', 'bored', 'proud', 'strong', 'weak'
    }

    PROFESSIONS = {
        'teacher', 'doctor', 'student', 'nurse', 'engineer', 'police',
        'friend', 'lawyer', 'driver', 'worker', 'child', 'parent', 'boss',
        'interpreter', 'artist', 'scientist', 'chef', 'manager'
    }

    VERBS = {
        'work': 'working', 'eat': 'eating', 'go': 'going', 'talk': 'talking',
        'study': 'studying', 'sleep': 'sleeping', 'play': 'playing', 'help': 'helping',
        'learn': 'learning', 'read': 'reading', 'write': 'writing', 'cook': 'cooking',
        'walk': 'walking', 'run': 'running', 'wait': 'waiting', 'watch': 'watching',
        'see': 'seeing', 'listen': 'listening', 'start': 'starting', 'stop': 'stopping',
        'buy': 'buying', 'drink': 'drinking', 'meet': 'meeting', 'leave': 'leaving',
        'come': 'coming', 'teach': 'teaching', 'practice': 'practicing', 'drive': 'driving',
        'clean': 'cleaning', 'understand': 'understanding', 'know': 'knowing', 'think': 'thinking'
    }

    QUESTION_WORDS = {'what', 'where', 'when', 'who', 'why', 'how', 'which'}

    def __init__(self):
        pass

    # ─── Bounded memoization ────────────────────────────────────────────────
    # Guards pure translate() results. Insertion-ordered dict evicts the oldest
    # entry once the cap is reached (deterministic, no external deps).
    _MEMO_MAX = 512
    _memo: dict[tuple, str] = {}

    @classmethod
    def _memo_get(cls, key: tuple):
        return cls._memo.get(key)

    @classmethod
    def _translate_cached_key(cls, key: tuple) -> str:
        """translate() entry for a pre-cleaned token tuple (recursion helper)."""
        return default_translator.translate(list(key))

    @classmethod
    def clear_translation_cache(cls) -> None:
        """Invalidate all memoized translation results."""
        cls._memo.clear()

    def clean_tokens(self, text_or_tokens) -> list[str]:
        """Cleans and tokenizes input words or string."""
        if isinstance(text_or_tokens, str):
            raw = re.findall(r'[a-zA-Z0-9_]+', text_or_tokens.lower())
        else:
            raw = [str(t).lower().strip() for t in text_or_tokens]

        tokens = []
        for t in raw:
            # Handle compound underscore words like thank_you
            if '_' in t:
                parts = t.split('_')
                tokens.extend(p for p in parts if p)
            elif t:
                tokens.append(t)
        return tokens

    def translate(self, text_or_tokens) -> str:
        """
        Translates a sequence of signed words into a fluent natural English sentence.
        """
        tokens = self.clean_tokens(text_or_tokens)
        if not tokens:
            return ""

        # Canonical key: pure function of the cleaned tokens, shared by the
        # memoized and uncached paths (greeting decomposition recurses via the
        # same key form, so sub-results are cached too).
        key = tuple(tokens) if isinstance(text_or_tokens, str) else ('@seq', tuple(tokens))

        hit = self._memo_get(key)
        if hit is not None:
            return hit

        result = self._translate_uncached(tokens)

        if len(self._memo) >= self._MEMO_MAX:
            self._memo.pop(next(iter(self._memo)))
        self._memo[key] = result
        return result

    def _translate_uncached(self, tokens: list[str]) -> str:
        """Unmemoized translation core; identical logic/order to the original."""
        query = " ".join(tokens)

        # 1. Check exact pattern matches
        for pattern, replacement in self.IDIO_PATTERNS:
            if pattern.match(query):
                return replacement

        # 2. Greeting prefix decomposition (e.g. "Hello who you" -> "Hello, who are you?")
        if len(tokens) > 1 and tokens[0] in {'hello', 'hi', 'hey'}:
            greeting = tokens[0].capitalize()
            rest_translated = self._translate_cached_key(tuple(tokens[1:]))
            if rest_translated:
                words = rest_translated.split()
                fw = words[0]
                fw_formatted = fw if fw == 'I' else fw.lower()
                rem = rest_translated[len(fw):]
                return f"{greeting}, {fw_formatted}{rem}"

        # 2b. Two-word greeting prefix ("good morning", "good afternoon", etc.)
        if len(tokens) > 2 and tokens[0] == 'good' and tokens[1] in {'morning', 'afternoon', 'evening', 'night', 'day'}:
            greeting = f"Good {tokens[1]}"
            rest_translated = self._translate_cached_key(tuple(tokens[2:]))
            if rest_translated:
                words = rest_translated.split()
                fw = words[0]
                fw_formatted = fw if fw == 'I' else fw.lower()
                rem = rest_translated[len(fw):]
                return f"{greeting}, {fw_formatted}{rem}"

        # 3. General Rule-based Conjointing & Grammar Smoothing
        return self._smooth_grammar(tokens)

    def _smooth_grammar(self, tokens: list[str]) -> str:
        n = len(tokens)
        if n == 1:
            w = tokens[0]
            if w in {'hello', 'hi', 'hey'}:
                return f"{w.capitalize()}!"
            if w in {'thanks', 'thank'}:
                return "Thank you!"
            if w in self.ADJECTIVES:
                return f"Feeling {w}."
            if w in self.PROFESSIONS:
                article = 'an' if w[0] in 'aeiou' else 'a'
                return f"A {w}."
            return f"{w.capitalize()}."

        # Case A: Conjoint Pronouns ("You" + "I" / "Me" + "You")
        if ((tokens[0] in {'you', 'i'} and tokens[1] in {'you', 'i'} and tokens[0] != tokens[1]) or
            (tokens[0] == 'you' and tokens[1] == 'me') or
            (tokens[0] == 'me' and tokens[1] == 'you')):
            rest = tokens[2:]
            if not rest:
                return "We are together."

            has_now = 'now' in rest
            verb_candidate = [w for w in rest if w in self.VERBS or w in {'work', 'eat', 'go', 'start'}]

            if verb_candidate:
                v = verb_candidate[0]
                if has_now or v in {'work', 'eat', 'go', 'start'}:
                    v_ing = self.VERBS.get(v, v + 'ing')
                    return f"Let's start {v_ing}."
                return f"Let's {v}."
            else:
                if rest[0] in self.PROFESSIONS or rest[0] in {'friend', 'team', 'partner'}:
                    plural = rest[0] + ('s' if not rest[0].endswith('s') else '')
                    return f"We are {plural}."
                sub_sentence = " ".join(rest)
                return f"We {sub_sentence}."

        # Case B: Questions starting with Wh-word
        if tokens[0] in self.QUESTION_WORDS:
            q_word = tokens[0].capitalize()
            rest = tokens[1:]
            if not rest:
                return f"{q_word}?"

            if rest[0] in self.PROFESSIONS or rest[0] in {'bathroom', 'toilet', 'hospital', 'store', 'school', 'office'}:
                return f"{q_word} is the {' '.join(rest)}?"
            if rest[0] in {'this', 'that', 'it'}:
                return f"{q_word} is {rest[0]}?"
            if rest[0] == 'time':
                return "What time is it?"
            if rest[0] in {'you', 'we', 'they'}:
                subj = rest[0]
                be = 'are'
                v_words = rest[1:]
                if v_words:
                    v = v_words[0]
                    v_ing = self.VERBS.get(v, v + 'ing')
                    return f"{q_word} {be} {subj} {v_ing}?"
                return f"{q_word} {be} {subj}?"
            if rest[0] in {'he', 'she'}:
                subj = rest[0]
                v_words = rest[1:]
                if v_words:
                    v = v_words[0]
                    v_ing = self.VERBS.get(v, v + 'ing')
                    return f"{q_word} is {subj} {v_ing}?"
                return f"{q_word} is {subj}?"

        # Case C: Question with Wh-word at end (e.g. "you go where" -> "Where are you going?")
        if tokens[-1] in self.QUESTION_WORDS:
            q_word = tokens[-1].capitalize()
            front = tokens[:-1]
            if front[0] in self.PRONOUNS_SUBJ:
                subj = front[0]
                be = 'am' if subj == 'i' else ('is' if subj in {'he', 'she', 'it'} else 'are')
                verbs = [w for w in front[1:] if w in self.VERBS]
                if verbs:
                    v_ing = self.VERBS[verbs[0]]
                    return f"{q_word} {be} {subj} {v_ing}?"
                return f"{q_word} {be} {subj}?"

        # Case D: Missing copulas (is, am, are) for subject pronouns
        subj = tokens[0]
        if subj in self.PRONOUNS_SUBJ:
            be_form = 'am' if subj == 'i' else ('is' if subj in {'he', 'she', 'it'} else 'are')
            rest = tokens[1:]
            if rest:
                target = rest[0]
                # If target is adjective: "I hungry" -> "I am hungry."
                if target in self.ADJECTIVES:
                    return f"{subj.capitalize()} {be_form} {' '.join(rest)}."
                # If target is profession: "She doctor" -> "She is a doctor."
                if target in self.PROFESSIONS:
                    article = 'an' if target[0] in 'aeiou' else 'a'
                    return f"{subj.capitalize()} {be_form} {article} {' '.join(rest)}."
                # If target is verb + now: "I work now" -> "I am working now."
                if target in self.VERBS and 'now' in rest:
                    v_ing = self.VERBS[target]
                    other = [w for w in rest[1:] if w != 'now']
                    other_str = (" " + " ".join(other)) if other else ""
                    return f"{subj.capitalize()} {be_form} {v_ing}{other_str} now."
                # If target is verb alone
                if target in self.VERBS:
                    return f"{subj.capitalize()} {target} {' '.join(rest[1:])}.".strip()

        # Case E: Missing copulas for singular nouns: "Teacher good" -> "The teacher is good."
        if tokens[0] in self.PROFESSIONS and len(tokens) >= 2 and tokens[1] in self.ADJECTIVES:
            return f"The {tokens[0]} is {tokens[1]}."

        # Case F: Infinitive marker "to" after volition verbs (want, need, like + verb)
        out_tokens = []
        i = 0
        while i < len(tokens):
            w = tokens[i]
            out_tokens.append(w)
            if w in {'want', 'need', 'like', 'love', 'hope', 'try', 'learn'} and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt in self.VERBS or nxt in {'eat', 'go', 'work', 'sleep', 'help', 'sign', 'talk'}:
                    out_tokens.append('to')
            i += 1

        # General Fallback smoothing:
        sentence = " ".join(out_tokens).strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith(('.', '!', '?')):
                sentence += "."
        return sentence

# Global singleton instance for easy import
default_translator = FSLSentenceTranslator()

def translate_sentence(words_or_string) -> str:
    """Convenience function to translate FSL words to an English sentence."""
    return default_translator.translate(words_or_string)
