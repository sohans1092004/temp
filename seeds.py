"""Hand-written seed lists for the template-generated intent dataset.

Written independently of prepped_data (which stays a regression suite); the
generator reports and quarantines anything that still lands close to a
prepped transcript.

LABEL POLICY
  1 = the speaker is asking for help / signalling danger to THEMSELVES, now.
  0 = the words point elsewhere: negated, quoted/reported, about an object or a
      task, hyperbole, ordinary chatter.
  policy_sensitive = the label depends on a product decision (exact UDK words in
      the speaker's own non-danger speech, or quoted). Default 0, matching the
      user's prepped_data labels; flip in one place if the policy changes.
"""

# ---- positives: paraphrases per UDK (the exact phrase is added automatically) ----
PARAPHRASES = {
    "UDK_01": ["call the police I need help", "somebody call the police please", "please call the police right now",
               "call 112 I need help", "call 100 please I need help", "phone the police, help me",
               "get the police here, I need help", "someone please call the cops"],
    "UDK_02": ["I am not safe", "I don't feel safe at all", "I'm not safe here", "this isn't safe for me",
               "I'm really not safe right now", "I am not safe please come"],
    "UDK_03": ["get away from me", "get off me", "move away from me", "go away from me",
               "get away from me right now", "back off, get away from me", "stay off me, get away"],
    "UDK_04": ["someone is following me", "somebody is following me", "a man is following me",
               "there's someone following me", "I think someone is following me", "this guy keeps following me"],
    "UDK_05": ["I need help right now", "I need help immediately", "please I need help now",
               "I need help, please hurry", "help, I need help right now"],
    "UDK_06": ["don't touch me", "do not touch me", "stop touching me", "don't you dare touch me",
               "take your hands off me", "keep your hands off me"],
    "UDK_07": ["I'm being followed", "I am being followed", "I'm being followed by someone",
               "I think I'm being followed", "I'm being followed, I'm scared"],
    "UDK_08": ["help me please", "please help me", "somebody please help me", "help me, please help",
               "please, someone help me", "can someone help me please"],
    "UDK_09": ["I'm scared, stay back", "stay back, I'm scared", "I'm scared, don't come near me",
               "stay back please, I'm so scared", "I'm afraid, stay away"],
    "UDK_10": ["let go of me", "let me go", "let go of my hand", "let go, let me go",
               "leave my arm, let go of me", "release me, let go"],
    "UDK_11": ["I need the police here", "send the police here", "I need police right now",
               "please send the police here", "we need the police here"],
    "UDK_12": ["someone is trying to hurt me", "somebody is trying to hurt me", "he's trying to hurt me",
               "a man is trying to hurt me", "someone wants to hurt me"],
    "UDK_13": ["please don't hurt me", "don't hurt me", "please don't hurt me, please",
               "I beg you don't hurt me", "no no please don't hurt me"],
    "UDK_14": ["I don't feel safe here", "I don't feel safe", "I don't feel safe in this place",
               "this place doesn't feel safe", "I really don't feel safe here"],
    "UDK_15": ["stay away from me", "keep away from me", "stay away, don't come closer",
               "stay away from me please", "you stay away from me"],
    "UDK_16": ["I'm in danger", "I am in danger", "I'm in serious danger", "I think I'm in danger",
               "please, I'm in danger"],
    "UDK_17": ["somebody help me now", "someone help me now", "anybody help me now",
               "somebody please help me now", "help me now, somebody"],
    "UDK_18": ["I want to go home now", "I want to go home right now", "please let me go home now",
               "take me home now", "I just want to go home now, please", "can I please go home now",
               "I want to go home, take me home"],
    "UDK_19": ["please call for help", "call for help please", "someone call for help",
               "please get help", "call someone for help please"],
    "UDK_20": ["I need to get out of here", "I have to get out of here", "I need to leave right now",
               "get me out of here", "I need to get out of here now"],
}
POS_PREFIX = ["", "", "please, ", "oh god, ", "hey, ", "no no, ", "somebody, ", "arre, ", "ayyo, ", "listen, "]
POS_SUFFIX = ["", "", " please", " right now", ", I'm serious", ", please hurry", " now", " yaar", " na", ", he won't stop"]

# ---- negatives (hard) ------------------------------------------------------------
NEGATION = [
    "I'm safe", "I am totally safe", "I'm not in danger", "nobody is following me", "I don't need any help",
    "no need to call the police", "I'm not scared at all", "don't worry, nobody is hurting me", "I feel safe here",
    "no one is trying to hurt me", "you don't need to help me", "I'm fine, I don't need the police",
    "I'm not being followed, relax", "nobody touched me", "I'm okay, no danger at all", "I'm not in any trouble",
]
NEG_WRAP = ["{x}", "{x}, don't worry", "it's fine, {x}", "mom, {x}", "{x}, I'll call you later", "relax, {x}"]

QUOTE_SUBJ = ["she", "the girl", "my friend", "the actor", "the heroine", "my little sister", "the kid", "that character"]
QUOTE_VERB = ["kept shouting", "was yelling", "said", "screamed", "kept saying", "shouted"]
QUOTE_END = ["in the movie", "in the serial last night", "during the play", "and everyone started laughing",
             "it was so dramatic", "in that song", "in the video you sent", "in the web series"]

HYPERBOLE = [
    "help me, this {task} is killing me", "someone help me finish this {food}", "I'm in danger of {fail}",
    "call the police, this {food} is criminally good", "I need to get out of here, this {event} is so boring",
    "I'm scared of my {exam} results", "this {task} is trying to hurt me", "save me from this {event}",
    "I want to go home now, I'm so tired of this {event}", "get away from me with that {food}",
]
INSTRUCTION = [
    "don't touch the {hazard}", "get away from the {edge}", "let go of the {obj}", "stay away from the {hazard}",
    "stay back from the {edge}", "don't touch that {obj}, it's hot", "please don't hurt the {pet}",
    "keep your hands off the {food}", "move away from the {edge}, it's slippery", "don't let the {pet} get away",
]
NEAR_MISS = [
    "help me with this {obj}", "don't touch my {item}", "someone is following me on {app}",
    "I'm being followed by bad luck today", "can you help me with the {task}", "let go of the past",
    "stay away from {food}, it's too oily", "I need help with my {task}", "the police were at the {place} today",
    "don't touch my {item}, I just cleaned it", "get away from my {item}", "I need to get out of here by {time}",
    "is somebody following this {app} page", "my friend was in danger of missing the {vehicle}",
]
OWN_SPEECH_NONTHREAT = [  # exact UDK words, own speech, no danger -> policy_sensitive
    "get away from me, you're driving too close", "stay away from me, I have a cold", "don't touch me, my hands are dirty",
    "let go of me, I need to catch the {vehicle}", "I want to go home now, the {event} is over",
    "I need to get out of here, I'm late for the {event}", "don't touch me, I just got a tattoo",
]
CHATTER = [
    "I'll reach {place} in {n} minutes", "{name}, did you eat?", "the {vehicle} is so late today",
    "traffic is very bad near {place}", "don't wait up, I'll be late", "can you pick up {food} on the way",
    "what time is the {event}", "the auto guy is asking {n} rupees", "my phone battery is at {pct} percent",
    "I'm stuck in traffic near {place}", "call me when you reach", "the {vehicle} is full, I'll take the next one",
    "{name} is coming for dinner", "did you pay the electricity bill", "the {event} got postponed to {time}",
    "I'm at {place}, where are you", "send me the location", "it's raining heavily here",
    "let's meet at {place} at {time}", "I forgot my {item} at home", "the shop near {place} is closed",
    "{name}, can you hear me, the network is bad", "okay bye, talk to you later", "have you booked the tickets",
]
HINGLISH = [
    "arre yaar {place} mein bahut traffic hai", "{name} ko bolo {time} tak aa jaye", "chalo let's go, late ho raha hai",
    "kya scene hai, {event} kab hai", "main {place} pahunch gaya", "khana kha liya kya {name}",
    "bas {n} minute mein aata hoon", "yaar {vehicle} abhi tak nahi aayi",
]

SLOTS = {
    "task": ["homework", "assignment", "project", "meeting", "code", "report", "workout"],
    "food": ["biryani", "samosa", "pizza", "chai", "dosa", "pani puri", "cake"],
    "fail": ["failing the exam", "missing the deadline", "losing this game", "missing the train"],
    "event": ["meeting", "lecture", "wedding", "movie", "class", "function", "match"],
    "exam": ["exam", "board exam", "semester", "interview"],
    "hazard": ["stove", "wire", "iron", "gas", "socket", "fire"],
    "edge": ["edge", "platform edge", "road", "railing", "balcony"],
    "obj": ["rope", "box", "bag", "handle", "kite string", "door"],
    "pet": ["dog", "cat", "puppy", "bird"],
    "item": ["phone", "laptop", "stuff", "bag", "charger", "books", "bike"],
    "app": ["instagram", "twitter", "youtube", "linkedin"],
    "place": ["the station", "MG Road", "the mall", "college", "the metro", "Koramangala", "Ameerpet", "Andheri"],
    "vehicle": ["bus", "train", "metro", "auto", "cab"],
    "name": ["Priya", "Rahul", "Amma", "Kiran", "Sneha", "Arjun", "Divya"],
    "n": ["five", "ten", "twenty", "fifteen", "two hundred", "forty"],
    "pct": ["five", "ten", "twenty", "fifteen", "forty"],
    "time": ["six", "seven thirty", "tomorrow", "Monday", "evening"],
}

# ---- minimal pairs: SAME ambiguous phrase, meaning decided by neighbouring words ----
# Each phrase keeps all its contexts in ONE split (so test pairs are unseen phrases).
# (danger contexts -> label 1, harmless contexts -> label 0); the phrase comes first.
MINIMAL_PAIRS = {
    "leave me": (["you're hurting me", "stop following me", "I'll scream, I swear", "please, just go away"],
                 ["at the metro station", "some biryani for dinner", "a message when you reach", "out of this trip plan"]),
    "stop it": (["please, it hurts", "let go of my hand", "you're scaring me, stop"],
                ["you're making me laugh", "the song is so annoying", "you'll spill the chai"]),
    "get off me": (["you're hurting me", "somebody help", "I said get off"],
                   ["you're sitting on my charger", "the dog is licking my face, ugh", "I need to finish this call"]),
    "let go": (["you're hurting my arm", "I'll call the police", "please, let me leave"],
               ["of the handle slowly", "of the stress yaar, relax", "of the kite string now"]),
    "don't come near me": (["I'm warning you", "I'll scream for help", "stay where you are"],
                           ["I have a bad cold", "I just painted my nails", "I'm all sweaty from the gym"]),
    "help": (["he's grabbing me", "someone is chasing me", "he won't let me go"],
             ["I can't open this jar", "which bus goes to the airport", "carry these bags to the car"]),
    "go away": (["stop following me", "or I'll scream", "don't touch me"],
                ["I'm studying for the exam", "the rain will go away soon", "and come back after lunch"]),
    "please don't": (["please don't hurt me", "don't take me anywhere", "I'm begging you"],
                     ["please don't forget the milk", "don't wait for me for dinner", "don't tell mom about the marks"]),
    "hold me": (["don't let him take me", "I'm so scared, he's outside", "he's coming back"],
                ["the bus is shaking too much", "while I click the photo", "I'll fall off the scooter"]),
    "someone is outside": (["banging on the door, I'm scared", "and he won't leave, please come", "trying to open the window"],
                           ["must be the delivery guy", "it's probably the milkman", "can you check who it is"]),
}
# ---- negation minimal pairs: (danger, same sentence negated) per UDK ----------------
# Teaches what "not/no/never/nobody" DOES: the two sides differ only in the negation, so
# the model can't separate them on topic words. Both sides of a UDK share one split and
# the split is by UDK, so test measures negation on phrases never seen negated.
# UDKs that already carry a negation (02, 06, 13, 14) are paired with the positive form.
NEG_PAIRS = {
    "UDK_01": [("call the police, I need help", "don't call the police, I don't need help"),
               ("I need help, call the police", "I don't need help, no need to call the police"),
               ("please call the police", "please don't call the police"),
               ("we need the police and I need help", "we don't need the police and I don't need help")],
    "UDK_02": [("I'm not safe", "I'm safe"), ("I'm not safe here", "I'm safe here"),
               ("I am not safe right now", "I am safe right now"), ("it's not safe for me here", "it's safe for me here"),
               ("I don't think I'm safe", "I think I'm safe")],
    "UDK_03": [("get away from me", "he got away from me"), ("he won't get away from me", "he got away from me, it's fine"),
               ("I can't get him away from me", "I got him away from me, it's fine"),
               ("get away from me now", "you don't have to get away from me")],
    "UDK_04": [("someone is following me", "no one is following me"), ("someone is following me", "someone isn't following me"),
               ("a man is following me", "no man is following me"), ("I think someone is following me", "I don't think anyone is following me"),
               ("he's still following me", "he's not following me anymore"), ("somebody's following me", "nobody's following me")],
    "UDK_05": [("I need help right now", "I don't need help right now"), ("I need help now", "I don't need any help now"),
               ("I need help right now", "I never said I need help"), ("I really need help", "I really don't need help")],
    "UDK_06": [("don't touch me", "he didn't touch me"), ("he's touching me", "he isn't touching me"),
               ("stop touching me", "nobody is touching me"), ("he keeps touching me", "he never touched me"),
               ("someone touched me", "no one touched me")],
    "UDK_07": [("I'm being followed", "I'm not being followed"), ("I am being followed", "I am not being followed"),
               ("I think I'm being followed", "I don't think I'm being followed"),
               ("I'm being followed again", "I'm not being followed anymore")],
    "UDK_08": [("help me, please", "you don't need to help me"), ("please help me", "please don't help me, I can do it"),
               ("somebody help me", "nobody needs to help me"), ("I need someone to help me", "I don't need anyone to help me")],
    "UDK_09": [("I'm scared, stay back", "I'm not scared, you can come"), ("I'm scared", "I'm not scared"),
               ("I'm so scared", "I'm not scared at all"), ("I'm really scared right now", "I'm not really scared right now")],
    "UDK_10": [("let go of me", "he let go of me, I'm fine"), ("he won't let go of me", "he didn't grab me at all"),
               ("he's holding me, let go", "he's not holding me"), ("he's grabbing my arm", "he isn't grabbing my arm")],
    "UDK_11": [("I need the police here", "I don't need the police here"), ("send the police here", "don't send the police here"),
               ("we need the police", "we don't need the police"), ("I need the police right now", "I don't need the police right now")],
    "UDK_12": [("someone is trying to hurt me", "no one is trying to hurt me"),
               ("someone is trying to hurt me", "someone isn't trying to hurt me"),
               ("he's trying to hurt me", "he's not trying to hurt me"), ("he wants to hurt me", "he doesn't want to hurt me"),
               ("they are trying to hurt me", "they never tried to hurt me")],
    "UDK_13": [("please don't hurt me", "he didn't hurt me"), ("he's hurting me", "he isn't hurting me"),
               ("you're hurting me", "you're not hurting me"), ("he hurt me", "he never hurt me"),
               ("it hurts, he's hurting me", "it doesn't hurt, nobody is hurting me")],
    "UDK_14": [("I don't feel safe here", "I feel safe here"), ("I don't feel safe", "I do feel safe"),
               ("I don't feel safe with him", "I feel safe with him"), ("this place doesn't feel safe", "this place feels safe")],
    "UDK_15": [("stay away from me", "you don't have to stay away from me"),
               ("he won't stay away from me", "he's staying away from me, it's fine"),
               ("stay away from me, please", "no, don't stay away from me"),
               ("keep him away from me", "he isn't near me")],
    "UDK_16": [("I'm in danger", "I'm not in danger"), ("I am in danger", "I am not in danger"),
               ("I think I'm in danger", "I don't think I'm in danger"), ("I'm in danger right now", "I'm not in any danger right now"),
               ("I'm in real danger", "I was never in danger")],
    "UDK_17": [("somebody help me now", "nobody needs to help me now"), ("someone help me", "no one has to help me"),
               ("I need somebody to help me now", "I don't need anybody to help me now"),
               ("somebody please help me", "somebody please don't bother helping me, I'm fine")],
    "UDK_18": [("I want to go home now", "I don't want to go home now"), ("I want to go home", "I don't want to go home yet"),
               ("please let me go home now", "I'm not in a hurry to go home"),
               ("I really want to go home now", "I don't really want to go home now")],
    "UDK_19": [("please call for help", "please don't call for help"), ("call for help", "don't call for help"),
               ("someone call for help", "no one needs to call for help"), ("we should call for help", "we shouldn't call for help")],
    "UDK_20": [("I need to get out of here", "I don't need to get out of here"),
               ("I have to get out of here", "I don't have to get out of here"),
               ("I need to leave right now", "I don't need to leave right now"),
               ("get me out of here", "I don't want to get out of here")],
}
# negation words that are part of real distress -> label 1. Without these the model
# learns the shortcut "has 'not' -> safe" instead of what the negation applies to.
NEG_DANGER = [
    "I'm not okay, please help me", "I'm not joking, call the police", "no, no, let go of me",
    "no, stop, don't touch me", "I can't breathe, help", "I can't get away from him",
    "he won't let me go", "I don't know where I am and someone is following me",
    "I'm not sure but I think someone is following me", "I can't find a way out, help me",
    "nobody is helping me, please come", "no one will help me, call the police",
    "don't leave me alone with him", "I'm not kidding, he's trying to hurt me",
    "he won't stop following me", "I can't stop shaking, I'm scared", "I never felt this unsafe, please come",
    "no, please, not again, he's hurting me", "I don't want to be here, he's scaring me",
    "I'm not fine, I'm in danger", "don't hang up, someone is outside my door", "I can't move, he's holding me",
    "no, I won't go with you, let go", "why won't anyone help me",
]
# said, then taken back -> label 0, policy_sensitive (should a retraction cancel an alert?)
RETRACTIONS = [
    "help me! no sorry, I'm fine, I was talking about the game",
    "I'm in danger, haha just kidding, it's the exam tomorrow",
    "someone is following me, oh wait, it's just Rahul",
    "don't touch me, oh it's you, you scared me",
    "call the police, no no, it's fine, the dog just knocked the vase",
    "let go of me, okay okay, I'll come, I'm coming",
    "get away from me, sorry, I thought you were a stranger",
    "I'm scared, stay back, oh it was just the cat",
]
# sound-alike substitutions of the kind Whisper/Parakeet produce on noisy speech
# (negation words deliberately absent: a corruption must never flip a label)
ASR_CONFUSIONS = {
    "help": ["health", "hell", "held"], "away": ["a way", "way"], "touch": ["tough", "dutch"],
    "police": ["please", "polish"], "scared": ["scarred", "skated"], "safe": ["save", "say"],
    "following": ["falling", "follow in"], "hurt": ["heard", "hot"], "hurting": ["hurdling", "hurting"],
    "leave": ["live", "leaf"], "danger": ["dangerous", "ranger"], "stay": ["say", "stray"],
    "back": ["pack", "bag"], "let": ["led", "late"], "someone": ["some one", "someday"],
    "go": ["goal", "so"], "me": ["mean", "m"], "scream": ["stream", "cream"], "grabbing": ["grab in", "crabbing"],
}
