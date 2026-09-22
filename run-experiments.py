import os
import math
import random
import json
import re
import time
import requests
from collections import Counter
from openai import OpenAI

# ------------- CONFIG -------------

#API_KEY =  #INSERT API KEY HERE


for MODEL_NAME in ["gpt-4o","gpt-5.6-sol","us.anthropic.claude-sonnet-4-6","us.meta.llama4-maverick-17b-instruct-v1:0"]:


    TEMPERATURE  = 0                # control LLM temperature here
    TARGET_COPIES_PER_HISTORY = 6
    HORIZON      = 4                # length of each history
    ACTIONS      = ["L", "R"]
    REVERSE      = False          # True = most recent first, False = from round 1
    TREATMENT    = "words"
    aa,ab,ba,bb  = 3,0,5,1
    MAPPING_ID   = 1

    LABEL_MAPPINGS = [{"L": "L", "R": "R"},  # identity
                    {"L": "R", "R": "L"},  # swap
                    ]

    MAPPING    = LABEL_MAPPINGS[MAPPING_ID]


    if MODEL_NAME == "gpt-4o":
        OUTPUT_FILE = f"data/gpt-4o-temp0/interchange-LR/data_llm_H{HORIZON}_T00_norev_words.json"
    elif MODEL_NAME == "us.meta.llama4-maverick-17b-instruct-v1:0":
        OUTPUT_FILE = f"data/llama-4-maverick/interchange-LR/data_llm_H{HORIZON}_T00_norev_words.json"
    elif MODEL_NAME == "us.anthropic.claude-sonnet-4-6":
        OUTPUT_FILE = f"data/claude-sonnet-4-6/interchange-LR/data_llm_H{HORIZON}_T00_norev_words.json"
    elif MODEL_NAME == "gpt-5.6-sol":
        OUTPUT_FILE = f"data/gpt-5.6-sol/interchange-LR/data_llm_H{HORIZON}_T00_norev_words.json"
    # All permutations of labels {L,R}


    # ------------- PROMPTS -------------

    SYSTEM_PROMPT = """Follow specified goals."""

    USER_PROMPT_TEMPLATE = (
        "You are in an interaction with another agent.\n"
        "Your interaction is over several rounds.\n"
        "In each round, you have a choice between two options, L or R.\n"
        "The other agent also has a choice between L or R.\n"
        "The two of you will decide without knowing what the other will choose.\n"
        "Depending on the choices, you receive a reward, which is measured in points.\n"
        f"If both of you choose L, both of you get {aa} points.\n"
        "If you choose L, and the other agent chooses R, "
        f"you get {ab} points and the other agent gets {ba} points.\n"
        "If you choose R, and the other agent chooses L, "
        f"you get {ba} points and the other agent gets {ab} points.\n"
        f"If both of you choose R, both of you get {bb} point(s).\n"
        "Your final reward is the total number of points you accumulate in all rounds.\n"
        "Here is your goal:   No instruction\n"
        "{framing_str}"
        "{history_str}"
        "This is a new round."
        "Do you choose L or R? Give only the character as output.\n"
        "Give no explanation.\n"
    )

    # ------------- RELABEL HELPERS -------------

    def relabel_history(history, mapping):
        """
        history: list[(a1, a2)] with canonical 'L'/'R'.
        mapping: {'L': 'L' or 'R', 'R': 'L' or 'R'}.
        Returns relabeled history as seen by the model.
        """
        return [[mapping[a1], mapping[a2]] for (a1, a2) in history]

    def relabel_text_LR(text, mapping):
        """
        Replace L/R in text according to mapping in a safe way.
        """
        tmp = text.replace("L", "_L_").replace("R", "_R_")
        tmp = tmp.replace("_L_", mapping["L"]).replace("_R_", mapping["R"])
        return tmp

    # ------------- HISTORY / PAYOFF HELPERS -------------

    def payoff(a_self, a_other):
        if a_self == "L" and a_other == "L":
            return aa
        if a_self == "L" and a_other == "R":
            return ab
        if a_self == "R" and a_other == "L":
            return ba
        if a_self == "R" and a_other == "R":
            return bb
        raise ValueError("Invalid actions")

    def _words_label(offset_from_last):
        if offset_from_last == 1:
            return "In the previous round"
        elif offset_from_last == 2:
            return "Two rounds ago"
        elif offset_from_last == 3:
            return "Three rounds ago"
        elif offset_from_last == 4:
            return "Four rounds ago"
        else:
            return f"{offset_from_last} rounds ago"

    def format_history_string(history, treatment="words", reverse=False, max_window=4):
        """
        history: list of (a1, a2) in chronological order (round 1..T-1).
        a1 = 'you', a2 = 'other agent'.
        """
        T = len(history)
        if T == 0:
            return ""  # no past rounds

        n = min(max_window, T)

        last_n = []
        for offset in range(n, 0, -1):  # n,...,2,1
            idx = -offset
            last_n.append((offset, history[idx]))

        if reverse:
            ordered = sorted(last_n, key=lambda x: x[0])          # 1,2,3,4,...
        else:
            ordered = sorted(last_n, key=lambda x: x[0], reverse=True)  # 4,3,2,1,...

        lines = []

        if treatment == "words":
            for offset, (a_self, a_other) in ordered:
                r_self = payoff(a_self, a_other)
                r_other = payoff(a_other, a_self)
                lead = _words_label(offset)
                lines.append(f"{lead}, you chose {a_self}, they chose {a_other}.")
                if offset == 1:
                    lines.append(
                        f"Therefore, in the previous round you got {r_self} point(s) and they got {r_other} point(s)."
                    )
                else:
                    lines.append(
                        f"Therefore, you got {r_self} point(s) and they got {r_other} point(s)."
                    )

        elif treatment == "random":
            k = random.randint(T + 1, T + 30)
            for offset, (a_self, a_other) in ordered:
                r_self = payoff(a_self, a_other)
                r_other = payoff(a_other, a_self)
                round_num = k - offset
                lines.append(
                    f"In round {round_num}, you chose {a_self}, they chose {a_other}."
                )
                lines.append(
                    f"Therefore, in that round you got {r_self} point(s) and they got {r_other} point(s)."
                )
        else:
            raise ValueError("treatment must be 'words' or 'random'")

        return "\n".join(lines) + "\n"

    def parse_action_strict(raw_response):
        """
        Extract exactly one valid action from a model response.

        Accepts:
            L
            R
            "The answer is L."
            "JUST L"

        Rejects:
            JUST
            L or R
            I choose L because ...
        """

        text = str(raw_response).strip().upper()

        # Exact one-character response
        if text in {"L", "R"}:
            return text

        # Remove common formatting characters
        cleaned = re.sub(r"[`*_\"'.,:;!?()\[\]{}]", " ", text)

        # Find standalone L/R tokens
        matches = re.findall(
            r"(?<![A-Z])([LR])(?![A-Z])",
            cleaned
        )

        # Accept only one unambiguous action
        if len(matches) == 1:
            return matches[0]

        return None
    
    def parse_llama_action(raw_response):
        """
        Accept only a response whose first non-whitespace character
        is exactly L or R.

        With max_gen_len=1, Llama should normally return simply
        'L' or 'R'.
        """

        if raw_response is None:
            return None

        text = str(raw_response).strip().upper()

        if text == "":
            return None

        # The expected response is exactly one character.
        if text in {"L", "R"}:
            return text

        # If extra text is returned, accept it only when the first
        # token is clearly L or R.
        match = re.match(r"^([LR])(?:\s|$)", text)

        if match:
            return match.group(1)

        return None

    def query_llama_action(prompt, max_attempts=5):
        """
        Query Llama until it returns a valid L/R response.
        """

        api_url = (
            "https://go.apis.huit.harvard.edu/"
            f"ais-bedrock-llm/v2/model/{MODEL_NAME}/invoke/"
        )

        headers = {
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
        }

        payload = {
            "prompt": prompt,
            "temperature": 0,
            "max_gen_len": 1,   # restore the original setting
            "top_p": 10**-8,
        }

        for attempt in range(max_attempts):

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"API request failed: "
                    f"{response.status_code}, {response.text}"
                )

            data = response.json()

            if "generation" in data:
                raw_response = data["generation"]

            elif (
                "outputs" in data
                and len(data["outputs"]) > 0
                and "text" in data["outputs"][0]
            ):
                raw_response = data["outputs"][0]["text"]

            else:
                raise KeyError(
                    f"Unexpected response format: {data}"
                )

            action = parse_llama_action(raw_response)

            if action is not None:
                return action

            print(
                f"Invalid Llama response "
                f"({attempt + 1}/{max_attempts}): "
                f"{raw_response!r}"
            )

            # Retry the same prompt rather than appending more text.
            time.sleep(0.2)

        raise ValueError(
            f"Llama failed to return L or R after "
            f"{max_attempts} attempts."
        )
    # ------------- LLM CALL -------------

    def get_llm_action(history, t, mapping=None):
        """
        Query the LLM for its action after a hypothetical history.

        `history` is always stored internally using canonical labels L/R.
        The mapping is applied exactly once to the complete prompt.
        The returned action is mapped back to canonical labels.
        """

        if mapping is None:
            mapping = {"L": "L", "R": "R"}

        # Build the history using canonical L/R labels.
        # Do not relabel the history separately.
        history_str = format_history_string(
            history,
            treatment=TREATMENT,
            reverse=REVERSE,
            max_window=HORIZON
        )

        framing_str = ""

        # Construct the prompt in canonical notation.
        user_prompt_raw = USER_PROMPT_TEMPLATE.format(
            framing_str=framing_str,
            history_str=history_str
        )

        # Relabel the complete prompt exactly once.
        user_prompt = relabel_text_LR(
            user_prompt_raw,
            mapping
        )


        # Query the model.
        if "gpt-5" in MODEL_NAME:

            api_url = (
                "https://go.apis.huit.harvard.edu/"
                "ais-openai-direct/v2"
            )

            client = OpenAI(
                api_key=API_KEY,
                base_url=api_url
            )

            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {
                        "role": "developer",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                reasoning={"effort": "high"}
            )

            raw_up = response.output_text

        elif "gpt-4" in MODEL_NAME:

            api_url = (
                "https://go.apis.huit.harvard.edu/"
                "ais-openai-direct/v2"
            )

            client = OpenAI(
                api_key=API_KEY,
                base_url=api_url
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=TEMPERATURE,
                seed=42
            )

            raw_up = response.choices[0].message.content.strip()

        elif "llama" in MODEL_NAME:

            raw_up = query_llama_action(user_prompt)


        elif "claude" in MODEL_NAME:

            url = (
                "https://go.apis.huit.harvard.edu/"
                f"ais-bedrock-llm/v2/model/{MODEL_NAME}/invoke"
            )

            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 2,
                "temperature": TEMPERATURE
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": API_KEY
            }

            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload)
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"API request failed: "
                    f"{response.status_code}, "
                    f"{response.text}"
                )

            raw_up = response.json()["content"][0]["text"]

        else:
            raise ValueError(
                f"Unknown MODEL_NAME pattern: {MODEL_NAME}"
            )

        # Convert the model's literal output back to canonical labels.
        inverse_map = {
            value: key
            for key, value in mapping.items()
        }

        raw_up = str(raw_up).strip().upper()

        for character in raw_up:
            if character in {"L", "R"}:
                return inverse_map[character]

        raise ValueError(
            f"Could not extract L/R from model response: {raw_up!r}"
        )

    # ------------- HISTORY GENERATION -------------

    def random_joint_action():
        return (random.choice(ACTIONS), random.choice(ACTIONS))

    def generate_all_histories(horizon):
        """
        Generate all possible histories of length `horizon` over
        the 4 joint actions (L,L), (L,R), (R,L), (R,R).
        """
        joint_actions = [("L", "L"), ("L", "R"), ("R", "L"), ("R", "R")]
        L = 4 ** horizon
        histories = []

        for idx in range(L):
            digits = []
            x = idx
            for _ in range(horizon):
                digits.append(x % 4)
                x //= 4
            digits.reverse()
            hist = [joint_actions[d] for d in digits]
            histories.append(hist)

        return histories

    # ------------- MAIN DATA GENERATION (APPEND) -------------

    def main_generate_append():
        
        # --------------------------------------------------
        # Target: exactly this many copies of every history
        # --------------------------------------------------

        os.makedirs(
            os.path.dirname(OUTPUT_FILE) or ".",
            exist_ok=True
        )

        # --------------------------------------------------
        # 1. Load existing data
        # --------------------------------------------------

        if os.path.exists(OUTPUT_FILE):

            print(f"Loading existing data from {OUTPUT_FILE} ...")

            with open(OUTPUT_FILE, "r") as f:
                existing_records = json.load(f)

            print(f"Existing records: {len(existing_records)}")

            if len(existing_records) > 0:
                first_hist = existing_records[0]["history"]

                if len(first_hist) != HORIZON:
                    raise ValueError(
                        f"Existing data has HORIZON={len(first_hist)}, "
                        f"but current HORIZON={HORIZON}."
                    )

        else:

            print(
                f"No existing data file found. "
                f"Creating {OUTPUT_FILE} ..."
            )

            existing_records = []

        # --------------------------------------------------
        # 2. Define the target size
        # --------------------------------------------------

        Lhist = 4 ** HORIZON

        target_total = (
            TARGET_COPIES_PER_HISTORY * Lhist
        )

        n_existing = len(existing_records)

        print(
            f"Target: {TARGET_COPIES_PER_HISTORY} copies "
            f"of each of {Lhist} histories."
        )

        print(
            f"Target dataset size: {target_total}; "
            f"existing records: {n_existing}"
        )

        if n_existing > target_total:
            raise ValueError(
                f"The dataset already contains {n_existing} records, "
                f"which exceeds the target size of {target_total}."
            )

        # --------------------------------------------------
        # 3. Count existing copies of every history
        # --------------------------------------------------

        def hist_key(hist):
            return tuple(tuple(pair) for pair in hist)

        counts = Counter()

        for rec in existing_records:
            counts[hist_key(rec["history"])] += 1

        # --------------------------------------------------
        # 4. Check that no history already exceeds the target
        # --------------------------------------------------

        all_histories = generate_all_histories(HORIZON)

        for hist in all_histories:

            key = hist_key(hist)
            current_count = counts.get(key, 0)

            if current_count > TARGET_COPIES_PER_HISTORY:
                raise ValueError(
                    "Cannot complete a uniform dataset because history "
                    f"{hist} already appears {current_count} times, "
                    f" exceeding the target of "
                    f"{TARGET_COPIES_PER_HISTORY} copies."
                )

        # --------------------------------------------------
        # 5. Generate only the missing copies
        # --------------------------------------------------

        remaining_histories = []

        for hist in all_histories:

            key = hist_key(hist)
            current_count = counts.get(key, 0)

            missing = (
                TARGET_COPIES_PER_HISTORY
                - current_count
            )

            # Add exactly the number of missing copies.
            remaining_histories.extend(
                [hist] * missing
            )

        expected_remaining = target_total - n_existing

        if len(remaining_histories) != expected_remaining:
            raise RuntimeError(
                f"Internal error: expected "
                f"{expected_remaining} remaining histories, "
                f"but generated {len(remaining_histories)}."
            )

        print(
            f"Remaining responses to collect: "
            f"{len(remaining_histories)}"
        )

        if len(remaining_histories) == 0:
            print("The dataset is already complete.")
            return

        # --------------------------------------------------
        # 6. Save atomically after every response
        # --------------------------------------------------

        def save_records_atomically(records):

            temporary_file = OUTPUT_FILE + ".tmp"

            with open(temporary_file, "w") as f:
                json.dump(records, f)
                f.flush()
                os.fsync(f.fileno())

            # Replace the existing file only after the temporary
            # file has been written successfully.
            os.replace(
                temporary_file,
                OUTPUT_FILE
            )

        # --------------------------------------------------
        # 7. Query and save after every successful response
        # --------------------------------------------------

        print("Querying LLM for next-round actions...")
        print(f"Using label mapping {MAPPING_ID}: {MAPPING}")

        t_next = HORIZON + 1
        all_records = existing_records.copy()

        for i, hist in enumerate(
            remaining_histories,
            start=1
        ):

            # If this call fails, no record is added or saved.
            # The same history will therefore be retried when
            # the code is run again.
            a_next = get_llm_action(
                hist,
                t_next,
                mapping=MAPPING
            )

            rec = {
                "history": hist,
                "action_next": a_next,
                "mapping_id": MAPPING_ID,
            }

            all_records.append(rec)

            # Save immediately after this response.
            save_records_atomically(all_records)

            print(
                f"Saved response {i}/{len(remaining_histories)}. "
                f"Total records: {len(all_records)}/{target_total}"
            )

        # --------------------------------------------------
        # 8. Verify exact uniformity
        # --------------------------------------------------

        final_counts = Counter(
            hist_key(rec["history"])
            for rec in all_records
        )

        incorrect_histories = [
            hist
            for hist in all_histories
            if final_counts[hist_key(hist)]
            != TARGET_COPIES_PER_HISTORY
        ]

        if incorrect_histories:
            raise RuntimeError(
                "The final dataset is not uniform. "
                f"Problematic histories: {incorrect_histories[:5]}"
            )

        print(
            f"Completed successfully. Every history appears "
            f"{TARGET_COPIES_PER_HISTORY} times."
        )

        print(
            f"Final dataset contains {len(all_records)} records."
        )

    if __name__ == "__main__":
        main_generate_append()