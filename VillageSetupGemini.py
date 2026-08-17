import json
import os
import time
import datetime
from pathlib import Path

from google.genai.errors import APIError as GoogleAPIError
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

from google import genai

#Find this folder location
Script_folder= Path(__file__).resolve().parent

#Define some directories
Village_dir = Script_folder/ os.environ.get("Village_dir", "Village_DataV1.6")
Village_dir.mkdir(exist_ok=True)
 
#Recent events loaded
Village_State = Village_dir/"Village_State.json"
Recent= 200
 
#Full history of the village
Village_log= Village_dir/'Village_Log.json'

#Sync the clock  with everyone
Village_Clock = Village_dir/"Village_Clock.json"
Times_of_day = ["morning", "afternoon", "evening", 'night']


def load_village_clock() -> dict:
    """
    #Load the village clock, or set it if no file exists
    """
    if Village_Clock.exists():
        try:
            return json.loads(Village_Clock.read_text())
        except json.JSONDecodeError:
            pass

    #This way the village clock starts at day 0: morning
    return {"day": -1, "time": "night"}

def advance_village_clock() -> dict:
    """
    Check which index the current time is in Times_of_day, 
    to check if it needs to continue to the next day and reset the Times_of_day cycle.
    Write the current time to the village_clock file. 
    """
    clock = load_village_clock()
    
    idx = Times_of_day.index(clock["time"]) if clock["time"] in Times_of_day else 0
    idx = (idx + 1) % len(Times_of_day)
    clock["time"] = Times_of_day[idx]
    if idx == 0:
        clock["day"] += 1
    Village_Clock.write_text(json.dumps(clock, indent=2))
    return clock
 
#Two different Gemini models with 15 RPM, 250k TPM and 500 RPD (use 1 for consistancy)
#Can use the second to handle future tasks used for future features
Default_Gemini_Model = "gemini-3.5-flash-lite"
#Default_Gemini_Model = "gemini-3.1-flash-lite"
 
Model = os.environ.get("Village_Agent_Model", Default_Gemini_Model)

def get_llm_client():
    """
    Return the Google Gemini client
    Remnant of when I used different clients. 
    Could be useful for furture expansion.
    """
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

#---=== Relationship Update ===---

#Define relationship layout, this is the initial state all villagers are in
Default_Relationship= {
    "type": "stranger",
    "affinity": 0.0,
    "familiarity": 0,
    "note": "Haven't really interacted yet.",
    "last_interaction_day": None,
}

#Give this to the LLM in the prompt to correctly interpret the relationship
Relationship_Scale_Guide = """
Relationship scale reference:

AFFINITY (-1.0 to 1.0): how this villager feels about the other person:
  -1.0 to -0.6  :hostile/hatred
  -0.5 to -0.1:dislike
   0.0: neutral
   0.1 to 0.4: friendly acquaintance
   0.5 to 0.7: friend
   0.8 to 1.0: romantic partner

  Adjust gradually. A single pleasant chat might shift affinity by
  +0.02 to +0.05. A betrayal or major conflict might shift it by
  -0.2 to -0.3. Avoid huge jumps unless the moment truly warrants it.

FAMILIARITY (0+): roughly how many MEANINGFUL interactions you've had:
   0     : stranger
   1-5   : met a few times
   6-14   : know them decently
   15-29  : deep familiarity
   30+   : family-level closeness

TYPE: a short label consistent with the above:
  stranger, acquaintance, friend, close friend, rival, enemy,
  family, romantic, mentor, mentee.
"""

# ---=== Structured Output Schemas (Gemini response_schema) ===---
#Constraining the JSON shape is more reliable than asking
#in the prompt, and means we rarely need to fall back to the
#except-json.JSONDecodeError path below.

#The structure which allows us to read it properly for both sleep and action calls:
Sleep_Decision_Schema = {
    "type": "object",
    "properties": {
        "sleep_location": {"type": "string"},
        "action": {"type": "string"},
        "relationship_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "villager": {"type": "string"},
                    "affinity_change": {"type": "number"},
                    "type": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["villager"],
            },
        },
    },
    "required": ["sleep_location", "action"],
}

Act_Decision_Schema = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "inner_thought": {"type": "string"},
        "relationship_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "villager": {"type": "string"},
                    "affinity_change": {"type": "number"},
                    "type": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["villager"],
            },
        },
        "new_goal": {"type": "string", "nullable": True},
        "new_location": {"type": "string", "nullable": True},
    },
    "required": ["action", "inner_thought"],
}


#These two classes makes sure response.choices[0].message.content still works like OpenAI.
#Originally I used OpenAI.
#TODO: Convert code to a universal method instead of the makeshift solution. --> Once I am happy with the API/LLM used
class MockChoice:
    def __init__(self, text):
        self.message = type('MockMsg', (object,), {'content': text})()

class MockResponse:
    def __init__(self, text):
        self.choices = [MockChoice(text)]

class VillageAgent:
    """
    The class containing all capabilities of the villagers/agents.
    Everything they do and their specifics are being stored and called in here,
    Together with how they act.
    """
    def __init__(self, name: str, race: str, gender:str, personality: str, goals: list[str], profession: str, home: str, model= Model):

        #Define the initial specifics of the villager
        self.name= name
        self.race= race
        self.gender= gender
        self.personality= personality
        self.profession= profession
        self.goals= goals
        self.home= home
        self.model= model
        self.client= get_llm_client()

        #Give each villager their own memory file
        self.memory_file= Village_dir/f"{name.lower()}_memory.json"
        self.memories= self._load_memories()
        self.relationships: dict[str, dict] = self.memories.get('relationships', {})
        #self.day: int= self.memories.get("day", 0)
        #self.time: str = self.memories.get("time", "morning")
        self.location: str = self.memories.get("location", self.home)

        self._register_in_village()


#V1.3 Because we use free limiting API calls
    def _safe_llm_call(self, prompt_text: str, system_instruction, response_schema, max_retries = 5):
            """Call Gemini:
            - persona/rules go in system_instruction (cached by Gemini across the
            conversation instead of being re-parsed as part of the user turn)
            - contents holds only the turn-specific situation
            - response_mime_type + response_schema constrain the model to return
            exactly the JSON shape we need

            Added :
            Safety for when we crossed the API token or request limit
            """

            #Make sure the LLM responds in the wished format following instructions
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=1.0,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
            )
    
            attempt = 0
            while attempt < max_retries:
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt_text,
                        config=config,
                    )
                    return MockResponse(response.text)

                #If we receive an error:
                except GoogleAPIError as e:
                    attempt += 1
                    status = getattr(e, "code", None)
                    #Depending on the error we get, change the return:
                    wait_time = 15 #Fine for a small personal project
                    if status == 429:
                        #Rate limit
                        print(f"\n[Gemini Rate Limit] {self.name} hit 429. Waiting {wait_time:.1f}s "
                            f"(attempt {attempt}/{max_retries})...")
                    elif status is not None and 500 <= status < 600:
                        #Server-side error.
                        print(f"\n[Gemini Server Error {status}] {self.name} paused. Retrying in "
                            f"{wait_time:.1f}s (attempt {attempt}/{max_retries})...")
                    else:
                        #Something other error (bad request, auth, etc.) wont fix itself
                        #by retrying, but we still dont want one bad turn to
                        #crash the whole loop.
                        print(f"\n[Gemini API Error {status}] {self.name}: {e}. Retrying in "
                            f"{wait_time:.1f}s (attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
            return MockResponse("{}")


#---=== Reading and Writing ===---

    def _load_memories(self) -> dict:
        """
        Load memories if file exists, if it doesnt exist it loads an initial state
        """
        if self.memory_file.exists():
            return json.loads(self.memory_file.read_text())
        return{'observations':[], 'relationships': {}, 'day':0, 'time': 'morning', 'location': self.home}

    
    def _save_memories(self) -> None:
        """
        Save the relationships, location and the memories
        """
        self.memories["relationships"]= self.relationships
        self.memories['location']= self.location
        self.memory_file.write_text(json.dumps(self.memories, indent=2))


    def _load_village_state(self) -> dict:
        """
        Load the information from all agents
        """
        if Village_State.exists():
            try:
                return json.loads(Village_State.read_text())

            except json.JSONDecodeError:
                pass

        return{"agents": {}, 'log': []}

    def _append_to_log(self, entry: dict) -> None:
        """
        Write an entry to the log
        The log is a for fun record
        """
        with open(Village_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

    def _register_in_village(self) -> None:
        """
        Write agent information to the village_state, similar to _append_to_log
        """

        clock = load_village_clock()
        state = self._load_village_state()
        existing = state["agents"].get(self.name, {})
        state["agents"][self.name] = {
            'last_action': existing.get('last_action'),
            'day': clock['day'],
            'time': clock['time'],
            'location': self.location,
            'home': self.home,
            'update_time': datetime.datetime.now().isoformat(),
        }
        Village_State.write_text(json.dumps(state, indent=2))


    def  _update_village_state(self, action_summary:str) -> None:
        """
        Update the village log and state by saving the (last) action of the agent at a given day/time/location
        The state is what is actually used by the agent
        """
        clock = load_village_clock()
        entry={
                'agent': self.name,
                'day': clock['day'],
                'time': clock['time'],
                'location': self.location,
                'action': action_summary,
                'IRL_time': datetime.datetime.now().isoformat(),
                }
        #Append this entry to the log
        self._append_to_log(entry)
        
        state= self._load_village_state()
        state["agents"][self.name]={
            'last_action': action_summary,
            'day': clock['day'],
            'time': clock['time'],
            'location': self.location,
            'home': self.home,
            'update_time': entry['IRL_time'],
        }

        #Add this entry to the state, save only the last 'Recent' entries
        state["log"].append(entry)
        state["log"] = state["log"][-Recent:] 
        Village_State.write_text(json.dumps(state, indent=2))

        
# ---=== Memories ===---

    def add_memory(self, text: str, importance: int=3) -> None:
        clock = load_village_clock()
        self.memories['observations'].append(
            {
                "day": clock['day'],
                'time': clock['time'],
                "text": text,
                "importance": importance,
                'IRL_time': datetime.datetime.now().isoformat()
            }
        )
        
    def retrieve_relevant_memories(self,k:int=8) -> list[dict]:

        obs=self.memories['observations']
        indexed= list(enumerate(obs))
        scored= sorted(
            indexed,
            key=lambda pair: pair[1].get('importance', 1)  * 0.6 + pair[0] * 0.01,
            reverse=True
        )
        return [o for _, o in scored[:k]]

    #TODO: Make importance goal dependent and personal + change over time.
    def _estimate_importance(self, text: str) -> int:
        """
        Temporary way of addressing importance, would like to make it more dynamic in the future
        """
        keywords = ["married", "fight", "died", "born", "moved", "betrayed", "promised", "built", 'promise', 'kill', 'hate', 'love']
        return 8 if any(k in text.lower() for k in keywords) else 3

    def perceive_others(self):
        """
        Perceive all other villagers current location and recent actions across the village
        
        TODO: Change this so you only know who is at your current location
        """
        state = self._load_village_state()
        others = {person: info for person, info in state.get("agents", {}).items() if person != self.name}
        recent_log = [e for e in state.get("log", []) if e["agent"] != self.name][-15:]
        return others, recent_log

    #Location builder
    def get_locations(self) -> list[str]:
        """
        Add locations everyone can visit + homes
        """
        state= self._load_village_state()
        
        #For now a single common gathering spot
        #Maybe could add locations of interest by the villagers themselves in the future
        locations = {"Village Square"}
        
        #Add every registered villager's personal home
        for info in state.get("agents", {}).values():
            if "home" in info:
                locations.add(info["home"])
                
        return sorted(list(locations))

    #All known homes -> the only valid places to sleep
    def get_homes(self, others: dict) -> dict[str, str]:
        """
        Returns {home_location: owner_name}, owner_name is 'you' for your own home.
        """
        homes = {self.home: "you"}
        for person, info in others.items():
            if info.get("home"):
                homes[info["home"]] = person
        return homes

# V1.5 ---=== Relationship Update ===---

    def _relationship_defaults(self, others:dict) -> None:
        """
        Add default relationship status for everyone you dont know
        """
        Changed= False
        for person in others:
            if person != self.name and person not in self.relationships:
                self.relationships[person]= dict(Default_Relationship)
                Changed= True
        if Changed:
            self._save_memories()

    def _format_relationship(self) -> str:
        """
        Response format for relationship updates.
        """
        if not self.relationships:
            return "No relationships yet"
        lines=[]
        for person, rel in self.relationships.items():
            lines.append(
                f"-{person}: {rel['type']} (affinity:  {rel['affinity']:+.1f},"
                f"familiarity: {rel['familiarity']}) — {rel['note']}"
            )
        return "\n".join(lines)

    def _apply_relationship_updates(self, updates: list, clock: dict) -> None:
        """
        Update the relationship IF there is a relationship update returned by the agent
        """
        for update in (updates or []):
            villager = update.get("villager")
            if not villager:
                continue

            #Only change the involved villager(s)
            rel = self.relationships.setdefault(villager, dict(Default_Relationship))

            if "affinity_change" in update:
                rel["affinity"] = max(-1.0, min(1.0, rel["affinity"] + float(update["affinity_change"])))
            if update.get("note"):
                rel["note"] = update["note"]
            if update.get("type"):
                rel["type"] = update["type"]

            rel["familiarity"] = rel.get("familiarity", 0) + 1 #Increase familiarity after interactions
            rel["last_interaction_day"] = clock["day"] #Improtant for relationship status decreasing after certain amount of time

# ---=== Decision Loop===---

    def decide_sleep_location(self) -> dict:
        """
        Called instead of decide_and_act()'s daytime path during night.
        A smaller LLM call: the agent picks which known home to sleep
        at — their own, or someone else's, if that fits their relationship
        and the moment.
        """

        #Get the time of the day and other information from villagers/memories
        clock = load_village_clock()
        others, _ = self.perceive_others()
        self._relationship_defaults(others)
        relevant_memories = self.retrieve_relevant_memories(k=5)
        homes = self.get_homes(others)

        #Text for the LLM prompt
        homes_text = "\n".join(f"- {loc} (home of {owner})" for loc, owner in homes.items())
        memory_text = "\n".join(f"- {m['text']}" for m in relevant_memories) or "No memories yet."
        relationships_text = self._format_relationship()

        #Instructions to follow for the reply
        system_instruction = f"""You are {self.name}, a {self.race} {self.gender} {self.profession}.
        Personality: {self.personality}
 
        {Relationship_Scale_Guide}
 
        You choose where you sleep each night, staying true to your personality
        and relationships. Nearly all nights you'll simply go home, and that's
        completely fine. But if you're close with someone, or the moment calls
        for it, you may choose to sleep at another villager's home instead. A
        sleepover like that is a meaningful sign of closeness and can deepen a
        relationship. Don't invite yourself somewhere that would be out of
        character or presumptuous given how well you actually know them."""
 
        prompt = f"""It is now night on Day {clock['day']}. You are currently at {self.location}.
 
        Your relationships:
        {relationships_text}
 
        Relevant memories:
        {memory_text}
 
        Known homes you could sleep at tonight (not everyone will be home, check this list before interacting):
        {homes_text}
 
        Decide where you sleep tonight and produce JSON matching the response schema:
        - sleep_location: one of the known homes listed above, exactly as written
        - action: one or two sentences on how you settle in for the night
        - relationship_updates: optional list of {{"villager": "Name", "affinity_change": "e.g. +0.05", "type": "e.g. friend", "note": "..."}} (could be multiple in 1 prompt)"""

        #Call the LLM
        response = self._safe_llm_call(
            prompt,
            system_instruction=system_instruction,
            response_schema=Sleep_Decision_Schema,
        )

        #Read the LLM response and clean it a bit
        raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()

        #Update data and villager information (relationships, location, memories, action)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"sleep_location": self.home, "action": f"{self.name} heads home to sleep.", "relationship_updates": {}}

        chosen = data.get("sleep_location")
        #Fallback: dont let a non-existing location break the code
        if chosen not in homes:
            chosen = self.home  

        self.location = chosen
        action = data.get("action") or f"{self.name} settles in at {chosen} for the night."

        self._apply_relationship_updates(data.get("relationship_updates"), clock)

        importance = 3 if chosen != self.home else 1  # sleeping elsewhere is noteworthy; sleeping home is routine
        self.add_memory(action, importance=importance)
        self._update_village_state(action)
        self._save_memories()

        #Return in a print statement
        print(f"[Day {clock['day']}, {clock['time']}] {self.name} ({self.location}): {action}")
        return data

    def decide_and_act(self) -> dict:
        """
        A LLM call to advance the villager and their relationship.
        Making use of most other functions and read in the state of the villagers
        """
        clock = load_village_clock()

        if clock['time'] == "night":
            return self.decide_sleep_location()

        #Get information ready for the prompt
        others, recent_log= self.perceive_others()
        self._relationship_defaults(others)
        relevant_memories= self.retrieve_relevant_memories()
        locations= self.get_locations()

        #Can interact with these characters
        present_with_me= [person for person, info in others.items() if info.get("location") == self.location]

        memory_text= "\n".join(f"- (Day {m['day']}, {m['time']}) {m['text']}" for m in relevant_memories) or "No memories yet."
        others_text = (
            "\n".join(
                f"- {n}: at {info.get('location', 'unknown')}, last did '{info.get('last_action') or 'nothing yet'}' "
                f"(Day {info.get('day')}, {info.get('time')})"
                for n, info in others.items()
            )
            or "No other villagers known yet."
        )

        relationships_text = self._format_relationship()
        log_text = "\n".join(f"- Day {e['day']}, {e['time']}: {e['agent']} at {e['location']} — {e['action']}" for e in recent_log) or "Nothing observed."
        locations_text = ", ".join(locations) or "Village Square"

        if present_with_me:
            presence_text =(
                f"Right now, {', '.join(present_with_me)} is/are also at {self.location} with you. "
                "Since you are at the same location you could interact, though you don't have to if it doesn't fit the moment. "
            )
        else:
            presence_text = "No one else is with you at your current location right now."

        #Instruction for the LLM
        system_instruction = f"""You are {self.name}, a {self.race} {self.gender} villager working as a {self.profession}.
 
            Personality: {self.personality}
            Home: {self.home}
 
            {Relationship_Scale_Guide}
 
            You are a whole person, not only your job — you need to work toward
            your long-term goals, but not every single moment. It's normal to
            rest or do something just because you feel like it, as long as you
            stay true to your personality. If your action involves another
            villager, name them explicitly. You may move to a different known
            location if that fits what you're doing, or stay where you are."""
 
        prompt = f"""Long term goals: {", ".join(self.goals)}
 
            Current state: Day {clock['day']}, {clock['time']}, currently at {self.location}
 
            {presence_text}
 
            Locations in the village: {locations_text}
 
            Your recent memories:
            {memory_text}
 
            Your relationships:
            {relationships_text}
 
            Other villagers you know about (not everyone will be at their home, check this before interacting):
            {others_text}
 
            Recent events in the village:
            {log_text}
 
            Decide what you do right now and produce JSON matching the response schema:
            - action: two or three sentences of what you do during the {clock['time']}
            - inner_thought: brief private reflection on why you do your action
            - relationship_updates: optional list of {{"villager": "Name", "affinity_change": "e.g. +0.05", "type": "e.g. friend", "note": "..."}} (could be multiple in 1 prompt)
            - new_goal: optional new or updated goal, or null
            - new_location: one of the known locations, or null to stay put"""


        #Reply of the LLM
        response = self._safe_llm_call(
            prompt,
            system_instruction=system_instruction,
            response_schema=Act_Decision_Schema,
        )

        #Extract the respons and clean it a bit
        raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()

        #Add data to their memories and the village state
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"action": raw, "inner_thought": "", "relationship_updates": {}, "new_goal": None, "new_location": None}

        action = data.get("action", "did nothing notable")
        self.add_memory(action, importance=self._estimate_importance(action))

        
        if data.get("inner_thought"):
            self.add_memory(f"(thought) {data['inner_thought']}", importance=2)

        self._apply_relationship_updates(data.get("relationship_updates"), clock)

        new_goal = data.get("new_goal")
        if new_goal and new_goal not in self.goals:
            self.goals.append(new_goal)

        new_location = data.get("new_location")
        if new_location and new_location in locations:
            self.location = new_location

        self._update_village_state(action)
        self._save_memories()

        print(f"[Day {clock['day']}, {clock['time']}] {self.name} ({self.location}): {action}")
        return data


def run_village(agents: list[VillageAgent], cycle_seconds: int = 60) -> None:
    """
    Let the fun begin!
    Run the calls for and advance time of the village.

    TODO: Could be interesting to randomize the order so other villagers could take the first action of the next day for instance.
    """
    #Prevents going over the RPM
    spacing = cycle_seconds / max(len(agents), 1)
    #Let the simulation begin
    print(f"Starting village loop — {len(agents)} agents, one full cycle every {cycle_seconds}s.")
    while True:
        #Start and advance the time
        cycle_start = time.time()
        clock = advance_village_clock()
        print(f"--- Day {clock['day']}, {clock['time']} ---")
        #Let all agents act one time
        for agent in agents:
            try:
                agent.decide_and_act()
            except Exception as e:
                print(f"Error during {agent.name}'s turn: {e}")
            time.sleep(spacing)

        #So the full cycle takes 'cycle_seconds' amount of seconds in total. 
        #Each call also takes time so this is less than spacing, or even 0 if the calls take a long time.
        elapsed = time.time() - cycle_start
        remaining = cycle_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)