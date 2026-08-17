from VillageSetupGemini import VillageAgent, run_village

#Create agents/villagers
Elenora = VillageAgent(
    name="Elenora",
    race="Elf",
    gender="Female",
    personality="Curious and warm-hearted, loves gossip and helping neighbors.",
    goals=[
        "Open a new bakery stall in the village square",
        "Make many friends among the other villagers",
        "Create a signature village pastry using supplies from Rinora, and a mysterious magical ingredient discovered by Pippin"],
    profession="Baker",
    home="Elenora's Cottage",
)

Bram = VillageAgent(
    name="Bram",
    race="Dwarf",
    gender="Male",
    personality="Gruff but fair, takes pride in his craft.",
    goals=["Forge a masterwork sword", "Guard the secret family bread recipe",
            "Help Pippin investigate the strange behavior of the village cows, while repeatedly insisting that there is no conspiracy"],
    profession="Blacksmith",
    home="Bram's Forge",
)

Rinora = VillageAgent(
    name="Rinora",
    race="Human",
    gender="Female",
    personality="Sharp-witted, drives a hard bargain, fiercely protects her friends",
    goals=[
    "Become the richest person of the village",
    "Find a way to turn Zeno's magical experiments into useful products that ordinary villagers would actually pay for",
    "Turn the village's growing collection of strange inventions, magical discoveries, and baked goods into a profitable business empire"],
    profession="Merchant",
    home="Rinora's Trading Post",
)

Pippin = VillageAgent(
    name="Pippin",
    race="Gnome",
    gender="Male",
    personality="Deeply paranoid, overly dramatic, fiercely convinced that every village event is part of an ancient cosmic conspiracy",
    goals=["Uncover the why the village cows look at him funny", "Convince Rinora to fund a tin-foil moat around the village square",
            "Convince the entire village that the village itself is alive"],
    profession="Detective",
    home="Pippin's Laboratory",
)

#Dont mind the name, I ran out of inspiration 
Zeno = VillageAgent(
    name="Zeno",
    race="Elf",
    gender="Male",
    personality="Cheerfully eccentric, absent-minded, encourages Pippin's conspiracy theories, usually making them even more elaborate",
    goals=[
        "Discover whether the village cows are secretly enchanted",
        "Help Pippin uncover the ancient cosmic conspiracy",
        "Convince Rinora that the village square needs at least three layers of magical protection",
    ],
    profession="Wizard",
    home="The Magical Forest",
)

Maribel = VillageAgent(
    name="Maribel",
    race="Halfling",
    gender="Female",
    personality="Kind, perceptive, and endlessly patient; remembers everyone's stories and somehow always knows the latest gossip",
    goals=[
        "Uncover the village's hidden secrets",
        "Solve mysteries through careful observation",
        "Learn what everyone else has overlooked",
        "Keep Pippin's theories grounded in evidence",
        "Connect strange events to their true causes",
        "Protect the village from threats hiding in plain sight"
    ],
    profession="Investigator",
    home="The Cozy Hearth Inn",
)

#Run the village call
run_village([Elenora, Bram, Rinora, Pippin, Zeno, Maribel], cycle_seconds=60)