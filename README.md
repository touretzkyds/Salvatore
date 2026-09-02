# Salvatore

Salvatore Ferrante is a 19th-century Italian domino maker who lives on a VEX AIM
robot. He watches real dominoes through the robot's camera and teaches a person
to play the Block Game, working through the concepts in a fixed prerequisite
order so nothing is explained before its groundwork is laid.

## How the pieces fit together

Three layers, with a deliberately strict division of labour:

| Layer | Where | Job |
|---|---|---|
| Perception | `aim_fsm/domino.py`, `lab8/`, `DominoObj` in `aim_fsm/worldmap.py` | Find tiles in the camera image, read their pips, track them in the world map |
| Rules | `aim_fsm/domino_game.py` | Decide what is and is not a legal move. The only referee |
| Teaching | `Salvatore.fsm` preambles + OpenAI | Decide what to *say*, and in what order to teach it |

`aim_fsm/domino_bridge.py` joins the first two: it reads tracked tiles out of the
world map, works out which ones form the played layout and which are in whose
hand, and offers that to the rules engine.

The important rule, and the one to preserve in any future change: **the language
model never decides whether a move is legal, and neither does the bridge.**
The bridge reports what it saw and `DominoBlockGameState` rules on it. A misread
tile therefore becomes an ordinary "you cannot play that" conversation instead of
a silently corrupted board.

Two graph structures run through the system. The teaching order is a DAG of
concept prerequisites (`DOMINO_CONCEPT_PREREQS` in `Salvatore.fsm`), and the
board itself is a chain of tiles joined at matching ends, which the bridge
recovers from tile positions each time it looks.

## Installing

```bash
pip install -r requirements-macos.txt      # or -linux / -windows
pip install -r requirements-vision.txt     # only if you want the camera
```

The vision requirements (`ultralytics`, `torch`, `torchvision`) are large and
entirely optional. Without them Salvatore still teaches; he just asks the player
to say their tiles out loud.

### Model weights

Weights are not in git. Put them in `weights/` — see [`weights/README.md`](weights/README.md)
for the list and where to get them. The one that matters for the default setup is
`weights/domino_segment.pt`.

### API keys

All keys come from the environment; none are read from source.

| Variable | Used for | If missing |
|---|---|---|
| `OPENAI_API_KEY` | Salvatore's dialogue | He cannot talk; this one is required |
| `ELEVENLABS_API_KEY` | His Italian voice | Speech falls back to gTTS |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud TTS, if selected | Speech falls back to gTTS |

## Running

```bash
python main.py
```

then, at the prompt:

```python
runfsm('Salvatore')
```

Salvatore turns the camera on by himself. If the weights or the vision packages
are missing he prints why, and carries on teaching by ear.

`runfsm` constructs him with no arguments. To change how he starts — skipping the
camera, or refusing to start without it — build him yourself instead:

```python
from Salvatore import Salvatore
salvatore = Salvatore(domino=False)          # conversation only
salvatore = Salvatore(domino_optional=False) # insist on perception
salvatore.start()
```

`DominoWorldMap.py` is a smaller program for checking perception on its own:
run it and type `tm` to print every domino currently in the world map.

## Setting up the table

The bridge tells the played board from the two hands by geometry, so the
physical layout matters:

* The player sits across the table, facing the robot. Construct him with
  `Salvatore(domino_player_side='near')` if they sit on the robot's own side.
* Played tiles go end to end in a line. Two tiles count as joined when their
  ends are within 30 mm *and* the touching pips match, which is what keeps a
  tidily arranged hand from being mistaken for a board.
* To have Salvatore recognise the *opening* tile of a game, tell him where tiles
  get played:

  ```python
  Salvatore(domino_play_area=(0, 250, 80))   # x, y, radius in mm, world frame
  ```

  A lone tile on an empty table is otherwise indistinguishable from one lying
  beside somebody's hand, so without a play area Salvatore waits to be told
  about the first move and reads the rest himself.

Pip readings are settled by majority vote over several camera frames, so a tile
takes a moment to be named after it lands. That is deliberate: one bad frame
should not rename a tile.

## The voice

`SoundActuator` in `aim_fsm/actuators.py` selects the speech engine through three
class attributes. Salvatore ships with:

```python
TTS_API = 'elevenlabs'
TTS_VOICE = 'yowh82B72eMNrxcxHgBh'  # Lorenzo Prada - Refined Italian accent
TTS_PARAMS = {'model_id': 'eleven_multilingual_v2', 'output_format': 'mp3_44100_128'}
```

The voice has to be in the My Voices list of the account that owns
`ELEVENLABS_API_KEY`. Commented examples for Google and OpenAI sit just above the
active selection.

## Changing the state machine

Edit `Salvatore.fsm` and regenerate:

```bash
./genfsm Salvatore.fsm
```

Never edit `Salvatore.py` directly; it is generated and your changes will be
overwritten.

## Tests

```bash
python -m unittest aim_fsm.test_domino_game aim_fsm.test_domino_bridge aim_fsm.test_domino_fsm_wiring
```

These need no robot, camera, or weights. They cover the rules engine, the
bridge's geometry and pip voting, and the FSM node that applies a camera-seen
move — including that an illegal placement is refused by the rules engine rather
than accepted by the bridge.

## Troubleshooting

**"Domino perception unavailable"** — either `requirements-vision.txt` is not
installed or `weights/domino_segment.pt` is missing. Salvatore continues without
the camera.

**Tiles are seen but never named.** The pip vote has not settled. Check that the
tiles are within about 800 mm and that `domino_labeling` is on (it is by
default). The world map only updates while the robot is stopped.

**A hand is being read as the board.** The tiles in the hand are touching end to
end and happen to match. Spread them out, or move them further from the play
area.

**Salvatore speaks, but not in Italian.** `ELEVENLABS_API_KEY` is unset or the
voice is not in that account, so speech fell back to gTTS. The console says which.

## Credits

Built on vex-aim-tools by David Touretzky, Carnegie Mellon University, which
derives from cozmo-tools and, before that, the Tekkotsu framework by Ethan
Tira-Thompson and David Touretzky.

Duoduo Qian wrote the PyQt6 viewers. Boden Moraski wrote the safety section of
the system prompt. The Salvatore persona, the teaching DAG, and the Block Game
rules engine came from the Salvatore teaching project; the domino perception
stack came from the vision project.
